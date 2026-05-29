#!/usr/bin/env python
"""
Direct bilevel poison generation on distilled data.

For target class c_i, loop j = 1..IPC. For each slot j:

    min_{s_p}    L_adv(x_t, y_adv; theta*(s_p))
    s.t.         theta*(s_p) = argmin_theta
                    L_train((S_c \\ {s_{c,j}^i}) ∪ S_c' ∪ {s_p}, Y ; theta)

The inner argmin is unrolled as n_inner steps of full-batch (S)GD over the
surrogate set with create_graph=True, giving the exact bilevel gradient on
the unrolled trajectory (MetaPoison-style direct solution).
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torchvision.utils import save_image, make_grid

# CIFAR-10 normalization constants (matches utils.py / main_dc.py)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

def denorm(t):
    """Convert a normalized (C,H,W) or (N,C,H,W) tensor back to [0,1] for saving."""
    out = t.clone().float()
    mean = torch.tensor(CIFAR10_MEAN, device=t.device).view(-1, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=t.device).view(-1, 1, 1)
    if out.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    out = out * std + mean
    return out.clamp(0, 1)


# =====================================================================
# ConvNet used by DC / DM / MTT (width=128, depth=3, InstanceNorm)
# =====================================================================
class ConvNet(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3,
                 net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32)):
        super().__init__()
        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        self.classifier = nn.Linear(
            shape_feat[0] * shape_feat[1] * shape_feat[2], num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)

    @staticmethod
    def _act(name):
        return {'relu': nn.ReLU(inplace=True),
                'leakyrelu': nn.LeakyReLU(0.01),
                'sigmoid': nn.Sigmoid()}[name]

    @staticmethod
    def _pool(name):
        return {'maxpooling': nn.MaxPool2d(2, 2),
                'avgpooling': nn.AvgPool2d(2, 2),
                'none': nn.Identity()}[name]

    @staticmethod
    def _norm(name, shape):
        if name == 'batchnorm':    return nn.BatchNorm2d(shape[0], affine=True)
        if name == 'layernorm':    return nn.LayerNorm(shape, elementwise_affine=True)
        if name == 'instancenorm': return nn.GroupNorm(shape[0], shape[0], affine=True)
        if name == 'groupnorm':    return nn.GroupNorm(4, shape[0], affine=True)
        return nn.Identity()

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers, in_c = [], channel
        shape_feat = [in_c, im_size[0], im_size[1]]
        for _ in range(net_depth):
            layers.append(nn.Conv2d(in_c, net_width, 3, padding=1))
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers.append(self._norm(net_norm, shape_feat))
            layers.append(self._act(net_act))
            in_c = net_width
            if net_pooling != 'none':
                layers.append(self._pool(net_pooling))
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat


# =====================================================================
# Functional, differentiable inner loop (unrolled GD with create_graph)
# =====================================================================
def unrolled_inner_loop(model, init_params, X, Y,
                        n_steps, lr, momentum=0.0, bptt_truncate=None,
                        inner_batch_size=None):
    """
    Full-batch (S)GD starting from init_params on (X, Y) for n_steps.

    If `bptt_truncate=K`, only the last K steps carry a differentiable graph.
    If `inner_batch_size=B`, each step samples B examples (reduces memory ~N/B).
    """
    params = dict(init_params)
    vel = {k: torch.zeros_like(v) for k, v in params.items()} if momentum > 0 else None

    N = X.shape[0]
    for t in range(n_steps):
        if bptt_truncate is not None and (n_steps - t) > bptt_truncate:
            params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
            if vel is not None:
                vel = {k: v.detach() for k, v in vel.items()}

        if inner_batch_size is not None and inner_batch_size < N:
            idx = torch.randperm(N, device=X.device)[:inner_batch_size]
            Xb, Yb = X[idx], Y[idx]
        else:
            Xb, Yb = X, Y

        logits = functional_call(model, params, Xb)
        loss = F.cross_entropy(logits, Yb)

        grads = torch.autograd.grad(
            loss, list(params.values()),
            create_graph=True, allow_unused=False)

        new_params = {}
        for (k, p), g in zip(params.items(), grads):
            if momentum > 0:
                vel[k] = momentum * vel[k] + g
                new_params[k] = p - lr * vel[k]
            else:
                new_params[k] = p - lr * g
        params = new_params

    return params


# =====================================================================
# Surrogate construction
# =====================================================================
def build_surrogate(S_c, Y_c, surr_idx, replace_idx, s_p):
    """
    Build inner-loop surrogate from a pre-selected subset of S_c.

    surr_idx: 1-D LongTensor of global indices into S_c to include,
              must NOT contain replace_idx (s_p takes that slot).
    s_p's label is taken from Y_c[replace_idx].
    """
    X = torch.cat([S_c[surr_idx], s_p.unsqueeze(0)], dim=0)
    Y = torch.cat([Y_c[surr_idx], Y_c[replace_idx:replace_idx + 1]], dim=0)
    return X, Y


# =====================================================================
# Outer loop for a single poison slot
# =====================================================================
def optimize_one_poison(make_model, S_c, Y_c, surr_idx,
                        replace_idx, s_init,
                        target_x, target_y_adv,
                        n_outer, outer_lr,
                        n_inner, inner_lr, inner_momentum,
                        n_init_ensemble, bptt_truncate, inner_batch_size,
                        pixel_clamp, device, verbose_every=5):
    s_p = s_init.clone().detach().to(device).requires_grad_(True)
    opt = torch.optim.Adam([s_p], lr=outer_lr)

    for step in range(n_outer):
        X, Y = build_surrogate(S_c, Y_c, surr_idx, replace_idx, s_p)

        adv_acc = 0.0
        for _ in range(n_init_ensemble):
            # Fresh theta_0 per outer step per ensemble member
            fresh = make_model().to(device)
            init_params = {n: p.detach().clone().requires_grad_(True)
                           for n, p in fresh.named_parameters()}

            final_params = unrolled_inner_loop(
                fresh, init_params, X, Y,
                n_steps=n_inner, lr=inner_lr,
                momentum=inner_momentum,
                bptt_truncate=bptt_truncate,
                inner_batch_size=inner_batch_size,
            )

            logits_t = functional_call(fresh, final_params, target_x)
            adv_acc = adv_acc + F.cross_entropy(logits_t, target_y_adv)

        adv_loss = adv_acc / n_init_ensemble

        opt.zero_grad(set_to_none=True)
        adv_loss.backward()
        opt.step()

        if pixel_clamp is not None:
            with torch.no_grad():
                s_p.clamp_(pixel_clamp[0], pixel_clamp[1])

        if step == 0 or (step + 1) % verbose_every == 0 or step == n_outer - 1:
            print(f'      outer {step+1:>3d}/{n_outer}   L_adv = {adv_loss.item():.4f}')

    return s_p.detach().cpu()


# =====================================================================
# Main: loop over IPC slots of the target class
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--S_c_path',
        default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DC_CIFAR10_ConvNet_100ipc.pt')
    ap.add_argument('--target_class', type=int,default = 2 ,
                    help='c_i: class whose IPC slots get poisoned.')
    ap.add_argument('--adv_label',    type=int, default = 5 ,
                    help='y_adv: label the attacker wants x_t to be predicted as.')
    ap.add_argument('--target_x_path', default=None,
                    help='.pt tensor holding x_t (same preprocessing as distilled data). '
                         'Shape (3,32,32) or (1,3,32,32). If omitted a placeholder is used.')

    ap.add_argument('--ipc',         type=int, default=100)
    ap.add_argument('--num_classes', type=int, default=10)

    # outer
    ap.add_argument('--n_outer',  type=int,   default=30)
    ap.add_argument('--outer_lr', type=float, default=0.1)

    # inner
    ap.add_argument('--n_inner',        type=int,   default=10)
    ap.add_argument('--inner_lr',       type=float, default=0.01)
    ap.add_argument('--inner_momentum', type=float, default=0.0)

    # robustness / memory knobs
    ap.add_argument('--n_init_ensemble', type=int, default=2,
                    help='Resample theta_0 this many times per outer step, average L_adv.')
    ap.add_argument('--bptt_truncate',   type=int, default=None,
                    help='If set, only last K inner steps carry grad (memory saver).')
    ap.add_argument('--inner_batch_size', type=int, default=None,
                    help='Mini-batch size for inner-loop gradient steps. '
                         'None = full surrogate set (recommended with --inner_n_per_class).')
    ap.add_argument('--inner_n_per_class', type=int, default=10,
                    help='Images per class to use in the inner-loop surrogate (from S_c only). '
                         'Smaller = stronger gradient signal for s_p. None = use all ipc.')
    ap.add_argument('--pixel_clamp', type=float, nargs=2, default=None,
                    help='Optional (low, high) clamp on s_p after each outer step.')

    ap.add_argument('--save_path', default='./poisoned_class.pt')
    ap.add_argument('--img_dir',   default='./poison_images',
                    help='Directory to save per-slot PNG images (init / optimized / delta).')
    ap.add_argument('--device',    default='cuda')
    ap.add_argument('--seed',      type=int, default=0)
    args = ap.parse_args()
    

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'device = {device}')

    # ---- load distilled dataset ----
    b1 = torch.load(args.S_c_path, map_location='cpu')
    S_c, Y_c = b1['data']
    S_c, Y_c = S_c.to(device), Y_c.to(device).long()
    print(f'S_c shape = {tuple(S_c.shape)}   labels = {tuple(Y_c.shape)}')

    # ---- target example x_t ----
    if args.target_x_path is not None:
        x_t = torch.load(args.target_x_path, map_location=device)
    else:
        print('[WARN] --target_x_path not provided; using placeholder (S_c[0]).')
        x_t = S_c[0:1].clone()
    if x_t.dim() == 3:
        x_t = x_t.unsqueeze(0)
    x_t   = x_t.to(device)
    y_adv = torch.tensor([args.adv_label], device=device, dtype=torch.long)

    # ---- per-class index map into S_c ----
    class_idx = (Y_c == args.target_class).nonzero(as_tuple=True)[0].tolist()
    assert len(class_idx) == args.ipc, \
        f'found {len(class_idx)} slots for class {args.target_class}, expected IPC={args.ipc}'

    # indices for every class (used to build the inner-loop surrogate)
    all_class_indices = {
        c: (Y_c == c).nonzero(as_tuple=True)[0].tolist()
        for c in range(args.num_classes)
    }
    n_per = args.inner_n_per_class  # None = use all

    # ---- model factory: fresh init per inner run ----
    def make_model():
        return ConvNet(channel=3, num_classes=args.num_classes,
                       net_width=128, net_depth=3,
                       net_norm='instancenorm', im_size=(32, 32))

    os.makedirs(args.img_dir, exist_ok=True)

    # ---- main loop: 100 poison slots ----
    poisons_init = []
    poisons = []
    for j, replace_idx in enumerate(class_idx):
        print(f'[class {args.target_class}] slot {j+1}/{args.ipc}  (global idx {replace_idx})')
        s_init = S_c[replace_idx].detach().clone().cpu()

        # build surrogate index: N images per class from S_c, replace_idx excluded
        surr_parts = []
        for c in range(args.num_classes):
            pool = [i for i in all_class_indices[c] if i != replace_idx]
            k = min(n_per, len(pool)) if n_per is not None else len(pool)
            chosen = torch.tensor(pool, device=device)[torch.randperm(len(pool), device=device)[:k]]
            surr_parts.append(chosen)
        surr_idx = torch.cat(surr_parts)
        print(f'  surrogate size: {surr_idx.numel()} images  (s_p is 1/{surr_idx.numel()+1})')

        s_star = optimize_one_poison(
            make_model=make_model,
            S_c=S_c, Y_c=Y_c, surr_idx=surr_idx,
            replace_idx=replace_idx, s_init=s_init,
            target_x=x_t, target_y_adv=y_adv,
            n_outer=args.n_outer,         outer_lr=args.outer_lr,
            n_inner=args.n_inner,         inner_lr=args.inner_lr,
            inner_momentum=args.inner_momentum,
            n_init_ensemble=args.n_init_ensemble,
            bptt_truncate=args.bptt_truncate,
            inner_batch_size=args.inner_batch_size,
            pixel_clamp=tuple(args.pixel_clamp) if args.pixel_clamp else None,
            device=device,
        )
        poisons_init.append(s_init.unsqueeze(0))
        poisons.append(s_star.unsqueeze(0))

        # save init | optimized | delta side by side
        delta = s_star - s_init
        delta_vis = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
        grid = make_grid([denorm(s_init), denorm(s_star), delta_vis], nrow=3, padding=2)
        img_path = os.path.join(args.img_dir, f'class{args.target_class}_slot{j+1:04d}.png')
        save_image(grid, img_path)

        # incremental save so a crash mid-run is not total loss
        S_p_init_partial = torch.cat(poisons_init, dim=0)
        S_p_partial      = torch.cat(poisons,      dim=0)
        torch.save({'S_p_init': S_p_init_partial,
                    'S_p': S_p_partial,
                    'Y_p': torch.full((S_p_partial.size(0),),
                                      args.target_class, dtype=torch.long),
                    'target_class': args.target_class,
                    'adv_label':    args.adv_label,
                    'args': vars(args),
                    'completed_slots': j + 1},
                   args.save_path)

    print(f'Saved final {tuple(S_p_partial.shape)} tensor to {args.save_path}')


if __name__ == '__main__':
    main()