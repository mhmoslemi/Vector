"""
Curvature-Penalized Dataset Distillation (CPDD) for sanitizing unlearnable CIFAR-10.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T


# ----------------------------- model -----------------------------
class ConvNet(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3,
                 im_size=(32, 32)):
        super().__init__()
        layers = []
        in_c = channel
        cur = im_size[0]
        for _ in range(net_depth):
            layers += [
                nn.Conv2d(in_c, net_width, 3, padding=1),
                nn.GroupNorm(net_width, net_width, affine=True),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2, 2),
            ]
            in_c = net_width
            cur //= 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(net_width * cur * cur, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


# ----------------------------- data -----------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(root="./data"):
    tf = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)

    def to_tensor(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
        return xs, ys

    xtr, ytr = to_tensor(train)
    xte, yte = to_tensor(test)
    return xtr, ytr, xte, yte


def apply_perturbation(x_clean, pert_path):
    delta = torch.load(pert_path, map_location="cpu")
    if isinstance(delta, dict):
        for k in ("noise", "perturbation", "delta"):
            if k in delta:
                delta = delta[k]
                break
    delta = torch.as_tensor(delta).float()
    if delta.max() > 1.5:
        delta = delta / 255.0
    if delta.shape[0] != x_clean.shape[0]:
        raise ValueError(f"perturbation N={delta.shape[0]} != data N={x_clean.shape[0]}")
    if delta.shape[1:] != x_clean.shape[1:]:
        if delta.shape[-1] == 3:
            delta = delta.permute(0, 3, 1, 2).contiguous()
    x_poison = (x_clean + delta).clamp(0.0, 1.0)
    print(f"[data] perturbation loaded: shape={tuple(delta.shape)}, "
          f"max|delta|={delta.abs().max().item():.4f} "
          f"({delta.abs().max().item()*255:.1f}/255)")
    return x_poison


def normalize(x):
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ----------------------------- gradient utils -----------------------------
def model_grad(model, x, y, criterion):
    out = model(normalize(x))
    loss = criterion(out, y)
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                create_graph=False, retain_graph=False)
    return [g.detach() for g in grads]


def model_grad_with_graph(model, x, y, criterion):
    out = model(normalize(x))
    loss = criterion(out, y)
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                create_graph=True, retain_graph=True)
    return grads


# ----------------------------- inner: surrogate curvature -----------------------------
def surrogate_perturbation(model, x, y, criterion, eps, n_steps=1, alpha=None):
    if alpha is None:
        alpha = eps
    x = x.detach()
    g0 = model_grad(model, x, y, criterion)
    g0_flat = torch.cat([g.reshape(-1) for g in g0]).detach()

    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(n_steps):
        out = model(normalize(x + delta))
        loss = criterion(out, y)
        g = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        g_flat = torch.cat([gi.reshape(-1) for gi in g])
        obj = ((g_flat - g0_flat) ** 2).sum()
        grad_delta = torch.autograd.grad(obj, delta)[0]
        delta = (delta + alpha * grad_delta.sign()).clamp(-eps, eps).detach()
        delta.requires_grad_(True)
    return delta.detach()


# ----------------------------- defensive proxy gradient (eq.3) -----------------------------
def defensive_target_gradient(model, x_u, y, criterion, eps, gamma, r, inner_steps=1):
    delta_p = surrogate_perturbation(model, x_u, y, criterion, eps=eps, n_steps=inner_steps)
    g_x = model_grad(model, x_u, y, criterion)
    g_xrd = model_grad(model, (x_u + r * delta_p).clamp(0, 1), y, criterion)
    target = []
    for gx, gxr in zip(g_x, g_xrd):
        hvp = (gxr - gx) / r
        target.append((gx - gamma * hvp).detach())
    return target


# ----------------------------- distance metric -----------------------------
# def match_loss(g_syn, g_real):
#     loss = torch.tensor(0.0, device=g_syn[0].device)
#     for gs, gr in zip(g_syn, g_real):
#         if gs.dim() == 4:
#             gs_ = gs.reshape(gs.size(0), -1)
#             gr_ = gr.reshape(gr.size(0), -1)
#         elif gs.dim() == 2:
#             gs_ = gs
#             gr_ = gr
#         else:
#             gs_ = gs.reshape(1, -1)
#             gr_ = gr.reshape(1, -1)
#         num = (gs_ * gr_).sum(dim=-1)
#         den = gs_.norm(dim=-1) * gr_.norm(dim=-1) + 1e-6
#         loss = loss + (1 - num / den).sum()
#     return loss


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(gwr, gws)
    return dis



# ----------------------------- training / eval -----------------------------
def train_eval(x_train, y_train, x_test, y_test, device,
               epochs=50, bs=256, lr=0.01, tag=""):
    net = ConvNet(num_classes=10).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True)

    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if x_train.shape[0] < 1000:
                xb = T.RandomCrop(32, padding=4)(xb)
                xb = T.RandomHorizontalFlip()(xb)
            opt.zero_grad()
            loss = crit(net(normalize(xb)), yb)
            loss.backward()
            opt.step()
        sched.step()

    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), 512):
            xb = x_test[i:i+512].to(device)
            yb = y_test[i:i+512].to(device)
            pred = net(normalize(xb)).argmax(1)
            correct += (pred == yb).sum().item()
    acc = correct / len(x_test)
    print(f"[eval][{tag}] test acc = {acc*100:.2f}%")
    return acc





# ----------------------------- distillation loop -----------------------------
def distill(x_u, y_u, num_classes, ipc, args, device, x_te=None, y_te=None):
    s_imgs, s_labels = [], []
    for c in range(num_classes):
        idx = (y_u == c).nonzero(as_tuple=True)[0]
        sel = idx[torch.randperm(len(idx))[:ipc]]
        s_imgs.append(x_u[sel].clone())
        s_labels.append(torch.full((ipc,), c, dtype=torch.long))
    syn_x = torch.cat(s_imgs).to(device).requires_grad_(True)
    syn_y = torch.cat(s_labels).to(device)

    opt_syn = torch.optim.SGD([syn_x], lr=args.lr_syn, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"[distill] synthetic set: {syn_x.shape}, ipc={ipc}")

    def _periodic_eval(step_tag):
        if args.eval_every > 0 and x_te is not None:
            print(f"  [periodic-eval@{step_tag}] (epochs={args.epochs_eval_mid})")
            train_eval(syn_x.detach().cpu(), syn_y.cpu(), x_te, y_te, device,
                       epochs=args.epochs_eval_mid, tag=f"distilled@{step_tag}")

    _periodic_eval("init")

    t0 = time.time()
    for it in range(args.outer_steps):
        net = ConvNet(num_classes=num_classes).to(device)
        net.train()

        loss_total = 0.0
        for c in range(num_classes):
            idx_real = (y_u == c).nonzero(as_tuple=True)[0]
            sel = idx_real[torch.randint(0, len(idx_real), (args.batch_real,))]
            xb = x_u[sel].to(device)
            yb = y_u[sel].to(device)

            g_target = defensive_target_gradient(
                net, xb, yb, criterion,
                eps=args.eps, gamma=args.gamma, r=args.r,
                inner_steps=args.inner_steps,
            )

            mask = (syn_y == c)
            sxb = syn_x[mask]
            syb = syn_y[mask]
            g_syn = model_grad_with_graph(net, sxb, syb, criterion)

            loss_c = match_loss(g_syn, g_target, args)
            opt_syn.zero_grad()
            loss_c.backward()
            opt_syn.step()
            loss_total += loss_c.item()

            with torch.no_grad():
                syn_x.clamp_(0.0, 1.0)

        if (it + 1) % args.log_every == 0:
            print(f"  iter {it+1:5d}/{args.outer_steps}  "
                  f"match_loss={loss_total/num_classes:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

        if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            _periodic_eval(f"iter{it+1}")

    return syn_x.detach().cpu(), syn_y.cpu()


# ----------------------------- main -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pert", type=str,
                   default="experiments/CIFAR10_samplewise_min-min/perturbation.pt")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--ipc", type=int, default=100)
    p.add_argument("--outer-steps", type=int, default=5000)
    p.add_argument("--inner-steps", type=int, default=10)
    p.add_argument("--batch-real", type=int, default=256)
    p.add_argument("--lr-syn", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=8/255)
    p.add_argument("--gamma", type=float, default=1)
    p.add_argument("--r", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=80,
                   help="evaluate distilled set on test every N outer steps (0=off). Also evals at init.")
    p.add_argument("--epochs-eval", type=int, default=50,
                   help="epochs for final evaluation")
    p.add_argument("--epochs-eval-mid", type=int, default=50,
                   help="epochs for periodic mid-distillation evaluations")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--save", type=str, default="distilled_cpdd.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"[env] device={device}")

    print("[data] loading CIFAR-10")
    x_clean_tr, y_tr, x_te, y_te = load_cifar10(args.data_root)
    print(f"[data] train={tuple(x_clean_tr.shape)} test={tuple(x_te.shape)}")

    print(f"[data] applying perturbation from {args.pert}")
    x_poison_tr = apply_perturbation(x_clean_tr, args.pert)
    print(x_poison_tr.shape)
    print(x_clean_tr.shape)
    


    print("[distill] starting CPDD")
    syn_x, syn_y = distill(x_poison_tr, y_tr, num_classes=10,
                           ipc=args.ipc, args=args, device=device,
                           x_te=x_te, y_te=y_te)
    torch.save({"x": syn_x, "y": syn_y, "args": vars(args)}, args.save)
    print(f"[distill] saved -> {args.save}")

    args.eval = True
    if args.eval:
        print("\n=== Final Evaluation ===")
        # train_eval(x_clean_tr, y_tr, x_te, y_te, device,
        #            epochs=args.epochs_eval, tag="clean (upper bound)")
        # train_eval(x_poison_tr, y_tr, x_te, y_te, device,
        #            epochs=args.epochs_eval, tag="poisoned (no defense)")
        train_eval(syn_x, syn_y, x_te, y_te, device,
                   epochs=args.epochs_eval, tag="CPDD distilled")


if __name__ == "__main__":
    main()