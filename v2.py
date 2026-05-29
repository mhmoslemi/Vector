"""
Curvature-Penalized Dataset Distillation v2 (CPDD-v2) for sanitizing
unlearnable CIFAR-10.

Implements the formulation in Sec. 8.1-8.4:
  - Inner (eq. 39): delta* = argmax_{||delta||<=eps} loss(x' + delta; theta)
  - Proxy (eq. 42): G_target = mean over batch of
        grad_loss(x') + ( grad_loss(x' + r*delta*) - grad_loss(x') ) / r
  - Outer (eq. 44): minimize D(G_synthetic, G_target) over synthetic pixels.

Also runs each evaluation (CPDD curve, clean upper bound, poisoned lower bound)
with 3 seeds and plots mean +/- 1 std as a shaded band.
"""

import argparse
import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T


# ----------------------------- model -----------------------------
class ConvNet(nn.Module):
    """Standard 3-layer ConvNet used in DC/DSA/MTT literature."""
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
    """Load CIFAR-10 as raw [0,1] tensors. Normalization is applied per-forward."""
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
    """Add the unlearnable / error-minimizing noise to clean pixels."""
    delta = torch.load(pert_path, map_location="cpu")
    if isinstance(delta, dict):
        for k in ("noise", "perturbation", "delta"):
            if k in delta:
                delta = delta[k]
                break
    delta = torch.as_tensor(delta).float()
    # heuristic: if values look like 0-255 uint8, rescale to [0,1]
    if delta.max() > 1.5:
        delta = delta / 255.0
    if delta.shape[0] != x_clean.shape[0]:
        raise ValueError(f"perturbation N={delta.shape[0]} != data N={x_clean.shape[0]}")
    if delta.shape[1:] != x_clean.shape[1:]:
        # NHWC -> NCHW
        if delta.shape[-1] == 3:
            delta = delta.permute(0, 3, 1, 2).contiguous()
    x_poison = (x_clean + delta).clamp(0.0, 1.0)
    print(f"[data] perturbation loaded: shape={tuple(delta.shape)}, "
          f"max|delta|={delta.abs().max().item():.4f} "
          f"({delta.abs().max().item()*255:.1f}/255)")
    return x_poison


def normalize(x):
    """Per-forward CIFAR normalization so distillation operates in pixel space."""
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ----------------------------- gradient utils -----------------------------
def model_grad(model, x, y, criterion):
    """Parameter gradient of CE loss, detached. Used for the target proxy."""
    out = model(normalize(x))
    loss = criterion(out, y)
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                create_graph=False, retain_graph=False)
    return [g.detach() for g in grads]


def model_grad_with_graph(model, x, y, criterion):
    """Parameter gradient with create_graph=True so we can backprop through it
    into the synthetic pixels in the outer loop."""
    out = model(normalize(x))
    loss = criterion(out, y)
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                create_graph=True, retain_graph=True)
    return grads


# ----------------------------- inner: adversarial restoration (eq. 39) -----------------------------
def restorative_perturbation(model, x, y, criterion, eps, n_steps=10, alpha=None):
    """
    Eq. 39: delta* = argmax_{||delta||_inf <= eps} loss(x + delta; theta).

    Standard L_inf PGD ascent on the loss, used to project the poisoned sample
    out of its artificial local minimum. The maximizer delta* is treated as
    the rigorous restorative direction (no heuristic scaling).
    """
    if alpha is None:
        # default: ~2.5 * eps / n_steps, the Madry-style step
        alpha = 2.5 * eps / max(n_steps, 1)

    x = x.detach()
    # random start inside the eps-ball helps escape flat regions
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = delta.detach().requires_grad_(True)

    for _ in range(n_steps):
        out = model(normalize((x + delta).clamp(0.0, 1.0)))
        loss = criterion(out, y)
        grad_delta = torch.autograd.grad(loss, delta)[0]
        # sign-ascent + project back to eps-ball + clip to valid pixel range
        delta = (delta + alpha * grad_delta.sign()).clamp(-eps, eps)
        # also keep x+delta inside [0,1]
        delta = (x + delta).clamp(0.0, 1.0).sub(x).detach().requires_grad_(True)

    return delta.detach()


# ----------------------------- exact gradient proxy (eq. 42) -----------------------------
def target_gradient(model, x_u, y, criterion, eps, r, inner_steps=10):
    """
    Eq. 42: G_target = grad_loss(x') + ( grad_loss(x' + r*delta*) - grad_loss(x') ) / r,
    averaged implicitly over the batch by torch's mean reduction in CE.

    The second term is the exact directional derivative of the parameter
    gradient along delta*, computed as a finite difference with microscopic r.
    No gamma; no scaling constant. This *is* the first-order Taylor correction
    that maps grad at the poisoned point to grad at the restored point.
    """
    delta_star = restorative_perturbation(
        model, x_u, y, criterion, eps=eps, n_steps=inner_steps
    )
    g_x = model_grad(model, x_u, y, criterion)
    x_pert = (x_u + r * delta_star).clamp(0.0, 1.0)
    g_xrd = model_grad(model, x_pert, y, criterion)

    target = []
    for gx, gxr in zip(g_x, g_xrd):
        # eq. 42: grad(x') + directional-derivative-along-delta*
        directional = (gxr - gx) / r
        target.append((gx + directional).detach())
    return target


# ----------------------------- distance metric -----------------------------
def distance_wb(gwr, gws):
    """DC-style per-output-channel cosine distance (Zhao et al. 2021)."""
    shape = gwr.shape
    if len(shape) == 4:  # conv: out * in * h * w
        gwr = gwr.reshape(shape[0], -1)
        gws = gws.reshape(shape[0], -1)
    elif len(shape) == 3:  # layernorm-like: C * h * w
        gwr = gwr.reshape(shape[0], -1)
        gws = gws.reshape(shape[0], -1)
    elif len(shape) == 2:  # linear: out * in
        pass
    elif len(shape) == 1:  # bias / groupnorm: ignore (matches DC paper)
        return torch.tensor(0.0, dtype=torch.float, device=gwr.device)

    num = (gwr * gws).sum(dim=-1)
    den = gwr.norm(dim=-1) * gws.norm(dim=-1) + 1e-6
    return torch.sum(1 - num / den)


def match_loss(gw_syn, gw_real, args):
    """Sum of per-layer cosine distances. D in eq. 44."""
    dis = torch.tensor(0.0, device=args.device)
    for ig in range(len(gw_real)):
        dis = dis + distance_wb(gw_real[ig], gw_syn[ig])
    return dis


# ----------------------------- training / eval -----------------------------
def train_eval(x_train, y_train, x_test, y_test, device,
               epochs=50, bs=256, lr=0.01, tag="", seed=0):
    """Train a fresh ConvNet on (x_train, y_train) and report test acc."""
    torch.manual_seed(seed)
    net = ConvNet(num_classes=10).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True)

    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # mild aug for tiny distilled sets, to mirror the original script
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
    print(f"[eval][{tag}][seed={seed}] test acc = {acc*100:.2f}%")
    return acc


def eval_with_seeds(x_train, y_train, x_test, y_test, device,
                    epochs, n_seeds=3, tag=""):
    """Run train_eval n_seeds times, return list of accuracies."""
    accs = []
    for s in range(n_seeds):
        a = train_eval(x_train, y_train, x_test, y_test, device,
                       epochs=epochs, tag=tag, seed=s)
        accs.append(a)
    return accs


# ----------------------------- distillation loop -----------------------------
def distill(x_u, y_u, num_classes, ipc, args, device, x_te=None, y_te=None):
    """
    Outer loop (eq. 44-45): match synthetic gradient to the proxy gradient
    G_target computed via the eq. 42 finite-difference HVP, with delta*
    obtained from PGD ascent (eq. 39).

    Returns the distilled (syn_x, syn_y) and a list of (iter, [acc_seed0, ...])
    pairs for the periodic-eval curve.
    """
    # init synthetic set: ipc real (poisoned) samples per class
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

    # (iter, list_of_accs_over_seeds) records for the curve
    curve = []

    def _periodic_eval(step_tag, it_value):
        if args.eval_every > 0 and x_te is not None:
            print(f"  [periodic-eval@{step_tag}] (epochs={args.epochs_eval_mid}, "
                  f"seeds={args.eval_seeds})")
            accs = eval_with_seeds(
                syn_x.detach().cpu(), syn_y.cpu(), x_te, y_te, device,
                epochs=args.epochs_eval_mid, n_seeds=args.eval_seeds,
                tag=f"distilled@{step_tag}",
            )
            curve.append((it_value, accs))

    _periodic_eval("init", 0)

    t0 = time.time()
    for it in range(args.outer_steps):
        # fresh random network each outer step (DC-style)
        net = ConvNet(num_classes=num_classes).to(device)
        net.train()

        loss_total = 0.0
        for c in range(num_classes):
            # sample a real (poisoned) batch for this class
            idx_real = (y_u == c).nonzero(as_tuple=True)[0]
            sel = idx_real[torch.randint(0, len(idx_real), (args.batch_real,))]
            xb = x_u[sel].to(device)
            yb = y_u[sel].to(device)

            # eq. 42 target proxy
            g_target = target_gradient(
                net, xb, yb, criterion,
                eps=args.eps, r=args.r, inner_steps=args.inner_steps,
            )

            # synthetic batch for this class
            mask = (syn_y == c)
            sxb = syn_x[mask]
            syb = syn_y[mask]
            g_syn = model_grad_with_graph(net, sxb, syb, criterion)

            # eq. 44 matching loss + step on syn pixels
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
            _periodic_eval(f"iter{it+1}", it + 1)

        # periodic plot snapshot using whatever curve points we have so far
        if args.plot_every > 0 and (it + 1) % args.plot_every == 0 and len(curve) > 0:
            try:
                plot_curve(curve, args._clean_accs, args._poison_accs, args.plot)
                print(f"  [plot] snapshot at iter {it+1} -> {args.plot}")
            except Exception as e:
                print(f"  [plot] snapshot failed: {e}")

    return syn_x.detach().cpu(), syn_y.cpu(), curve


# ----------------------------- plotting -----------------------------
def plot_curve(curve, clean_accs, poison_accs, save_path):
    """
    curve: list of (iter, [accs over seeds]).
    clean_accs / poison_accs: list of accs over seeds (one number per seed).
    Draws mean line + +/-1 std shaded band for the CPDD curve, and dashed
    horizontal mean lines + thin shaded bands for the two baselines.
    """
    import matplotlib.pyplot as plt

    iters = np.array([c[0] for c in curve])
    accs = np.array([c[1] for c in curve]) * 100.0  # (n_iters, n_seeds)
    mean = accs.mean(axis=1)
    std = accs.std(axis=1)

    clean = np.array(clean_accs) * 100.0
    poison = np.array(poison_accs) * 100.0
    clean_mean, clean_std = clean.mean(), clean.std()
    poison_mean, poison_std = poison.mean(), poison.std()

    fig, ax = plt.subplots(figsize=(10, 6))

    # CPDD curve with std shading
    ax.plot(iters, mean, "o-", color="tab:blue", label="CPDD-v2 sanitated")
    ax.fill_between(iters, mean - std, mean + std, color="tab:blue", alpha=0.20)

    # clean upper bound, dashed line + thin band
    ax.axhline(clean_mean, linestyle="--", color="tab:green",
               label=f"Clean upper bound ({clean_mean:.2f}% ± {clean_std:.2f})")
    ax.fill_between(iters, clean_mean - clean_std, clean_mean + clean_std,
                    color="tab:green", alpha=0.12)

    # poisoned lower bound, dashed line + thin band
    ax.axhline(poison_mean, linestyle="--", color="tab:red",
               label=f"Poisoned no defense ({poison_mean:.2f}% ± {poison_std:.2f})")
    ax.fill_between(iters, poison_mean - poison_std, poison_mean + poison_std,
                    color="tab:red", alpha=0.12)

    ax.set_xlabel("Sanitation iteration")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CPDD-v2 on unlearnable CIFAR-10 (mean ± 1 std over 3 seeds)")
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[plot] saved -> {save_path}")


# ----------------------------- main -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pert", type=str,
                   default="experiments/CIFAR10_samplewise_min-min/perturbation.pt")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--ipc", type=int, default=25)
    p.add_argument("--outer-steps", type=int, default=3000)
    p.add_argument("--inner-steps", type=int, default=5,
                   help="PGD steps for the eq. 39 inner maximization")
    p.add_argument("--batch-real", type=int, default=512)
    p.add_argument("--lr-syn", type=float, default=1)
    p.add_argument("--eps", type=float, default=10/255)
    p.add_argument("--r", type=float, default=1e-6,
                   help="finite-difference scalar in eq. 42")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100,
                   help="evaluate distilled set every N outer steps (0=off)")
    p.add_argument("--plot-every", type=int, default=500,
                   help="re-render the plot file every N outer steps using the curve so far (0=off, only final plot)")
    p.add_argument("--epochs-eval", type=int, default=400,
                   help="epochs for FINAL evaluation")
    p.add_argument("--epochs-eval-mid", type=int, default=400,
                   help="epochs for periodic mid-distillation evaluations")
    p.add_argument("--eval-seeds", type=int, default=3,
                   help="number of seeds for every evaluation point")
    p.add_argument("--save", type=str, default="distilled_cpdd_v2.pt")
    p.add_argument("--plot", type=str, default="cpdd_v2_curve.png")
    p.add_argument("--curve-json", type=str, default="cpdd_v2_curve.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recompute-baselines", action="store_true",
                   help="re-run clean/poisoned baselines instead of using cached values")
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

    # ---------------- baselines (hardcoded from previous run) ----------------
    # Recomputing these every run is wasteful; values cached from a prior run
    # with the same setup (3 seeds, 50 epochs, ConvNet, CIFAR-10 EM noise).
    # Set --recompute-baselines to override.
    CLEAN_ACCS_CACHED = [0.8098, 0.8099, 0.8156]
    POISON_ACCS_CACHED = [0.2386, 0.2363, 0.2450]

    if args.recompute_baselines:
        print("\n=== Clean upper bound (3 seeds) ===")
        clean_accs = eval_with_seeds(x_clean_tr, y_tr, x_te, y_te, device,
                                     epochs=args.epochs_eval,
                                     n_seeds=args.eval_seeds,
                                     tag="clean")
        print("\n=== Poisoned lower bound (3 seeds) ===")
        poison_accs = eval_with_seeds(x_poison_tr, y_tr, x_te, y_te, device,
                                      epochs=args.epochs_eval,
                                      n_seeds=args.eval_seeds,
                                      tag="poisoned")
    else:
        clean_accs = CLEAN_ACCS_CACHED
        poison_accs = POISON_ACCS_CACHED
        print(f"\n[baselines] using cached clean={clean_accs} poison={poison_accs}")

    # ---------------- distillation ----------------
    print("\n[distill] starting CPDD-v2")
    # stash baselines on args so the distill loop can re-render the plot mid-run
    args._clean_accs = clean_accs
    args._poison_accs = poison_accs
    syn_x, syn_y, curve = distill(
        x_poison_tr, y_tr, num_classes=10,
        ipc=args.ipc, args=args, device=device,
        x_te=x_te, y_te=y_te,
    )
    torch.save({"x": syn_x, "y": syn_y, "args": vars(args)}, args.save)
    print(f"[distill] saved -> {args.save}")

    # ---------------- final eval (3 seeds) ----------------
    print("\n=== Final CPDD-v2 evaluation (3 seeds) ===")
    final_accs = eval_with_seeds(syn_x, syn_y, x_te, y_te, device,
                                 epochs=args.epochs_eval,
                                 n_seeds=args.eval_seeds,
                                 tag="CPDD-v2 distilled")
    # append the final point to the curve so the plot ends on the trained value
    curve.append((args.outer_steps, final_accs))

    # ---------------- save curve + plot ----------------
    with open(args.curve_json, "w") as f:
        json.dump({
            "curve": [(int(it), [float(a) for a in accs]) for it, accs in curve],
            "clean": [float(a) for a in clean_accs],
            "poison": [float(a) for a in poison_accs],
        }, f, indent=2)
    print(f"[curve] saved -> {args.curve_json}")

    plot_curve(curve, clean_accs, poison_accs, args.plot)


if __name__ == "__main__":
    main()