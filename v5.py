"""
CPDD-v4: runs three distillation experiments and plots them on a single axis.

Modes:
  clean         - distill on clean CIFAR-10, vanilla DC (eps=0, no FD term)
  poisoned-dc   - distill on unlearnable CIFAR-10, vanilla DC (eps=0, no FD)
  poisoned-cpdd - distill on unlearnable CIFAR-10, with eq. 39-42 proxy

A single run of this script executes all three (or a subset via --modes) and
saves one combined plot with three curves + clean/poisoned baselines.
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
from torchvision.utils import save_image


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

    return (*to_tensor(train), *to_tensor(test))


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


def restorative_perturbation(model, x, y, criterion, eps, n_steps=10, alpha=None):
    """L_inf PGD ascent on the loss (eq. 39). Returns delta* in [-eps, eps]."""
    if eps == 0 or n_steps == 0:
        return torch.zeros_like(x)
    if alpha is None:
        alpha = 2.5 * eps / max(n_steps, 1)
    x = x.detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (x + delta).clamp(0.0, 1.0).sub(x).detach().requires_grad_(True)
    for _ in range(n_steps):
        out = model(normalize((x + delta).clamp(0.0, 1.0)))
        loss = criterion(out, y)
        g = torch.autograd.grad(loss, delta)[0]
        delta = (delta + alpha * g.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0.0, 1.0).sub(x).detach().requires_grad_(True)
    return delta.detach()


def target_gradient(model, x_u, y, criterion, eps, r, inner_steps, use_proxy):
    """
    If use_proxy is True: G_target = grad(l(x')) + (grad(l(x'+r*delta*)) - grad(l(x'))) / r
    If False: G_target = grad(l(x'))   (vanilla DC)
    """
    g_x = model_grad(model, x_u, y, criterion)
    if not use_proxy:
        return g_x
    delta_star = restorative_perturbation(model, x_u, y, criterion,
                                          eps=eps, n_steps=inner_steps)
    g_xrd = model_grad(model, (x_u + r * delta_star).clamp(0.0, 1.0), y, criterion)
    return [(gx + (gxr - gx) / r).detach() for gx, gxr in zip(g_x, g_xrd)]


# ----------------------------- distance metric -----------------------------
def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:
        gwr = gwr.reshape(shape[0], -1)
        gws = gws.reshape(shape[0], -1)
    elif len(shape) == 3:
        gwr = gwr.reshape(shape[0], -1)
        gws = gws.reshape(shape[0], -1)
    elif len(shape) == 2:
        pass
    elif len(shape) == 1:
        return torch.tensor(0.0, dtype=torch.float, device=gwr.device)
    num = (gwr * gws).sum(dim=-1)
    den = gwr.norm(dim=-1) * gws.norm(dim=-1) + 1e-6
    return torch.sum(1 - num / den)


def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0, device=args.device)
    for ig in range(len(gw_real)):
        dis = dis + distance_wb(gw_real[ig], gw_syn[ig])
    return dis


# ----------------------------- inner net training on synthetic data -----------------------------
def train_net_on_syn(net, image_syn, label_syn, args):
    dst = TensorDataset(image_syn.detach().clone(), label_syn.detach().clone())
    loader = DataLoader(dst, batch_size=args.batch_train, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.5)
    crit = nn.CrossEntropyLoss().to(args.device)
    net.train()
    for _ in range(args.inner_loop):
        for xb, yb in loader:
            xb = xb.to(args.device); yb = yb.to(args.device)
            opt.zero_grad()
            crit(net(normalize(xb)), yb).backward()
            opt.step()


# ----------------------------- training / eval -----------------------------
def train_eval(x_train, y_train, x_test, y_test, device,
               epochs=50, bs=256, lr=0.01, tag="", seed=0):
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
            if x_train.shape[0] < 1000:
                xb = T.RandomCrop(32, padding=4)(xb)
                xb = T.RandomHorizontalFlip()(xb)
            opt.zero_grad()
            crit(net(normalize(xb)), yb).backward()
            opt.step()
        sched.step()
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), 512):
            xb = x_test[i:i+512].to(device); yb = y_test[i:i+512].to(device)
            correct += (net(normalize(xb)).argmax(1) == yb).sum().item()
    acc = correct / len(x_test)
    print(f"[eval][{tag}][seed={seed}] test acc = {acc*100:.2f}%")
    return acc


def eval_with_seeds(x_train, y_train, x_test, y_test, device, epochs, n_seeds=3, tag=""):
    return [train_eval(x_train, y_train, x_test, y_test, device,
                       epochs=epochs, tag=tag, seed=s) for s in range(n_seeds)]


# ----------------------------- distillation loop -----------------------------
def distill(x_u, y_u, num_classes, ipc, args, device, x_te, y_te,
            use_proxy, eps, mode_tag):
    """One distillation run. Returns (syn_x, syn_y, curve)."""
    s_imgs, s_labels = [], []
    for c in range(num_classes):
        idx = (y_u == c).nonzero(as_tuple=True)[0]
        sel = idx[torch.randperm(len(idx))[:ipc]]
        s_imgs.append(x_u[sel].clone())
        s_labels.append(torch.full((ipc,), c, dtype=torch.long))
    image_syn = torch.cat(s_imgs).to(device).detach().requires_grad_(True)
    label_syn = torch.cat(s_labels).to(device)

    save_image(image_syn.detach().cpu().clamp(0.0, 1.0),
               f"vis_{mode_tag}_init.png", nrow=ipc)

    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_syn, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(device)
    indices_class = [(y_u == c).nonzero(as_tuple=True)[0] for c in range(num_classes)]

    def get_real_batch(c, n):
        idx = indices_class[c][torch.randint(0, len(indices_class[c]), (n,))]
        return x_u[idx].to(device), y_u[idx].to(device)

    print(f"[distill:{mode_tag}] synthetic set: {image_syn.shape}, ipc={ipc}, "
          f"outer_loop={args.outer_loop}, inner_loop={args.inner_loop}, "
          f"use_proxy={use_proxy}, eps={eps}")

    curve = []

    def _periodic_eval(step_tag, it_value):
        if x_te is None:
            return
        print(f"  [{mode_tag}@{step_tag}] (epochs={args.epochs_eval_mid}, "
              f"seeds={args.eval_seeds})")
        accs = eval_with_seeds(image_syn.detach().cpu(), label_syn.cpu(),
                               x_te, y_te, device,
                               epochs=args.epochs_eval_mid,
                               n_seeds=args.eval_seeds,
                               tag=f"{mode_tag}@{step_tag}")
        curve.append((it_value, accs))

    _periodic_eval("init", 0)

    t0 = time.time()
    if mode_tag == 'clean':
        LL = 10
    else:
        LL = args.Iteration + 1
    for it in range(LL):
        net = ConvNet(num_classes=num_classes).to(device)
        net.train()
        loss_avg = 0.0

        for ol in range(args.outer_loop):
            loss = torch.tensor(0.0, device=device)
            for c in range(num_classes):
                xb_real, yb_real = get_real_batch(c, args.batch_real)
                # gw_real = target_gradient(net, xb_real, yb_real, criterion,
                #                           eps=eps, r=args.r,
                #                           inner_steps=args.inner_steps,
                #                           use_proxy=use_proxy)
                # img_syn_c = image_syn[c*ipc:(c+1)*ipc]
                # lab_syn_c = label_syn[c*ipc:(c+1)*ipc]
                # gw_syn = model_grad_with_graph(net, img_syn_c, lab_syn_c, criterion)
                # loss = loss + match_loss(gw_syn, gw_real, args)

                gw_real = target_gradient(net, xb_real, yb_real, criterion,
                                          eps=eps, r=args.r,
                                          inner_steps=args.inner_steps,
                                          use_proxy=use_proxy)
                
                # --- FIX: Sample a mini-batch from the synthetic class ---
                # Ensure we don't sample more than the available ipc
                n_syn_samples = min(args.batch_train, ipc) 
                rand_idx = torch.randint(0, ipc, (n_syn_samples,))
                
                # Offset by c*ipc to grab from the correct class segment
                img_syn_c = image_syn[c*ipc + rand_idx]
                lab_syn_c = label_syn[c*ipc + rand_idx]
                # ---------------------------------------------------------
                
                gw_syn = model_grad_with_graph(net, img_syn_c, lab_syn_c, criterion)
                loss = loss + match_loss(gw_syn, gw_real, args)


            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            with torch.no_grad():
                image_syn.clamp_(0.0, 1.0)

            if ol == args.outer_loop - 1:
                break
            train_net_on_syn(net, image_syn, label_syn, args)

        loss_avg /= (num_classes * args.outer_loop)
        if it % args.log_every == 0:
            print(f"  [{mode_tag}] iter {it:4d}/{args.Iteration}  "
                  f"match_loss={loss_avg:.4f}  elapsed={time.time()-t0:.1f}s")

        do_eval = False
        if it > 0 and it < 10:
            do_eval = True
        elif args.eval_every > 0 and it > 0 and it % args.eval_every == 0:
            do_eval = True

        if do_eval:
            _periodic_eval(f"iter{it}", it)
            save_image(image_syn.detach().cpu().clamp(0.0, 1.0),
                       f"vis_{mode_tag}.png", nrow=ipc)

    return image_syn.detach().cpu(), label_syn.cpu(), curve


# ----------------------------- cache helpers -----------------------------
def cache_path(cache_dir, mode):
    return os.path.join(cache_dir, f"curve_{mode}.json")


def save_curve_cache(cache_dir, mode, curve):
    """Save one mode's curve to JSON + a human-readable txt."""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, mode)
    payload = {
        "mode": mode,
        "curve": [(int(it), [float(a) for a in accs]) for it, accs in curve],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    # also dump a plain-text version for quick reading / grepping
    txt_path = path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        f.write(f"# mode: {mode}\n")
        f.write(f"# iter\tmean_acc\taccs_per_seed\n")
        for it, accs in curve:
            mean = sum(accs) / len(accs) if accs else 0.0
            f.write(f"{it}\t{mean:.4f}\t{','.join(f'{a:.4f}' for a in accs)}\n")
    print(f"[cache] saved {mode} -> {path}")


def load_curve_cache(cache_dir, mode):
    """Return cached curve list[(it, [accs])] or None if missing."""
    path = cache_path(cache_dir, mode)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    curve = [(int(it), [float(a) for a in accs]) for it, accs in payload["curve"]]
    print(f"[cache] loaded {mode} <- {path} ({len(curve)} points)")
    return curve


# ----------------------------- plotting -----------------------------
MODE_STYLE = {
    "clean":         {"color": "tab:blue",   "label": "Distill on clean"},
    "poisoned-dc":   {"color": "tab:orange", "label": "Distill on poisoned (vanilla DC)"},
    "poisoned-cpdd": {"color": "tab:purple", "label": "Distill on poisoned (CPDD proxy)"},
}


def plot_combined(all_curves, clean_accs, poison_accs, save_path):
    import matplotlib.pyplot as plt

    clean = np.array(clean_accs) * 100.0
    poison = np.array(poison_accs) * 100.0
    clean_mean, clean_std = clean.mean(), clean.std()
    poison_mean, poison_std = poison.mean(), poison.std()

    fig, ax = plt.subplots(figsize=(10, 6))

    # collect a sensible x-range for the baseline shading
    all_iters = sorted({c[0] for curve in all_curves.values() for c in curve}) or [0, 1]
    xspan = np.array([min(all_iters), max(all_iters)])

    for mode, curve in all_curves.items():
        if not curve:
            continue
        iters = np.array([c[0] for c in curve])
        accs = np.array([c[1] for c in curve]) * 100.0
        mean = accs.mean(axis=1)
        std = accs.std(axis=1)
        style = MODE_STYLE[mode]
        ax.plot(iters, mean, "o-", color=style["color"], label=style["label"])
        ax.fill_between(iters, mean - std, mean + std, color=style["color"], alpha=0.20)

    ax.hlines(clean_mean, xspan[0], xspan[1], linestyle="--", color="tab:green",
              label=f"Clean upper bound ({clean_mean:.2f}% ± {clean_std:.2f})")
    ax.fill_between(xspan, clean_mean - clean_std, clean_mean + clean_std,
                    color="tab:green", alpha=0.10)

    ax.hlines(poison_mean, xspan[0], xspan[1], linestyle="--", color="tab:red",
              label=f"Poisoned no defense ({poison_mean:.2f}% ± {poison_std:.2f})")
    ax.fill_between(xspan, poison_mean - poison_std, poison_mean + poison_std,
                    color="tab:red", alpha=0.10)

    ax.set_xlabel("Sanitation iteration")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CPDD-v4: distillation on clean vs poisoned vs poisoned+CPDD")
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved -> {save_path}")


# ----------------------------- main -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pert", type=str, default="perturbation.pt")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--ipc", type=int, default=3000)
    p.add_argument("--Iteration", type=int, default=30)
    p.add_argument("--inner-steps", type=int, default=30,
                   help="PGD steps for eq. 39 (only used by poisoned-cpdd mode)")
    p.add_argument("--batch-real", type=int, default=256)
    p.add_argument("--batch-train", type=int, default=256)
    p.add_argument("--lr-syn", type=float, default=0.1)
    p.add_argument("--lr-net", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--r", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--epochs-eval", type=int, default=50)
    p.add_argument("--epochs-eval-mid", type=int, default=50)
    p.add_argument("--eval-seeds", type=int, default=1)
    p.add_argument("--plot", type=str, default="cpdd_v6_combined.png")
    p.add_argument("--curve-json", type=str, default="cpdd_v4_curves.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modes", type=str, nargs="+",
                   default=["poisoned-cpdd"],
                   choices=["clean", "poisoned-dc", "poisoned-cpdd"],
                   help="which experiments to run; default = all three")
    p.add_argument("--rerun", type=str, nargs="*", default=[],
                   choices=["clean", "poisoned-dc", "poisoned-cpdd"],
                   help="modes to re-run even if a cached curve file exists")
    p.add_argument("--cache-dir", type=str, default="curve_cache",
                   help="directory with per-mode cached curve files")
    args = p.parse_args()

    # DC loops
    args.outer_loop = 100
    args.inner_loop = 4

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"[env] device={device}")

    print("[data] loading CIFAR-10")
    x_clean_tr, y_tr, x_te, y_te = load_cifar10(args.data_root)
    print(f"[data] train={tuple(x_clean_tr.shape)} test={tuple(x_te.shape)}")

    # poisoned data only needed if a poisoned mode is requested
    needs_poison = any(m.startswith("poisoned") for m in args.modes)
    if needs_poison:
        print(f"[data] applying perturbation from {args.pert}")
        x_poison_tr = apply_perturbation(x_clean_tr, args.pert)
    else:
        x_poison_tr = None

    # cached baselines
    clean_accs = [0.8098, 0.8099, 0.8156]
    poison_accs = [0.2386, 0.2363, 0.2450]
    print(f"\n[baselines] clean={clean_accs} poison={poison_accs}")

    # ---- run each requested mode ----
    all_curves = {}
    mode_specs = {
        "clean":         dict(x_train="clean",   use_proxy=False, eps=0.0),
        "poisoned-dc":   dict(x_train="poison",  use_proxy=False, eps=0.0),
        "poisoned-cpdd": dict(x_train="poison",  use_proxy=True,  eps=args.eps),
    }

    for mode in args.modes:
        spec = mode_specs[mode]

        # try to use cached curve unless user explicitly asked to rerun this mode
        cached = None
        if mode not in args.rerun:
            
            cached = load_curve_cache(args.cache_dir, mode)

        if cached is not None:
            all_curves[mode] = cached
            print(f"[{mode}] using cached curve; skipping distillation. "
                  f"Pass --rerun {mode} to force re-run.")
            plot_combined(all_curves, clean_accs, poison_accs, args.plot)
            continue

        x_train = x_clean_tr if spec["x_train"] == "clean" else x_poison_tr
        print(f"\n========== running mode: {mode} ==========")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        syn_x, syn_y, curve = distill(
            x_train, y_tr, num_classes=10,
            ipc=args.ipc, args=args, device=device,
            x_te=x_te, y_te=y_te,
            use_proxy=spec["use_proxy"], eps=spec["eps"],
            mode_tag=mode,
        )

        # final eval
        print(f"\n=== Final eval [{mode}] ===")
        final_accs = eval_with_seeds(syn_x, syn_y, x_te, y_te, device,
                                     epochs=args.epochs_eval,
                                     n_seeds=args.eval_seeds,
                                     tag=f"final-{mode}")
        curve.append((args.Iteration, final_accs))
        all_curves[mode] = curve

        torch.save({"x": syn_x, "y": syn_y, "args": vars(args), "mode": mode},
                   f"distilled_{mode}.pt")

        # cache the curve so future runs can skip this mode
        save_curve_cache(args.cache_dir, mode, curve)

        # incremental plot after each mode finishes
        plot_combined(all_curves, clean_accs, poison_accs, args.plot)

    # dump curves
    with open(args.curve_json, "w") as f:
        json.dump({
            "modes": {
                m: [(int(it), [float(a) for a in accs]) for it, accs in c]
                for m, c in all_curves.items()
            },
            "clean": [float(a) for a in clean_accs],
            "poison": [float(a) for a in poison_accs],
        }, f, indent=2)
    print(f"[curve] saved -> {args.curve_json}")

    plot_combined(all_curves, clean_accs, poison_accs, args.plot)


if __name__ == "__main__":
    main()
