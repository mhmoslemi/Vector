"""
Experiment 1-1: Gradient cosine similarity under AT training.

Same metric as exp1 — cos(g_rel, g_tar) measured on the poisoned probe
batch — but the model is trained with PGD-AT instead of normal SGD.

Key question: does AT training undo the gradient alignment?
  - DGC: cosine should STAY high (Theorem 2 — AT cannot recover)
  - SW:  cosine should DROP (AT removes shortcut, gradient direction recovers)
  - clean: cosine should go negative (well-trained robust model)

Run at two AT budgets (4/255 and 8/255) to show SW breaks at 4 but DGC persists.

Saves:
  exp1-1_AT{budget}_plot_data.pt
  exp1-1_gradient_cosine_AT.png
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from util import get_dataset, build_emn_model

# ── Config ────────────────────────────────────────────────────────────────────
DATASET    = 'CIFAR10'
MODEL_NAME = 'ResNet18'
DATA_PATH  = '/home/mmoslem3/scratch/UE-DD/data/'
SAVE_PATH  = '/home/mmoslem3/scratch/UE-DD/extraEXP'

DGC_FILE = '/home/mmoslem3/scratch/UE-DD/result-fianle/res_CIFAR10_iter150_bug8_lamexcess0.5.pt'
SW_FILE  = '/home/mmoslem3/scratch/UE-DD/noise-EMN/CIFAR10_SW.pt'

EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 60
EMN_BATCH        = 128
PROBE_BATCH      = 512
AT_STEPS         = 7
AT_BUDGETS       = [4, 8]   # /255 — SW breaks at 4, DGC persists at 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_wrong_labels(labs, num_classes):
    offset = torch.randint(1, num_classes, labs.shape, device=labs.device)
    return (labs + offset) % num_classes


def grad_vector(net, imgs, labs, criterion):
    net.zero_grad()
    loss = criterion(net(imgs), labs)
    loss.backward()
    return torch.cat([p.grad.detach().flatten() for p in net.parameters() if p.grad is not None])


def cosine_sim(a, b):
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def pgd_attack(net, images, labels, criterion, eps, alpha, steps, mean, std):
    device = images.device
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
    x_raw  = (images * std_t + mean_t).detach()
    x_adv  = torch.clamp(x_raw + torch.empty_like(x_raw).uniform_(-eps, eps), 0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = criterion(net((x_adv - mean_t) / std_t), labels)
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        with torch.no_grad():
            x_adv = torch.clamp(x_raw + torch.clamp(x_adv + alpha * grad.sign() - x_raw, -eps, eps), 0.0, 1.0)
    return ((x_adv - mean_t) / std_t).detach()


def load_poisoned(noise_type, mean, std, dst_train, device):
    if noise_type == 'DGC':
        data = torch.load(DGC_FILE, map_location=device, weights_only=False)
        return data['images_poisoned'].to(device), data['labels'].to(device)

    imgs = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
    labs = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

    if noise_type == 'SW':
        raw   = torch.load(SW_FILE, map_location=device, weights_only=False)
        noise = raw if not isinstance(raw, dict) else next(raw[k] for k in ('noise', 'perturbation', 'delta') if k in raw)
        noise = torch.as_tensor(noise, dtype=torch.float32, device=device)
        if noise.max() > 1.5:
            noise = noise / 255.0
        if noise.ndim == 4 and noise.shape[-1] in (1, 3):
            noise = noise.permute(0, 3, 1, 2).contiguous()
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
        imgs = (torch.clamp(imgs * std_t + mean_t + noise, 0.0, 1.0) - mean_t) / std_t

    return imgs, labs


# ── AT training + measurement ─────────────────────────────────────────────────

def train_at_and_measure(noise_type, imgs_all, labs_all, net, num_classes,
                         device, probe_imgs, probe_labs, mean, std, at_eps):
    """AT-train net; measure cos(g_rel, g_tar) on the poisoned probe each epoch."""
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR, momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(EMN_EPOCHS * 0.5), int(EMN_EPOCHS * 0.75)], gamma=0.1
    )
    loader  = DataLoader(TensorDataset(imgs_all, labs_all), batch_size=EMN_BATCH, shuffle=True, num_workers=0)
    alpha   = at_eps / 5.0

    loss_curve   = []
    cosine_curve = []

    for ep in range(EMN_EPOCHS):
        loss_sum = n = 0
        for imgs, labs in loader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)
            net.eval()
            x_adv = pgd_attack(net, imgs, labs, criterion, at_eps, alpha, AT_STEPS, mean, std)
            net.train()
            optimizer.zero_grad()
            out  = net(x_adv)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labs.size(0)
            n        += labs.size(0)
        scheduler.step()
        epoch_loss = loss_sum / n
        loss_curve.append(epoch_loss)

        # Measure cosine on original poisoned probe (NOT adversarial examples)
        net.eval()
        wrong_labs = random_wrong_labels(probe_labs, num_classes)
        g_rel = grad_vector(net, probe_imgs, probe_labs, criterion)
        g_tar = grad_vector(net, probe_imgs, wrong_labs, criterion)
        cos   = cosine_sim(g_rel, g_tar)
        cosine_curve.append(cos)
        net.train()
        print(f'  [{noise_type} | AT-{int(at_eps*255)}] epoch={ep+1:03d}  '
              f'loss={epoch_loss:.4f}  cos(g_rel,g_tar)={cos:.4f}')

    return np.array(loss_curve), np.array(cosine_curve)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    (channel, im_size, num_classes, _cn, mean, std, dst_train, _dt, _tl) = get_dataset(DATASET, DATA_PATH)

    # Probe batch: same poisoned images used as probe across all conditions
    # For each noise_type, the probe is drawn from that condition's poisoned data
    # so cosine reflects gradient alignment ON THE POISONED DATA under AT model
    noise_types = ('clean', 'DGC', 'SW')

    # Pre-load all poisoned datasets and fix probe batches
    poisoned_data = {}
    probe_batches = {}
    for nt in noise_types:
        imgs, labs = load_poisoned(nt, mean, std, dst_train, device)
        poisoned_data[nt] = (imgs, labs)
        idx = torch.randperm(len(imgs))[:PROBE_BATCH]
        probe_batches[nt] = (imgs[idx].float(), labs[idx])

    colors  = {'clean': 'green', 'DGC': 'royalblue', 'SW': 'tomato'}
    epochs  = np.arange(1, EMN_EPOCHS + 1)

    # One figure with 2 columns (one per AT budget) × 2 rows (loss, cosine)
    fig, axes = plt.subplots(2, len(AT_BUDGETS), figsize=(7 * len(AT_BUDGETS), 8))

    all_plot_data = {}

    for col, budget in enumerate(AT_BUDGETS):
        at_eps = budget / 255.0
        print(f"\n{'='*65}")
        print(f"  AT budget = {budget}/255")
        print(f"{'='*65}")

        results = {}
        for nt in noise_types:
            imgs, labs = poisoned_data[nt]
            probe_imgs, probe_labs = probe_batches[nt]
            net = build_emn_model(MODEL_NAME, num_classes, channel, im_size).to(device)
            loss_curve, cosine_curve = train_at_and_measure(
                nt, copy.deepcopy(imgs), copy.deepcopy(labs),
                net, num_classes, device,
                probe_imgs, probe_labs,
                mean, std, at_eps
            )
            results[nt] = (loss_curve, cosine_curve)

        all_plot_data[f'AT-{budget}'] = {
            nt: {'loss': r[0].tolist(), 'cosine': r[1].tolist()}
            for nt, r in results.items()
        }

        # Loss subplot
        ax = axes[0, col]
        for nt, (lc, _) in results.items():
            ax.plot(epochs, lc, color=colors[nt], label=nt, linewidth=1.8)
        ax.set_xlabel('Epoch'); ax.set_ylabel('AT Training Loss (CE)')
        ax.set_title(f'Training Loss — AT {budget}/255  ({DATASET})')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Cosine subplot
        ax = axes[1, col]
        for nt, (_, cc) in results.items():
            ax.plot(epochs, cc, color=colors[nt], label=nt, linewidth=1.8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\cos(g_\mathrm{rel},\, g_\mathrm{tar})$')
        ax.set_title(f'Gradient Cosine Similarity — AT {budget}/255  ({DATASET})')
        ax.legend(); ax.grid(True, alpha=0.3)

    all_plot_data['epochs'] = epochs.tolist()
    torch.save(all_plot_data, os.path.join(SAVE_PATH, 'exp1-1_plot_data.pt'))

    plt.tight_layout()
    out = os.path.join(SAVE_PATH, 'exp1-1_gradient_cosine_AT.png')
    plt.savefig(out, dpi=150)
    print(f'\n  Saved: {out}')
    plt.close()


if __name__ == '__main__':
    main()
