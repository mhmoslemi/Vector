"""
Experiment 1: Gradient cosine similarity + training loss curves.

Visualises Proposition 1: on DGC-poisoned data the gradient computed
with the correct label (g_rel) aligns with the gradient computed with a
random wrong label (g_tar), whereas on clean or EMN-SW data it does not.

Two plots are saved:
  - exp1_cosine_similarity.png
  - exp1_training_loss.png
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
PROBE_BATCH      = 512   # batch used for gradient measurement each epoch
PROBE_FREQ       = 1     # measure every N epochs


# ── Helpers ───────────────────────────────────────────────────────────────────

def random_wrong_labels(labs, num_classes):
    """Sample one uniformly random wrong label per sample."""
    offset = torch.randint(1, num_classes, labs.shape, device=labs.device)
    return (labs + offset) % num_classes


def grad_vector(net, imgs, labs, criterion):
    """Return flattened gradient w.r.t. all parameters (no in-place net update)."""
    net.zero_grad()
    loss = criterion(net(imgs), labs)
    loss.backward()
    return torch.cat([p.grad.detach().flatten() for p in net.parameters() if p.grad is not None])


def cosine_sim(a, b):
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def load_poisoned(noise_type, mean, std, dst_train, device, num_classes):
    if noise_type == 'DGC':
        data = torch.load(DGC_FILE, map_location=device, weights_only=False)
        imgs = data['images_poisoned'].to(device)
        labs = data['labels'].to(device)

    elif noise_type == 'SW':
        imgs = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
        labs = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)
        raw  = torch.load(SW_FILE, map_location=device, weights_only=False)
        noise = raw if not isinstance(raw, dict) else next(raw[k] for k in ('noise', 'perturbation', 'delta') if k in raw)
        noise = torch.as_tensor(noise, dtype=torch.float32, device=device)
        if noise.max() > 1.5:
            noise = noise / 255.0
        if noise.ndim == 4 and noise.shape[-1] in (1, 3):
            noise = noise.permute(0, 3, 1, 2).contiguous()
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
        imgs = (torch.clamp(imgs * std_t + mean_t + noise, 0.0, 1.0) - mean_t) / std_t

    else:  # clean
        imgs = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
        labs = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

    return imgs, labs


# ── Training + measurement ────────────────────────────────────────────────────

def train_and_measure(noise_type, imgs_all, labs_all, net, num_classes, device, probe_imgs, probe_labs):
    """Train net and record per-epoch loss and gradient cosine similarity."""
    criterion  = nn.CrossEntropyLoss().to(device)
    current_lr = 0.01 if 'ConvNet' in net.__class__.__name__ else EMN_LR
    optimizer  = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EMN_EPOCHS, eta_min=0.0)
    loader     = DataLoader(TensorDataset(imgs_all, labs_all), batch_size=EMN_BATCH, shuffle=True, num_workers=0)

    loss_curve = []
    cosine_curve = []

    for ep in range(EMN_EPOCHS):
        net.train()
        loss_sum = n = 0
        for imgs, labs in loader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)
            optimizer.zero_grad()
            out  = net(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labs.size(0)
            n        += labs.size(0)
        scheduler.step()
        epoch_loss = loss_sum / n
        loss_curve.append(epoch_loss)

        if (ep % PROBE_FREQ) == 0:
            net.eval()
            wrong_labs = random_wrong_labels(probe_labs, num_classes)
            g_rel = grad_vector(net, probe_imgs, probe_labs, criterion)
            g_tar = grad_vector(net, probe_imgs, wrong_labs, criterion)
            cos   = cosine_sim(g_rel, g_tar)
            cosine_curve.append(cos)
            net.train()
            print(f'  [{noise_type}] epoch={ep+1:03d}  loss={epoch_loss:.4f}  cos(g_rel,g_tar)={cos:.4f}')

    return np.array(loss_curve), np.array(cosine_curve)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    (channel, im_size, num_classes, _cn, mean, std, dst_train, _dt, _tl) = get_dataset(DATASET, DATA_PATH)

    # Fixed probe batch (same images across all conditions for fair comparison)
    all_imgs_clean = torch.stack([dst_train[i][0] for i in range(len(dst_train))])
    all_labs_clean = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long)
    idx = torch.randperm(len(all_imgs_clean))[:PROBE_BATCH]
    probe_imgs = all_imgs_clean[idx].float().to(device)
    probe_labs = all_labs_clean[idx].to(device)

    results = {}
    for noise_type in ('clean', 'DGC', 'SW'):
        print(f"\n{'='*60}\n  noise_type={noise_type}\n{'='*60}")
        imgs, labs = load_poisoned(noise_type, mean, std, dst_train, device, num_classes)
        net = build_emn_model(MODEL_NAME, num_classes, channel, im_size).to(device)
        loss_curve, cosine_curve = train_and_measure(
            noise_type, imgs, labs, net, num_classes, device, probe_imgs, probe_labs
        )
        results[noise_type] = (loss_curve, cosine_curve)
        torch.save({'loss': loss_curve, 'cosine': cosine_curve},
                   os.path.join(SAVE_PATH, f'exp1_{noise_type}_curves.pt'))

    # ── Plot: training loss ────────────────────────────────────────────────
    colors = {'clean': 'green', 'DGC': 'royalblue', 'SW': 'tomato'}
    epochs = np.arange(1, EMN_EPOCHS + 1)
    cosine_epochs = np.arange(0, EMN_EPOCHS, PROBE_FREQ) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for nt, (lc, _) in results.items():
        ax.plot(epochs, lc, color=colors[nt], label=nt, linewidth=1.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss (CE)')
    ax.set_title(f'Training Loss  ({DATASET}, {MODEL_NAME})')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for nt, (_, cc) in results.items():
        ax.plot(cosine_epochs, cc, color=colors[nt], label=nt, linewidth=1.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel(r'$\cos(g_\mathrm{rel},\, g_\mathrm{tar})$')
    ax.set_title(f'Gradient Cosine Similarity  ({DATASET}, {MODEL_NAME})')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_data = {nt: {'loss': r[0].tolist(), 'cosine': r[1].tolist(),
                      'epochs': epochs.tolist(), 'cosine_epochs': cosine_epochs.tolist()}
                 for nt, r in results.items()}
    torch.save(plot_data, os.path.join(SAVE_PATH, 'exp1_plot_data.pt'))
    out = os.path.join(SAVE_PATH, 'exp1_gradient_cosine.png')
    plt.savefig(out, dpi=150)
    print(f'\n  Saved: {out}')
    plt.close()


if __name__ == '__main__':
    main()
