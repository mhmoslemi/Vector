"""
Experiment 4: SimCLR self-supervised defense.

SSL pretraining does NOT use labels — if DGC corrupts semantic signal
rather than just hiding it, SSL representations should also degrade.
If SSL recovers clean accuracy → the defence breaks DGC.

Protocol:
  1. Pretrain SimCLR encoder on *poisoned* training set (no labels).
  2. Freeze encoder; train a linear head on the *full clean* training
     labels (linear probe).  This is the standard SSL evaluation.
  3. Report test accuracy.

Compares: clean / DGC / EMN-SW pretraining.

Saves:
  exp4_simclr_results.txt
  exp4_simclr_CIFAR10.png   (bar chart)
  exp4_simclr_CIFAR100.png
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T
from util import get_dataset, build_emn_model

DATA_PATH = '/home/mmoslem3/scratch/UE-DD/data/'
SAVE_PATH = '/home/mmoslem3/scratch/UE-DD/extraEXP'

DGC_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/UE-DD/result-fianle/res_CIFAR10_iter150_bug8_lamexcess0.5.pt',
    'CIFAR100': '/home/mmoslem3/scratch/UE-DD/result-fianle/res_bike_C100_MO_AT_CIFAR100_ConvNet_8.pt',
}
SW_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/UE-DD/noise-EMN/CIFAR10_SW.pt',
    'CIFAR100': '/home/mmoslem3/scratch/UE-DD/UE-EMN/experiments/CIFAR100_samplewise_min-min/CIFAR100_SW.pt',
}

SIMCLR_EPOCHS  = 200
SIMCLR_BATCH   = 512
SIMCLR_LR      = 0.5       # linear scaling rule: 0.5 * batch/256
SIMCLR_TEMP    = 0.5
PROJ_DIM       = 128

LINEAR_EPOCHS  = 100
LINEAR_BATCH   = 256
LINEAR_LR      = 1.0


# ── SimCLR augmentation (CIFAR-adapted) ───────────────────────────────────────

def simclr_aug(mean, std):
    """Two independent stochastic views of the same image."""
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.Normalize(mean, std),   # images are already in [0,1] after denorm below
    ])


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps a tensor dataset and applies two random augmentations per sample."""
    def __init__(self, images, aug):
        # images: [N, C, H, W] float, already normalized
        self.images = images
        self.aug    = aug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]   # already normalized
        return self.aug(img), self.aug(img)


# ── Model: ResNet18 encoder + projection head ──────────────────────────────────

class SimCLRNet(nn.Module):
    def __init__(self, encoder, feat_dim, proj_dim=PROJ_DIM):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)          # [B, feat_dim]
        z = self.projector(h)        # [B, proj_dim]
        return F.normalize(z, dim=1)

    def encode(self, x):
        return self.encoder(x)


def build_encoder(num_classes, channel, im_size):
    """Return ResNet18 with the FC head replaced by Identity."""
    from util import build_emn_model
    net = build_emn_model('ResNet18', num_classes, channel, im_size)
    feat_dim = net.linear.in_features
    net.linear = nn.Identity()
    return net, feat_dim


# ── NT-Xent loss ──────────────────────────────────────────────────────────────

def nt_xent_loss(z1, z2, temperature=SIMCLR_TEMP):
    """Normalised temperature-scaled cross-entropy (NT-Xent)."""
    B  = z1.size(0)
    z  = torch.cat([z1, z2], dim=0)          # [2B, D]
    sim = torch.mm(z, z.t()) / temperature   # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pair for index i is index i+B (and vice-versa)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])  # [2B]

    return F.cross_entropy(sim, labels)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_poisoned(noise_type, dataset, mean, std, dst_train, device):
    imgs = torch.stack([dst_train[i][0] for i in range(len(dst_train))])
    labs = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long)

    if noise_type == 'DGC':
        data = torch.load(DGC_FILES[dataset], map_location='cpu', weights_only=False)
        imgs = data['images_poisoned'].cpu()
        labs = data['labels'].cpu()

    elif noise_type == 'SW':
        raw   = torch.load(SW_FILES[dataset], map_location='cpu', weights_only=False)
        noise = raw if not isinstance(raw, dict) else next(raw[k] for k in ('noise', 'perturbation', 'delta') if k in raw)
        noise = torch.as_tensor(noise, dtype=torch.float32)
        if noise.max() > 1.5:
            noise = noise / 255.0
        if noise.ndim == 4 and noise.shape[-1] in (1, 3):
            noise = noise.permute(0, 3, 1, 2).contiguous()
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t  = torch.tensor(std).view(1, -1, 1, 1)
        imgs   = (torch.clamp(imgs * std_t + mean_t + noise, 0.0, 1.0) - mean_t) / std_t

    # Keep on CPU for the dataset wrapper; move to GPU batch-by-batch
    return imgs.float(), labs.long()


# ── SimCLR pretraining ────────────────────────────────────────────────────────

def pretrain_simclr(model, imgs_poisoned, mean, std, device):
    """Pretrain SimCLR on poisoned images (no labels)."""
    # Denormalize to [0,1] so augmentation colour jitter operates in pixel space
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t  = torch.tensor(std).view(1, -1, 1, 1)
    imgs_raw = (imgs_poisoned * std_t + mean_t).clamp(0.0, 1.0)  # [N, C, H, W]

    aug     = simclr_aug(mean, std)
    dataset = TwoViewDataset(imgs_raw, aug)
    loader  = DataLoader(dataset, batch_size=SIMCLR_BATCH, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=SIMCLR_LR,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SIMCLR_EPOCHS, eta_min=0.0)

    model.to(device)
    model.train()
    start = time.time()

    for ep in range(SIMCLR_EPOCHS):
        loss_sum = n = 0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model(v1), model(v2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n        += 1
        scheduler.step()
        if (ep + 1) % (SIMCLR_EPOCHS // 10) == 0:
            print(f'    SSL epoch={ep+1:03d}/{SIMCLR_EPOCHS}  loss={loss_sum/n:.4f}  '
                  f'time={int(time.time()-start)}s')

    return model


# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe(encoder, imgs_clean, labs_clean, testloader, num_classes, device):
    """Freeze encoder, train linear head on clean labels."""
    encoder.eval()

    # Pre-compute features
    with torch.no_grad():
        feats_list, labs_list = [], []
        for i in range(0, len(imgs_clean), 512):
            x = imgs_clean[i:i+512].to(device)
            feats_list.append(encoder.encode(x).cpu())
            labs_list.append(labs_clean[i:i+512])
        feats_all = torch.cat(feats_list)
        labs_all  = torch.cat(labs_list)

    feat_dim = feats_all.size(1)
    head     = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=LINEAR_LR,
                                momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LINEAR_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(TensorDataset(feats_all, labs_all),
                           batch_size=LINEAR_BATCH, shuffle=True)

    for ep in range(LINEAR_EPOCHS):
        head.train()
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(head(f), y).backward()
            optimizer.step()
        scheduler.step()

    # Evaluate on test set using the full model (encoder + head)
    correct = total = 0
    head.eval()
    with torch.no_grad():
        for imgs, labs in testloader:
            imgs, labs = imgs.to(device), labs.to(device)
            feats = encoder.encode(imgs)
            preds = head(feats).argmax(1)
            correct += (preds == labs).sum().item()
            total   += labs.size(0)
    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',    nargs='+', default=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--noise_types', nargs='+', default=['clean', 'DGC', 'SW'],
                        choices=['clean', 'DGC', 'SW'])
    parser.add_argument('--num_eval',    type=int,  default=1)
    args = parser.parse_args()

    os.makedirs(SAVE_PATH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n{'='*65}")
    print(f"  exp4-simclr.py  —  {ts}")
    print(f"  SSL epochs={SIMCLR_EPOCHS}  Linear epochs={LINEAR_EPOCHS}  batch={SIMCLR_BATCH}")
    print(f"{'='*65}")

    all_results = {}  # dataset -> noise_type -> acc

    for dataset in args.datasets:
        (channel, im_size, num_classes, _cn, mean, std, dst_train, _dt, testloader) = get_dataset(dataset, DATA_PATH)

        # Clean images/labels for linear probe (same regardless of noise type)
        imgs_clean = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).float()
        labs_clean = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long)

        all_results[dataset] = {}

        for noise_type in args.noise_types:
            print(f"\n  {'─'*55}")
            print(f"  Dataset={dataset}  noise={noise_type}")

            imgs_poisoned, _ = load_poisoned(noise_type, dataset, mean, std, dst_train, device)

            accs = []
            for run in range(args.num_eval):
                encoder, feat_dim = build_encoder(num_classes, channel, im_size)
                simclr_model = SimCLRNet(encoder, feat_dim).to(device)

                print(f'\n  [Pretraining SimCLR  run={run}]')
                pretrain_simclr(simclr_model, imgs_poisoned, mean, std, device)

                print(f'  [Linear probe  run={run}]')
                acc = linear_probe(simclr_model, imgs_clean, labs_clean, testloader, num_classes, device)
                accs.append(acc)
                print(f'  linear probe acc={acc:.4f}')

            m, s = float(np.mean(accs)), float(np.std(accs))
            all_results[dataset][noise_type] = (m, s)
            print(f'  RESULT  dataset={dataset}  noise={noise_type}  mean={m:.4f}  std={s:.4f}')

    # ── Save text ─────────────────────────────────────────────────────────
    W = 65
    lines = ["="*W, f"  SIMCLR DEFENSE  —  Generated: {ts}",
             f"  SSL epochs={SIMCLR_EPOCHS}  Linear epochs={LINEAR_EPOCHS}", "="*W]
    for dataset, nt_dict in all_results.items():
        lines += ["", f"  Dataset: {dataset}", "  " + "-"*(W-2),
                  f"  {'noise':<10s}  {'mean':>8}  {'std':>8}", "  " + "-"*(W-2)]
        for noise_type, (m, s) in nt_dict.items():
            lines.append(f"  {noise_type:<10s}  {m:>8.4f}  {s:>8.4f}")
    lines.append("")
    out_txt = os.path.join(SAVE_PATH, "exp4_simclr_results.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out_txt}")

    # ── Plot bar chart ────────────────────────────────────────────────────
    colors = {'clean': 'green', 'DGC': 'royalblue', 'SW': 'tomato'}

    for dataset, nt_dict in all_results.items():
        labels = list(nt_dict.keys())
        means  = [nt_dict[k][0] * 100 for k in labels]
        stds   = [nt_dict[k][1] * 100 for k in labels]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(labels, means, yerr=stds, color=[colors.get(k, 'gray') for k in labels],
               capsize=4, width=0.5)
        ax.set_ylabel('Linear Probe Test Accuracy (%)')
        ax.set_title(f'{dataset}  SimCLR Linear Probe')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plot_data = {'dataset': dataset, 'labels': labels, 'means': means, 'stds': stds}
        torch.save(plot_data, os.path.join(SAVE_PATH, f'exp4_{dataset}_plot_data.pt'))
        out_fig = os.path.join(SAVE_PATH, f'exp4_simclr_{dataset}.png')
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"  Saved: {out_fig}")


if __name__ == '__main__':
    main()
