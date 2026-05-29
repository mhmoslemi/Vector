"""
Experiment 2: Per-class accuracy heatmap.

For each noise type (DGC, EMN-SW) and each training regime
(Normal, AT-2, AT-4, AT-6, AT-8), compute per-class test accuracy
and plot as a heatmap.  Reveals which classes are hardest to protect
and which are most vulnerable to AT recovery.

Saves:
  exp2_classwise_CIFAR10.png
  exp2_classwise_CIFAR100.png   (class indices only — too many for names)
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from util import get_dataset, epoch, build_emn_model

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

EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 60
EMN_BATCH        = 128

AT_BUDGETS = [2, 4, 6, 8]   # /255
AT_STEPS   = 7


# ── Data loading ──────────────────────────────────────────────────────────────

def load_poisoned(noise_type, dataset, mean, std, dst_train, device):
    if noise_type == 'DGC':
        data = torch.load(DGC_FILES[dataset], map_location=device, weights_only=False)
        return data['images_poisoned'].to(device), data['labels'].to(device)

    imgs = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
    labs = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

    if noise_type == 'SW':
        raw   = torch.load(SW_FILES[dataset], map_location=device, weights_only=False)
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


# ── PGD ───────────────────────────────────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────────────────────

def train(net, imgs_all, labs_all, device, mean, std, at_eps=None):
    criterion  = nn.CrossEntropyLoss().to(device)
    optimizer  = torch.optim.SGD(net.parameters(), lr=EMN_LR, momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    if at_eps is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EMN_EPOCHS, eta_min=0.0)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EMN_EPOCHS*0.5), int(EMN_EPOCHS*0.75)], gamma=0.1)
    loader = DataLoader(TensorDataset(imgs_all, labs_all), batch_size=EMN_BATCH, shuffle=True, num_workers=0)

    for ep in range(EMN_EPOCHS):
        net.train()
        for imgs, labs in loader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)
            if at_eps is not None:
                net.eval()
                imgs = pgd_attack(net, imgs, labs, criterion, at_eps, at_eps/5.0, AT_STEPS, mean, std)
                net.train()
            optimizer.zero_grad()
            loss = criterion(net(imgs), labs)
            loss.backward()
            optimizer.step()
        scheduler.step()


# ── Per-class accuracy ────────────────────────────────────────────────────────

def per_class_accuracy(net, testloader, num_classes, device):
    correct = torch.zeros(num_classes)
    total   = torch.zeros(num_classes)
    net.eval()
    with torch.no_grad():
        for imgs, labs in testloader:
            imgs, labs = imgs.to(device), labs.to(device)
            preds = net(imgs).argmax(1)
            for c in range(num_classes):
                mask = labs == c
                correct[c] += (preds[mask] == c).sum().item()
                total[c]   += mask.sum().item()
    return (correct / total.clamp(min=1)).numpy() * 100.0


# ── Plot heatmap ──────────────────────────────────────────────────────────────

def plot_heatmap(data, row_labels, col_labels, title, out_path, class_names=None):
    """data: [num_trainers, num_classes]"""
    try:
        import seaborn as sns
        use_sns = True
    except ImportError:
        use_sns = False

    fig_h = max(4, len(row_labels) * 0.5)
    fig_w = max(8, len(col_labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(class_names if class_names else [str(c) for c in col_labels], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('Class'); ax.set_ylabel('Training Regime')
    ax.set_title(title)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', fontsize=6, color='black')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',    nargs='+', default=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--noise_types', nargs='+', default=['DGC', 'SW'], choices=['DGC', 'SW'])
    args = parser.parse_args()

    os.makedirs(SAVE_PATH, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer_tags = ['Normal'] + [f'AT-{b}' for b in AT_BUDGETS]

    for dataset in args.datasets:
        (channel, im_size, num_classes, class_names, mean, std, dst_train, _dt, testloader) = get_dataset(dataset, DATA_PATH)

        for noise_type in args.noise_types:
            print(f"\n{'='*60}\n  {dataset}  noise={noise_type}\n{'='*60}")
            imgs_all, labs_all = load_poisoned(noise_type, dataset, mean, std, dst_train, device)

            # [num_trainers, num_classes]
            heatmap = np.zeros((len(trainer_tags), num_classes))

            for t_idx, tag in enumerate(trainer_tags):
                at_eps = None if tag == 'Normal' else int(tag.split('-')[1]) / 255.0
                print(f'  Training: {tag}  at_eps={at_eps}')
                net = build_emn_model('ResNet18', num_classes, channel, im_size).to(device)
                train(net, copy.deepcopy(imgs_all), copy.deepcopy(labs_all), device, mean, std, at_eps=at_eps)
                acc_per_class = per_class_accuracy(net, testloader, num_classes, device)
                heatmap[t_idx] = acc_per_class
                print(f'    mean={acc_per_class.mean():.2f}%  min={acc_per_class.min():.2f}%  max={acc_per_class.max():.2f}%')

            pt_path = os.path.join(SAVE_PATH, f'exp2_{dataset}_{noise_type}_heatmap.pt')
            torch.save({'heatmap': heatmap, 'trainer_tags': trainer_tags,
                        'class_names': class_names, 'dataset': dataset,
                        'noise_type': noise_type}, pt_path)
            print(f'  Saved plot data: {pt_path}')

            cnames = class_names if len(class_names) <= 20 else None
            plot_heatmap(
                heatmap, trainer_tags, list(range(num_classes)),
                title=f'{dataset}  noise={noise_type}  — Per-class Test Accuracy (%)',
                out_path=os.path.join(SAVE_PATH, f'exp2_{dataset}_{noise_type}.png'),
                class_names=cnames
            )


if __name__ == '__main__':
    main()
