"""
Experiment 3: Extended AT budgets (2 – 64 / 255).

Tests whether AT at very large budgets can recover clean accuracy
from DGC-poisoned data.  For EMN-SW, recovery is expected at ~4/255;
for DGC, Theorem 2 predicts it should not recover regardless of budget.

Saves results to exp3_high_at_budgets.txt
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
from datetime import datetime
from torch.utils.data import TensorDataset
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

MODEL_EVAL_POOLS = {
    'CIFAR10':  ['ResNet18'],
    'CIFAR100': ['ResNet18'],
}

EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 60
EMN_BATCH        = 128

AT_BUDGETS = [2, 4, 6, 8, 16, 32, 64]   # /255
AT_STEPS   = 7


# ── Data loading ──────────────────────────────────────────────────────────────

def load_poisoned(noise_type, dataset, mean, std, dst_train, device):
    if noise_type == 'DGC':
        data = torch.load(DGC_FILES[dataset], map_location=device, weights_only=False)
        imgs = data['images_poisoned'].to(device)
        labs = data['labels'].to(device)
        print(f'  [DGC] {tuple(imgs.shape)}')
        return imgs, labs

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
        print(f'  [SW]  {tuple(imgs.shape)}')

    return imgs, labs


# ── PGD attack ────────────────────────────────────────────────────────────────

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

def train_normal(it_eval, net, imgs, labs, testloader, args, mean, std):
    device    = args.device
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR, momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EMN_EPOCHS, eta_min=0.0)
    loader    = torch.utils.data.DataLoader(TensorDataset(imgs, labs), batch_size=EMN_BATCH, shuffle=True, num_workers=0)
    start     = time.time()

    for ep in range(EMN_EPOCHS):
        net.train()
        for x, y in loader:
            x, y = x.float().to(device), y.long().to(device)
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            optimizer.step()
        scheduler.step()

    _, acc = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'  [Normal] Eval_{it_eval}  time={int(time.time()-start)}s  test_acc={acc:.4f}')
    return acc


def train_at(it_eval, net, imgs, labs, testloader, args, mean, std, at_eps):
    device    = args.device
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR, momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EMN_EPOCHS*0.5), int(EMN_EPOCHS*0.75)], gamma=0.1)
    loader    = torch.utils.data.DataLoader(TensorDataset(imgs, labs), batch_size=EMN_BATCH, shuffle=True, num_workers=0)
    alpha     = at_eps / 5.0
    start     = time.time()

    for ep in range(EMN_EPOCHS):
        for x, y in loader:
            x, y = x.float().to(device), y.long().to(device)
            net.eval()
            x_adv = pgd_attack(net, x, y, criterion, at_eps, alpha, AT_STEPS, mean, std)
            net.train()
            optimizer.zero_grad()
            criterion(net(x_adv), y).backward()
            optimizer.step()
        scheduler.step()

    _, acc = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'  [AT eps={at_eps*255:.0f}/255] Eval_{it_eval}  time={int(time.time()-start)}s  test_acc={acc:.4f}')
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval',    type=int,   default=1)
    parser.add_argument('--datasets',    nargs='+',  default=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--noise_types', nargs='+',  default=['DGC', 'SW'], choices=['DGC', 'SW'])
    args = parser.parse_args()

    args.dsa = False; args.dc_aug_param = None; args.eval_mode = 'S'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(SAVE_PATH, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n{'='*65}")
    print(f"  exp3-high-at-budgets.py  —  {ts}")
    print(f"  AT budgets: {AT_BUDGETS} /255")
    print(f"{'='*65}")

    # results[dataset][noise_type][trainer_tag] = {model: (mean, std)}
    all_results = {}

    for dataset in args.datasets:
        (channel, im_size, num_classes, _cn, mean, std, dst_train, _dt, testloader) = get_dataset(dataset, DATA_PATH)
        all_results[dataset] = {}

        for noise_type in args.noise_types:
            print(f"\n  Dataset={dataset}  noise={noise_type}")
            imgs_all, labs_all = load_poisoned(noise_type, dataset, mean, std, dst_train, args.device)
            all_results[dataset][noise_type] = {}

            for model_name in MODEL_EVAL_POOLS[dataset]:
                # Normal baseline
                tag = 'Normal'
                accs = []
                for it in range(args.num_eval):
                    net = build_emn_model(model_name, num_classes, channel, im_size).to(args.device)
                    accs.append(train_normal(it, net, copy.deepcopy(imgs_all), copy.deepcopy(labs_all), testloader, args, mean, std))
                m, s = float(np.mean(accs)), float(np.std(accs))
                all_results[dataset][noise_type][tag] = {model_name: (m, s)}
                print(f"  RESULT  {dataset}  {noise_type}  {tag}  {model_name}  mean={m:.4f}  std={s:.4f}")

                # AT at each budget
                for budget in AT_BUDGETS:
                    at_eps = budget / 255.0
                    tag = f'AT-{budget}'
                    accs = []
                    for it in range(args.num_eval):
                        net = build_emn_model(model_name, num_classes, channel, im_size).to(args.device)
                        accs.append(train_at(it, net, copy.deepcopy(imgs_all), copy.deepcopy(labs_all), testloader, args, mean, std, at_eps))
                    m, s = float(np.mean(accs)), float(np.std(accs))
                    all_results[dataset][noise_type][tag] = {model_name: (m, s)}
                    print(f"  RESULT  {dataset}  {noise_type}  {tag}  {model_name}  mean={m:.4f}  std={s:.4f}")

    # ── Save text results ─────────────────────────────────────────────────
    W = 78
    trainer_tags = ['Normal'] + [f'AT-{b}' for b in AT_BUDGETS]
    lines = ["="*W, f"  EXTENDED AT BUDGETS  —  Generated: {ts}", f"  Budgets: {AT_BUDGETS} /255", "="*W]

    for dataset, nt_dict in all_results.items():
        lines += ["", f"  Dataset: {dataset}", "  " + "-"*(W-2),
                  f"  {'noise':<8s}  {'Trainer':<12s}  {'Model':<14s}  {'mean':>8}  {'std':>8}",
                  "  " + "-"*(W-2)]
        for noise_type, tag_dict in nt_dict.items():
            for tag in trainer_tags:
                if tag not in tag_dict:
                    continue
                for model_name, (m, s) in tag_dict[tag].items():
                    lines.append(f"  {noise_type:<8s}  {tag:<12s}  {model_name:<14s}  {m:>8.4f}  {s:>8.4f}")
    lines.append("")

    out = os.path.join(SAVE_PATH, "exp3_high_at_budgets.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out}")

    # ── Plot: accuracy vs AT budget ───────────────────────────────────────
    x_ticks = [0] + AT_BUDGETS   # 0 = Normal
    x_labels = ['Normal'] + [str(b) for b in AT_BUDGETS]
    colors   = {'DGC': 'royalblue', 'SW': 'tomato'}
    markers  = {'DGC': 'o', 'SW': 's'}

    for dataset in args.datasets:
        fig, ax = plt.subplots(figsize=(8, 4))
        for noise_type in args.noise_types:
            tag_dict = all_results[dataset][noise_type]
            model_name = list(MODEL_EVAL_POOLS[dataset])[0]
            ys = []
            for tag in ['Normal'] + [f'AT-{b}' for b in AT_BUDGETS]:
                if tag in tag_dict and model_name in tag_dict[tag]:
                    ys.append(tag_dict[tag][model_name][0] * 100)
                else:
                    ys.append(float('nan'))
            ax.plot(range(len(x_ticks)), ys, color=colors[noise_type],
                    marker=markers[noise_type], label=noise_type, linewidth=1.8)

        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('AT Budget (/255)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{dataset}  {model_name}  — Accuracy vs AT Budget')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_data = {'dataset': dataset, 'x_labels': x_labels, 'x_ticks': x_ticks,
                     'noise_types': args.noise_types, 'curves': {}}
        for noise_type in args.noise_types:
            tag_dict = all_results[dataset][noise_type]
            model_name = list(MODEL_EVAL_POOLS[dataset])[0]
            ys = [tag_dict[t][model_name][0] * 100 if t in tag_dict and model_name in tag_dict[t] else float('nan')
                  for t in ['Normal'] + [f'AT-{b}' for b in AT_BUDGETS]]
            plot_data['curves'][noise_type] = ys
        torch.save(plot_data, os.path.join(SAVE_PATH, f'exp3_{dataset}_plot_data.pt'))
        out_fig = os.path.join(SAVE_PATH, f'exp3_{dataset}_at_budgets.png')
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"  Saved: {out_fig}")



if __name__ == '__main__':
    main()
