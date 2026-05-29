"""
mix_ratio_eval.py  :  partial-poisoning sweep for UE methods.

For each (method, alpha) pair, constructs a training set where an alpha
fraction of samples are clean and a (1 - alpha) fraction are protected,
index-matched so that the same underlying image contributes one version
or the other. Trains under normal training and reports clean test
accuracy. Appends to a CSV after every run so partial results survive
crashes.

Theory prediction (from Prop. 1 of the DGC paper):
    effective clean-direction gradient on mixed batch =
        ||g_clean||^2 * (alpha - (1 - alpha)/(C - 1))   for DGC
        ||g_clean||^2 * alpha                           for shortcut UEs
    DGC crossover alpha* = 1/C.
    CIFAR-10: alpha* = 0.1. CIFAR-100: alpha* = 0.01.

Expected shape of the result on CIFAR-10:
    - At alpha = 0:   all UEs protect (test acc near baseline).
    - At alpha = 0.1: shortcut UEs already recovering strongly;
                      DGC roughly stalled (near the crossover).
    - At alpha = 0.25 and above: DGC also recovers but should trail
                                 shortcut UEs by a visible margin.
    - At alpha = 1.0: clean baseline (same for all methods; run once).
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from util import get_dataset, epoch, build_emn_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_EVAL_POOLS = {
    'CIFAR10':  ['ResNet18'],   # recommended first pass: C=10 puts alpha* in-sweep
    # 'CIFAR100': ['ResNet18'], # C=100 puts alpha* below the minimal sweep; enable later
}
ALL_DATASETS = list(MODEL_EVAL_POOLS.keys())
METHODS      = ['DGC', 'EMN', 'GUE', 'TUE']
ALPHAS       = [0.0, 0.1, 0.25, 0.5, 1.0]

EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 30
EMN_BATCH        = 128


# DGC and EMN-SW file paths per dataset
DGC_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/Unlearnable-Examples-DD/all-pt/res_CIFAR10_iter150_bug8_lamexcess0.5.pt',
}
EMN_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN/CIFAR10_SW.pt',
}
GUE_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-GUE/gue_cifar10_SW.pt',
}
TUE_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-TUE/TUE_simclr_cifar10.pt',
}
_SW_FILE_MAP = {'EMN': EMN_FILES, 'GUE': GUE_FILES, 'TUE': TUE_FILES}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_poisoned_images(noise_type, dataset, mean, std, dst_train, device):
    if noise_type == 'DGC':
        data = torch.load(DGC_FILES[dataset], map_location=device, weights_only=False)
        images_all = data['images_poisoned'].to(device)
        labels_all = data['labels'].to(device)
        print(f"  [DGC] Loaded {tuple(images_all.shape)} from {os.path.basename(DGC_FILES[dataset])}")

    elif noise_type in ('EMN', 'GUE', 'TUE'):
        images_all = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
        labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

        noise_file = _SW_FILE_MAP[noise_type][dataset]
        raw = torch.load(noise_file, map_location=device, weights_only=False)
        noise_tensor = raw if not isinstance(raw, dict) else next(raw[k] for k in ('noise', 'perturbation', 'delta') if k in raw)
        noise_tensor = torch.as_tensor(noise_tensor, dtype=torch.float32, device=device)
        if noise_tensor.max() > 1.5:
            noise_tensor = noise_tensor / 255.0
        if noise_tensor.ndim == 4 and noise_tensor.shape[-1] in (1, 3):
            noise_tensor = noise_tensor.permute(0, 3, 1, 2).contiguous()
        if noise_tensor.shape[0] != len(images_all):
            raise ValueError(f"[{noise_type}] noise shape {noise_tensor.shape} vs dataset {images_all.shape}")

        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
        images_all = (torch.clamp(images_all * std_t + mean_t + noise_tensor, 0.0, 1.0) - mean_t) / std_t
        print(f"  [{noise_type}] Poisoned {tuple(images_all.shape)}, noise linf={noise_tensor.abs().max():.4f}")

    elif noise_type == 'clean':
        images_all = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
        labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)
        print(f"  [clean] Loaded {tuple(images_all.shape)} from dst_train (no noise)")
    else:
        raise ValueError(f"noise_type must be DGC, EMN, GUE, TUE, or clean; got {noise_type}")

    return images_all, labels_all




# ---------------------------------------------------------------------------
# Training (copied verbatim from clean-eval.py for self-containment)
# ---------------------------------------------------------------------------

def train_normal(it_eval, net, images_train, labels_train, testloader, args, mean=None, std=None):
    device = args.device
    net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    current_lr = 0.01 if 'ConvNet' in net.__class__.__name__ else EMN_LR
    optimizer = torch.optim.SGD(net.parameters(), lr=current_lr,
                                momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EMN_EPOCHS, eta_min=0.0)
    trainloader = torch.utils.data.DataLoader(
        TensorDataset(images_train, labels_train),
        batch_size=EMN_BATCH, shuffle=True, num_workers=0,
    )

    loss_train = acc_train = 0.0
    start = time.time()

    for ep in range(EMN_EPOCHS):
        # print(f'[Nor] Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d} starting...', flush=True)
        net.train()
        loss_sum = acc_sum = n = 0
        for imgs, labs in trainloader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)
            optimizer.zero_grad()
            out = net(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                acc_sum  += (out.argmax(1) == labs).sum().item()
                loss_sum += loss.item() * labs.size(0)
                n        += labs.size(0)
        scheduler.step()
        loss_train = loss_sum / n
        acc_train  = acc_sum  / n
        eval_interval = max(1, EMN_EPOCHS // 10)
        if (ep + 1) % eval_interval == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            net.train()
            print(f'[Nor] Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d} '
                  f'({100*(ep+1)//EMN_EPOCHS:3d}%)  '
                  f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  '
                  f'test_acc={acc_test_mid:.4f}', flush=True)
        else:
            print(f'[Nor] Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d}  '
                  f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}', flush=True)

    time_train = time.time() - start
    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'[Nor] Eval_{it_eval:02d}: FINAL  train_time={int(time_train)}s  '
          f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test:.4f}')
    return acc_test


# ---------------------------------------------------------------------------
# New: protected-dataset loading and index-matched mixing
# ---------------------------------------------------------------------------

def load_protected_dataset(method, dataset, protected_dir):
    """Load (images, labels) for a protected dataset. Flexible about format."""
    path = os.path.join(protected_dir, f"{method.lower()}_{dataset.lower()}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Protected dataset not found: {path}")

    data = torch.load(path, map_location='cpu')

    if isinstance(data, dict):
        images = data.get('images', data.get('data', data.get('x')))
        labels = data.get('labels', data.get('targets', data.get('y')))
        if images is None or labels is None:
            raise ValueError(f"Could not find images/labels in dict at {path}; keys={list(data.keys())}")
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        images, labels = data
    else:
        raise ValueError(f"Unexpected format in {path}: {type(data)}")

    return images.float(), labels.long()


def build_mixed_tensor(clean_images, clean_labels, prot_images, prot_labels, alpha, seed=0):
    """
    Index-matched mixing. A random alpha-fraction of sample indices uses the
    clean image; the rest use the protected image. Labels must align exactly
    between clean and protected (UEs are clean-label by definition).
    """
    N = len(clean_images)
    if len(prot_images) != N:
        raise ValueError(f"Size mismatch: clean={N}, protected={len(prot_images)}")
    if not torch.equal(clean_labels, prot_labels):
        raise ValueError(
            "Labels misaligned between clean and protected tensors. "
            "The index-matched mixing assumption is broken; check .pt generation."
        )

    n_clean = int(round(alpha * N))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    clean_idx = torch.as_tensor(perm[:n_clean], dtype=torch.long)

    mixed_images = prot_images.clone()
    if n_clean > 0:
        mixed_images[clean_idx] = clean_images[clean_idx]
    mixed_labels = clean_labels.clone()  # identical to prot_labels by assertion above

    return mixed_images, mixed_labels


# def tensor_stats(x):
#     return f"range=[{x.min().item():.3f},{x.max().item():.3f}]  mean={x.mean().item():.3f}  std={x.std().item():.3f}"


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval',      type=int,   default=1)
    parser.add_argument('--data_path',     type=str,   default='/home/mmoslem3/scratch/UE-DD/data/')
    parser.add_argument('--save_path',     type=str,   default='/home/mmoslem3/scratch/UE-DD/result-mix-ratio')
    parser.add_argument('--datasets',      nargs='+',  default=ALL_DATASETS)
    parser.add_argument('--methods',       nargs='+',  default=METHODS)
    parser.add_argument('--alphas',        nargs='+',  type=float, default=ALPHAS)
    args = parser.parse_args()

    args.dsa = False
    args.dc_aug_param = None
    args.eval_mode = 'S'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_path, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    csv_path = os.path.join(args.save_path, 'mix_ratio_results.csv')
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(['timestamp', 'dataset', 'architecture', 'method', 'alpha', 'seed', 'test_acc'])
    csv_file.flush()

    def log_row(dataset, arch, method, alpha, seed, acc):
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            dataset, arch, method, f'{alpha:.4f}', seed, f'{acc:.4f}',
        ])
        csv_file.flush()

    print("\n" + "=" * 78)
    print(f"  mix_ratio_eval.py  :  {ts}")
    print(f"  Device: {args.device}   Datasets: {args.datasets}")
    print(f"  Methods: {args.methods}   Alphas: {args.alphas}")
    print(f"  Output CSV: {csv_path}")
    print("=" * 78)

    for dataset in args.datasets:
        if dataset not in MODEL_EVAL_POOLS:
            print(f"WARNING: unknown dataset '{dataset}', skipping.")
            continue

        (channel, im_size, num_classes, _cn, mean, std,
         dst_train, _dt, testloader) = get_dataset(dataset, args.data_path)

        clean_images = torch.stack([dst_train[i][0] for i in range(len(dst_train))])
        clean_labels = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long)

        print(f"\n  Dataset={dataset}  shape={tuple(clean_images.shape)}  C={num_classes}")
        print(f"  Theory: crossover alpha* = 1/C = {1.0 / num_classes:.4f}")
        # print(f"  Clean  : {tensor_stats(clean_images)}")

        # Load all protected tensors up front; fail fast if any missing.
        protected = {}
        for method in args.methods:
            prot_imgs, prot_labs = load_poisoned_images(method, dataset, mean, std, dst_train, device='cpu')
            if not torch.equal(prot_labs.cpu(), clean_labels):
                raise ValueError(
                    f"{method} labels do not match clean labels for {dataset}. "
                    "Mixing requires index-matched tensors."
                )
            protected[method] = (prot_imgs, prot_labs)
            # print(f"  {method:4s}  : {tensor_stats(prot_imgs)}  shape={tuple(prot_imgs.shape)}")

        for arch in MODEL_EVAL_POOLS[dataset]:
            # --- alpha = 1.0 baseline (pure clean), run once per seed -------
            if 1.0 in args.alphas:
                for seed in range(args.num_eval):
                    print(f"\n  >>> [{dataset}/{arch}] method=clean  alpha=1.0  seed={seed}")
                    net = build_emn_model(arch, num_classes, channel, im_size).to(args.device)
                    acc = train_normal(
                        seed, net,
                        clean_images.clone(), clean_labels.clone(),
                        testloader, args, mean, std,
                    )
                    log_row(dataset, arch, 'clean', 1.0, seed, acc)

            # --- method x alpha sweep, excluding alpha = 1.0 ---------------
            for method in args.methods:
                prot_images, prot_labels = protected[method]
                for alpha in args.alphas:
                    if alpha == 1.0:
                        continue
                    for seed in range(args.num_eval):
                        print(f"\n  >>> [{dataset}/{arch}] method={method}  alpha={alpha}  seed={seed}")
                        mixed_images, mixed_labels = build_mixed_tensor(
                            clean_images, clean_labels,
                            prot_images, prot_labels,
                            alpha, seed=seed,
                        )
                        net = build_emn_model(arch, num_classes, channel, im_size).to(args.device)
                        acc = train_normal(
                            seed, net,
                            mixed_images, mixed_labels,
                            testloader, args, mean, std,
                        )
                        log_row(dataset, arch, method, alpha, seed, acc)

    csv_file.close()
    print(f"\n  DONE.  Results: {csv_path}")


if __name__ == '__main__':
    main()