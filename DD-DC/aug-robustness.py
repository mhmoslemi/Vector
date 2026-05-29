
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.data import TensorDataset
from util import get_dataset, epoch, build_emn_model

MODEL_EVAL_POOLS = {
    'CIFAR10':  ['ResNet18'],
    'CIFAR100': ['ResNet18'],
}
ALL_DATASETS = list(MODEL_EVAL_POOLS.keys())

EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 60
EMN_BATCH        = 128

# DGC and EMN-SW file paths per dataset
DGC_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/UE-DD/result-fianle/res_CIFAR10_iter150_bug8_lamexcess0.5.pt',
    'CIFAR100': '/home/mmoslem3/scratch/UE-DD/result-fianle/res_bike_C100_MO_AT_CIFAR100_ConvNet_8.pt',
}
SW_FILES = {
    'CIFAR10':  '/home/mmoslem3/scratch/UE-DD/noise-EMN/CIFAR10_SW.pt',
    'CIFAR100': '/home/mmoslem3/scratch/UE-DD/UE-EMN/experiments/CIFAR100_samplewise_min-min/CIFAR100_SW.pt',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_poisoned_images(noise_type, dataset, pt_file, noise_file, mean, std, dst_train, device):
    if noise_type == 'DGC':
        data = torch.load(pt_file, map_location=device, weights_only=False)
        images_all = data['images_poisoned'].to(device)
        labels_all = data['labels'].to(device)
        print(f"  [DGC] Loaded {tuple(images_all.shape)} from {os.path.basename(pt_file)}")

    elif noise_type == 'SW':
        images_all = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
        labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

        raw = torch.load(noise_file, map_location=device, weights_only=False)
        noise_tensor = raw if not isinstance(raw, dict) else next(raw[k] for k in ('noise', 'perturbation', 'delta') if k in raw)
        noise_tensor = torch.as_tensor(noise_tensor, dtype=torch.float32, device=device)
        if noise_tensor.max() > 1.5:
            noise_tensor = noise_tensor / 255.0
        if noise_tensor.ndim == 4 and noise_tensor.shape[-1] in (1, 3):
            noise_tensor = noise_tensor.permute(0, 3, 1, 2).contiguous()
        if noise_tensor.shape[0] != len(images_all):
            raise ValueError(f"SW noise shape {noise_tensor.shape} vs dataset {images_all.shape}")

        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
        images_raw = images_all * std_t + mean_t
        images_all = (torch.clamp(images_raw + noise_tensor, 0.0, 1.0) - mean_t) / std_t
        print(f"  [SW] Poisoned {tuple(images_all.shape)}, noise linf={noise_tensor.abs().max():.4f}")
    elif noise_type == 'clean':
        images_all = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
        labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)
        print(f"  [clean] Loaded {tuple(images_all.shape)} from dst_train (no noise)")
    else:
        raise ValueError(f"noise_type must be DGC, SW, or clean, got {noise_type}")

    return images_all, labels_all


# ── Augmentation functions ────────────────────────────────────────────────────
# Each returns either (imgs, labs) for 2-tuple or (imgs, labs_a, labs_b, lam) for mixed-label

def aug_cutout(imgs, labs, hole_frac=0.5):
    B, C, H, W = imgs.shape
    size = int(H * hole_frac)
    imgs = imgs.clone()
    for i in range(B):
        cx = torch.randint(H, (1,)).item()
        cy = torch.randint(W, (1,)).item()
        x1, x2 = max(0, cx - size // 2), min(H, cx + size // 2)
        y1, y2 = max(0, cy - size // 2), min(W, cy + size // 2)
        imgs[i, :, x1:x2, y1:y2] = 0.0
    return imgs, labs

def aug_mixup(imgs, labs, alpha=1.0):
    lam = float(torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha)).sample())
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labs, labs[idx], lam

def aug_cutmix(imgs, labs, alpha=1.0):
    lam = float(torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha)).sample())
    B, C, H, W = imgs.shape
    idx = torch.randperm(B, device=imgs.device)
    r = (1 - lam) ** 0.5
    cut_h, cut_w = int(H * r), int(W * r)
    cx, cy = torch.randint(H, (1,)).item(), torch.randint(W, (1,)).item()
    x1, x2 = max(0, cx - cut_h // 2), min(H, cx + cut_h // 2)
    y1, y2 = max(0, cy - cut_w // 2), min(W, cy + cut_w // 2)
    imgs = imgs.clone()
    imgs[:, :, x1:x2, y1:y2] = imgs[idx, :, x1:x2, y1:y2]
    lam_actual = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    return imgs, labs, labs[idx], lam_actual

def aug_gaussian(imgs, labs, sigma=0.1):
    return imgs + torch.randn_like(imgs) * sigma, labs

def aug_random_erasing(imgs, labs):
    import torchvision.transforms as T
    re = T.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    return torch.stack([re(img) for img in imgs]), labs

def aug_cutout_gaussian(imgs, labs):
    imgs, labs = aug_cutout(imgs, labs)
    return aug_gaussian(imgs, labs)

def aug_mixup_cutout(imgs, labs):
    imgs, labs = aug_cutout(imgs, labs)
    return aug_mixup(imgs, labs)

def aug_cutmix_gaussian(imgs, labs):
    return aug_cutmix(*aug_gaussian(imgs, labs))

AUGMENTATIONS = {
    # 'none':             None,
    'cutout':           aug_cutout,
    'mixup':            aug_mixup,
    'cutmix':           aug_cutmix,
    'gaussian':         aug_gaussian,
    'random_erasing':   aug_random_erasing,
    'cutout+gaussian':  aug_cutout_gaussian,
    'mixup+cutout':     aug_mixup_cutout,
    'cutmix+gaussian':  aug_cutmix_gaussian,
}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_with_aug(it_eval, net, images_train, labels_train, testloader, args, aug_name, aug_fn, mean=None, std=None):
    device = args.device
    net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    current_lr = 0.01 if 'ConvNet' in net.__class__.__name__ else EMN_LR
    optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EMN_EPOCHS, eta_min=0.0)
    trainloader = torch.utils.data.DataLoader(TensorDataset(images_train, labels_train), batch_size=EMN_BATCH, shuffle=True, num_workers=0)

    loss_train = acc_train = 0.0
    start = time.time()
    tag = f'[{aug_name[:12]:12s}]'

    for ep in range(EMN_EPOCHS):
        net.train()
        loss_sum = acc_sum = n = 0

        for imgs, labs in trainloader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)

            if aug_fn is None:
                out = net(imgs)
                loss = criterion(out, labs)
                ref_labs = labs
            else:
                result = aug_fn(imgs, labs)
                if len(result) == 4:
                    imgs_aug, labs_a, labs_b, lam = result
                    out  = net(imgs_aug)
                    loss = lam * criterion(out, labs_a) + (1 - lam) * criterion(out, labs_b)
                    ref_labs = labs_a
                else:
                    imgs_aug, ref_labs = result
                    out  = net(imgs_aug)
                    loss = criterion(out, ref_labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc_sum  += (out.argmax(1) == ref_labs).sum().item()
                loss_sum += loss.item() * ref_labs.size(0)
                n        += ref_labs.size(0)

        scheduler.step()
        loss_train = loss_sum / n
        acc_train  = acc_sum  / n

        if (ep + 1) % (EMN_EPOCHS // 10) == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            net.train()
            print(f'{tag} Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d} ({100*(ep+1)//EMN_EPOCHS:3d}%)  '
                  f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test_mid:.4f}')

    time_train = time.time() - start
    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'{tag} Eval_{it_eval:02d}: FINAL  train_time={int(time_train)}s  '
          f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test:.4f}')
    return acc_test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval',    type=int,   default=1)
    parser.add_argument('--data_path',   type=str,   default='/home/mmoslem3/scratch/UE-DD/data/')
    parser.add_argument('--save_path',   type=str,   default='/home/mmoslem3/scratch/UE-DD/extraEXP')
    parser.add_argument('--datasets',    nargs='+',  default=ALL_DATASETS)
    parser.add_argument('--noise_types', nargs='+',  default=['clean', 'DGC', 'SW'], choices=['clean', 'DGC', 'SW'])
    parser.add_argument('--augmentations', nargs='+', default=list(AUGMENTATIONS.keys()))
    args = parser.parse_args()

    args.dsa = False; args.dc_aug_param = None; args.eval_mode = 'S'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_path, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n{'='*70}")
    print(f"  aug-robustness.py  —  {ts}")
    print(f"  Device: {args.device}  Datasets: {args.datasets}  noise_types: {args.noise_types}")
    print(f"  Augmentations: {args.augmentations}")
    print(f"{'='*70}")

    # all_results[noise_type][dataset][aug_name][model_name] = (mean, std)
    all_results = {nt: {} for nt in args.noise_types}

    for dataset in args.datasets:
        if dataset not in MODEL_EVAL_POOLS:
            print(f"WARNING: Unknown dataset '{dataset}' — skipping.")
            continue

        (channel, im_size, num_classes, _cn, mean, std, dst_train, _dt, testloader) = get_dataset(dataset, args.data_path)

        for noise_type in args.noise_types:
            print(f"\n{'─'*70}")
            print(f"  Dataset={dataset}  noise_type={noise_type}")
            print(f"{'─'*70}")

            images_all, labels_all = load_poisoned_images(
                noise_type, dataset,
                DGC_FILES[dataset], SW_FILES[dataset],
                mean, std, dst_train, args.device
            )
            all_results[noise_type][dataset] = {}

            for aug_name in args.augmentations:
                if aug_name not in AUGMENTATIONS:
                    print(f"  Unknown augmentation '{aug_name}' — skipping.")
                    continue
                aug_fn = AUGMENTATIONS[aug_name]
                print(f"\n  aug={aug_name}")
                all_results[noise_type][dataset][aug_name] = {}

                for model_name in MODEL_EVAL_POOLS[dataset]:
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = build_emn_model(model_name, num_classes, channel, im_size).to(args.device)
                        acc_test = train_with_aug(
                            it_eval, net_eval,
                            copy.deepcopy(images_all.detach()),
                            copy.deepcopy(labels_all.detach()),
                            testloader, args, aug_name, aug_fn, mean, std
                        )
                        accs.append(acc_test)
                    m, s = float(np.mean(accs)), float(np.std(accs))
                    all_results[noise_type][dataset][aug_name][model_name] = (m, s)
                    print(f"  RESULT  noise={noise_type}  dataset={dataset}  aug={aug_name}  "
                          f"model={model_name}  mean={m:.4f}  std={s:.4f}")

    # Save results
    W = 82
    lines = ["="*W, f"  AUGMENTATION ROBUSTNESS  —  Generated: {ts}", "="*W]
    for noise_type, ds_dict in all_results.items():
        for dataset, aug_dict in ds_dict.items():
            lines += ["", f"  noise={noise_type}  dataset={dataset}",
                      "  " + "-"*(W-2),
                      f"  {'Augmentation':<22s}  {'Model':<20s}  {'mean':>8}  {'std':>8}",
                      "  " + "-"*(W-2)]
            for aug_name, model_dict in aug_dict.items():
                for model_name, (m, s) in model_dict.items():
                    lines.append(f"  {aug_name:<22s}  {model_name:<20s}  {m:>8.4f}  {s:>8.4f}")
    lines.append("")

    torch.save(all_results, os.path.join(args.save_path, "aug_robustness_data.pt"))
    out = os.path.join(args.save_path, "results_aug_robustness.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out}")
    print(f"  Saved plot data: {os.path.join(args.save_path, 'aug_robustness_data.pt')}")


if __name__ == '__main__':
    main()
