import os
import copy
import argparse
import numpy as np
import torch
from datetime import datetime
from utils import get_dataset, get_network, evaluate_synset

# ── Per-dataset model evaluation pools ────────────────────────────────────────
MODEL_EVAL_POOLS = {
    # 'CIFAR10':      ['ConvNet', 'DenseNet121BN', 'ResNet18BN_AP', 'VGG11BN'], # clean : ConvNet: mean=0.8133  std=0.0031 |  DenseNet121BN: mean=0.8953  std=0.0010
    # 'MNIST':        ['ConvNet', 'DenseNet121',   'ResNet18_mnist', 'VGG11_AP'],  # clean : ConvNet: mean=0.9954  std=0.0001 |  DenseNet121: mean=0.9965  std=0.0002 |  ResNet18_mnist: mean=0.9950  std=0.0003 | VGG11_AP: mean=0.9955  std=0.0003
    'FashionMNIST': ['ConvNet', 'DenseNet121BN', 'ResNet18BN_AP', 'VGG11_AP'],   # clean :
    'SVHN':         ['ConvNet', 'DenseNet_SVHN',  'ResNet18_AP',  'VGG11'],   # clean : ConvNet: mean=0.9266  std=0.0008 |  DenseNet121: 0.9604 |
}
MODEL_EVAL_POOLS = {
    'CIFAR10':      ['ConvNet',  'ResNet18BN_AP', 'VGG11BN'], # clean : ConvNet: mean=0.8133  std=0.0031 |  DenseNet121BN: mean=0.8953  std=0.0010
    'MNIST':        ['ConvNet',    'ResNet18_mnist', 'VGG11_AP'],  # clean : ConvNet: mean=0.9954  std=0.0001 |  DenseNet121: mean=0.9965  std=0.0002 |  ResNet18_mnist: mean=0.9950  std=0.0003 | VGG11_AP: mean=0.9955  std=0.0003
    'FashionMNIST': ['ConvNet',  'ResNet18BN_AP', 'VGG11_AP'],   # clean :
    'SVHN':         ['ConvNet',  'ResNet18_AP',  'VGG11'],   # clean : ConvNet: mean=0.9266  std=0.0008 |  DenseNet121: 0.9604 |
}

# MODEL_EVAL_POOLS = {
#     'CIFAR10':      [ 'ResNet18BN_AP', 'VGG11BN'],
#     'FashionMNIST': ['ConvNet', 'DenseNet121BN', 'ResNet18BN_AP', 'VGG11_AP'],
#     'SVHN':         ['ResNet18_AP',  'VGG11'],   
# }

ALL_DATASETS  = list(MODEL_EVAL_POOLS.keys())
ALL_CONDITIONS = ['clean', 'CW', 'SW']
ALL_CONDITIONS = ['CW', 'SW']
# ALL_CONDITIONS = ['clean']


# ── Noise loading ──────────────────────────────────────────────────────────────
def load_and_poison(images_all, labels_all, dataset, noise_type,
                    noise_path, num_classes, device, mean, std):
    noise_filename  = f"{dataset}_{noise_type}.pt"
    full_noise_path = os.path.join(noise_path, noise_filename)

    if not os.path.exists(full_noise_path):
        raise FileNotFoundError(f"Noise file not found: {full_noise_path}")

    print(f"    Loading {noise_type} noise from {full_noise_path} ...")
    noise_data = torch.load(full_noise_path, map_location=device)

    if isinstance(noise_data, dict):
        for k in ("noise", "perturbation", "delta"):
            if k in noise_data:
                noise_tensor = noise_data[k]
                break
    else:
        noise_tensor = noise_data

    noise_tensor = torch.as_tensor(noise_tensor, dtype=torch.float32, device=device)

    if noise_tensor.max() > 1.5:
        noise_tensor = noise_tensor / 255.0

    if (noise_tensor.ndim == 4
            and noise_tensor.shape[1:] != images_all.shape[1:]
            and noise_tensor.shape[-1] in [1, 3]):
        noise_tensor = noise_tensor.permute(0, 3, 1, 2).contiguous()

    if noise_tensor.shape[0] == num_classes:           # CW: [num_classes, C, H, W]
        noise_to_add = noise_tensor[labels_all]
    elif noise_tensor.shape[0] == images_all.shape[0]: # SW: [N, C, H, W]
        noise_to_add = noise_tensor
    else:
        raise ValueError(
            f"Noise shape mismatch — noise: {noise_tensor.shape}, "
            f"images: {images_all.shape}"
        )

    # Noise was generated in raw [0, 1] pixel space (UE-EMN uses no normalization).
    # images_all is normalized, so denormalize first, add noise, clamp, then renormalize.
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
    images_raw    = images_all * std_t + mean_t                        # undo normalization → [0, 1]
    poisoned_raw  = torch.clamp(images_raw + noise_to_add, 0.0, 1.0)  # add noise in pixel space
    poisoned      = (poisoned_raw - mean_t) / std_t                    # renormalize
    print(f"    Dataset poisoned. Noise l-inf: {noise_to_add.abs().max().item():.4f}")
    return poisoned


# ── Single dataset × condition evaluation ─────────────────────────────────────
def evaluate_dataset_condition(dataset, condition, args):
    """Return dict {model_name: (mean_acc, std_acc)} for one dataset/condition."""
    print(f"\n  ── {dataset}  [{condition}] {'─'*40}")

    (channel, im_size, num_classes, _class_names,
     mean, std, dst_train, _dst_test, testloader) = get_dataset(dataset, args.data_path)

    images_all = torch.cat(
        [torch.unsqueeze(dst_train[i][0], 0) for i in range(len(dst_train))], dim=0
    ).to(args.device)
    labels_all = torch.tensor(
        [dst_train[i][1] for i in range(len(dst_train))],
        dtype=torch.long, device=args.device
    )

    if condition in ('CW', 'SW'):
        images_all = load_and_poison(
            images_all, labels_all, dataset, condition,
            args.noise_path, num_classes, args.device, mean, std
        )
    else:
        print("    Clean dataset (no noise).")

    results = {}
    for model_name in MODEL_EVAL_POOLS[dataset]:
        accs = []
        for it_eval in range(args.num_eval):
            net_eval  = get_network(model_name, channel, num_classes, im_size).to(args.device)
            imgs_eval = copy.deepcopy(images_all.detach())
            lbls_eval = copy.deepcopy(labels_all.detach())
            _, _, acc_test = evaluate_synset(
                it_eval, net_eval, imgs_eval, lbls_eval, testloader, args
            )
            accs.append(acc_test)
        m, s = float(np.mean(accs)), float(np.std(accs))
        results[model_name] = (m, s)
        print(f"    {model_name:<22s}  mean={m:.4f}  std={s:.4f}")

    return results


# ── Output writers ─────────────────────────────────────────────────────────────
def write_clean_txt(all_clean, save_path, ts):
    W = 60
    lines = [
        "=" * W,
        "  CLEAN DATASET — FULL TRAINING ACCURACY",
        f"  Generated : {ts}",
        "=" * W,
    ]
    for dataset, results in all_clean.items():
        lines += [
            "",
            f"  Dataset : {dataset}",
            "  " + "-" * (W - 2),
            f"  {'Model':<22s}  {'mean':>8}   {'std':>8}",
            "  " + "-" * (W - 2),
        ]
        for model, (m, s) in results.items():
            lines.append(f"  {model:<22s}  {m:>8.4f}   {s:>8.4f}")
    lines.append("")

    out = os.path.join(save_path, f"results_clean.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out}")
    return out


def write_emn_txt(all_emn, save_path, ts):
    W = 80
    lines = [
        "=" * W,
        "  EMN NOISE — FULL TRAINING ACCURACY  (CW = Class-Wise | SW = Sample-Wise)",
        f"  Generated : {ts}",
        "=" * W,
    ]
    for dataset, cond_results in all_emn.items():
        cw = cond_results.get('CW', {})
        sw = cond_results.get('SW', {})
        all_models = list(cw.keys()) or list(sw.keys())

        lines += [
            "",
            f"  Dataset : {dataset}",
            "  " + "-" * (W - 2),
            f"  {'Model':<22s}  {'CW mean':>9}  {'CW std':>8}  |  {'SW mean':>9}  {'SW std':>8}",
            "  " + "-" * (W - 2),
        ]
        for model in all_models:
            cw_m, cw_s = cw.get(model, (float('nan'), float('nan')))
            sw_m, sw_s = sw.get(model, (float('nan'), float('nan')))
            lines.append(
                f"  {model:<22s}  {cw_m:>9.4f}  {cw_s:>8.4f}  |  {sw_m:>9.4f}  {sw_s:>8.4f}"
            )
    lines.append("")

    out = os.path.join(save_path, f"results_EMN.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Full-dataset evaluation (all datasets × clean/CW/SW)')
    parser.add_argument('--num_eval',          type=int,   default=1)
    parser.add_argument('--epoch_eval_train',  type=int,   default=50,
                        help='Epochs to train each eval model on the full dataset')
    parser.add_argument('--lr_net',            type=float, default=0.01)
    parser.add_argument('--lr_img',            type=float, default=0.1)
    parser.add_argument('--batch_real',        type=int,   default=256)
    parser.add_argument('--batch_train',       type=int,   default=256)
    parser.add_argument('--init',              type=str,   default='real')
    parser.add_argument('--data_path',         type=str,   default='../data')
    parser.add_argument('--save_path',         type=str,   default='result')
    parser.add_argument('--noise_path',        type=str,
                        default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN')
    parser.add_argument('--datasets',   nargs='+', default=ALL_DATASETS,
                        help='Subset of datasets to run (default: all)')
    parser.add_argument('--conditions', nargs='+', default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help='Conditions to run: clean  CW  SW  (default: all three)')

    args = parser.parse_args()
    args.dsa          = False
    args.dc_aug_param = None
    args.eval_mode    = 'S'
    args.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    print(f"\n{'='*65}")
    print(f"  test_full.py  —  started {ts}")
    print(f"  Device     : {args.device}")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Conditions : {args.conditions}")
    print(f"  num_eval   : {args.num_eval}   epoch_eval_train: {args.epoch_eval_train}")
    print(f"{'='*65}")

    all_clean = {}   # { dataset: { model: (mean, std) } }
    all_emn   = {}   # { dataset: { 'CW'|'SW': { model: (mean, std) } } }

    for dataset in args.datasets:
        if dataset not in MODEL_EVAL_POOLS:
            print(f"\nWARNING: Unknown dataset '{dataset}' — skipping.")
            continue
        for condition in args.conditions:
            results = evaluate_dataset_condition(dataset, condition, args)
            if condition == 'clean':
                all_clean[dataset] = results
            else:
                all_emn.setdefault(dataset, {})[condition] = results

    print(f"\n{'='*65}")
    print("  Writing output files ...")

    if all_clean:
        write_clean_txt(all_clean, args.save_path, ts)
    if all_emn:
        write_emn_txt(all_emn, args.save_path, ts)

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
