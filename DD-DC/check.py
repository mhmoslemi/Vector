import torch
import numpy as np
from util import get_dataset


def measure_perturbation_magnitude(pt_file, dataset_name, data_path,
                                    device='cuda', sample_frac=0.3, seed=0):
    """
    Measure the effective perturbation (poisoned - clean) in raw [0,1] pixel
    space. Reports L-inf, L2, and per-pixel L1 statistics, plus a comparison
    to the standard UE budgets (4/255, 8/255, 16/255).

    Args:
        pt_file:       Path to the .pt file containing 'images_poisoned' and 'labels'.
        dataset_name:  e.g. 'CIFAR10'.
        data_path:     Where get_dataset looks for the clean data.
        device:        'cuda' or 'cpu'.
        sample_frac:   Fraction of the dataset to evaluate. Use 1.0 for full.
        seed:          RNG seed for subsampling.
    """
    print(f"\n{'='*70}")
    print(f"  Perturbation magnitude analysis")
    print(f"  PT file : {pt_file}")
    print(f"  Dataset : {dataset_name}   sample_frac = {sample_frac}")
    print(f"{'='*70}")

    # ---------- Load clean dataset ----------
    (channel, im_size, num_classes, _cn,
     mean, std, dst_train, _dst_test, _testloader) = get_dataset(
        dataset_name, data_path
    )

    # Stack the clean training images. get_dataset returns them normalized,
    # so we de-normalize back to [0,1] raw pixel space for a fair comparison.
    clean_imgs_norm = torch.stack([dst_train[i][0] for i in range(len(dst_train))])
    clean_labels    = torch.tensor([dst_train[i][1] for i in range(len(dst_train))])

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, -1, 1, 1)

    clean_imgs_raw = clean_imgs_norm * std_t + mean_t   # [0,1] space

    # ---------- Load poisoned dataset ----------
    data = torch.load(pt_file, map_location='cpu', weights_only=True)
    poisoned_imgs = data['images_poisoned'].float().cpu()
    poisoned_labels = data['labels'].cpu()

    # Figure out whether the saved tensor is already in [0,1] or normalized.
    # Heuristic: if min < -0.1 or max > 1.1, it's normalized.
    pmin, pmax = poisoned_imgs.min().item(), poisoned_imgs.max().item()
    print(f"  Poisoned tensor range: [{pmin:.3f}, {pmax:.3f}]")
    if pmin < -0.1 or pmax > 1.1:
        print(f"  -> Detected normalized space; de-normalizing to [0,1].")
        poisoned_imgs_raw = poisoned_imgs * std_t + mean_t
    else:
        print(f"  -> Detected raw [0,1] space; using as-is.")
        poisoned_imgs_raw = poisoned_imgs

    # ---------- Align shapes / order ----------
    assert poisoned_imgs_raw.shape == clean_imgs_raw.shape, (
        f"Shape mismatch: poisoned {poisoned_imgs_raw.shape} vs "
        f"clean {clean_imgs_raw.shape}. "
        f"If ordering differs between the two, match by label/index first."
    )

    # Sanity check: label ordering should match. If not, you'd need to
    # re-align before computing differences.
    if not torch.equal(poisoned_labels, clean_labels):
        print("  WARNING: poisoned and clean labels do not match elementwise.")
        print("  Delta statistics below assume elementwise pairing — verify "
              "your dataset-saving code preserves order.")

    # ---------- Subsample ----------
    N = clean_imgs_raw.shape[0]
    if sample_frac < 1.0:
        g = torch.Generator().manual_seed(seed)
        k = int(N * sample_frac)
        idx = torch.randperm(N, generator=g)[:k]
        clean_sub    = clean_imgs_raw[idx]
        poisoned_sub = poisoned_imgs_raw[idx]
    else:
        clean_sub    = clean_imgs_raw
        poisoned_sub = poisoned_imgs_raw

    print(f"  Evaluating on {clean_sub.shape[0]} of {N} samples.")

    # ---------- Compute deltas ----------
    delta = poisoned_sub - clean_sub                       # [N, C, H, W] in [0,1]
    delta_flat = delta.view(delta.shape[0], -1)            # [N, C*H*W]

    # Per-image norms
    linf_per = delta_flat.abs().max(dim=1).values          # [N]
    l2_per   = delta_flat.norm(p=2, dim=1)                 # [N]
    l1_per   = delta_flat.abs().mean(dim=1)                # [N]  mean |delta|

    # Global pixel-level
    abs_all = delta.abs().view(-1)

    # ---------- Report ----------
    def stats(t):
        return (t.mean().item(), t.std().item(),
                t.median().item(),
                t.min().item(), t.max().item(),
                torch.quantile(t, 0.95).item(),
                torch.quantile(t, 0.99).item())

    labels_ = ['mean', 'std', 'median', 'min', 'max', 'p95', 'p99']

    def print_row(name, t, scale_255=False):
        s = stats(t)
        if scale_255:
            s = tuple(x * 255 for x in s)
            fmt = '{:>10.3f}'
            suffix = '  (in /255 units)'
        else:
            fmt = '{:>10.5f}'
            suffix = ''
        row = '   '.join(fmt.format(v) for v in s)
        print(f"  {name:<30s} {row}{suffix}")

    print(f"\n  {'metric':<30s} " + '   '.join(f'{l:>10}' for l in labels_))
    print("  " + "-" * 110)
    print_row("L-inf per image (raw)",   linf_per)
    print_row("L-inf per image (/255)",  linf_per, scale_255=True)
    print_row("L2 per image (raw)",      l2_per)
    print_row("mean |delta| per img",    l1_per)
    print_row("mean |delta| (/255)",     l1_per, scale_255=True)
    print_row("abs pixel delta (raw)",   abs_all)
    print_row("abs pixel delta (/255)",  abs_all, scale_255=True)

    # ---------- Budget comparison ----------
    print(f"\n  Fraction of images within standard L-inf budgets:")
    for budget_255 in [2, 4, 6, 8, 12, 16, 32, 64]:
        budget = budget_255 / 255.0
        frac = (linf_per <= budget).float().mean().item()
        print(f"    ||delta||_inf <= {budget_255:>3d}/255 : "
              f"{frac*100:6.2f}%")

    # ---------- Summary ----------
    print(f"\n  Summary numbers for the paper:")
    print(f"    Mean  ||delta||_inf = {linf_per.mean().item()*255:.2f}/255  "
          f"(std {linf_per.std().item()*255:.2f}/255)")
    print(f"    Mean  ||delta||_2   = {l2_per.mean().item():.3f}  "
          f"(std {l2_per.std().item():.3f})")
    print(f"    Mean  |delta|_pixel = {abs_all.mean().item()*255:.3f}/255")
    print(f"{'='*70}\n")

    return {
        'linf_per_image': linf_per,
        'l2_per_image':   l2_per,
        'l1_per_image':   l1_per,
        'delta_abs_all':  abs_all,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_file', type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-clmap')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--data_path', type=str,
                        default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/data')
    parser.add_argument('--sample_frac', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()


    args.pt_file = args.pt_file + f'/res_MO_AT_final_{args.dataset}_ConvNet.pt'
    args.pt_file  = '/home/mmoslem3/scratch/UE-DD/result/res_MO_AT_final_CIFAR10_ConvNet.pt'
    measure_perturbation_magnitude(
        pt_file=args.pt_file,
        dataset_name=args.dataset,
        data_path=args.data_path,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )