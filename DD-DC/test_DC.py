import os
import copy
import argparse
import numpy as np
import torch
from datetime import datetime
from utils import get_dataset, get_network, evaluate_synset

# ── Per-dataset model evaluation pools ────────────────────────────────────────
MODEL_EVAL_POOLS = {
    'CIFAR10':      ['ConvNet', 'DenseNet121BN', 'ResNet18BN_AP', 'VGG11BN'],
    'MNIST':        ['ConvNet', 'DenseNet121',   'ResNet18_mnist', 'VGG11_AP'],
    'FashionMNIST': ['ConvNet', 'DenseNet121BN', 'ResNet18BN_AP', 'VGG11_AP'],
    'SVHN':         ['ConvNet', 'DenseNet_SVHN',  'ResNet18_AP',  'VGG11'],
}

MODEL_EVAL_POOLS = {
    'CIFAR10':      ['ConvNet', 'ResNet18BN_AP', 'VGG11BN'],
    # 'MNIST':        ['ConvNet',   'ResNet18_mnist', 'VGG11_AP'],
    # 'FashionMNIST': ['ConvNet', 'ResNet18BN_AP', 'VGG11_AP'],
    'SVHN':         ['ConvNet',  'ResNet18_AP',  'VGG11'],
}

ALL_DATASETS   = list(MODEL_EVAL_POOLS.keys())
ALL_CONDITIONS = ['clean', 'CW', 'SW']
ALL_CONDITIONS = ['SW']


# ── Load condensed data from .pt ───────────────────────────────────────────────
def load_condensed(pt_path, device):
    """
    Load the .pt saved by main.py.
    Returns (image_syn, label_syn) tensors on `device`.

    File structure:
        { 'data': [ [image_cpu, label_cpu], ... ],  # one entry per num_exp
          'accs_all_exps': { model: [acc, ...] } }
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Condensed .pt not found: {pt_path}")

    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    data_save = ckpt['data']  # list of [image_tensor, label_tensor]

    # Concatenate across experiments (usually just one)
    images = torch.cat([entry[0] for entry in data_save], dim=0).to(device)
    labels = torch.cat([entry[1] for entry in data_save], dim=0).to(device)
    return images, labels


def pt_filename(method, dataset, distill_model, ipc, condition):
    noise_str = f"_{condition}" if condition in ('CW', 'SW') else '_clean'
    return f"res_{method}_{dataset}_{distill_model}_{ipc}ipc{noise_str}.pt"


# ── Single dataset × condition evaluation ─────────────────────────────────────
def evaluate_dataset_condition(dataset, condition, args):
    """Return dict {model_name: (mean_acc, std_acc)} for one dataset/condition."""
    print(f"\n  ── {dataset}  [{condition}] {'─'*40}")

    fname   = pt_filename(args.method, dataset, args.distill_model, args.ipc, condition)
    pt_path = os.path.join(args.pt_path, fname)
    print(f"    Loading: {pt_path}")

    image_syn, label_syn = load_condensed(pt_path, args.device)
    print(f"    Condensed: images={image_syn.shape}  labels={label_syn.shape}")

    # Real testloader only — condensed data is the training set
    (channel, im_size, num_classes, _class_names,
     _mean, _std, _dst_train, _dst_test, testloader) = get_dataset(dataset, args.data_path)

    results = {}
    for model_name in MODEL_EVAL_POOLS[dataset]:
        accs = []
        for it_eval in range(args.num_eval):
            net_eval  = get_network(model_name, channel, num_classes, im_size).to(args.device)
            imgs_eval = copy.deepcopy(image_syn.detach())
            lbls_eval = copy.deepcopy(label_syn.detach())
            _, _, acc_test = evaluate_synset(
                it_eval, net_eval, imgs_eval, lbls_eval, testloader, args
            )
            accs.append(acc_test)
        m, s = float(np.mean(accs)), float(np.std(accs))
        results[model_name] = (m, s)
        print(f"    {model_name:<22s}  mean={m:.4f}  std={s:.4f}")

    return results


# ── Output writers ─────────────────────────────────────────────────────────────
def write_clean_txt(all_clean, method, ipc, distill_model, save_path, ts):
    W = 62
    lines = [
        "=" * W,
        f"  DC CONDENSED — CLEAN",
        f"  Method: {method}   ipc={ipc}   distilled with: {distill_model}",
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

    out = os.path.join(save_path, "results_DC_clean.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out}")
    return out


def write_emn_txt(all_emn, method, ipc, distill_model, save_path, ts):
    W = 80
    lines = [
        "=" * W,
        f"  DC CONDENSED — EMN NOISE  (CW = Class-Wise | SW = Sample-Wise)",
        f"  Method: {method}   ipc={ipc}   distilled with: {distill_model}",
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

    out = os.path.join(save_path, "results_DC_EMN.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DC condensed data across all datasets × clean/CW/SW'
    )
    # Condensed-data identity — must match main.py arguments
    parser.add_argument('--method',        type=str, default='DC',
                        help='Condensation method (DC or DSA)')
    parser.add_argument('--distill_model', type=str, default='ConvNet',
                        help='Model used during distillation (--model in main.py)')
    parser.add_argument('--ipc',           type=int, default=100,
                        help='Images per class used during distillation')
    parser.add_argument('--pt_path',       type=str, default='result',
                        help='Directory containing res_*.pt files from main.py')

    # Eval settings
    parser.add_argument('--num_eval',         type=int,   default=4)
    parser.add_argument('--epoch_eval_train', type=int,   default=300,
                        help='Epochs to train each eval model on condensed data')
    parser.add_argument('--lr_net',           type=float, default=0.01)
    parser.add_argument('--lr_img',           type=float, default=0.1)
    parser.add_argument('--batch_real',       type=int,   default=256)
    parser.add_argument('--batch_train',      type=int,   default=256)
    parser.add_argument('--init',             type=str,   default='real')
    parser.add_argument('--data_path',        type=str,   default='../data')
    parser.add_argument('--save_path',        type=str,   default='result')
    parser.add_argument('--datasets',   nargs='+', default=ALL_DATASETS,
                        help='Datasets to run (default: all)')
    parser.add_argument('--conditions', nargs='+', default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help='Conditions: clean  CW  SW  (default: all three)')

    args = parser.parse_args()
    args.dsa          = False
    args.dc_aug_param = None
    args.eval_mode    = 'S'
    args.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    print(f"\n{'='*65}")
    print(f"  test_DC.py  —  started {ts}")
    print(f"  Device        : {args.device}")
    print(f"  Method        : {args.method}   ipc={args.ipc}   distill_model={args.distill_model}")
    print(f"  .pt directory : {args.pt_path}")
    print(f"  Datasets      : {args.datasets}")
    print(f"  Conditions    : {args.conditions}")
    print(f"  num_eval      : {args.num_eval}   epoch_eval_train: {args.epoch_eval_train}")
    print(f"{'='*65}")

    all_clean = {}   # { dataset: { model: (mean, std) } }
    all_emn   = {}   # { dataset: { 'CW'|'SW': { model: (mean, std) } } }

    for dataset in args.datasets:
        if dataset not in MODEL_EVAL_POOLS:
            print(f"\nWARNING: Unknown dataset '{dataset}' — skipping.")
            continue
        for condition in args.conditions:
            try:
                results = evaluate_dataset_condition(dataset, condition, args)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue
            if condition == 'clean':
                all_clean[dataset] = results
            else:
                all_emn.setdefault(dataset, {})[condition] = results

    print(f"\n{'='*65}")
    print("  Writing output files ...")

    if all_clean:
        write_clean_txt(all_clean, args.method, args.ipc, args.distill_model,
                        args.save_path, ts)
    if all_emn:
        write_emn_txt(all_emn, args.method, args.ipc, args.distill_model,
                      args.save_path, ts)

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
