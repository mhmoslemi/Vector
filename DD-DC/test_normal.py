
import os
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import datetime
from torch.utils.data import TensorDataset
from util import get_dataset, epoch, build_emn_model




# ── Per-dataset model evaluation pools ────────────────────────────────────────

MODEL_EVAL_POOLS = {
    'CIFAR10':      ['ResNet18'],
    # 'CIFAR100':      ['ResNet18'],
    # 'SVHN':         ['ResNet18', 'ResNet50'],
}

EPSS = 8/255



EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 60
EMN_BATCH        = 128


def evaluate_synset_Normal(it_eval, net, images_train, labels_train, testloader,
                           args, mean=None, std=None):

    device = args.device
    net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)

    Epoch     = EMN_EPOCHS
    criterion = nn.CrossEntropyLoss().to(device)

    current_lr = 0.01 if 'ConvNet' in net.__class__.__name__ else EMN_LR

    optimizer = torch.optim.SGD(net.parameters(), lr=current_lr,
                                momentum=EMN_MOMENTUM,
                                weight_decay=EMN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Epoch, eta_min=0.0
    )

    trainloader = torch.utils.data.DataLoader(
        TensorDataset(images_train, labels_train),
        batch_size=EMN_BATCH, shuffle=True, num_workers=0
    )

    loss_train = acc_train = 0.0
    start = time.time()

    for ep in range(Epoch):
        net.train()
        loss_sum = acc_sum = n = 0

        for imgs, labs in trainloader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)

            optimizer.zero_grad()
            out  = net(imgs)
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

        if (ep + 1) % (Epoch // 10) == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer,
                                    criterion, args, aug=False)
            net.train()
            print(
                f'[Nor] Eval_{it_eval:02d}: epoch={ep+1:03d}/{Epoch:03d} '
                f'({100*(ep+1)//Epoch:3d}%)  '
                f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  '
                f'test_acc={acc_test_mid:.4f}'
            )

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer,
                                criterion, args, aug=False)

    print(
        f'[Nor] Eval_{it_eval:02d}: epoch={Epoch:03d}  '
        f'train_time={int(time_train)}s  '
        f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  '
        f'test_acc={acc_test:.4f}'
    )
    return net, acc_train, acc_test



# ══════════════════════════════════════════════════════════════════════════════
#  PGD helper
# ══════════════════════════════════════════════════════════════════════════════
ALL_DATASETS   = list(MODEL_EVAL_POOLS.keys())
def pgd_attack(net, images, labels, criterion, eps, alpha, steps, mean=None, std=None):
    """
    Madry PGD L-inf attack natively operating in raw pixel space.
    """
    device = images.device

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
    else:
        mean_t = torch.tensor(0.0, device=device)
        std_t  = torch.tensor(1.0, device=device)

    # 1. Denormalize images to [0, 1] raw pixel space
    x_raw = images * std_t + mean_t
    x_raw = x_raw.detach()

    # Initialize random perturbation in pixel space
    x_adv = x_raw + torch.empty_like(x_raw).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        # 2. Re-normalize to feed into the network
        x_norm = (x_adv - mean_t) / std_t
        loss = criterion(net(x_norm), labels)
        
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]

        with torch.no_grad():
            # 3. Step in pixel space
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_raw, -eps, eps)
            x_adv = torch.clamp(x_raw + delta, 0.0, 1.0)

    # 4. Return final adversarial batch in normalized space
    x_adv_norm = (x_adv - mean_t) / std_t
    return x_adv_norm.detach()


def evaluate_synset_AT(it_eval, net, images_train, labels_train, testloader,
                       args, mean=None, std=None):

    device = args.device
    net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)

    Epoch     = EMN_EPOCHS
    criterion = nn.CrossEntropyLoss().to(device)
    
    # ConvNet diverges instantly with LR=0.1, restrict it to 0.01 safely
    current_lr = 0.01 if 'ConvNet' in net.__class__.__name__ else EMN_LR

    optimizer = torch.optim.SGD(net.parameters(), lr=current_lr,
                                momentum=EMN_MOMENTUM,
                                weight_decay=EMN_WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=Epoch, eta_min=0.0
    # )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 81], gamma=0.1
    )

    trainloader = torch.utils.data.DataLoader(
        TensorDataset(images_train, labels_train),
        batch_size=EMN_BATCH, shuffle=True, num_workers=0
    )

    loss_train = acc_train = 0.0
    start = time.time()

    for ep in range(Epoch):
        loss_sum = acc_sum = n = 0

        for imgs, labs in trainloader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)

            # Step 1: craft adversarial examples (MUST BE IN EVAL MODE)
            net.eval()
            # x_adv = pgd_attack(net, imgs, labs, criterion,
            #                     eps=args.at_eps, alpha=args.at_alpha,
            #                     steps=args.at_steps, mean=mean, std=std)
            
            current_at_alpha = args.at_eps / 5.0
            # ... inside your loop ...
            x_adv = pgd_attack(net, imgs, labs, criterion,
                        eps=args.at_eps, alpha=current_at_alpha, # Use the dynamic alpha
                        steps=args.at_steps, mean=mean, std=std)

            # Step 2: cross-entropy on adversarial batch (MUST BE IN TRAIN MODE)
            net.train()
            optimizer.zero_grad()
            out  = net(x_adv)
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

        if (ep + 1) % (Epoch // 10) == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer,
                                    criterion, args, aug=False)
            net.train()
            print(
                f'[AT ] Eval_{it_eval:02d}: epoch={ep+1:03d}/{Epoch:03d} '
                f'({100*(ep+1)//Epoch:3d}%)  '
                f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  '
                f'test_acc={acc_test_mid:.4f}'
            )

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer,
                                criterion, args, aug=False)

    print(
        f'[AT ] Eval_{it_eval:02d}: epoch={Epoch:03d}  '
        f'train_time={int(time_train)}s  '
        f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  '
        f'test_acc={acc_test:.4f}'
    )
    return net, acc_train, acc_test




# ══════════════════════════════════════════════════════════════════════════════
#  Single dataset × condition evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dataset_condition(dataset, condition, args):
    """Return { trainer: { model_name: (mean_acc, std_acc) } }."""
    print(f"\n  ── {dataset}  [{condition}] {'─'*40}")

    (channel, im_size, num_classes, _class_names,
     mean, std, dst_train, _dst_test, testloader) = get_dataset(
        dataset, args.data_path
    )

    
    # New flow: load the fully poisoned dataset directly from the PT file
    if not os.path.exists(args.pt_file):
        raise FileNotFoundError(f"PT file not found: {args.pt_file}")
        
    print(f"    Loading poisoned dataset from {args.pt_file} ...")
    data = torch.load(args.pt_file, map_location=args.device,  weights_only=False)
    images_all = data['images_poisoned'].to(args.device)
    labels_all = data['labels'].to(args.device)
    print(f"    Loaded: images {images_all.shape}, labels {labels_all.shape}")
    

    trainers = {
'Normal': evaluate_synset_Normal,
        'AT':     evaluate_synset_AT,
    }
    results = {t: {} for t in trainers}

    for trainer_name, train_fn in trainers.items():
        print(f"\n    [{trainer_name}]")
        for model_name in MODEL_EVAL_POOLS[dataset]:
            accs = []
            for it_eval in range(args.num_eval):
                net_eval = build_emn_model(
                    model_name, num_classes, channel, im_size
                ).to(args.device)
                imgs_eval = copy.deepcopy(images_all.detach())
                lbls_eval = copy.deepcopy(labels_all.detach())
                
                # Pass mean and std down to the trainers
                _, _, acc_test = train_fn(
                    it_eval, net_eval, imgs_eval, lbls_eval, testloader, args, mean, std
                )
                accs.append(acc_test)
            m, s = float(np.mean(accs)), float(np.std(accs))
            results[trainer_name][model_name] = (m, s)
            print(f"    {trainer_name}  {model_name:<26s}  "
                  f"mean={m:.4f}  std={s:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Output writers
# ══════════════════════════════════════════════════════════════════════════════

def write_clean_txt(all_clean, save_path, ts):
    W = 70
    lines = [
        "=" * W,
        "  CLEAN DATASET — TRAINING  (Normal | AT | DRO)  [EMN models]",
        f"  Generated : {ts}",
        "=" * W,
    ]
    for dataset, trainer_dict in all_clean.items():
        for trainer, results in trainer_dict.items():
            lines += [
                "",
                f"  Dataset : {dataset}   Trainer : {trainer}",
                "  " + "-" * (W - 2),
                f"  {'Model':<26s}  {'mean':>8}   {'std':>8}",
                "  " + "-" * (W - 2),
            ]
            for model, (m, s) in results.items():
                lines.append(f"  {model:<26s}  {m:>8.4f}   {s:>8.4f}")
    lines.append("")
    out = os.path.join(save_path, "results_robust_clean.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out}")
    return out


def write_emn_txt(all_emn, save_path, ts):
    W = 84
    lines = [
        "=" * W,
        "  EMN NOISE — TRAINING  (Normal | AT | DRO)  [EMN models]  "
        "(CW | SW)",
        f"  Generated : {ts}",
        "=" * W,
    ]
    for dataset, cond_dict in all_emn.items():
        cw_trainers = cond_dict.get('CW', {})
        sw_trainers = cond_dict.get('SW', {})

        for trainer in ('Normal', 'AT', 'DRO'):
            cw = cw_trainers.get(trainer, {})
            sw = sw_trainers.get(trainer, {})
            all_models = list(cw.keys()) or list(sw.keys())
            if not all_models:
                continue
            lines += [
                "",
                f"  Dataset : {dataset}   Trainer : {trainer}",
                "  " + "-" * (W - 2),
                f"  {'Model':<26s}  {'CW mean':>9}  {'CW std':>8}"
                f"  |  {'SW mean':>9}  {'SW std':>8}",
                "  " + "-" * (W - 2),
            ]
            for model in all_models:
                cw_m, cw_s = cw.get(model, (float('nan'), float('nan')))
                sw_m, sw_s = sw.get(model, (float('nan'), float('nan')))
                lines.append(
                    f"  {model:<26s}  {cw_m:>9.4f}  {cw_s:>8.4f}"
                    f"  |  {sw_m:>9.4f}  {sw_s:>8.4f}"
                )
    lines.append("")
    out = os.path.join(save_path, "results_robust_EMN.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out}")
    return out


def write_pt_txt(all_pt, save_path, ts, pt_file):
    W = 70
    lines = [
        "=" * W,
        "  WARPED DATASET (PT) — TRAINING  (Normal | AT | DRO)",
        f"  File : {pt_file}",
        f"  Generated : {ts}",
        "=" * W,
    ]
    for dataset, trainer_dict in all_pt.items():
        for trainer, results in trainer_dict.items():
            lines += [
                "",
                f"  Dataset : {dataset}   Trainer : {trainer}",
                "  " + "-" * (W - 2),
                f"  {'Model':<26s}  {'mean':>8}   {'std':>8}",
                "  " + "-" * (W - 2),
            ]
            for model, (m, s) in results.items():
                lines.append(f"  {model:<26s}  {m:>8.4f}   {s:>8.4f}")
    lines.append("")
    out = os.path.join(save_path, "results_robust_PT.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Full-dataset robust evaluation (AT and DRO) '
                    'with exact EMN architectures and training config'
    )
    parser.add_argument('--num_eval',   type=int, default=1)
    parser.add_argument('--data_path',  type=str, default='/home/mmoslem3/scratch/UE-DD/data/')
    # parser.add_argument('--data_path',  type=str, default='../data')
    parser.add_argument('--save_path',  type=str, default='/home/mmoslem3/scratch/UE-DD/result-AT-Mo')
    # parser.add_argument('--save_path',  type=str, default='result')

    # parser.add_argument('--pt_file', type=str, default='/home/m/mmoslem3/SCRATCH')
    # parser.add_argument('--pt_file', type=str, default='/home/mmoslem3/scratch/UE-DD/result')
    
    
    parser.add_argument('--pt_file', type=str, default='/home/mmoslem3/scratch/UE-DD/result-fianle')

    parser.add_argument('--datasets',   nargs='+', default=ALL_DATASETS)
    parser.add_argument('--budget',   type = int,  default=2)
    parser.add_argument('--conditions', nargs='+', default=['PT'],
                        choices=['clean', 'CW', 'SW', 'PT']) # Added PT

    # AT hyper-parameters
    parser.add_argument('--at_eps',   type=float, default=EPSS,
                        help='PGD L-inf budget (default 8/255)')
    parser.add_argument('--at_steps', type=int,   default=7,
                        help='PGD iterations per batch (default 7)')
    
    # DRO (CVaR) hyper-parameter
    parser.add_argument('--dro_alpha', type=float, default=0.5,
                        help='Worst-case fraction (1.0=ERM, 0.1=hardest 10%%, default 0.5)')

    args = parser.parse_args()
    
    args.dsa          = False
    args.dc_aug_param = None
    args.eval_mode    = 'S'
    args.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.pt_file = args.pt_file + f'/res_MO_AT_final_{ALL_DATASETS[0]}_ConvNet.pt'

    # args.pt_file = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-DGC/res_MO_AT_CIFAR10_ConvNet_8.0.pt'
    # args.pt_file  = '/home/mmoslem3/scratch/UE-DD/result/test6-AT.pt'
    
    # for num in [8,6,4,2]:
    for num in [args.budget]:
        for at_eps in [1]:
            args.at_eps = at_eps / 255
            args.at_alpha = args.at_eps / 5
            args.pt_file = f'/home/mmoslem3/scratch/UE-DD/result-fianle/res_MO_AT_CIFAR10_ConvNet_{str(num)}.pt'
            # args.pt_file  = '/home/mmoslem3/scratch/UE-DD/result/res_MO_AT_CIFAR10_ConvNet_8.0.pt'

            

            os.makedirs(args.data_path, exist_ok=True)
            os.makedirs(args.save_path, exist_ok=True)

            ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
            print(f"\n{'='*65}")
            print(f"  test_robust_pt.py  —  started {ts}")
            print(f"  num — --> {num}")
            print(f"  Device     : {args.device}")
            print(f"  Datasets   : {args.datasets}")
            print(f"  Conditions : {args.conditions}")
            if 'PT' in args.conditions:
                print(f"  PT file    : {args.pt_file}")
            print(f"  num_eval   : {args.num_eval}")
            print(f"  EMN config : SGD lr={EMN_LR}  momentum={EMN_MOMENTUM}  "
                f"wd={EMN_WEIGHT_DECAY}  epochs={EMN_EPOCHS}  batch={EMN_BATCH}")
            print(f"  Scheduler  : CosineAnnealingLR T_max={EMN_EPOCHS} eta_min=0.0")
            print(f"  AT         : eps={args.at_eps*255:.4f}  alpha={args.at_alpha:.4f}  "
                f"steps={args.at_steps}")
            print(f"{'='*65}")

            all_clean = {}
            all_emn   = {}
            all_pt    = {}

            for dataset in args.datasets:
                if dataset not in MODEL_EVAL_POOLS:
                    print(f"\nWARNING: Unknown dataset '{dataset}' — skipping.")
                    continue
                for condition in args.conditions:
                    results = evaluate_dataset_condition(dataset, condition, args)
                    if condition == 'clean':
                        all_clean[dataset] = results
                    elif condition == 'PT':
                        all_pt[dataset] = results
                    else:
                        all_emn.setdefault(dataset, {})[condition] = results

            print(f"\n{'='*65}")
            print("  Writing output files ...")

            if all_clean:
                write_clean_txt(all_clean, args.save_path, ts)
            if all_emn:
                write_emn_txt(all_emn, args.save_path, ts)
            if all_pt:
                write_pt_txt(all_pt, args.save_path, ts, args.pt_file)

            print(f"\n  Done.")


if __name__ == '__main__':
    main()

