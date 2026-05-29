
import copy
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import TensorDataset
from util import get_dataset, epoch, build_emn_model

MODEL_EVAL_POOLS = {
#     'CIFAR10':  ['ResNet18', 'VGG11', 'ResNet50', 'DenseNet121', 'ViT'],
    'CIFAR100': ['ResNet18', 'VGG11', 'ResNet50', 'DenseNet121', 'ViT'],
}

MODEL_EVAL_POOLS = {
     'CIFAR10':  ['ResNet18'],
#    'CIFAR100': ['ResNet18'],
}

ALL_DATASETS = list(MODEL_EVAL_POOLS.keys())

EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 60
EMN_BATCH        = 128


def train_normal(it_eval, net, images_train, labels_train, testloader, args, mean=None, std=None):
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

    for ep in range(EMN_EPOCHS):
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
        if (ep + 1) % (EMN_EPOCHS // 10) == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            net.train()
            print(f'[Nor] Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d} ({100*(ep+1)//EMN_EPOCHS:3d}%)  '
                  f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test_mid:.4f}')

    time_train = time.time() - start
    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'[Nor] Eval_{it_eval:02d}: FINAL  train_time={int(time_train)}s  '
          f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test:.4f}')
    return acc_test


def pgd_attack(net, images, labels, criterion, eps, alpha, steps, mean=None, std=None):
    device = images.device
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
    else:
        mean_t = torch.tensor(0.0, device=device)
        std_t  = torch.tensor(1.0, device=device)

    x_raw = (images * std_t + mean_t).detach()
    x_adv = torch.clamp(x_raw + torch.empty_like(x_raw).uniform_(-eps, eps), 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = criterion(net((x_adv - mean_t) / std_t), labels)
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        with torch.no_grad():
            x_adv = torch.clamp(x_raw + torch.clamp(x_adv + alpha * grad.sign() - x_raw, -eps, eps), 0.0, 1.0)

    return ((x_adv - mean_t) / std_t).detach()


def train_at(it_eval, net, images_train, labels_train, testloader, args, mean=None, std=None):
    device = args.device
    net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    current_lr = 0.01 if 'ConvNet' in net.__class__.__name__ else EMN_LR
    optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=EMN_MOMENTUM, weight_decay=EMN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EMN_EPOCHS * 0.5), int(EMN_EPOCHS * 0.75)], gamma=0.1)
    trainloader = torch.utils.data.DataLoader(TensorDataset(images_train, labels_train), batch_size=EMN_BATCH, shuffle=True, num_workers=0)

    loss_train = acc_train = 0.0
    start = time.time()

    for ep in range(EMN_EPOCHS):
        loss_sum = acc_sum = n = 0
        for imgs, labs in trainloader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)
            net.eval()
            x_adv = pgd_attack(net, imgs, labs, criterion, eps=args.at_eps, alpha=args.at_eps / 5.0, steps=args.at_steps, mean=mean, std=std)
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
        if (ep + 1) % (EMN_EPOCHS // 10) == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            net.train()
            print(f'[AT  eps={args.at_eps*255:.0f}] Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d} ({100*(ep+1)//EMN_EPOCHS:3d}%)  '
                  f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test_mid:.4f}')

    time_train = time.time() - start
    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'[AT  eps={args.at_eps*255:.0f}] Eval_{it_eval:02d}: FINAL  train_time={int(time_train)}s  '
          f'train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test:.4f}')
    return acc_test


def main():
    import argparse
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval',  type=int,   default=1)
    parser.add_argument('--data_path', type=str,   default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/data')
    # parser.add_argument('--data_path', type=str,   default='/home/mmoslem3/scratch/UE-DD/data/')
    parser.add_argument('--save_path', type=str,   default='extraEXP')
    parser.add_argument('--datasets',  nargs='+',  default=ALL_DATASETS)
    parser.add_argument('--at_eps_list', nargs='+', type=int, default=[ 16],
                        help='AT budgets in integer /255 units')
    parser.add_argument('--at_steps', type=int, default=7)
    args = parser.parse_args()

    args.dsa = False; args.dc_aug_param = None; args.eval_mode = 'S'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_path, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n{'='*70}")
    print(f"  clean-eval.py  —  {ts}")
    print(f"  Device: {args.device}  Datasets: {args.datasets}")
    print(f"  AT budgets: {args.at_eps_list}/255  steps: {args.at_steps}")
    print(f"{'='*70}")

    # results[dataset][trainer_tag][model_name] = (mean, std)
    all_results = {}

    for dataset in args.datasets:
        if dataset not in MODEL_EVAL_POOLS:
            print(f"WARNING: Unknown dataset '{dataset}' — skipping.")
            continue

        (channel, im_size, num_classes, _cn, mean, std, dst_train, _dt, testloader) = get_dataset(dataset, args.data_path)

        all_images = torch.stack([dst_train[i][0] for i in range(len(dst_train))])
        all_labels = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long)
        print(f"\n  Dataset={dataset}  images={tuple(all_images.shape)}")

        all_results[dataset] = {}

        # # ── Normal training ───────────────────────────────────────────────────
        # tag = 'Normal'
        # print(f"\n  [{tag}]")
        # all_results[dataset][tag] = {}
        # for model_name in MODEL_EVAL_POOLS[dataset]:
        #     accs = []
        #     for it_eval in range(args.num_eval):
        #         net_eval = build_emn_model(model_name, num_classes, channel, im_size).to(args.device)
        #         acc_test = train_normal(
        #             it_eval, net_eval,
        #             copy.deepcopy(all_images), copy.deepcopy(all_labels),
        #             testloader, args, mean, std
        #         )
        #         accs.append(acc_test)
        #     m, s = float(np.mean(accs)), float(np.std(accs))
        #     all_results[dataset][tag][model_name] = (m, s)
        #     print(f"  RESULT  dataset={dataset}  trainer={tag}  model={model_name}  mean={m:.4f}  std={s:.4f}")

        # ── AT training at each budget ─────────────────────────────────────────
        for budget in args.at_eps_list:
            args.at_eps = budget / 255.0
            tag = f'AT-{budget}'
            print(f"\n  [{tag}]  eps={args.at_eps:.5f}")
            all_results[dataset][tag] = {}
            for model_name in MODEL_EVAL_POOLS[dataset]:
                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = build_emn_model(model_name, num_classes, channel, im_size).to(args.device)
                    acc_test = train_at(
                        it_eval, net_eval,
                        copy.deepcopy(all_images), copy.deepcopy(all_labels),
                        testloader, args, mean, std
                    )
                    accs.append(acc_test)
                m, s = float(np.mean(accs)), float(np.std(accs))
                all_results[dataset][tag][model_name] = (m, s)
                print(f"  RESULT  dataset={dataset}  trainer={tag}  model={model_name}  mean={m:.4f}  std={s:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    W = 78
    lines = ["="*W, f"  CLEAN DATA EVAL  (Normal + AT)  —  Generated: {ts}", "="*W]
    for dataset, trainer_dict in all_results.items():
        lines += ["", f"  Dataset: {dataset}", "  " + "-"*(W-2),
                  f"  {'Trainer':<12s}  {'Model':<20s}  {'mean':>8}  {'std':>8}",
                  "  " + "-"*(W-2)]
        for trainer_tag, model_dict in trainer_dict.items():
            for model_name, (m, s) in model_dict.items():
                lines.append(f"  {trainer_tag:<12s}  {model_name:<20s}  {m:>8.4f}  {s:>8.4f}")
    lines.append("")

    torch.save(all_results, os.path.join(args.save_path, "clean_eval_data.pt"))
    out = os.path.join(args.save_path, "results_clean_eval.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out}")
    print(f"  Saved data: {os.path.join(args.save_path, 'clean_eval_data.pt')}")


if __name__ == '__main__':
    main()
