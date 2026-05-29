
import copy
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import TensorDataset
from util import get_dataset, epoch, build_emn_model


MODEL_EVAL_POOLS = {
    'CIFAR10':  ['ResNet18'],
    'CIFAR100': ['ResNet18'],
}

# MODEL_EVAL_POOLS = {
#     'CIFAR10':  ['ResNet50','ViT','VGG11','DenseNet121'],
#     'CIFAR100': ['ResNet50','ViT','VGG11','DenseNet121'],
# }


ALL_DATASETS = list(MODEL_EVAL_POOLS.keys())

EMN_LR           = 0.1
EMN_MOMENTUM     = 0.9
EMN_WEIGHT_DECAY = 5e-4
EMN_EPOCHS       = 50
EMN_BATCH        = 256


def evaluate_synset_Normal(it_eval, net, images_train, labels_train, testloader, args, mean=None, std=None):
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
        if (ep + 1) % (EMN_EPOCHS // 10) == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            net.train()
            print(f'[Nor] Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d} ({100*(ep+1)//EMN_EPOCHS:3d}%)  train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test_mid:.4f}')

    time_train = time.time() - start
    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'[Nor] Eval_{it_eval:02d}: epoch={EMN_EPOCHS:03d}  train_time={int(time_train)}s  train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test:.4f}')
    return net, acc_train, acc_test


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


def evaluate_synset_AT(it_eval, net, images_train, labels_train, testloader, args, mean=None, std=None):
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
            out = net(x_adv)
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
        if (ep + 1) % (EMN_EPOCHS // 20) == 0:
            net.eval()
            _, acc_test_mid = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            net.train()
            print(f'[AT ] Eval_{it_eval:02d}: epoch={ep+1:03d}/{EMN_EPOCHS:03d} ({100*(ep+1)//EMN_EPOCHS:3d}%)  train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test_mid:.4f}')

    time_train = time.time() - start
    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'[AT ] Eval_{it_eval:02d}: epoch={EMN_EPOCHS:03d}  train_time={int(time_train)}s  train_loss={loss_train:.4f}  train_acc={acc_train:.4f}  test_acc={acc_test:.4f}')
    return net, acc_train, acc_test


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval',  type=int, default=1)
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/UE-DD/data/')
    parser.add_argument('--datasets',  nargs='+', default=ALL_DATASETS)
    parser.add_argument('--at_steps',  type=int, default=7)
    args = parser.parse_args()

    args.dsa          = False
    args.dc_aug_param = None
    args.eval_mode    = 'S'
    args.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # trainers = {'Normal': evaluate_synset_Normal, 'AT': evaluate_synset_AT}
    trainers = {'Normal': evaluate_synset_Normal}

    for dataset in args.datasets:
        if dataset not in MODEL_EVAL_POOLS:
            print(f"WARNING: Unknown dataset '{dataset}' — skipping.")
            continue

        (channel, im_size, num_classes, _class_names,
         mean, std, dst_train, _dst_test, testloader) = get_dataset(dataset, args.data_path)

        # Build clean tensors directly from the dataset
        all_images = torch.stack([dst_train[i][0] for i in range(len(dst_train))])
        all_labels = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long)
        print(f"\n  Dataset: {dataset}  images={all_images.shape}  labels={all_labels.shape}")

        # for budget in [2, 4, 6, 8]:
        for budget in [8]:
            args.at_eps = budget / 255.0

            print(f"\n{'='*65}")
            print(f"  Dataset={dataset}  budget={budget}/255  device={args.device}")
            print(f"{'='*65}")

            for trainer_name, train_fn in trainers.items():
                # Normal trainer only runs once (budget-independent), but we print it per-budget for comparison
                print(f"\n  [{trainer_name}]  at_eps={args.at_eps:.5f}" if trainer_name == 'AT' else f"\n  [{trainer_name}]")
                for model_name in MODEL_EVAL_POOLS[dataset]:
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = build_emn_model(model_name, num_classes, channel, im_size).to(args.device)
                        imgs_eval = copy.deepcopy(all_images.detach())
                        lbls_eval = copy.deepcopy(all_labels.detach())
                        _, _, acc_test = train_fn(it_eval, net_eval, imgs_eval, lbls_eval, testloader, args, mean, std)
                        accs.append(acc_test)
                    m, s = float(np.mean(accs)), float(np.std(accs))
                    print(f"  RESULT  dataset={dataset}  budget={budget}/255  trainer={trainer_name}  model={model_name}  mean={m:.4f}  std={s:.4f}")


if __name__ == '__main__':
    main()
