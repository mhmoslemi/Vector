import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import get_dataset, get_network, get_eval_pool, get_daparam, TensorDataset, epoch, ParamDiffAug


def train_with_batch_size(batch_size, images_train, labels_train, testloader,
                          model_eval, channel, num_classes, im_size,
                          args, Epoch, full_batch=False):
    net = get_network(model_eval, channel, num_classes, im_size).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    lr = float(args.lr_net)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    dst_train = TensorDataset(images_train, labels_train)
    # full-batch GD: load everything at once, no shuffling needed
    effective_bs = len(dst_train) if full_batch else batch_size
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=effective_bs, shuffle=not full_batch, num_workers=0
    )

    label = f"Full-Batch GD (bs={effective_bs})" if full_batch else f"SGD bs={batch_size}"
    print(f"\n{'='*60}")
    print(f"  {label}  |  trainset size: {len(dst_train)}")
    print(f"{'='*60}")

    start = time.time()
    loss_avg, acc_avg = 0.0, 0.0

    for ep in range(Epoch + 1):
        loss_avg, acc_avg, num_exp = 0.0, 0.0, 0
        net.train()

        for datum in trainloader:
            img = datum[0].float().to(args.device)
            lab = datum[1].long().to(args.device)
            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(
                np.argmax(output.cpu().data.numpy(), axis=-1),
                lab.cpu().data.numpy()
            ))

            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_avg /= num_exp
        acc_avg /= num_exp

        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        if ep % 50 == 0:
            _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            print(f"  [ep {ep:04d}]  train loss: {loss_avg:.4f}  train acc: {acc_avg:.4f}  test acc: {acc_test:.4f}")

    time_train = time.time() - start
    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f"\n  FINAL [{label}]  time: {int(time_train)}s  "
          f"train loss: {loss_avg:.4f}  train acc: {acc_avg:.4f}  test acc: {acc_test:.4f}")
    return acc_test


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='ConvNet')
    parser.add_argument('--ipc', type=int, default=100)
    parser.add_argument('--eval_mode', type=str, default='S')
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=4)
    parser.add_argument('--epoch_eval_train', type=int, default=300)
    parser.add_argument('--Iteration', type=int, default=300)
    parser.add_argument('--lr_img', type=float, default=0.1)
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--init', type=str, default='real')
    parser.add_argument('--dsa_strategy', type=str, default='None')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/UE-DD/data')
    parser.add_argument('--save_path', type=str, default='result-dc')
    parser.add_argument('--dis_metric', type=str, default='ours')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False

    channel, im_size, num_classes, _, _, _, _, _, testloader = \
        get_dataset(args.dataset, args.data_path)
    model_eval = get_eval_pool(args.eval_mode, args.model, args.model)[0]

    pt_file = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DC_CIFAR10_ConvNet_100ipc.pt'
    data = torch.load(pt_file, map_location=args.device, weights_only=False)
    data = data['data']
    image_syn = data[0].to(args.device)
    label_syn = data[1].to(args.device)

    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)

    images_train = copy.deepcopy(image_syn.detach())
    labels_train = copy.deepcopy(label_syn.detach())

    Epoch = 400
    total_size = len(images_train)  # 1000

    # Batch sizes: small → large → full batch GD
    sgd_batch_sizes = [4,8, 16,32, 64,128, 256,512]
    sgd_batch_sizes = [64,128, 256,512]

    print(f"\nDataset: {args.dataset}  |  Total train size: {total_size}  |  Epochs: {Epoch}")
    print(f"Will run SGD with batch sizes {sgd_batch_sizes}, then full-batch GD (bs={total_size})\n")

    results = {}

    for bs in sgd_batch_sizes:
        acc = train_with_batch_size(
            bs, images_train, labels_train, testloader,
            model_eval, channel, num_classes, im_size, args, Epoch, full_batch=False
        )
        results[f"SGD bs={bs}"] = acc

    # Full-batch gradient descent
    acc = train_with_batch_size(
        total_size, images_train, labels_train, testloader,
        model_eval, channel, num_classes, im_size, args, Epoch, full_batch=True
    )
    results[f"Full-Batch GD"] = acc

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY  (test accuracy after {Epoch} epochs)")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name:<25s}  test acc: {acc:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
