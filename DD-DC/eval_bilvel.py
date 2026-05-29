# #!/usr/bin/env python
# """
# Evaluate a distilled-space poisoning attack.

# Loads saved poison tensors (produced by direct_bilevel_poison.py), injects them
# into the full clean CIFAR-10 training set D_c, trains a fresh victim model, and
# reports:
#     - clean test accuracy on the CIFAR-10 test set
#     - ASR on x_t (model predicts y_adv?)
# across multiple random seeds, with an optional clean-baseline comparison.
# """
# import argparse
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms as T
# from torch.utils.data import Dataset, ConcatDataset, DataLoader


# # Standard CIFAR-10 normalization (matches DC / DM / MTT preprocessing)
# CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# # ======================================================================
# # ConvNet (same architecture used to craft poisons)
# # ======================================================================
# class ConvNet(nn.Module):
#     def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3,
#                  net_act='relu', net_norm='instancenorm',
#                  net_pooling='avgpooling', im_size=(32, 32)):
#         super().__init__()
#         self.features, shape_feat = self._make_layers(
#             channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
#         self.classifier = nn.Linear(
#             shape_feat[0] * shape_feat[1] * shape_feat[2], num_classes)

#     def forward(self, x):
#         h = self.features(x)
#         h = h.view(h.size(0), -1)
#         return self.classifier(h)

#     @staticmethod
#     def _act(name):
#         return {'relu': nn.ReLU(inplace=True),
#                 'leakyrelu': nn.LeakyReLU(0.01)}[name]

#     @staticmethod
#     def _pool(name):
#         return {'avgpooling': nn.AvgPool2d(2, 2),
#                 'maxpooling': nn.MaxPool2d(2, 2)}[name]

#     @staticmethod
#     def _norm(name, shape):
#         if name == 'instancenorm': return nn.GroupNorm(shape[0], shape[0], affine=True)
#         if name == 'batchnorm':    return nn.BatchNorm2d(shape[0], affine=True)
#         if name == 'groupnorm':    return nn.GroupNorm(4, shape[0], affine=True)
#         if name == 'layernorm':    return nn.LayerNorm(shape, elementwise_affine=True)
#         return nn.Identity()

#     def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
#         layers, in_c = [], channel
#         shape_feat = [in_c, im_size[0], im_size[1]]
#         for _ in range(net_depth):
#             layers.append(nn.Conv2d(in_c, net_width, 3, padding=1))
#             shape_feat[0] = net_width
#             if net_norm != 'none':
#                 layers.append(self._norm(net_norm, shape_feat))
#             layers.append(self._act(net_act))
#             in_c = net_width
#             if net_pooling != 'none':
#                 layers.append(self._pool(net_pooling))
#                 shape_feat[1] //= 2
#                 shape_feat[2] //= 2
#         return nn.Sequential(*layers), shape_feat


# # ======================================================================
# # Poisoned training set = D_c  ∪  S_p
# # ======================================================================
# class TensorDS(Dataset):
#     """Already-normalized (C,H,W) tensors with int labels (matches the format of the saved poisons)."""
#     def __init__(self, X, Y):
#         self.X = X.float()
#         self.Y = Y.long()
#     def __len__(self): return len(self.Y)
#     def __getitem__(self, i):
#         return self.X[i], int(self.Y[i].item())


# def build_loaders(data_root, poison_files, batch_size=256, augment=True,
#                   num_workers=4):
#     train_tf_list = []
#     if augment:
#         train_tf_list += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
#     train_tf_list += [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
#     train_tf = T.Compose(train_tf_list)
#     test_tf  = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

#     clean_train = torchvision.datasets.CIFAR10(
#         root=data_root, train=True,  download=False, transform=train_tf)
#     clean_test  = torchvision.datasets.CIFAR10(
#         root=data_root, train=False, download=False, transform=test_tf)

#     # load all poison shards
#     S_parts, Y_parts, meta = [], [], []
#     for f in poison_files:
#         blob = torch.load(f, map_location='cpu')
#         S_parts.append(blob['S_p'].float())
#         Y_parts.append(blob['Y_p'].long())
#         meta.append({
#             'file': f,
#             'target_class': blob.get('target_class', None),
#             'adv_label':    blob.get('adv_label', None),
#             'n_poison':     int(blob['S_p'].shape[0]),
#         })
#         print(f"loaded {f}: S_p {tuple(blob['S_p'].shape)} "
#               f"(class {blob.get('target_class')} -> y_adv {blob.get('adv_label')})")

#     S_all = torch.cat(S_parts, dim=0)
#     Y_all = torch.cat(Y_parts, dim=0)
#     print(f'total poisons injected: {S_all.shape[0]}  (clean train size: {len(clean_train)})')

#     poison_ds = TensorDS(S_all, Y_all)
#     combined  = ConcatDataset([clean_train, poison_ds])

#     poisoned_loader = DataLoader(combined,    batch_size=batch_size, shuffle=True,
#                                  num_workers=num_workers, pin_memory=True, drop_last=False)
#     clean_loader    = DataLoader(clean_train, batch_size=batch_size, shuffle=True,
#                                  num_workers=num_workers, pin_memory=True, drop_last=False)
#     test_loader     = DataLoader(clean_test,  batch_size=512,        shuffle=False,
#                                  num_workers=num_workers, pin_memory=True)

#     return clean_loader, poisoned_loader, test_loader, meta


# # ======================================================================
# # Train / eval
# # ======================================================================
# def train_model(model, loader, epochs, lr, device, log_every=5):
#     model.train()
#     opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
#     sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
#     for ep in range(epochs):
#         loss_sum, correct, total = 0.0, 0, 0
#         for x, y in loader:
#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
#             logits = model(x)
#             loss = F.cross_entropy(logits, y)
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#             loss_sum += loss.item() * x.size(0)
#             correct  += (logits.argmax(-1) == y).sum().item()
#             total    += x.size(0)
#         sched.step()
#         if (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1:
#             print(f'  ep {ep+1:3d}/{epochs}  loss {loss_sum/total:.4f}  '
#                   f'train_acc {correct/total:.4f}  lr {opt.param_groups[0]["lr"]:.4f}')


# @torch.no_grad()
# def clean_acc(model, loader, device):
#     model.eval()
#     correct, total = 0, 0
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         correct += (model(x).argmax(-1) == y).sum().item()
#         total   += x.size(0)
#     return correct / total


# @torch.no_grad()
# def asr_on_target(model, x_t, y_adv, device):
#     """Returns (hit, pred_label, confidence_on_y_adv, full_probs)."""
#     model.eval()
#     x = x_t.to(device)
#     if x.dim() == 3: x = x.unsqueeze(0)
#     logits = model(x)
#     probs  = F.softmax(logits, dim=-1).squeeze(0)
#     pred   = int(probs.argmax().item())
#     return int(pred == y_adv), pred, float(probs[y_adv].item()), probs.cpu().numpy()


# # ======================================================================
# # Main
# # ======================================================================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--data_root',     default='/home/mmoslem3/scratch/UE-DD/data')
#     ap.add_argument('--poison_files',  nargs='+', required='./poisoned_class.pt',
#                     help='One or more .pt files saved by direct_bilevel_poison.py')
#     ap.add_argument('--target_x_path', required=True,
#                     help='.pt file holding x_t (same preprocessing as training data).')
#     ap.add_argument('--adv_label',     type=int, required=5,
#                     help='y_adv: label the attacker wants x_t to be predicted as.')

#     ap.add_argument('--epochs',     type=int,   default=20)
#     ap.add_argument('--lr',         type=float, default=0.01)
#     ap.add_argument('--batch_size', type=int,   default=256)
#     ap.add_argument('--no_augment', action='store_true',
#                     help='Disable random crop / flip on clean data.')

#     ap.add_argument('--seeds', type=int, nargs='+', default=[0])
#     ap.add_argument('--also_clean_baseline', action='store_true',
#                     help='Also train a model on clean D_c only (same seeds) for reference.')

#     ap.add_argument('--save_path', default='./eval_results.pt')
#     ap.add_argument('--device',    default='cuda')
#     args = ap.parse_args()

#     device = args.device if torch.cuda.is_available() else 'cpu'
#     print(f'device = {device}')

#     clean_loader, poisoned_loader, test_loader, poison_meta = build_loaders(
#         data_root=args.data_root,
#         poison_files=args.poison_files,
#         batch_size=args.batch_size,
#         augment=(not args.no_augment),
#     )

#     # load x_t
#     x_t = torch.load(args.target_x_path, map_location='cpu')
#     if x_t.dim() == 3: x_t = x_t.unsqueeze(0)
#     x_t = x_t.float()

#     results = {'poisoned': [], 'clean': [], 'meta': poison_meta, 'args': vars(args)}

#     # optional clean baseline
#     if args.also_clean_baseline:
#         for seed in args.seeds:
#             print(f'\n=== CLEAN baseline, seed {seed} ===')
#             torch.manual_seed(seed); np.random.seed(seed)
#             model = ConvNet().to(device)
#             train_model(model, clean_loader, args.epochs, args.lr, device)
#             acc = clean_acc(model, test_loader, device)
#             hit, pred, conf, probs = asr_on_target(model, x_t, args.adv_label, device)
#             print(f'  clean_acc {acc:.4f}   target_pred {pred}   '
#                   f'hit {hit}   P(y_adv)={conf:.4f}')
#             results['clean'].append(dict(seed=seed, clean_acc=acc,
#                                          target_pred=pred, hit=hit,
#                                          conf_y_adv=conf, probs=probs))

#     # poisoned runs
#     for seed in args.seeds:
#         print(f'\n=== POISONED (D_c ∪ S_p), seed {seed} ===')
#         torch.manual_seed(seed); np.random.seed(seed)
#         model = ConvNet().to(device)
#         train_model(model, poisoned_loader, args.epochs, args.lr, device)
#         acc = clean_acc(model, test_loader, device)
#         hit, pred, conf, probs = asr_on_target(model, x_t, args.adv_label, device)
#         print(f'  clean_acc {acc:.4f}   target_pred {pred}   '
#               f'hit {hit}   P(y_adv)={conf:.4f}')
#         results['poisoned'].append(dict(seed=seed, clean_acc=acc,
#                                         target_pred=pred, hit=hit,
#                                         conf_y_adv=conf, probs=probs))

#     # summary
#     def summarize(tag):
#         items = results[tag]
#         if not items: return
#         accs  = np.array([it['clean_acc']   for it in items])
#         hits  = np.array([it['hit']         for it in items])
#         confs = np.array([it['conf_y_adv']  for it in items])
#         print(f'\n[{tag.upper()}]  '
#               f'clean_acc {accs.mean():.4f} ± {accs.std():.4f}   '
#               f'ASR {hits.mean():.4f} ({int(hits.sum())}/{len(hits)})   '
#               f'P(y_adv) {confs.mean():.4f} ± {confs.std():.4f}')

#     print('\n========== SUMMARY ==========')
#     summarize('clean')
#     summarize('poisoned')

#     # delta
#     if args.also_clean_baseline and results['clean'] and results['poisoned']:
#         a_c = np.mean([it['clean_acc']  for it in results['clean']])
#         a_p = np.mean([it['clean_acc']  for it in results['poisoned']])
#         h_c = np.mean([it['hit']        for it in results['clean']])
#         h_p = np.mean([it['hit']        for it in results['poisoned']])
#         print(f'\nΔ clean_acc  = {a_p - a_c:+.4f}  (poisoned - clean)')
#         print(f'Δ ASR        = {h_p - h_c:+.4f}  (poisoned - clean)')

#     torch.save(results, args.save_path)
#     print(f'\nresults saved to {args.save_path}')


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python
"""
Minimal evaluation for distilled-space poisons.

Injects saved S_p into the full clean CIFAR-10 training set, trains a fresh
ConvNet from scratch, and reports:
    - overall clean test accuracy
    - per-class test accuracy
    - confusion pattern on the attacked class(es) (metadata read from poison file)
across multiple seeds, with an optional clean baseline.

No x_t, no ASR - just training sabotage metrics.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, ConcatDataset, DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ======================================================================
# ConvNet (must match the arch used to craft poisons)
# ======================================================================
class ConvNet(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3,
                 net_act='relu', net_norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32)):
        super().__init__()
        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        self.classifier = nn.Linear(
            shape_feat[0] * shape_feat[1] * shape_feat[2], num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)

    @staticmethod
    def _act(name):
        return {'relu': nn.ReLU(inplace=True),
                'leakyrelu': nn.LeakyReLU(0.01)}[name]
    @staticmethod
    def _pool(name):
        return {'avgpooling': nn.AvgPool2d(2, 2),
                'maxpooling': nn.MaxPool2d(2, 2)}[name]
    @staticmethod
    def _norm(name, shape):
        if name == 'instancenorm': return nn.GroupNorm(shape[0], shape[0], affine=True)
        if name == 'batchnorm':    return nn.BatchNorm2d(shape[0], affine=True)
        return nn.Identity()

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers, in_c = [], channel
        shape_feat = [in_c, im_size[0], im_size[1]]
        for _ in range(net_depth):
            layers.append(nn.Conv2d(in_c, net_width, 3, padding=1))
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers.append(self._norm(net_norm, shape_feat))
            layers.append(self._act(net_act))
            in_c = net_width
            if net_pooling != 'none':
                layers.append(self._pool(net_pooling))
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        return nn.Sequential(*layers), shape_feat


# ======================================================================
# Dataset helpers
# ======================================================================
class TensorDS(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X.float(), Y.long()
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], int(self.Y[i].item())


def build_loaders(data_root, poison_files, batch_size, augment, num_workers=4):
    train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
    train_tf += [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    train_tf = T.Compose(train_tf)
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    clean_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=False, transform=train_tf)
    clean_test  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=test_tf)

    S_parts, Y_parts, meta = [], [], []
    for f in poison_files:
        b = torch.load(f, map_location='cpu')
        S_parts.append(b['S_p'].float())
        Y_parts.append(b['Y_p'].long())
        meta.append(dict(file=f,
                         target_class=b.get('target_class'),
                         adv_label=b.get('adv_label'),
                         n_poison=int(b['S_p'].shape[0])))
        print(f"loaded {f}: S_p {tuple(b['S_p'].shape)} "
              f"(c_i={b.get('target_class')}, y_adv={b.get('adv_label')})")

    S_all = torch.cat(S_parts, dim=0)
    Y_all = torch.cat(Y_parts, dim=0)
    print(f'injected {S_all.shape[0]} poisons into {len(clean_train)} clean samples')

    poison_ds = TensorDS(S_all, Y_all)
    poisoned  = ConcatDataset([clean_train, poison_ds])

    pois_ld  = DataLoader(poisoned,    batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    clean_ld = DataLoader(clean_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(clean_test,  batch_size=512,        shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return clean_ld, pois_ld, test_ld, meta


# ======================================================================
# Train / eval
# ======================================================================
def train_model(model, loader, epochs, lr, device, log_every=10):
    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        loss_sum, corr, tot = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            corr     += (logits.argmax(-1) == y).sum().item()
            tot      += x.size(0)
        sched.step()
        if ep == 0 or (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f'  ep {ep+1:3d}/{epochs}  loss {loss_sum/tot:.4f}  '
                  f'train_acc {corr/tot:.4f}  lr {opt.param_groups[0]["lr"]:.4f}')


@torch.no_grad()
def eval_full(model, loader, device, num_classes=10):
    """Return overall acc, per-class acc, confusion matrix."""
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(-1)
        for t, pr in zip(y.cpu(), p.cpu()):
            conf[t.item(), pr.item()] += 1
    per_class = (conf.diag().float() / conf.sum(dim=1).clamp(min=1).float()).numpy()
    overall   = conf.diag().sum().item() / conf.sum().item()
    return overall, per_class, conf.numpy()


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',    default='/home/mmoslem3/scratch/UE-DD/data')
    ap.add_argument('--poison_files', nargs='+', default=['/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/poisoned_class.pt'])

    ap.add_argument('--epochs',     type=int,   default=20)
    ap.add_argument('--lr',         type=float, default=0.01)
    ap.add_argument('--batch_size', type=int,   default=256)
    ap.add_argument('--no_augment', action='store_true')

    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--also_clean_baseline', action='store_true')

    ap.add_argument('--save_path', default='./eval_results.pt')
    ap.add_argument('--device',    default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'device = {device}')

    clean_ld, pois_ld, test_ld, meta = build_loaders(
        args.data_root, args.poison_files,
        batch_size=args.batch_size, augment=(not args.no_augment))

    target_classes = sorted({m['target_class'] for m in meta if m['target_class'] is not None})

    results = {'poisoned': [], 'clean': [], 'meta': meta, 'args': vars(args)}

    def run(tag, loader, seed):
        print(f'\n=== {tag}, seed {seed} ===')
        torch.manual_seed(seed); np.random.seed(seed)
        model = ConvNet().to(device)
        train_model(model, loader, args.epochs, args.lr, device)
        overall, per_class, conf = eval_full(model, test_ld, device)
        print(f'  overall test_acc = {overall:.4f}')
        print(f'  per_class_acc    = ' + '  '.join(f'{a:.3f}' for a in per_class))
        for c in target_classes:
            row = conf[c]
            top3 = np.argsort(-row)[:3]
            leaks = '  '.join(f'{int(i)}:{int(row[i])}' for i in top3)
            print(f'  true class {c} -> top-3 preds: {leaks}')
        return dict(seed=seed, overall=overall, per_class=per_class, conf=conf)

    if args.also_clean_baseline:
        for s in args.seeds:
            results['clean'].append(run('CLEAN baseline', clean_ld, s))

    for s in args.seeds:
        results['poisoned'].append(run('POISONED (D_c ∪ S_p)', pois_ld, s))

    # summary
    print('\n========== SUMMARY ==========')
    def summarize(tag):
        items = results[tag]
        if not items: return
        ovs = np.array([r['overall'] for r in items])
        pcs = np.stack([r['per_class'] for r in items], 0)
        print(f'[{tag.upper()}] overall = {ovs.mean():.4f} ± {ovs.std():.4f}')
        print(f'[{tag.upper()}] per-class mean: ' + '  '.join(f'{a:.3f}' for a in pcs.mean(0)))

    summarize('clean')
    summarize('poisoned')

    if args.also_clean_baseline and results['clean'] and results['poisoned']:
        c_ov = np.mean([r['overall']   for r in results['clean']])
        p_ov = np.mean([r['overall']   for r in results['poisoned']])
        c_pc = np.stack([r['per_class'] for r in results['clean']], 0).mean(0)
        p_pc = np.stack([r['per_class'] for r in results['poisoned']], 0).mean(0)
        print(f'\nΔ overall acc = {p_ov - c_ov:+.4f}  (poisoned - clean)')
        print(f'Δ per-class   = ' + '  '.join(f'{d:+.3f}' for d in (p_pc - c_pc)))
        for tc in target_classes:
            print(f'Δ acc on attacked class {tc} = {p_pc[tc] - c_pc[tc]:+.4f}')

    torch.save(results, args.save_path)
    print(f'\nresults saved to {args.save_path}')


if __name__ == '__main__':
    main()