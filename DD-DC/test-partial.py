import os
import copy
import numpy as np
import torch
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.data import TensorDataset
from util import get_dataset, epoch, build_emn_model

# ── Paths (edit these) ────────────────────────────────────────────────────────
DATASET      = 'CIFAR10'
DATA_PATH    = '/home/mmoslem3/scratch/UE-DD/data/'
# SAVE_PATH    = '/home/mmoslem3/scratch/UE-DD/result-partial'

# Subset index file produced by DGC6.py
SUBSET_PT    = '/home/mmoslem3/scratch/UE-DD/partial/subset_CIFAR10_0.30.pt'

# Samplewise perturbation.pt produced by UE-EMN/make_noise.sh
EMN_NOISE_PT = '/home/mmoslem3/scratch/UE-DD/partial/perturbation-30.pt'

# Poisoned dataset .pt produced by DGC6.py (contains images_poisoned + labels)
DGC_PT       = '/home/mmoslem3/scratch/UE-DD/partial/cifar10-partial_30.pt'

# ── Training hyper-parameters ─────────────────────────────────────────────────
NUM_EVAL   = 3
EMN_LR     = 0.1
EMN_MOM    = 0.9
EMN_WD     = 5e-4
EMN_EPOCHS = 25
EMN_BATCH  = 256
MODELS     = ['ResNet18']

# If True: after adding noise to the subset, fill the rest of the training set
# with the remaining clean images (so the full dataset is used for training).
# If False: train only on the subset (poisoned or clean).
USE_FULL_TRAIN = True


# ── Train + evaluate one model ────────────────────────────────────────────────

def train_and_eval(label, net, images_train, labels_train, testloader, args):
    device = args.device
    net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR,
                                momentum=EMN_MOM, weight_decay=EMN_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EMN_EPOCHS, eta_min=0.0)

    loader = torch.utils.data.DataLoader(
        TensorDataset(images_train, labels_train),
        batch_size=EMN_BATCH, shuffle=True, num_workers=0)

    t0 = time.time()
    for ep in range(EMN_EPOCHS):
        net.train()
        loss_sum = acc_sum = n = 0
        for imgs, labs in loader:
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

        if (ep + 1) % (EMN_EPOCHS // 10) == 0:
            net.eval()
            _, acc_mid = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            print(f'  [{label}] ep {ep+1:03d}/{EMN_EPOCHS}  '
                  f'train_loss={loss_sum/n:.4f}  train_acc={acc_sum/n:.4f}  test_acc={acc_mid:.4f}')

    _, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print(f'  [{label}] FINAL  train_time={int(time.time()-t0)}s  '
          f'train_acc={acc_sum/n:.4f}  test_acc={acc_test:.4f}')
    return acc_test


# ── Build subset images/labels tensors from dst_train + keep_idx ──────────────

def build_clean_tensors(dst_train, keep_idx, device):
    images = torch.stack([dst_train[i][0] for i in keep_idx]).to(device)
    labels = torch.tensor([dst_train[i][1] for i in keep_idx],
                          dtype=torch.long, device=device)
    return images, labels


# ── Apply SW noise to clean images ────────────────────────────────────────────

def apply_sw_noise(images_clean, mean, std, device):
    raw = torch.load(EMN_NOISE_PT, map_location='cpu', weights_only=False)
    noise = raw if not isinstance(raw, dict) else raw['noise']
    noise = torch.as_tensor(noise, dtype=torch.float32)

    # normalize to [0, 1] range if stored as uint8-scale
    if noise.abs().max() > 1.5:
        noise = noise / 255.0

    # NHWC → NCHW if needed
    if noise.ndim == 4 and noise.shape[-1] in (1, 3):
        noise = noise.permute(0, 3, 1, 2).contiguous()

    noise = noise.to(device)

    if noise.shape[0] != images_clean.shape[0]:
        raise ValueError(f'SW noise size {noise.shape[0]} != images size {images_clean.shape[0]}')

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)

    images_raw = images_clean * std_t + mean_t
    images_poisoned = (torch.clamp(images_raw + noise, 0.0, 1.0) - mean_t) / std_t
    print(f'  [SW] noise linf={noise.abs().max():.4f}  shape={tuple(images_poisoned.shape)}')
    return images_poisoned


# ── Load DGC poisoned dataset ─────────────────────────────────────────────────

def load_dgc_tensors(device):
    data = torch.load(DGC_PT, map_location=device, weights_only=False)
    images = data['images_poisoned'].to(device)
    labels = data['labels'].to(device)
    print(f'  [DGC] loaded {tuple(images.shape)} from {os.path.basename(DGC_PT)}')
    return images, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Args:
        pass
    args = Args()
    args.device       = device
    args.dsa          = False
    args.dc_aug_param = None
    args.eval_mode    = 'S'

    # os.makedirs(SAVE_PATH, exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n{"="*60}')
    print(f'  test-partial.py  started {ts}')
    print(f'  device={device}  epochs={EMN_EPOCHS}  models={MODELS}')
    print(f'{"="*60}')

    # ── Load dataset ──────────────────────────────────────────────────────────
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = \
        get_dataset(DATASET, DATA_PATH)

    # ── Load subset indices ───────────────────────────────────────────────────
    subset_data = torch.load(SUBSET_PT, map_location='cpu', weights_only=False)
    keep_idx    = subset_data['indices']
    print(f'  Subset: {len(keep_idx)} samples from {os.path.basename(SUBSET_PT)}')
    print(f'  USE_FULL_TRAIN={USE_FULL_TRAIN}')

    # ── Build clean subset tensors ────────────────────────────────────────────
    images_clean, labels_clean = build_clean_tensors(dst_train, keep_idx, device)

    # ── Build remaining clean tensors (indices not in subset) ─────────────────
    if USE_FULL_TRAIN:
        rest_idx = sorted(set(range(len(dst_train))) - set(keep_idx))
        images_rest, labels_rest = build_clean_tensors(dst_train, rest_idx, device)
        print(f'  Remaining clean samples: {len(rest_idx)}')

    def maybe_add_rest(images_poisoned, labels_poisoned):
        """Append remaining clean images when USE_FULL_TRAIN is enabled."""
        if not USE_FULL_TRAIN:
            return images_poisoned, labels_poisoned
        return (torch.cat([images_poisoned, images_rest], dim=0),
                torch.cat([labels_poisoned, labels_rest], dim=0))

    results = {}

    # # ── Condition 1: clean ────────────────────────────────────────────────────
    # print(f'\n{"─"*60}')
    # print('  Condition: clean')
    # imgs_c, lbls_c = maybe_add_rest(images_clean, labels_clean)
    # results['clean'] = {}
    # for model_name in MODELS:
    #     accs = []
    #     for it in range(NUM_EVAL):
    #         net = build_emn_model(model_name, num_classes, channel, im_size).to(device)
    #         acc = train_and_eval(
    #             f'clean/{model_name}/run{it}', net,
    #             copy.deepcopy(imgs_c), copy.deepcopy(lbls_c),
    #             testloader, args)
    #         accs.append(acc)
    #     m, s = float(np.mean(accs)), float(np.std(accs))
    #     results['clean'][model_name] = (m, s)
    #     print(f'  clean  {model_name}  mean={m:.4f}  std={s:.4f}')

    # ── Condition 1b: clean2 (whole dataset minus the subset) ────────────────
    print(f'\n{"─"*60}')
    print('  Condition: clean2 (full dataset excluding subset)')
    rest_idx_c2 = sorted(set(range(len(dst_train))) - set(keep_idx))
    imgs_c2, lbls_c2 = build_clean_tensors(dst_train, rest_idx_c2, device)
    print(f'  clean2 training size: {len(rest_idx_c2)}')
    results['clean2'] = {}
    for model_name in MODELS:
        accs = []
        for it in range(NUM_EVAL):
            net = build_emn_model(model_name, num_classes, channel, im_size).to(device)
            acc = train_and_eval(
                f'clean2/{model_name}/run{it}', net,
                copy.deepcopy(imgs_c2), copy.deepcopy(lbls_c2),
                testloader, args)
            accs.append(acc)
        m, s = float(np.mean(accs)), float(np.std(accs))
        results['clean2'][model_name] = (m, s)
        print(f'  clean2 {model_name}  mean={m:.4f}  std={s:.4f}')

    # ── Condition 2: SW (EMN samplewise noise on the subset) ──────────────────
    print(f'\n{"─"*60}')
    print('  Condition: SW (EMN samplewise)')
    images_sw = apply_sw_noise(images_clean, mean, std, device)
    imgs_sw, lbls_sw = maybe_add_rest(images_sw, labels_clean)
    results['SW'] = {}
    for model_name in MODELS:
        accs = []
        for it in range(NUM_EVAL):
            net = build_emn_model(model_name, num_classes, channel, im_size).to(device)
            acc = train_and_eval(
                f'SW/{model_name}/run{it}', net,
                copy.deepcopy(imgs_sw), copy.deepcopy(lbls_sw),
                testloader, args)
            accs.append(acc)
        m, s = float(np.mean(accs)), float(np.std(accs))
        results['SW'][model_name] = (m, s)
        print(f'  SW     {model_name}  mean={m:.4f}  std={s:.4f}')

    # ── Condition 3: DGC (partial DGC6 poisoned dataset) ─────────────────────
    print(f'\n{"─"*60}')
    print('  Condition: DGC')
    images_dgc, labels_dgc = load_dgc_tensors(device)
    imgs_dgc, lbls_dgc = maybe_add_rest(images_dgc, labels_dgc)
    results['DGC'] = {}
    for model_name in MODELS:
        accs = []
        for it in range(NUM_EVAL):
            net = build_emn_model(model_name, num_classes, channel, im_size).to(device)
            acc = train_and_eval(
                f'DGC/{model_name}/run{it}', net,
                copy.deepcopy(imgs_dgc), copy.deepcopy(lbls_dgc),
                testloader, args)
            accs.append(acc)
        m, s = float(np.mean(accs)), float(np.std(accs))
        results['DGC'][model_name] = (m, s)
        print(f'  DGC    {model_name}  mean={m:.4f}  std={s:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 62
    print(f'\n{"="*W}')
    print(f'  {"Condition":<10} {"Model":<20} {"mean test acc":>14}  {"std":>8}')
    print(f'  {"─"*(W-2)}')
    for cond, cond_res in results.items():
        for model_name, (m, s) in cond_res.items():
            print(f'  {cond:<10} {model_name:<20} {m:>14.4f}  {s:>8.4f}')
    print(f'{"="*W}')

    # # ── Save summary ──────────────────────────────────────────────────────────
    # out_path = os.path.join(SAVE_PATH, 'results_partial.txt')
    # with open(out_path, 'w') as f:
    #     f.write(f'test-partial.py  {ts}\n')
    #     f.write(f'dataset={DATASET}  subset={len(keep_idx)} samples\n')
    #     f.write(f'subset_pt={SUBSET_PT}\n')
    #     f.write(f'emn_noise_pt={EMN_NOISE_PT}\n')
    #     f.write(f'dgc_pt={DGC_PT}\n\n')
    #     f.write(f'{"Condition":<10} {"Model":<20} {"mean":>10}  {"std":>8}\n')
    #     f.write(f'{"─"*52}\n')
    #     for cond, cond_res in results.items():
    #         for model_name, (m, s) in cond_res.items():
    #             f.write(f'{cond:<10} {model_name:<20} {m:>10.4f}  {s:>8.4f}\n')
    # print(f'\n  Saved: {out_path}')


if __name__ == '__main__':
    main()
