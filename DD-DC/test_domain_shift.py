import os
import copy
import numpy as np
import torch
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as T
from util import get_dataset, build_emn_model

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET      = 'CIFAR10'
DATA_PATH    = '/home/mmoslem3/scratch/UE-DD/data/'

# Full dataset perturbation.pt
EMN_NOISE_PT = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN/CIFAR10_SW.pt'

# Full poisoned dataset .pt
DGC_PT       = '/home/mmoslem3/scratch/UE-DD/result-fianle/res_CIFAR10_iter150_bug8_lamexcess0.5.pt'

# ── Training hyper-parameters ─────────────────────────────────────────────────
NUM_EVAL   = 1
EMN_LR     = 0.1
EMN_MOM    = 0.9
EMN_WD     = 5e-4
EMN_EPOCHS = 15
EMN_BATCH  = 256
MODELS     = ['ResNet18']


# ── Domain Adaptation Label Alignment (CIFAR-10 <-> STL-10) ───────────────────
def align_to_9_classes(images, labels, source: str):
    """
    Drops the mismatched class (CIFAR Frog=6, STL Monkey=7) and aligns 
    the remaining 9 classes to a shared 0-8 indexing system.
    """
    # Target mapping: 
    # 0:plane, 1:car, 2:bird, 3:cat, 4:deer, 5:dog, 6:horse, 7:ship, 8:truck
    if source == 'CIFAR10':
        mask = labels != 6  # Drop Frog
        mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 7:6, 8:7, 9:8}
    elif source == 'STL10':
        mask = labels != 7  # Drop Monkey
        mapping = {0:0, 2:1, 1:2, 3:3, 4:4, 5:5, 6:6, 8:7, 9:8}
    else:
        raise ValueError("Source must be CIFAR10 or STL10")

    images_filtered = images[mask]
    labels_filtered = labels[mask].clone()

    for old_val, new_val in mapping.items():
        labels_filtered[labels[mask] == old_val] = new_val

    return images_filtered, labels_filtered


def get_stl10_testloader(device, batch_size=256):
    """Loads STL-10, resizes to 32x32, and applies the 9-class alignment."""
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) # CIFAR norms
    ])
    stl10_test = torchvision.datasets.STL10(
        root=DATA_PATH, split='test', download=True, transform=transform)
    
    # Extract to tensors for filtering
    imgs = torch.stack([item[0] for item in stl10_test]).to(device)
    lbls = torch.tensor([item[1] for item in stl10_test], dtype=torch.long, device=device)
    
    imgs_aligned, lbls_aligned = align_to_9_classes(imgs, lbls, source='STL10')
    dataset = TensorDataset(imgs_aligned, lbls_aligned)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ── Train + evaluate one model ────────────────────────────────────────────────
def train_and_eval(label, net, images_train, labels_train, testloader, args):
    device = args.device
    net.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR,
                                momentum=EMN_MOM, weight_decay=EMN_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EMN_EPOCHS, eta_min=0.0)

    loader = DataLoader(
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
        train_acc = acc_sum / n
        train_loss = loss_sum / n

        net.eval()
        test_acc_sum = test_n = 0
        with torch.no_grad():
            for imgs, labs in testloader:
                imgs, labs = imgs.float().to(device), labs.long().to(device)
                out = net(imgs)
                test_acc_sum += (out.argmax(1) == labs).sum().item()
                test_n += labs.size(0)
        acc_test = test_acc_sum / test_n

        elapsed = int(time.time() - t0)
        print(f'  [{label}] ep {ep+1:3d}/{EMN_EPOCHS}  '
              f'loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
              f'STL10_acc={acc_test:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.5f}  elapsed={elapsed}s')

    print(f'  [{label}] FINAL  train_time={int(time.time()-t0)}s  '
          f'train_acc={train_acc:.4f}  STL10_test_acc={acc_test:.4f}')
    return acc_test


# ── Apply SW noise to full clean dataset ──────────────────────────────────────
def apply_sw_noise(images_clean, mean, std, device):
    raw = torch.load(EMN_NOISE_PT, map_location='cpu', weights_only=False)
    noise = raw if not isinstance(raw, dict) else raw['noise']
    noise = torch.as_tensor(noise, dtype=torch.float32)

    if noise.abs().max() > 1.5:
        noise = noise / 255.0

    if noise.ndim == 4 and noise.shape[-1] in (1, 3):
        noise = noise.permute(0, 3, 1, 2).contiguous()

    noise = noise.to(device)
    
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)

    images_raw = images_clean * std_t + mean_t
    images_poisoned = (torch.clamp(images_raw + noise, 0.0, 1.0) - mean_t) / std_t
    return images_poisoned


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class Args: pass
    args = Args()
    args.device = device

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n{"="*60}')
    print(f'  CIFAR10 -> STL10 Transfer Evaluation')
    print(f'  started {ts}')
    print(f'{"="*60}')

    # 1. Load standard CIFAR-10 data
    channel, im_size, num_classes, _, mean, std, dst_train, _, _ = \
        get_dataset(DATASET, DATA_PATH)
        
    images_all = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
    labels_all = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

    # 2. Build aligned STL-10 Testloader
    print("\nLoading and aligning STL-10 Test Set...")
    stl10_testloader = get_stl10_testloader(device, batch_size=EMN_BATCH)

    # 3. Setup the 3 Training Conditions (applying the 9-class filter to each)
    print("Preparing 9-Class Aligned Datasets...")
    
    # Clean
    # imgs_c, lbls_c = align_to_9_classes(images_all, labels_all, 'CIFAR10')
    
    # SW (Apply noise to 10-class first to keep index alignment, then filter)
    images_sw_full = apply_sw_noise(images_all, mean, std, device)
    imgs_sw, lbls_sw = align_to_9_classes(images_sw_full, labels_all, 'CIFAR10')
    
    # DGC (Load full 10-class poisoned data, then filter)
    dgc_data = torch.load(DGC_PT, map_location=device, weights_only=False)
    imgs_dgc, lbls_dgc = align_to_9_classes(dgc_data['images_poisoned'], dgc_data['labels'], 'CIFAR10')

    # Note: We pass num_classes=9 to build_emn_model now
    # RESULTS = {'Clean': (imgs_c, lbls_c), 'SW': (imgs_sw, lbls_sw), 'DGC': (imgs_dgc, lbls_dgc)}
    RESULTS = {'SW': (imgs_sw, lbls_sw), 'DGC': (imgs_dgc, lbls_dgc)}
    summary = {}

    for cond, (imgs, lbls) in RESULTS.items():
        print(f'\n{"─"*60}')
        print(f'  Condition: {cond} (Train size: {len(imgs)})')
        summary[cond] = {}
        
        for model_name in MODELS:
            accs = []
            for it in range(NUM_EVAL):
                net = build_emn_model(model_name, num_classes=9, channel=channel, im_size=im_size).to(device)
                acc = train_and_eval(
                    f'{cond}/{model_name}/run{it}', net,
                    copy.deepcopy(imgs), copy.deepcopy(lbls),
                    stl10_testloader, args)
                accs.append(acc)
            m, s = float(np.mean(accs)), float(np.std(accs))
            summary[cond][model_name] = (m, s)

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 62
    print(f'\n{"="*W}')
    print(f'  {"Condition":<10} {"Model":<20} {"STL10 mean acc":>16}  {"std":>8}')
    print(f'  {"─"*(W-2)}')
    for cond, cond_res in summary.items():
        for model_name, (m, s) in cond_res.items():
            print(f'  {cond:<10} {model_name:<20} {m:>16.4f}  {s:>8.4f}')
    print(f'{"="*W}')

if __name__ == '__main__':
    main()