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
# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET      = 'CIFAR10'
DATA_PATH    = '/home/mmoslem3/scratch/UE-DD/data/'

# Full dataset perturbation.pt
# EMN_NOISE_PT = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN/CIFAR10_SW.pt'
EMN_NOISE_PT = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-TUE/TUE_simclr_cifar10.pt'

# Full poisoned dataset .pt
DGC_PT       = '/home/mmoslem3/scratch/UE-DD/result-fianle/res_CIFAR10_iter150_bug8_lamexcess0.5.pt'

DGC_PT       = '/home/mmoslem3/scratch/UE-DD/partial/cifar10-avg.pt'

# ── Training hyper-parameters ─────────────────────────────────────────────────
NUM_EVAL   = 1
EMN_LR     = 0.1
EMN_MOM    = 0.9
EMN_WD     = 5e-4
EMN_EPOCHS = 10
EMN_BATCH  = 256
# MODELS     = ['ResNet18']
# LINF_BUDGET = 8.0 # Standard 8/255 budget for random noise



# ── Training hyper-parameters ─────────────────────────────────────────────────
NUM_EVAL   = 2
EMN_LR     = 0.01  # Lower LR for MLPs to prevent diverging
EMN_MOM    = 0.9
EMN_WD     = 5e-4
EMN_EPOCHS = 30
EMN_BATCH  = 256

MODELS     = ['ConvNet', 'MLP', 'TinyMLP' ] 


class LinearClassifier(nn.Module):
    def __init__(self, channel=3, im_size=(32, 32), num_classes=10):
        super(LinearClassifier, self).__init__()
        input_dim = channel * im_size[0] * im_size[1]  # 3072
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class TinyMLP(nn.Module):
    def __init__(self, channel=3, im_size=(32, 32), num_classes=10):
        super(TinyMLP, self).__init__()
        input_dim = channel * im_size[0] * im_size[1]  # 3072
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ── Simple MLP Architecture ───────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, channel=3, im_size=(32, 32), num_classes=10):
        super(SimpleMLP, self).__init__()
        input_dim = channel * im_size[0] * im_size[1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ── Train + evaluate one model ────────────────────────────────────────────────
def train_and_eval(label, net, images_train, labels_train, testloader, args, epochs=None):
    device = args.device
    net.to(device)
    n_epochs = epochs if epochs is not None else EMN_EPOCHS

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR,
                                momentum=EMN_MOM, weight_decay=EMN_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=0.0)

    loader = DataLoader(
        TensorDataset(images_train, labels_train),
        batch_size=EMN_BATCH, shuffle=True, num_workers=0)

    t0 = time.time()
    for ep in range(n_epochs):
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
        if (ep + 1) % max(1, n_epochs // 5) == 0:
            print(f'  [{label}] ep {ep+1:3d}/{n_epochs}  '
                f'loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
                f'CIFAR10_acc={acc_test:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.5f}  elapsed={elapsed}s')

    print(f'  [{label}] FINAL  train_time={int(time.time()-t0)}s  '
          f'train_acc={train_acc:.4f}  CIFAR10_test_acc={acc_test:.4f}')
    return acc_test

# ── Noise Functions ───────────────────────────────────────────────────────────
def apply_sw_noise(images_clean, mean, std, device):
    """Applies the pre-computed EMN samplewise noise."""
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
    print(f'  Capacity Test: MLP vs Unlearnable Examples')
    print(f'  started {ts}')
    print(f'{"="*60}')

    # 1. Load standard CIFAR-10 data
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = \
        get_dataset(DATASET, DATA_PATH)
        
    images_clean = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
    labels_clean = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

    print("Preparing Datasets...")
    
    # EMN Base
    imgs_emn = apply_sw_noise(images_clean, mean, std, device)
    
    # DGC Base
    dgc_data = torch.load(DGC_PT, map_location=device, weights_only=False)
    imgs_dgc = dgc_data['images_poisoned'].to(device)
    lbls_dgc = dgc_data['labels'].to(device)

    # Pre-computed Clean baseline (60 epochs, 2 runs each)
    CLEAN_RESULTS = {
        'MLP':     (0.5554, 0.0031),
        'TinyMLP': (0.5011, 0.0011),
        'Linear':  (0.3897, 0.0002),
    }

    RESULTS = {
        'EMN': (imgs_emn, labels_clean),
        # 'DGC': (imgs_dgc, lbls_dgc),
    }
    summary = {'Clean': CLEAN_RESULTS}

    for cond, (imgs, lbls) in RESULTS.items():
        print(f'\n{"─"*60}')
        print(f'  Condition: {cond} (Train size: {len(imgs)})')
        summary[cond] = {}

        for model_name in MODELS:
            accs = []
            for it in range(NUM_EVAL):
                if model_name == 'TinyMLP':
                    net = TinyMLP(channel=channel, im_size=im_size, num_classes=10).to(device)
                elif model_name == 'Linear':
                    net = LinearClassifier(channel=channel, im_size=im_size, num_classes=10).to(device)
                elif model_name == 'MLP':
                    net = SimpleMLP(channel=channel, im_size=im_size, num_classes=10).to(device)
                else:
                    net = build_emn_model(model_name, num_classes=10, channel=channel, im_size=im_size).to(device)

                acc = train_and_eval(
                    f'{cond}/{model_name}/run{it}', net,
                    copy.deepcopy(imgs), copy.deepcopy(lbls),
                    testloader, args)
                accs.append(acc)
            m, s = float(np.mean(accs)), float(np.std(accs))
            summary[cond][model_name] = (m, s)

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 60
    print(f'\n{"="*W}')
    print(f'  {"Condition":<10} {"Model":<16} {"CIFAR-10 acc":>14}  {"std":>8}')
    print(f'  {"─"*(W-2)}')
    for cond, cond_res in summary.items():
        for model_name, (m, s) in cond_res.items():
            print(f'  {cond:<10} {model_name:<16} {m:>14.4f}  {s:>8.4f}')
    print(f'{"="*W}')

if __name__ == '__main__':
    main()



# # ── Train + evaluate one model ────────────────────────────────────────────────
# def train_and_eval(label, net, images_train, labels_train, testloader, args):
#     device = args.device
#     net.to(device)
    
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR,
#                                 momentum=EMN_MOM, weight_decay=EMN_WD)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=EMN_EPOCHS, eta_min=0.0)

#     loader = DataLoader(
#         TensorDataset(images_train, labels_train),
#         batch_size=EMN_BATCH, shuffle=True, num_workers=0)

#     t0 = time.time()
#     for ep in range(EMN_EPOCHS):
#         net.train()
#         loss_sum = acc_sum = n = 0
#         for imgs, labs in loader:
#             imgs, labs = imgs.float().to(device), labs.long().to(device)
#             optimizer.zero_grad()
#             out = net(imgs)
#             loss = criterion(out, labs)
#             loss.backward()
#             optimizer.step()
#             with torch.no_grad():
#                 acc_sum  += (out.argmax(1) == labs).sum().item()
#                 loss_sum += loss.item() * labs.size(0)
#                 n        += labs.size(0)
#         scheduler.step()

#     # Final Evaluation on standard CIFAR-10 testloader
#     net.eval()
#     test_acc_sum = test_n = 0
#     with torch.no_grad():
#         for imgs, labs in testloader:
#             imgs, labs = imgs.float().to(device), labs.long().to(device)
#             out = net(imgs)
#             test_acc_sum += (out.argmax(1) == labs).sum().item()
#             test_n += labs.size(0)
            
#     acc_test = test_acc_sum / test_n
#     print(f'  [{label}] FINAL  train_time={int(time.time()-t0)}s  '
#           f'train_acc={acc_sum/n:.4f}  CIFAR10_test_acc={acc_test:.4f}')
#     return acc_test

# # ── Noise Functions ───────────────────────────────────────────────────────────
# def apply_random_noise(images_clean, mean, std, device, eps_255=8.0):
#     """Applies uniform random noise within an L-inf budget."""
#     budget = eps_255 / 255.0
#     # Uniform noise in [-budget, budget]
#     noise = (torch.rand_like(images_clean) * 2 - 1) * budget 
    
#     mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
#     std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)

#     images_raw = images_clean * std_t + mean_t
#     images_poisoned = (torch.clamp(images_raw + noise, 0.0, 1.0) - mean_t) / std_t
#     return images_poisoned

# def apply_sw_noise(images_clean, mean, std, device):
#     """Applies the pre-computed EMN samplewise noise."""
#     raw = torch.load(EMN_NOISE_PT, map_location='cpu', weights_only=False)
#     noise = raw if not isinstance(raw, dict) else raw['noise']
#     noise = torch.as_tensor(noise, dtype=torch.float32)

#     if noise.abs().max() > 1.5:
#         noise = noise / 255.0

#     if noise.ndim == 4 and noise.shape[-1] in (1, 3):
#         noise = noise.permute(0, 3, 1, 2).contiguous()

#     noise = noise.to(device)
    
#     mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
#     std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)

#     images_raw = images_clean * std_t + mean_t
#     images_poisoned = (torch.clamp(images_raw + noise, 0.0, 1.0) - mean_t) / std_t
#     return images_poisoned

# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     class Args: pass
#     args = Args()
#     args.device = device

#     ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print(f'\n{"="*60}')
#     print(f'  CIFAR-10 Poisoning Evaluation (Clean vs Random vs EMN vs DGC)')
#     print(f'  started {ts}')
#     print(f'{"="*60}')

#     # 1. Load standard CIFAR-10 data
#     channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = \
#         get_dataset(DATASET, DATA_PATH)
        
#     images_clean = torch.stack([dst_train[i][0] for i in range(len(dst_train))]).to(device)
#     labels_clean = torch.tensor([dst_train[i][1] for i in range(len(dst_train))], dtype=torch.long, device=device)

#     # 2. Setup the 4 Training Conditions
#     print("Preparing Datasets...")
    
#     # Random Noise
#     imgs_random = apply_random_noise(images_clean, mean, std, device, eps_255=LINF_BUDGET)
    
#     # EMN
#     imgs_sw = apply_sw_noise(images_clean, mean, std, device)
    
#     # DGC
#     dgc_data = torch.load(DGC_PT, map_location=device, weights_only=False)
#     imgs_dgc = dgc_data['images_poisoned'].to(device)

#     # EMN + Random  (random noise layered on top of EMN poison)
#     imgs_sw_random = apply_random_noise(imgs_sw, mean, std, device, eps_255=2*LINF_BUDGET)

#     # DGC + Random  (random noise layered on top of DGC poison)
#     imgs_dgc_random = apply_random_noise(imgs_dgc, mean, std, device, eps_255=2*LINF_BUDGET)

#     RESULTS = {
#         # 'Clean':      (images_clean, labels_clean),
#         # 'Random':     (imgs_random,  labels_clean),
#         # 'EMN':        (imgs_sw,      labels_clean),
#         'EMN+Random': (imgs_sw_random, labels_clean),
#         # 'DGC':        (imgs_dgc,     dgc_data['labels'].to(device)),
#         'DGC+Random': (imgs_dgc_random, dgc_data['labels'].to(device)),
#     }
#     summary = {}

#     for cond, (imgs, lbls) in RESULTS.items():
#         print(f'\n{"─"*60}')
#         print(f'  Condition: {cond} (Train size: {len(imgs)})')
#         summary[cond] = {}
        
#         for model_name in MODELS:
#             accs = []
#             for it in range(NUM_EVAL):
#                 net = build_emn_model(model_name, num_classes=10, channel=channel, im_size=im_size).to(device)
#                 acc = train_and_eval(
#                     f'{cond}/{model_name}/run{it}', net,
#                     copy.deepcopy(imgs), copy.deepcopy(lbls),
#                     testloader, args)
#                 accs.append(acc)
#             m, s = float(np.mean(accs)), float(np.std(accs))
#             summary[cond][model_name] = (m, s)

#     # ── Summary ───────────────────────────────────────────────────────────────
#     W = 66
#     print(f'\n{"="*W}')
#     print(f'  {"Condition":<10} {"Model":<20} {"CIFAR-10 mean acc":>18}  {"std":>8}')
#     print(f'  {"─"*(W-2)}')
#     for cond, cond_res in summary.items():
#         for model_name, (m, s) in cond_res.items():
#             print(f'  {cond:<10} {model_name:<20} {m:>18.4f}  {s:>8.4f}')
#     print(f'{"="*W}')

# if __name__ == '__main__':
#     main()