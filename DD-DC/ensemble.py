import os
import copy
import numpy as np
import torch
import torch.nn as nn
import time
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from util import get_dataset, build_emn_model

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET      = 'CIFAR10'
DATA_PATH    = '/home/mmoslem3/scratch/UE-DD/data/'

# Full dataset perturbation.pt
# EMN_NOISE_PT = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN/CIFAR10_SW.pt'
EMN_NOISE_PT = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-GUE/gue_cifar10_SW.pt'
# Full poisoned dataset .pt
DGC_PT       = '/home/mmoslem3/scratch/UE-DD/result-fianle/res_CIFAR10_iter150_bug8_lamexcess0.5.pt'
DGC_PT       = '/home/mmoslem3/scratch/UE-DD/partial/cifar10-avg.pt'


# ── Training hyper-parameters ─────────────────────────────────────────────────
NUM_EVAL   = 2          # How many full ensembles to train for mean/std
ENSEMBLE_SIZE = 5       # How many MLPs inside each ensemble
EMN_LR     = 0.01       
EMN_MOM    = 0.9
EMN_WD     = 5e-4
EMN_EPOCHS = 10
EMN_BATCH  = 256

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

# ── Train + evaluate ENSEMBLE ─────────────────────────────────────────────────
def train_and_eval_ensemble(label, images_train, labels_train, testloader, args):
    device = args.device
    ensemble_models = []
    
    t0_ensemble = time.time()
    
    # Train N independent MLPs
    for m_idx in range(ENSEMBLE_SIZE):
        # Different initialization each time
        # net = SimpleMLP(channel=3, im_size=(32, 32), num_classes=10).to(device)


        net = build_emn_model('ConvNet', num_classes=10, channel=3, im_size=(32,32)).to(device)


        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=EMN_LR,
                                    momentum=EMN_MOM, weight_decay=EMN_WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EMN_EPOCHS, eta_min=0.0)

        loader = DataLoader(
            TensorDataset(images_train, labels_train),
            batch_size=EMN_BATCH, shuffle=True, num_workers=0)

        for ep in range(EMN_EPOCHS):
            net.train()
            for imgs, labs in loader:
                imgs, labs = imgs.float().to(device), labs.long().to(device)
                optimizer.zero_grad()
                out = net(imgs)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        net.eval()
        ensemble_models.append(net)

    # Final Evaluation: Average the logits across the ensemble
    test_acc_sum = test_n = 0
    with torch.no_grad():
        for imgs, labs in testloader:
            imgs, labs = imgs.float().to(device), labs.long().to(device)
            
            # Get predictions from all models: shape (Ensemble_Size, Batch, Classes)
            outputs = torch.stack([model(imgs) for model in ensemble_models])
            
            # Average the logits across dim=0 (the ensemble models)
            avg_output = outputs.mean(dim=0)
            
            test_acc_sum += (avg_output.argmax(1) == labs).sum().item()
            test_n += labs.size(0)
            
    acc_test = test_acc_sum / test_n
    print(f'  [{label}] FINAL  train_time={int(time.time()-t0_ensemble)}s  '
          f'CIFAR10_test_acc={acc_test:.4f}')
    return acc_test

# ── Noise Functions ───────────────────────────────────────────────────────────
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
    print(f'  Capacity Test: MLP Deep Ensemble ({ENSEMBLE_SIZE} models) vs UE')
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

    RESULTS = {
        # 'Clean': (images_clean, labels_clean), 
        'GUE':   (imgs_emn, labels_clean),
        # 'DGC':   (imgs_dgc, lbls_dgc),
    }
    summary = {}

    for cond, (imgs, lbls) in RESULTS.items():
        print(f'\n{"─"*60}')
        print(f'  Condition: {cond} (Train size: {len(imgs)})')
        summary[cond] = {}
        
        accs = []
        for it in range(NUM_EVAL):
            acc = train_and_eval_ensemble(
                f'{cond}/MLP_Ensemble/run{it}', 
                copy.deepcopy(imgs), copy.deepcopy(lbls),
                testloader, args)
            accs.append(acc)
        m, s = float(np.mean(accs)), float(np.std(accs))
        summary[cond]['MLP_Ensemble'] = (m, s)

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