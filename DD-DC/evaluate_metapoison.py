# # """
# # Evaluate MetaPoison targeted attack -- distilled-poisoning threat model.

# # Victim training matches the paper exactly:
# #   - ConvNetBN (6-layer, batch norm)
# #   - 200 epochs, lr=0.1, decayed 10x at epochs 100 and 150
# #   - batch_size=125, SGD momentum=0.9, no weight decay, no data augmentation
# #   - "the same hyperparameters and architectures are used for victim evaluation"

# # Evaluation matches the paper:
# #   - Single target: report success/fail over N_seeds victim models
# #   - Multi-target: N targets x M seeds, report ASR = successes / total
# #   - Paper: "6 models are trained with different random seeds for each of
# #            10 target birds, totaling 60 victim models"

# # Conditions:
# #   (1) Dc only           -> clean baseline
# #   (2) Dc ∪ S            -> control
# #   (3) Dc ∪ Sp           -> attack
# # """
# # import argparse
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader, TensorDataset
# # import torchvision
# # import torchvision.transforms as T

# # from craft_sp_metapoison import (ConvNetBN, craft_sp_metapoison,
# #                                   pick_single_target_confident)


# # # ========================= data ========================= #
# # def get_cifar10(root='/home/mmoslem3/scratch/UE-DD/data'):
# #     mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
# #     tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
# #     train = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=tf)
# #     test  = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)
# #     return train, test


# # def to_tensors(ds):
# #     xs, ys = [], []
# #     for x, y in DataLoader(ds, batch_size=512, num_workers=0):
# #         xs.append(x); ys.append(y)
# #     return torch.cat(xs), torch.cat(ys)


# # # ========================= victim training (paper protocol) ========================= #
# # def train_victim(train_ds, epochs=200, lr=0.1, bs=125, device='cuda', seed=0):
# #     """
# #     Paper: "We train each victim to 200 epochs, decaying the learning rate
# #     by 10x at epochs 100 and 150."
# #     """
# #     torch.manual_seed(seed); np.random.seed(seed)
# #     loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
# #                         drop_last=False)
# #     net = ConvNetBN(num_classes=10).to(device)
# #     opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
# #     # Step decay at epoch 100 and 150
# #     sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 150], gamma=0.1)
# #     crit = nn.CrossEntropyLoss()

# #     for ep in range(epochs):
# #         net.train()
# #         for x, y in loader:
# #             x, y = x.to(device), y.to(device)
# #             opt.zero_grad(); crit(net(x), y).backward(); opt.step()
# #         sched.step()
# #         if (ep + 1) % 50 == 0:
# #             print(f"    victim epoch {ep+1}/{epochs}")

# #     return net


# # # ========================= eval ========================= #
# # @torch.no_grad()
# # def eval_attack(net, test_x, test_y, target_idx, targets_y_t, targets_y_adv,
# #                 device='cuda'):
# #     net.eval()
# #     # clean acc on non-target test points
# #     mask = torch.ones(len(test_x), dtype=torch.bool)
# #     mask[target_idx] = False
# #     non_tgt_x = test_x[mask]; non_tgt_y = test_y[mask]

# #     preds = []
# #     for i in range(0, len(non_tgt_x), 512):
# #         preds.append(net(non_tgt_x[i:i+512].to(device)).argmax(1).cpu())
# #     preds = torch.cat(preds)
# #     clean_acc = 100.0 * (preds == non_tgt_y).float().mean().item()

# #     # target predictions
# #     tgt_x = test_x[target_idx].to(device)
# #     tgt_pred = net(tgt_x).argmax(1).cpu()
# #     asr = 100.0 * (tgt_pred == targets_y_adv).float().mean().item()
# #     tmr = 100.0 * (tgt_pred != targets_y_t).float().mean().item()

# #     return clean_acc, asr, tmr, tgt_pred


# # # ========================= main ========================= #
# # def run(args):
# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

# #     train_ds, test_ds = get_cifar10()
# #     Dc_x, Dc_y = to_tensors(train_ds)
# #     test_x, test_y = to_tensors(test_ds)

# #     # load S
# #     S = torch.load(args.s_path, map_location='cpu', weights_only=False)
# #     S = S['data']
# #     S_x, S_y = S[0], S[1]
# #     print(f"[S] shape {S_x.shape}, per-class {torch.bincount(S_y).tolist()}")

# #     # pick targets (bird → dog by default)
# #     source_class = args.source_class if args.source_class is not None else 2  # bird
# #     target_class = args.target_class if args.target_class is not None else 5  # dog

# #     tgt_x, tgt_y, tgt_adv, tgt_idx = pick_single_target_confident(
# #         test_x, test_y,
# #         source_class=source_class,
# #         target_class=target_class,
# #         device=device, seed=args.target_seed)
# #     n_tgt = len(tgt_y)
# #     print(f"[targets] {n_tgt} picked")
# #     print(f"  true labels : {tgt_y.tolist()}")
# #     print(f"  adv  labels : {tgt_adv.tolist()}")

# #     # Split S: only the source class is optimized; the rest stays as static context
# #     src_mask = (S_y == source_class)
# #     S_static_x = S_x[~src_mask]   # non-bird synthetic images, static inner-training context
# #     S_static_y = S_y[~src_mask]
# #     Sp_init_x  = S_x[src_mask]    # bird synthetic images → will be optimized into Sp
# #     Sp_init_y  = S_y[src_mask]
# #     print(f"[split] static S: {len(S_static_x)} images, "
# #           f"Sp init (class {source_class}): {len(Sp_init_x)} images")

# #     # craft Sp — only the source-class images are optimized
# #     print("\n=== Crafting Sp via MetaPoison (bird-class only) ===")
# #     Sp_x, Sp_y = craft_sp_metapoison(
# #         S_static_x, S_static_y,
# #         tgt_x, tgt_adv, Dc_x, Dc_y,
# #         num_classes=10,
# #         outer_iters=args.craft_iters,
# #         outer_lr=args.outer_lr,
# #         K_unroll=args.K,
# #         inner_lr=args.inner_lr,
# #         inner_bs=args.inner_bs,
# #         num_ensemble=args.ensemble,
# #         device=device, verbose=True,
# #         Sp_init_x=Sp_init_x,
# #         Sp_init_y=Sp_init_y,
# #     )
# #     torch.save({'images': Sp_x, 'labels': Sp_y,
# #                 'target_idx': tgt_idx, 'target_y': tgt_y, 'target_adv': tgt_adv},
# #                args.sp_out)
# #     print(f"[Sp] saved -> {args.sp_out}  ({len(Sp_x)} poisoned images, class {source_class})")

# #     # conditions
# #     # Condition 2 adds only the static (non-bird) S — no poisoning, serves as control
# #     conditions = {
# #         '1_clean_Dc':    TensorDataset(Dc_x, Dc_y),
# #         '2_Dc_plus_S':   TensorDataset(torch.cat([Dc_x, S_static_x]),
# #                                         torch.cat([Dc_y, S_static_y])),
# #         '3_Dc_plus_Sp':  TensorDataset(torch.cat([Dc_x, Sp_x]), torch.cat([Dc_y, Sp_y])),
# #     }
# #     eps1_eff = len(Sp_x) / len(Dc_x)
# #     print(f"\neps1_eff = |Sp|/|Dc| = {len(Sp_x)}/{len(Dc_x)} = {eps1_eff:.5f}")

# #     # --- Paper: 6 seeds per target, report count out of total ---
# #     results = {}
# #     for name, ds in conditions.items():
# #         all_clean, all_asr, all_tmr = [], [], []
# #         all_preds = []
# #         print(f"\n--- {name} ---")
# #         for seed in range(args.seeds):
# #             print(f"  training victim seed={seed} ...")
# #             net = train_victim(ds, epochs=args.epochs, lr=args.victim_lr,
# #                                bs=args.victim_bs, device=device, seed=seed)
# #             ca, asr, tmr, preds = eval_attack(
# #                 net, test_x, test_y, tgt_idx, tgt_y, tgt_adv, device=device)
# #             print(f"  [{name}] seed={seed}  clean={ca:.2f}  "
# #                   f"ASR={asr:.1f}  TMR={tmr:.1f}  preds={preds.tolist()}")
# #             all_clean.append(ca); all_asr.append(asr); all_tmr.append(tmr)
# #             all_preds.append(preds)

# #         results[name] = {
# #             'clean': (np.mean(all_clean), np.std(all_clean)),
# #             'asr':   (np.mean(all_asr),   np.std(all_asr)),
# #             'tmr':   (np.mean(all_tmr),   np.std(all_tmr)),
# #             'preds': torch.stack(all_preds),  # [seeds, n_targets]
# #         }

# #     # report
# #     print("\n" + "=" * 76)
# #     print(f"RESULTS  eps1_eff={eps1_eff:.5f}  N_targets={n_tgt}  "
# #           f"seeds={args.seeds}  total_evals={n_tgt * args.seeds}")
# #     print("=" * 76)
# #     print(f"  {'condition':20s} {'clean acc':>16s}  {'ASR':>12s}  {'TMR':>12s}")
# #     for name, r in results.items():
# #         print(f"  {name:20s} "
# #               f"{r['clean'][0]:6.2f} +/- {r['clean'][1]:4.2f}  "
# #               f"{r['asr'][0]:5.1f} +/- {r['asr'][1]:4.1f}  "
# #               f"{r['tmr'][0]:5.1f} +/- {r['tmr'][1]:4.1f}")

# #     # Paper-style count: "Number of times out of N the target is classified as class X"
# #     atk_preds = results['3_Dc_plus_Sp']['preds']  # [seeds, n_targets]
# #     total_evals = atk_preds.numel()
# #     for c in range(10):
# #         cnt = (atk_preds == c).sum().item()
# #         if cnt > 0:
# #             label = ['airplane','automobile','bird','cat','deer',
# #                      'dog','frog','horse','ship','truck'][c]
# #             marker = ""
# #             if c in tgt_adv.tolist():
# #                 marker = " <-- adversarial class"
# #             if c in tgt_y.tolist():
# #                 marker += " <-- true class"
# #             print(f"  predicted as {label:12s}: {cnt:3d}/{total_evals}{marker}")

# #     base_asr = results['1_clean_Dc']['asr'][0]
# #     ctrl_asr = results['2_Dc_plus_S']['asr'][0]
# #     atk_asr  = results['3_Dc_plus_Sp']['asr'][0]
# #     print(f"\n  ASR lift (attack - clean):   {atk_asr - base_asr:+.1f}")
# #     print(f"  ASR lift (attack - control): {atk_asr - ctrl_asr:+.1f}")


# # if __name__ == '__main__':
# #     p = argparse.ArgumentParser()
# #     p.add_argument('--s_path',   type=str, default= '/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DSA_CIFAR10_ConvNet_50ipc.pt')
# #     p.add_argument('--sp_out',   type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DSA_CIFAR10_ConvNet_50ipc_Sp_meta.pt')
# #     # craft (paper defaults)
# #     p.add_argument('--craft_iters', type=int, default=60)
# #     p.add_argument('--ensemble',    type=int, default=24)
# #     p.add_argument('--K',           type=int, default=2)
# #     p.add_argument('--inner_lr',    type=float, default=0.1)
# #     p.add_argument('--inner_bs',    type=int, default=125)
# #     p.add_argument('--outer_lr',    type=float, default=0.5)
# #     # target
# #     p.add_argument('--n_targets',    type=int, default=10)
# #     p.add_argument('--target_seed',  type=int, default=0)
# #     p.add_argument('--source_class', type=int, default=None)
# #     p.add_argument('--target_class', type=int, default=None)
# #     # victim (paper defaults)
# #     p.add_argument('--epochs',    type=int, default=30)
# #     p.add_argument('--victim_lr', type=float, default=0.01)
# #     p.add_argument('--victim_bs', type=int, default=125)
# #     p.add_argument('--seeds',     type=int, default=1)
# #     args = p.parse_args()
# #     run(args)


# # # if __name__ == '__main__':
# # #     p = argparse.ArgumentParser()
# # #     p.add_argument('--s_path',   type=str, default= '/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DSA_CIFAR10_ConvNet_50ipc.pt')
# # #     p.add_argument('--sp_out',   type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DSA_CIFAR10_ConvNet_50ipc_Sp_meta2.pt')
# # #     # craft
# # #     p.add_argument('--craft_iters', type=int, default=300)
# # #     p.add_argument('--ensemble',    type=int, default=8)
# # #     p.add_argument('--K',           type=int, default=5)
# # #     p.add_argument('--inner_lr',    type=float, default=0.01)
# # #     p.add_argument('--outer_lr',    type=float, default=0.1)
# # #     p.add_argument('--n_targets',   type=int, default=1)
# # #     p.add_argument('--target_seed', type=int, default=0)
# # #     # victim training
# # #     p.add_argument('--arch',   type=str, default='convnet', choices=['convnet','resnet18'])
# # #     p.add_argument('--epochs', type=int, default=30)
# # #     p.add_argument('--lr',     type=float, default=0.01)
# # #     p.add_argument('--bs',     type=int, default=256)
# # #     p.add_argument('--seeds',  type=int, default=3)
# # #     args = p.parse_args()
# # #     run(args)

# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import torchvision
# import torchvision.transforms as T
# import numpy as np

# # ========================= 1. Model (No inplace ops!) =========================
# class ConvNetBN(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         def block(c_in, c_out):
#             return nn.Sequential(
#                 nn.Conv2d(c_in, c_out, 3, padding=1),
#                 nn.BatchNorm2d(c_out),
#                 nn.ReLU(), 
#                 nn.Conv2d(c_out, c_out, 3, padding=1),
#                 nn.BatchNorm2d(c_out),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),
#             )
#         self.block1 = block(3, 32)
#         self.block2 = block(32, 64)
#         self.block3 = block(64, 128)
#         self.classifier = nn.Linear(128 * 4 * 4, num_classes)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = x.flatten(1)
#         return self.classifier(x)

# # ========================= 2. Carlini & Wagner f6 Loss =========================
# def cw_margin_loss(logits, target_class, kappa=0.0):
#     """
#     Implements Eq 1 from the MetaPoison paper: f6 adversarial loss.
#     """
#     target_logit = logits[:, target_class]
#     mask = torch.ones_like(logits, dtype=torch.bool)
#     mask[:, target_class] = False
#     other_logits = logits[mask].view(logits.shape[0], -1)
#     max_other_logit = torch.max(other_logits, dim=1)[0]
#     loss = torch.clamp(max_other_logit - target_logit + kappa, min=0.0)
#     return loss.mean()

# # # ========================= 3. Native Bilevel Optimization =========================
# # def craft_sp_bilevel(S_static_x, S_static_y, Sp_init_x, Sp_init_y, 
# #                      target_x, target_y_adv, 
# #                      outer_iters=100, outer_lr=0.1, inner_lr=0.05, K_unroll=2):
    
# #     device = target_x.device
# #     net = ConvNetBN().to(device)
# #     net.eval() # MUST be in eval mode
    
# #     Sp_images = Sp_init_x.clone().detach().to(device).requires_grad_(True)
# #     outer_opt = torch.optim.Adam([Sp_images], lr=outer_lr)
    
# #     print(f"\n--- Solving Bilevel Optimization for Sp (K={K_unroll}) ---")
# #     for it in range(outer_iters):
# #         outer_opt.zero_grad()
        
# #         net.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
# #         fast_weights = {name: param.clone() for name, param in net.named_parameters()}
        
# #         combined_x = torch.cat([S_static_x, Sp_images], dim=0)
# #         combined_y = torch.cat([S_static_y, Sp_init_y], dim=0)
        
# #         for _ in range(K_unroll):
# #             logits_in = torch.func.functional_call(net, fast_weights, combined_x)
# #             loss_in = F.cross_entropy(logits_in, combined_y)
# #             grads = torch.autograd.grad(loss_in, fast_weights.values(), create_graph=True)
# #             fast_weights = {name: w - inner_lr * g for (name, w), g in zip(fast_weights.items(), grads)}
            
# #         logits_out = torch.func.functional_call(net, fast_weights, target_x)
# #         loss_out = cw_margin_loss(logits_out, target_y_adv.item(), kappa=0.5)
# #         loss_out.backward()
# #         outer_opt.step()
        
# #         with torch.no_grad():
# #             Sp_images.clamp_(-2.5, 2.5)
            
# #         if it % 10 == 0 or it == outer_iters - 1:
# #             tgt_logit = logits_out[0, target_y_adv.item()].item()
# #             true_logit = logits_out[0, Sp_init_y[0].item()].item()
# #             print(f"Iter {it:3d} | Outer C&W Loss: {loss_out.item():.4f} | Adv Logit: {tgt_logit:.2f} | True Logit: {true_logit:.2f}")

# #     return Sp_images.detach()

# def get_warm_checkpoints(S_x, S_y, device, num_epochs=30):
#     """
#     Pre-trains a surrogate network on the static distilled data to generate 
#     'warm' intermediate checkpoints. This gives the inner loop actual features
#     to manipulate instead of starting from random noise.
#     """
#     print("\n--- Generating Warm Checkpoints on S ---")
#     net = ConvNetBN().to(device)
#     opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    
#     ds = TensorDataset(S_x, S_y)
#     loader = DataLoader(ds, batch_size=125, shuffle=True)
    
#     checkpoints = []
#     net.train()
#     for ep in range(1, num_epochs + 1):
#         for x, y in loader:
#             opt.zero_grad()
#             F.cross_entropy(net(x), y).backward()
#             opt.step()
        
#         # Save checkpoints at intermediate stages of learning
#         if ep % 10 == 0:
#             checkpoints.append({k: v.clone() for k, v in net.state_dict().items()})
#             print(f"Saved warm checkpoint at epoch {ep}")
            
#     return checkpoints


# def craft_sp_bilevel(S_static_x, S_static_y, Sp_init_x, Sp_init_y, 
#                      target_x, target_y_adv, 
#                      outer_iters=100, outer_lr=0.1, inner_lr=0.05, K_unroll=2):
    
#     device = target_x.device
    
#     # 1. Generate warm checkpoints using the initial S data
#     combined_init_x = torch.cat([S_static_x, Sp_init_x])
#     combined_init_y = torch.cat([S_static_y, Sp_init_y])
#     checkpoints = get_warm_checkpoints(combined_init_x, combined_init_y, device)
    
#     net = ConvNetBN().to(device)
#     net.eval() # MUST be in eval mode for stable unrolling
    
#     Sp_images = Sp_init_x.clone().detach().to(device).requires_grad_(True)
#     outer_opt = torch.optim.Adam([Sp_images], lr=outer_lr)
    
#     print(f"\n--- Solving Bilevel Optimization for Sp (K={K_unroll}) ---")
#     for it in range(outer_iters):
#         outer_opt.zero_grad()
        
#         # 2. Load a warm checkpoint instead of random weights
#         ckpt = checkpoints[it % len(checkpoints)]
#         net.load_state_dict(ckpt)
#         fast_weights = {name: param.clone() for name, param in net.named_parameters()}
        
#         combined_x = torch.cat([S_static_x, Sp_images], dim=0)
#         combined_y = torch.cat([S_static_y, Sp_init_y], dim=0)
        
#         # 3. Inner Loop Unroll
#         for _ in range(K_unroll):
#             logits_in = torch.func.functional_call(net, fast_weights, combined_x)
#             loss_in = F.cross_entropy(logits_in, combined_y)
#             grads = torch.autograd.grad(loss_in, fast_weights.values(), create_graph=True)
#             fast_weights = {name: w - inner_lr * g for (name, w), g in zip(fast_weights.items(), grads)}
            
#         # 4. Outer Loop C&W Margin Loss
#         logits_out = torch.func.functional_call(net, fast_weights, target_x)
#         loss_out = cw_margin_loss(logits_out, target_y_adv.item(), kappa=0.5)
#         loss_out.backward()
#         outer_opt.step()
        
#         with torch.no_grad():
#             Sp_images.clamp_(-2.5, 2.5)
            
#         if it % 10 == 0 or it == outer_iters - 1:
#             tgt_logit = logits_out[0, target_y_adv.item()].item()
#             true_logit = logits_out[0, Sp_init_y[0].item()].item()
#             print(f"Iter {it:3d} | Outer Loss: {loss_out.item():.4f} | Adv Logit: {tgt_logit:.2f} | True Logit: {true_logit:.2f}")

#     return Sp_images.detach()


# # ========================= 4. Main Execution & Eval =========================
# def run(args):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 1. Load Clean Data (Dc) from specific path
#     print(f"Loading CIFAR-10 from {args.data_dir}...")
#     tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
#     train_ds = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tf)
#     test_ds  = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
    
#     # Convert full train_ds to tensors
#     xs, ys = [], []
#     for x, y in DataLoader(train_ds, batch_size=512, num_workers=0):
#         xs.append(x); ys.append(y)
#     Dc_x, Dc_y = torch.cat(xs), torch.cat(ys)
    
#     test_xs, test_ys = [], []
#     for x, y in DataLoader(test_ds, batch_size=512, num_workers=0):
#         test_xs.append(x); test_ys.append(y)
#     test_x, test_y = torch.cat(test_xs).to(device), torch.cat(test_ys).to(device)

#     # 2. Load Distilled Data S
#     print(f"Loading distilled S from {args.s_path}...")
#     S = torch.load(args.s_path, map_location='cpu', weights_only=False)
#     S_x, S_y = S['data'][0], S['data'][1]
#     print(f"[S] shape {S_x.shape}, per-class {torch.bincount(S_y).tolist()}")
    
#     # 3. Setup Target
#     source_class = args.source_class
#     target_class = args.target_class
    
#     # Pick a specific test target of the source class
#     src_mask_test = (test_y == source_class)
#     src_indices = src_mask_test.nonzero(as_tuple=True)[0]
#     target_idx = src_indices[args.target_seed] # Just picking one deterministically 
#     target_x = test_x[target_idx].unsqueeze(0)
#     target_y_adv = torch.tensor([target_class]).to(device)
    
#     print(f"[Target] Selected test index {target_idx.item()}: true class {source_class} -> adv class {target_class}")

#     # Split S into static context and the subset to be poisoned
#     src_mask_S = (S_y == source_class)
#     S_static_x = S_x[~src_mask_S].to(device)
#     S_static_y = S_y[~src_mask_S].to(device)
#     Sp_init_x  = S_x[src_mask_S].to(device)
#     Sp_init_y  = S_y[src_mask_S].to(device)
    
#     # 4. SOLVE FOR Sp
#     Sp_optimized = craft_sp_bilevel(
#         S_static_x, S_static_y, Sp_init_x, Sp_init_y, 
#         target_x, target_y_adv,
#         outer_iters=args.craft_iters, outer_lr=args.outer_lr, inner_lr=args.inner_lr, K_unroll=args.K
#     )
    
#     # 5. EVALUATE ON Dc U Sp
#     print("\n--- Evaluating Victim on Dc U Sp ---")
#     combined_train_x = torch.cat([Dc_x.to(device), Sp_optimized])
#     combined_train_y = torch.cat([Dc_y.to(device), Sp_init_y])
    
#     ds = TensorDataset(combined_train_x, combined_train_y)
#     loader = DataLoader(ds, batch_size=args.victim_bs, shuffle=True)
    
#     victim = ConvNetBN().to(device)
#     opt = torch.optim.SGD(victim.parameters(), lr=args.victim_lr, momentum=0.9, weight_decay=5e-4)
#     sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
#     for epoch in range(args.epochs):
#         victim.train()
#         for x, y in loader:
#             opt.zero_grad()
#             F.cross_entropy(victim(x), y).backward()
#             opt.step()
#         sched.step()
        
#         if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
#             victim.eval()
#             with torch.no_grad():
#                 # Check ASR on specific target
#                 pred_target = victim(target_x).argmax(1).item()
#                 is_success = (pred_target == target_class)
                
#                 # Check general clean accuracy on non-target test data
#                 mask = torch.ones(len(test_x), dtype=torch.bool)
#                 mask[target_idx] = False
#                 preds = victim(test_x[mask]).argmax(1)
#                 clean_acc = 100.0 * (preds == test_y[mask]).float().mean().item()
                
#                 print(f"Epoch {epoch+1:2d}/{args.epochs} | Clean Acc: {clean_acc:.2f}% | Target Pred: {pred_target} (ASR: {is_success})")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument('--s_path', type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DSA_CIFAR10_ConvNet_50ipc.pt')
#     p.add_argument('--data_dir', type=str, default='/home/mmoslem3/scratch/UE-DD/data')
    
#     p.add_argument('--craft_iters', type=int, default=100)
#     p.add_argument('--K', type=int, default=2)
#     p.add_argument('--inner_lr', type=float, default=0.1)
#     p.add_argument('--outer_lr', type=float, default=0.05)
    
#     p.add_argument('--source_class', type=int, default=2)
#     p.add_argument('--target_class', type=int, default=5)
#     p.add_argument('--target_seed', type=int, default=0)
    
#     p.add_argument('--epochs', type=int, default=40)
#     p.add_argument('--victim_lr', type=float, default=0.05)
#     p.add_argument('--victim_bs', type=int, default=128)
    
#     args = p.parse_args()
#     run(args)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
import numpy as np
import os

# ========================= 1. Model Architecture ========================= #
class ConvNetBN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # inplace=True MUST be omitted so PyTorch can compute second-order gradients
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(), 
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.block1 = block(3, 32)
        self.block2 = block(32, 64)
        self.block3 = block(64, 128)
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.classifier(x)


# ========================= 2. Carlini & Wagner f6 Loss ========================= #
def cw_margin_loss(logits, target_class, kappa=0.0):
    """
    Implements Eq 1 from the MetaPoison paper: f6 adversarial loss.
    """
    target_logit = logits[:, target_class]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, target_class] = False
    
    other_logits = logits[mask].view(logits.shape[0], -1)
    max_other_logit = torch.max(other_logits, dim=1)[0]
    
    loss = torch.clamp(max_other_logit - target_logit + kappa, min=0.0)
    return loss.mean()


# ========================= 3. Surrogate Generation ========================= #
def get_warm_checkpoints_on_S(S_x, S_y, device, num_epochs=40):
    """
    Trains the surrogate strictly on the distilled dataset S.
    Augmentation is strictly required here to prevent the network from overfitting 
    the highly compressed S and becoming 'blind' to real CIFAR-10 target images.
    """
    print(f"\n--- Generating Warm Checkpoints strictly on S ({len(S_x)} images) ---")
    net = ConvNetBN().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    
    aug = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ])
    
    checkpoints = []
    net.train()
    for ep in range(1, num_epochs + 1):
        S_x_aug = aug(S_x)
        ds = TensorDataset(S_x_aug, S_y)
        loader = DataLoader(ds, batch_size=125, shuffle=True)
        
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()
            
        if ep % 10 == 0:
            checkpoints.append({k: v.clone() for k, v in net.state_dict().items()})
            print(f"  Saved warm checkpoint at epoch {ep}")
            
    return checkpoints


# ========================= 4. Bilevel Optimization Solver ========================= #
def craft_sp_bilevel(S_x, S_y, Sp_init_x, Sp_init_y, 
                     target_x, target_y_adv, 
                     outer_iters=100, outer_lr=0.1, inner_lr=0.05, K_unroll=2):
    
    device = target_x.device
    
    # 1. Warm start strictly on S
    checkpoints = get_warm_checkpoints_on_S(S_x, S_y, device)
    
    net = ConvNetBN().to(device)
    net.eval() # Must remain in eval mode for stable unrolling over BatchNorm
    
    # 2. Setup Sp parameter
    Sp_images = Sp_init_x.clone().detach().to(device).requires_grad_(True)
    outer_opt = torch.optim.Adam([Sp_images], lr=outer_lr)
    
    print(f"\n--- Solving Bilevel Optimization for Sp on S U Sp (K={K_unroll}) ---")
    for it in range(outer_iters):
        outer_opt.zero_grad()
        
        # Load cyclic warm checkpoint
        ckpt = checkpoints[it % len(checkpoints)]
        net.load_state_dict(ckpt)
        fast_weights = {name: param.clone() for name, param in net.named_parameters()}
        
        # Inner loop trains on exactly S U Sp
        combined_x = torch.cat([S_x, Sp_images], dim=0)
        combined_y = torch.cat([S_y, Sp_init_y], dim=0)
        
        # 3. Inner Loop Unroll
        for _ in range(K_unroll):
            logits_in = torch.func.functional_call(net, fast_weights, combined_x)
            loss_in = F.cross_entropy(logits_in, combined_y)
            grads = torch.autograd.grad(loss_in, fast_weights.values(), create_graph=True)
            fast_weights = {name: w - inner_lr * g for (name, w), g in zip(fast_weights.items(), grads)}
            
        # 4. Outer Loop Meta-Loss
        logits_out = torch.func.functional_call(net, fast_weights, target_x)
        loss_out = cw_margin_loss(logits_out, target_y_adv.item(), kappa=0.5)
        loss_out.backward()
        
        outer_opt.step()
        
        # Clamp to realistic normalized CIFAR pixel bounds (~ [-2.5, 2.5])
        with torch.no_grad():
            Sp_images.clamp_(-2.5, 2.5)
            
        if it % 10 == 0 or it == outer_iters - 1:
            tgt_logit = logits_out[0, target_y_adv.item()].item()
            true_logit = logits_out[0, Sp_init_y[0].item()].item()
            print(f"Iter {it:3d} | Outer Loss: {loss_out.item():.4f} | Adv Logit: {tgt_logit:.2f} | True Logit: {true_logit:.2f}")

    return Sp_images.detach()


# ========================= 5. Main Execution & Eval Pipeline ========================= #
def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Clean Data (Dc)
    print(f"Loading CIFAR-10 from {args.data_dir}...")
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_ds = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tf)
    test_ds  = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
    
    xs, ys = [], []
    for x, y in DataLoader(train_ds, batch_size=512, num_workers=0):
        xs.append(x); ys.append(y)
    Dc_x, Dc_y = torch.cat(xs), torch.cat(ys)
    
    test_xs, test_ys = [], []
    for x, y in DataLoader(test_ds, batch_size=512, num_workers=0):
        test_xs.append(x); test_ys.append(y)
    test_x, test_y = torch.cat(test_xs).to(device), torch.cat(test_ys).to(device)

    # 2. Load Distilled Data S
    print(f"Loading distilled S from {args.s_path}...")
    S = torch.load(args.s_path, map_location='cpu', weights_only=False)
    S_x, S_y = S['data'][0], S['data'][1]
    print(f"[S] shape {S_x.shape}, per-class {torch.bincount(S_y).tolist()}")
    
    # 3. Setup Target
    source_class = args.source_class
    target_class = args.target_class
    
    src_mask_test = (test_y == source_class)
    src_indices = src_mask_test.nonzero(as_tuple=True)[0]
    target_idx = src_indices[args.target_seed] 
    target_x = test_x[target_idx].unsqueeze(0)
    target_y_adv = torch.tensor([target_class]).to(device)
    
    print(f"[Target] Selected test index {target_idx.item()}: true class {source_class} -> adv class {target_class}")

    # Initialize Sp as a direct copy of the source class from S
    src_mask_S = (S_y == source_class)
    Sp_init_x  = S_x[src_mask_S].to(device)
    Sp_init_y  = S_y[src_mask_S].to(device)
    
    # 4. SOLVE FOR Sp
    Sp_optimized = craft_sp_bilevel(
        S_x.to(device), S_y.to(device), Sp_init_x, Sp_init_y, 
        target_x, target_y_adv,
        outer_iters=args.craft_iters, outer_lr=args.outer_lr, inner_lr=args.inner_lr, K_unroll=args.K
    )
    
    # Optional: Save Sp
    if args.sp_out:
        torch.save({'images': Sp_optimized.cpu(), 'labels': Sp_init_y.cpu()}, args.sp_out)
        print(f"Saved optimized Sp to {args.sp_out}")
    
    # 5. EVALUATE ON Dc U Sp
    print("\n--- Evaluating Victim on Dc U Sp ---")
    combined_train_x = torch.cat([Dc_x.to(device), Sp_optimized])
    combined_train_y = torch.cat([Dc_y.to(device), Sp_init_y])
    
    ds = TensorDataset(combined_train_x, combined_train_y)
    loader = DataLoader(ds, batch_size=args.victim_bs, shuffle=True)
    
    victim = ConvNetBN().to(device)
    # Using stable SGD setup for victim evaluation
    opt = torch.optim.SGD(victim.parameters(), lr=args.victim_lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        victim.train()
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(victim(x), y).backward()
            opt.step()
        sched.step()
        
        if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
            victim.eval()
            with torch.no_grad():
                pred_target = victim(target_x).argmax(1).item()
                is_success = (pred_target == target_class)
                
                mask = torch.ones(len(test_x), dtype=torch.bool)
                mask[target_idx] = False
                preds = victim(test_x[mask]).argmax(1)
                clean_acc = 100.0 * (preds == test_y[mask]).float().mean().item()
                
                print(f"Epoch {epoch+1:2d}/{args.epochs} | Clean Acc: {clean_acc:.2f}% | Target Pred: {pred_target} (ASR: {is_success})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument('--s_path', type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DSA_CIFAR10_ConvNet_50ipc.pt')
    p.add_argument('--data_dir', type=str, default='/home/mmoslem3/scratch/UE-DD/data')
    p.add_argument('--sp_out', type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result-dc/res_DSA_CIFAR10_ConvNet_50ipc_Sp_meta.pt')
    
    # Crafting Hyperparameters
    p.add_argument('--craft_iters', type=int, default=200)
    p.add_argument('--K', type=int, default=20)
    p.add_argument('--inner_lr', type=float, default=0.1)
    p.add_argument('--outer_lr', type=float, default=0.1)
    
    # Target Setup (2=Bird, 5=Dog)
    p.add_argument('--source_class', type=int, default=2)
    p.add_argument('--target_class', type=int, default=5)
    p.add_argument('--target_seed', type=int, default=0)
    
    # Victim Training Params
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--victim_lr', type=float, default=0.05)
    p.add_argument('--victim_bs', type=int, default=128)
    
    args = p.parse_args()
    run(args)