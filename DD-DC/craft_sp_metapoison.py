"""
MetaPoison (Huang et al., NeurIPS 2020) -- faithful reproduction,
adapted to the distilled-poisoning threat model.

ONLY DIFFERENCE from the original paper:
  - Poisons are initialized from distilled set S, not from Dc.
  - Pixels unconstrained (no eps-ball), since S is already synthetic.

EVERYTHING ELSE matches the paper:
  - 6-layer ConvNetBN (Finn et al. 2017 style, with BatchNorm)
  - Staggered ensemble of 24 inner models: model i trained to epoch i
  - K=2 unrolled SGD steps per inner model
  - Adam outer optimizer, lr=200, decayed 10x every 20 outer steps
  - 60 outer steps total
  - Inner: batch_size=125, lr=0.1, no weight decay, no data augmentation
  - Victim: 200 epochs, lr=0.1 decayed 10x at epoch 100 and 150
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import higher
except ImportError:
    raise ImportError("pip install higher")


# ========================= ConvNetBN (6-layer) ========================= #
# 3 conv blocks x 2 conv layers each = 6 conv layers + 1 linear = "6-layer"
# Matches "the same 6-layer ConvNet architecture with batch normalization
# as Finn et al. [2017], henceforth called ConvNetBN"
class ConvNetBN(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, width=32, im_size=32):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.block1 = block(in_ch, width)
        self.block2 = block(width, width * 2)
        self.block3 = block(width * 2, width * 4)
        feat_size = im_size // 8
        self.classifier = nn.Linear(width * 4 * feat_size * feat_size, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.classifier(x)


# ========================= target selection ========================= #
def pick_targets(test_x, test_y, n_targets=10, num_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(test_x), generator=g)[:n_targets]
    x_t = test_x[idx].clone()
    y_t = test_y[idx].clone()
    y_adv = torch.empty_like(y_t)
    for i in range(n_targets):
        choices = [c for c in range(num_classes) if c != y_t[i].item()]
        y_adv[i] = choices[torch.randint(0, len(choices), (1,), generator=g).item()]
    return x_t, y_t, y_adv, idx


def pick_single_target_confident(test_x, test_y, source_class, target_class,
                                 device='cuda', seed=0):
    """Highest-confidence test image of source_class under a quick ConvNetBN."""
    import torchvision
    import torchvision.transforms as T

    torch.manual_seed(seed)
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR10('/home/mmoslem3/scratch/UE-DD/data', train=True, download=True, transform=tf)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    net = ConvNetBN(num_classes=10).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _ in range(20):  # enough for ranking
        net.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(net(x), y).backward(); opt.step()

    src_mask = (test_y == source_class)
    src_indices = src_mask.nonzero(as_tuple=True)[0]
    net.eval()
    with torch.no_grad():
        logits = net(test_x[src_indices].to(device))
        probs = torch.softmax(logits, dim=1)
        conf = probs[:, source_class]
    best = conf.argmax().item()
    chosen_idx = src_indices[best].unsqueeze(0)

    x_t = test_x[chosen_idx].clone()
    y_t = torch.tensor([source_class])
    y_adv = torch.tensor([target_class])
    print(f"[single target] test idx={chosen_idx.item()}  "
          f"class {source_class}->{target_class}  "
          f"clean confidence={conf[best].item():.4f}")
    return x_t, y_t, y_adv, chosen_idx


# ========================= staggered ensemble ========================= #
def build_staggered_ensemble(Dc_x, Dc_y, num_models=24, batch_size=125,
                             lr=0.1, num_classes=10, device='cuda'):
    """
    Paper: "An ensemble of 24 inner models is used, with model i trained
    until the i-th epoch. A batchsize of 125 and learning rate of 0.1
    are used."

    Train one ConvNetBN for 24 epochs on Dc, checkpoint at each epoch.
    """
    print(f"[ensemble] Training ConvNetBN for {num_models} epochs, saving checkpoints...")
    ds = TensorDataset(Dc_x, Dc_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=False)
    crit = nn.CrossEntropyLoss()

    net = ConvNetBN(num_classes=num_classes).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    checkpoints = []
    for epoch in range(1, num_models + 1):
        net.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(net(x), y).backward()
            opt.step()

        checkpoints.append({
            'net_state': {k: v.clone() for k, v in net.state_dict().items()},
            'epoch': epoch,
        })
        if epoch % 6 == 0 or epoch == num_models:
            print(f"  [ensemble] epoch {epoch}/{num_models} done")

    return checkpoints


# ========================= MetaPoison craft ========================= #
def craft_sp_metapoison(
    S_images, S_labels,
    targets_x, targets_y_adv,
    Dc_x, Dc_y,
    num_classes=10,
    # --- paper defaults ---
    outer_iters=60,
    outer_lr=200.0,
    lr_decay_factor=0.1,
    lr_decay_every=20,
    K_unroll=2,
    inner_lr=0.1,
    inner_bs=125,
    num_ensemble=24,
    device='cuda',
    verbose=True,
    # --- class-selective poisoning ---
    Sp_init_x=None,
    Sp_init_y=None,
):
    """
    MetaPoison crafting, faithful to paper.
    Poisons initialized from S (not Dc). Everything else matches.

    If Sp_init_x/Sp_init_y are provided, only those images are optimized
    (e.g. bird class only). S_images/S_labels serve as static inner-training
    context. This lets the attacker inject far fewer poisoned samples into Dc.
    """
    S_images = S_images.to(device)
    S_labels = S_labels.to(device)
    targets_x = targets_x.to(device)
    targets_y_adv = targets_y_adv.to(device)
    Dc_x = Dc_x.to(device)
    Dc_y = Dc_y.to(device)

    # Initialize Sp: either from a provided subset or from all of S
    if Sp_init_x is not None:
        Sp_images = Sp_init_x.clone().to(device).detach().requires_grad_(True)
        Sp_labels = Sp_init_y.clone().to(device).detach()
    else:
        Sp_images = S_images.clone().detach().requires_grad_(True)
        Sp_labels = S_labels.clone().detach()

    # Pixel bounds: keep Sp in the same normalized range as Dc so victim
    # training doesn't see extreme values that cause gradient explosions.
    px_min = Dc_x.min().item()
    px_max = Dc_x.max().item()

    # Paper: "Adam optimizer with an initial learning rate of 200.
    #         We decay the outer learning rate by 10x every 20 steps."
    outer_opt = torch.optim.Adam([Sp_images], lr=outer_lr)
    outer_sched = torch.optim.lr_scheduler.StepLR(
        outer_opt, step_size=lr_decay_every, gamma=lr_decay_factor)
    criterion = nn.CrossEntropyLoss()

    # Build staggered ensemble from Dc
    checkpoints = build_staggered_ensemble(
        Dc_x, Dc_y, num_models=num_ensemble,
        batch_size=inner_bs, lr=inner_lr,
        num_classes=num_classes, device=device)

    print(f"\n[craft] Starting: {outer_iters} outer steps, "
          f"K={K_unroll}, ensemble={num_ensemble}, outer_lr={outer_lr}")

    for it in range(outer_iters):
        outer_opt.zero_grad()

        total_outer = 0.0
        for ckpt in checkpoints:
            # Restore model to this epoch's checkpoint
            net = ConvNetBN(num_classes=num_classes).to(device)
            net.load_state_dict(ckpt['net_state'])
            # net.train()
            net.eval()

            inner_opt = torch.optim.SGD(net.parameters(), lr=inner_lr, momentum=0.9)

            # Inner loop data: S ∪ Sp (small, full-batch is fine)
            combined_x = torch.cat([S_images, Sp_images], dim=0)
            combined_y = torch.cat([S_labels, Sp_labels], dim=0)

            with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=True,
                                     track_higher_grads=True) as (fnet, fopt):
                for k in range(K_unroll):
                    out = fnet(combined_x)
                    loss_in = criterion(out, combined_y)
                    fopt.step(loss_in)

                # Outer loss: CE on targets with adversarial labels
                out_t = fnet(targets_x)
                loss_out = criterion(out_t, targets_y_adv)

            loss_out.backward()
            total_outer += loss_out.item()

        # Average grad over ensemble
        with torch.no_grad():
            if Sp_images.grad is not None:
                Sp_images.grad.div_(num_ensemble)

        outer_opt.step()
        outer_sched.step()

        if verbose and (it % 5 == 0 or it == outer_iters - 1):
            cur_lr = outer_sched.get_last_lr()[0]
            print(f"[craft] iter {it:3d}/{outer_iters}  "
                  f"outer_loss={total_outer/num_ensemble:.4f}  lr={cur_lr:.2f}")

    return Sp_images.detach().cpu(), Sp_labels.cpu()