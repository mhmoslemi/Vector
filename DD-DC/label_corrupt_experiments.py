"""
Two experiments on CIFAR-10:
  Exp 1: Random labels  — each training sample gets a random label in [0,9]
  Exp 2: Constant label — every training sample gets label 0
Test accuracy is logged every 5 epochs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import numpy as np

# ── config ──────────────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 128
LR          = 0.1
MOMENTUM    = 0.9
WD          = 5e-4
EVAL_EVERY  = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
SEED        = 42

# ── data ─────────────────────────────────────────────────────────────────────
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_loaders(corrupt_fn):
    """Return (train_loader, test_loader) with labels corrupted by corrupt_fn."""
    trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True,
                                            download=True, transform=train_tf)
    testset  = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False,
                                            download=True, transform=test_tf)

    # corrupt training labels
    np.random.seed(SEED)
    targets = np.array(trainset.targets)
    trainset.targets = list(corrupt_fn(targets))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True,  num_workers=4,
                                               pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=256,
                                               shuffle=False, num_workers=4,
                                               pin_memory=True)
    return train_loader, test_loader

# ── model ────────────────────────────────────────────────────────────────────
def get_model():
    model = torchvision.models.resnet18(num_classes=NUM_CLASSES)
    # CIFAR-10 uses 32×32 — shrink the first conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(DEVICE)

# ── train / eval ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        n       += imgs.size(0)
    return correct / n

# ── experiment runner ─────────────────────────────────────────────────────────
def run_experiment(name, corrupt_fn):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}")

    train_loader, test_loader = get_loaders(corrupt_fn)
    model     = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    results = []
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            test_acc = evaluate(model, test_loader)
            results.append((epoch, train_loss, train_acc, test_acc))
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  "
                  f"train_acc={train_acc*100:.1f}%  "
                  f"test_acc={test_acc*100:.1f}%")

    print(f"\n  Final test accuracy: {results[-1][3]*100:.2f}%")
    return results


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(SEED)

    # Experiment 1: random labels (uniform over [0, NUM_CLASSES))
    random_corrupt = lambda y: np.random.randint(0, NUM_CLASSES, size=len(y))
    results_random = run_experiment("Exp 1 — Random labels", random_corrupt)

    # Experiment 2: all same label (label = 0)
    const_corrupt  = lambda y: np.zeros(len(y), dtype=np.int64)
    results_const  = run_experiment("Exp 2 — All same label (0)", const_corrupt)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("SUMMARY — Test Accuracy Every 5 Epochs")
    print("="*60)
    header = f"{'Epoch':>6}  {'Exp1 Random':>12}  {'Exp2 Const':>12}"
    print(header)
    for (e1, _, _, t1), (e2, _, _, t2) in zip(results_random, results_const):
        assert e1 == e2
        print(f"{e1:6d}  {t1*100:11.2f}%  {t2*100:11.2f}%")
