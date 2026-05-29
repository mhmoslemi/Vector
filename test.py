import torch, torch.nn as nn, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from cpdd_unlearnable_cifar10 import ConvNet, normalize, load_cifar10

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- load distilled set ---
ckpt = torch.load("distilled_cpdd.pt", map_location="cpu")
syn_x, syn_y = ckpt["x"], ckpt["y"]
print(f"distilled: {tuple(syn_x.shape)}, labels {syn_y.unique().tolist()}")

# --- load CIFAR-10 test ---
_, _, x_te, y_te = load_cifar10("./data")

# --- train ConvNet on distilled, eval on test ---
net = ConvNet(num_classes=10).to(device)
opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
crit = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(syn_x, syn_y), batch_size=256, shuffle=True)

aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
for ep in range(50):
    net.train()
    for xb, yb in loader:
        xb, yb = aug(xb).to(device), yb.to(device)
        opt.zero_grad()
        crit(net(normalize(xb)), yb).backward()
        opt.step()
    sched.step()

net.eval()
correct = 0
with torch.no_grad():
    for i in range(0, len(x_te), 512):
        xb, yb = x_te[i:i+512].to(device), y_te[i:i+512].to(device)
        correct += (net(normalize(xb)).argmax(1) == yb).sum().item()
print(f"test acc = {correct/len(x_te)*100:.2f}%")