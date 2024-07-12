import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from BackboneModel import WideResNet
from torchvision.transforms import RandAugment  # 引入 RandAugment

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(-1, 8*8*128)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

def weak_augment(x):
    # 弱增强：随机裁剪和随机水平翻转
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 旋转和移位
    ])

    x_aug = torch.zeros_like(x)
    for i in range(x.size(0)):
        x_aug[i] = transform(x[i])

    return x_aug

def strong_augment(x):
    # 强增强：使用 RandAugment 进行更强的数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        RandAugment(num_ops=2, magnitude=9)  # 使用 RandAugment
    ])

    x_aug = torch.zeros_like(x)
    for i in range(x.size(0)):
        img = x[i]
        img = (img * 255).byte()  # 转换为 uint8 类型
        img = transform(img)
        img = img.float() / 255  # 转回 float32 类型
        x_aug[i] = img

    return x_aug

def fixmatch(model, x, y, u, tau=0.95, lambda_u=1.0):
    # 输入：有标签数据x,y；无标签数据u；模型model
    # tau：置信度阈值；lambda_u：无标签损失权重

    # 对无标签数据进行弱增强并计算伪标签
    u_w = weak_augment(u)
    with torch.no_grad():
        q = torch.softmax(model(u_w), dim=1)
        max_q, max_idx = torch.max(q, dim=1)
        mask = max_q > tau
        pseudo_labels = max_idx[mask]

    # 对有标签数据进行弱增强
    x_w = weak_augment(x)

    # 计算有标签数据的损失
    logits_x = model(x_w)
    loss_x = nn.CrossEntropyLoss()(logits_x, y)

    # 对无标签数据进行强增强
    u_s = strong_augment(u[mask])
    # 计算无标签数据的损失
    if u_s.size(0) > 0:
        logits_u = model(u_s)
        loss_u = nn.CrossEntropyLoss()(logits_u, pseudo_labels)
    else:
        loss_u = torch.tensor(0.0).to(loss_x.device)

    # 返回总损失
    return loss_x + lambda_u * loss_u

# 训练函数
def train(model, labeled_loader, unlabeled_loader, optimizer):
    model.train()
    running_loss = 0.0
    global steps, best_acc

    unlabeled_iter = iter(unlabeled_loader)
    labeled_iter = iter(labeled_loader)
    for _, (x, y) in enumerate(labeled_iter):
        try:
            u, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            u, _ = next(unlabeled_iter)

        x, y, u = x.to(device), y.to(device), u.to(device)

        loss = fixmatch(model, x, y, u)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1

        if steps % 100 == 0:
            test_acc = test(model, test_loader)
            print(f'steps [{steps}/{max_steps}], Test Accuracy: {test_acc:.2f}%')
            writer.add_scalar('Loss/train', loss.item(), steps)
            writer.add_scalar('Accuracy/test', test_acc, steps)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'./saved_model/FixMatch/model_{num}_{best_acc:.2f}%.pth')
                print(f'---------------Save model with accuracy {best_acc:.2f}%------------------')

        running_loss += loss.item()

        if steps > max_steps:
            break

# 测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
steps = 0
max_steps = 20001
best_acc = 0

# 分别使用40, 250, 4000张标注数据
num_labeled = [4000, 40, 250]

for num in num_labeled:
    labeled_set = torch.utils.data.Subset(trainset, list(range(num)))   # 在训练集中取前num个数据有标签数据集
    unlabeled_set = torch.utils.data.Subset(trainset, list(range(num, len(trainset)))) # 无标签数据集，在unlabeled_set中取num之后的数据

    labeled_loader = torch.utils.data.DataLoader(labeled_set, batch_size=64, shuffle=True)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet(depth=28, num_classes=10, widen_factor=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    num_epochs = 100000000

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir=f'./logs/fixmatch/{num}')

    for epoch in range(num_epochs):
        if(steps > max_steps):
            break
        train(model, labeled_loader, unlabeled_loader, criterion, optimizer)
        # print(f'Epoch [{epoch+1}/{num_epochs}]')

    steps = 0
    print(f'Final Accuracy with {num} ')
