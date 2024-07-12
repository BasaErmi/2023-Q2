import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from BackboneModel import WideResNet
from collections import defaultdict
import torch.nn.functional as F
import random


def mixmatch_loss(model, mixed_input, mixed_target, num_labeled, lambda_u=1.0):
    # 将混合输入传递给模型
    logits = model(mixed_input)

    # 计算有标签数据的损失
    Lx = F.cross_entropy(logits[:num_labeled], mixed_target[:num_labeled])
    # 计算无标签数据的损失
    Lu = F.mse_loss(logits[num_labeled:], mixed_target[num_labeled:])

    # 总损失
    L = Lx + lambda_u * Lu

    return L

def augment(x):
    # 数据增强函数，包含随机裁剪和随机水平翻转
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
    ])

    x_aug = torch.zeros_like(x)
    for i in range(x.size(0)):
        x_aug[i] = transform(x[i])

    return x_aug


# MixMatch算法核心函数
def mixmatch(model, x, y, u, K=2, T=0.5, alpha=0.75):
    # 输入：有标签数据x,y；无标签数据u；模型model
    # K：数据增强次数；T：温度系数；alpha：Beta分布参数

    # 对有标签数据进行增强
    x_aug = augment(x)

    # 对无标签数据进行K次增强
    u_aug = [augment(u) for _ in range(K)]

    with torch.no_grad():
        # 对每次增强后的无标签数据进行伪标签预测
        p_aug = [torch.softmax(model(u_aug_k), dim=1) for u_aug_k in u_aug]

        # 平均所有增强后的预测
        p_avg = sum(p_aug) / K

        # 温度缩放 (即论文中的 sharpening操作)
        p_avg = p_avg ** (1 / T)
        targets_u = p_avg / p_avg.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()

    # 合并有标签数据和无标签数据
    all_inputs = torch.cat([x_aug, u], dim=0)
    all_targets = torch.cat([y, targets_u], dim=0)

    # Shuffle操作
    shuffle_idx = torch.randperm(all_inputs.size(0))
    all_inputs = all_inputs[shuffle_idx]
    all_targets = all_targets[shuffle_idx]

    # Mixup操作将有标签数据和无标签数据混合
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    idx = torch.randperm(all_inputs.size(0))

    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target


# 训练函数
def train(model, labeled_loader, unlabeled_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    global steps, best_acc

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    for _, (x, y) in enumerate(labeled_iter):
        try:
            u, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            u, _ = next(unlabeled_iter)

        x, y, u = x.to(device), y.to(device), u.to(device)
        y_onehot = torch.zeros(y.size(0), 10).to(device).scatter_(1, y.view(-1, 1), 1)

        mixed_input, mixed_target = mixmatch(model, x, y_onehot, u)
        loss = criterion(model, mixed_input, mixed_target, num_labeled=x.size(0))

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
                torch.save(model.state_dict(), f'./saved_model/MixMatch/model_{num}_{best_acc:.2f}%.pth')
                print(f'---------------Save model with accuracy {best_acc:.2f}%------------------')

        running_loss += loss.item()

        if steps >= max_steps:
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
num = 80

# 按类别分类样本
class_samples = defaultdict(list)
for idx, (image, label) in enumerate(trainset):
    class_samples[label].append(idx)

# 设定每个类别取样数量
num_per_class = num//class_samples.keys().__len__()
print(num_per_class)
labeled_indices = []

# 从每个类别中均匀取样
for label, indices in class_samples.items():
    labeled_indices.extend(random.sample(indices, num_per_class))

# 剩下的作为无标签数据集
all_indices = set(range(len(trainset)))
unlabeled_indices = list(all_indices - set(labeled_indices))

# 创建数据子集
labeled_set = torch.utils.data.Subset(trainset, labeled_indices)
unlabeled_set = torch.utils.data.Subset(trainset, unlabeled_indices)

labeled_loader = torch.utils.data.DataLoader(labeled_set, batch_size=64, shuffle=True)
unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WideResNet(depth=28, num_classes=10, widen_factor=2).to(device)
criterion = mixmatch_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100000000

# 初始化 SummaryWriter
writer = SummaryWriter(log_dir=f'./logs/mixmatch/{num}')

for epoch in range(num_epochs):
    if(steps >= max_steps):
        break
    train(model, labeled_loader, unlabeled_loader, criterion, optimizer)
    # print(f'Epoch [{epoch+1}/{num_epochs}]')

steps = 0
print(f'Final Accuracy with {num} ')
