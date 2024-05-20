import numpy as np
import torch
import os
import utils
import torchvision
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from timm.models.layers import trunc_normal_, DropPath


class Module(nn.Module, utils.HyperParameters):


    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = utils.ProgressBoard(display=True, save_to_file=True)

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):

        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, utils.numpy(utils.to(value, utils.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):

        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):

        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)


class DataModule(utils.HyperParameters):
    """数据dataloader基类"""

    def __init__(self, root='../data', num_workers=12):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)

class EmotionDataset(Dataset):
    """数据集"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = os.listdir(root_dir) # 根据文件夹名字获取类别
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)} # 类别到索引的映射

        for cls in self.classes: # 依次读取每个类别的图片
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                # 将图片路径和标签加入到列表中
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 定义__getitem__方法返回图片和标签，用于DataLoader加载数据并打上标签
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class EmotionDataModule(DataModule):
    """情感数据集的DataModule"""
    def __init__(self, root='./data', batch_size=32, num_workers=12, resize=(224, 224)):
        self.save_hyperparameters()
        self.train_transform = transforms.Compose([  # 对训练集进行图像增强
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.root_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataloader(self, train):
        if train:
            dataset = EmotionDataset(root_dir=os.path.join(self.root_dir, 'train'), transform=self.train_transform)
            weights = calculate_weights(dataset)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
        else:
            dataset = EmotionDataset(root_dir=os.path.join(self.root_dir, 'test'), transform=self.val_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_workers)

    def calculate_weights(dataset):
        label_counts = np.bincount(dataset.labels)
        total_samples = len(dataset.labels)
        weights = 1.0 / label_counts
        sample_weights = weights[dataset.labels]
        return sample_weights

def calculate_weights(dataset):
    label_counts = np.bincount(dataset.labels)
    total_samples = len(dataset.labels)
    weights = 1.0 / label_counts
    sample_weights = weights[dataset.labels]
    return sample_weights


class Trainer(utils.HyperParameters):

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0, checkpoint_path='best_model.pth'):
        self.save_hyperparameters()
        self.gpus = [utils.gpu(i) for i in range(min(num_gpus, utils.num_gpus()))]
        self.best_val_accuracy = 0
        self.checkpoint_path = checkpoint_path

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [utils.to(a, self.gpus[0]) for a in batch]
        return batch

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            # 输出验证集和测试集的准确率
            train_acc = utils.evaluate_accuracy(model, self, self.train_dataloader)
            val_acc = utils.evaluate_accuracy(model, self, self.val_dataloader)
            print(f'Epoch {self.epoch+1}/{self.max_epochs}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

    def fit_epoch(self):

        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()

        val_acc = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                val_acc += self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
        val_acc /= self.num_val_batches
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            print(f'Validation accuracy improved. Saving model.')
            self.save_checkpoint()

    def save_checkpoint(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'accuracy': self.best_val_accuracy,
        }, self.checkpoint_path)

    def clip_gradients(self, grad_clip_val, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

class Classifier(Module):


    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        acc = utils.accuracy(Y_hat, batch[-1])
       # self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', acc, train=False)
        return acc

    def training_step(self, batch):
        Y_hat = self(*batch[:-1])
        l = self.loss(Y_hat, batch[-1])
        #self.plot('loss', l, train=True)
        self.plot('acc', utils.accuracy(Y_hat, batch[-1]), train=True)
        return l

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = utils.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = utils.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        X = utils.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)




"""

ResNeXt

"""
class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)


class ResNeXt(Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super(ResNeXt, self).__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i + 2}', self.block(*b, first_block=(i == 0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(utils.init_cnn)


    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def block(self, num_ResNeXt, num_channels, first_block=False):
        blk = []
        for i in range(num_ResNeXt):
            if i == 0 and not first_block:
                blk.append(ResNeXtBlock(num_channels, 32, 1, use_1x1conv=True, strides=2))
            elif i==num_ResNeXt-2:
                blk.append(ResNeXtBlock(num_channels, 32, 1))
                blk.append(nn.Dropout(0.2))
            elif i==num_ResNeXt-1:
                blk.append(ResNeXtBlock(num_channels, 32, 1))
                blk.append(nn.Dropout(0.4))
        return nn.Sequential(*blk)

    def configure_optimizers(self):
        """使用adam优化器"""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class ResNeXt18(ResNeXt):
    def __init__(self, lr, num_classes):
        super().__init__(((2, 64), (2, 128), (2, 256)),
                       lr, num_classes)



