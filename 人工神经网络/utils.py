import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
import collections
import inspect

def use_svg_display():

    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):

    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):

    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

class HyperParameters:

    def save_hyperparameters(self, ignore=[]):
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ProgressBoard(HyperParameters):


    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(10, 8), display=True, save_to_file=False, filename="output.png"):
        self.save_hyperparameters()
        self.save_to_file = save_to_file
        self.filename = filename

    def draw(self, x, y, label, every_n=1):

        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        if self.save_to_file:
            plt.savefig(self.filename)
        else:
            plt.show()
        if isinstance(display, type):
            display.display(self.fig)
            display.clear_output(wait=True)

def init_cnn(module):  #@save

    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


def cpu():

    return torch.device('cpu')

def gpu(i=0):

    return torch.device(f'cuda:{i}')

def num_gpus():

    return torch.cuda.device_count()

def try_gpu(i=0):

    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():

    return [gpu(i) for i in range(num_gpus())]

def accuracy(Y_hat, Y, averaged=True):

    Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = astype(argmax(Y_hat, axis=1), Y.dtype)
    compare = astype(preds == reshape(Y, -1), float32)
    return reduce_mean(compare) if averaged else compare


def evaluate_accuracy(model, trainer, dataloader):
    model.eval()
    model.to(trainer.gpus[0] if trainer.gpus else cpu())
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = trainer.prepare_batch(batch)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def evaluate_metrics(model, trainer, dataloader):
    model.eval()
    model.to(trainer.gpus[0] if trainer.gpus else 'cpu')
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = trainer.prepare_batch(batch)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    cm = confusion_matrix(all_labels, all_predictions)

    return precision, recall, macro_f1, cm


def inference(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)
float32 = torch.float32
