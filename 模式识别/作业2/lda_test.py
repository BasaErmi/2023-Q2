from scipy import io
from func import lda
import numpy as np
import matplotlib.pyplot as plt

x=io.loadmat('Yale_64x64.mat')
ins_perclass,class_number,train_test_split = 11,15,9
input_dim=x['fea'].shape[1]
feat=x['fea'].reshape(-1,ins_perclass,input_dim)
label=x['gnd'].reshape(-1,ins_perclass)

train_data,test_data = feat[:,:train_test_split,:].reshape(-1,input_dim),feat[:,train_test_split:,:].reshape(-1,input_dim)
train_label,test_label = label[:,:train_test_split].reshape(-1),label[:,train_test_split:].reshape(-1)

"""
train_data和test_data的维度 = (num_samples, num_features)
train_label和test_label的维度 = (num_samples, 1)
"""

# 进行LDA降维，保留前12个主成分
X_pca, eigVects = lda(train_data, train_label, 12)

plt.figure(figsize=(12, 6))
for i in range(8):
    component_image = np.real(eigVects[:, i]).reshape(64, 64)
    plt.subplot(2, 4, i + 1)
    plt.imshow(np.rot90(component_image,-1), cmap='gray')
    plt.title(f'Component {i + 1}')
    plt.axis('off')

plt.show()

eigVects_top2 = eigVects[:, :2]
X_lda = np.dot(train_data, eigVects_top2)
# 绘图
cmap = plt.get_cmap('tab20', class_number)

# 可视化降维后的训练和测试数据
def plot_lda_2d(X, labels, title):
    plt.figure(figsize=(10, 6))
    c=0;
    for label in np.unique(labels):
        plt.scatter(X[labels == label, 0], X[labels == label, 1], label=f'Class {label}', color=cmap(c))
        c = c+1
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.legend()
    plt.show()

plot_lda_2d(X_lda, train_label, 'LDA of Training Data')

# 二维数据
X_lda_test = np.dot(test_data, eigVects_top2)
# 绘图
cmap = plt.get_cmap('tab20', class_number)

# 可视化降维后的训练和测试数据
def plot_lda_2d(X, labels, title):
    plt.figure(figsize=(10, 6))
    c=0;
    for label in np.unique(labels):
        plt.scatter(X[labels == label, 0], X[labels == label, 1], label=f'Class {label}', color=cmap(c))
        c = c+1
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.legend()
    plt.show()

plot_lda_2d(X_lda_test, test_label, 'LDA of Training Data')