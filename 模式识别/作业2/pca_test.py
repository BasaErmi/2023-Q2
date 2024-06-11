from scipy import io
from func import pca, KNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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

# 进行PCA降维，保留前8个主成分
X_pca, components, explained_variance = pca(train_data, 8)
components = components.reshape(-1, 64, 64)

# 显示前8个特征向量
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.rot90(components[i],-1), cmap='gray')
    ax.set_title( f'Component {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()


# 进行PCA降维，保留前2个主成分
X_train_pca, components, explained_variance = pca(train_data, 2)
# 使用已经训练好的投影矩阵（即components）对测试数据进行分类
X_test_mean = test_data - np.mean(test_data, axis=0)
X_test_pca = np.dot(X_test_mean, components.T)

# 绘图
from matplotlib.colors import ListedColormap
cmap = plt.get_cmap('tab20', class_number)

# 可视化降维后的训练和测试数据
def plot_pca_2d(X, labels, title):
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

plot_pca_2d(X_train_pca, train_label, 'PCA of Training Data')
plot_pca_2d(X_test_pca, test_label, 'PCA of Testing Data')


# # 测试不同压缩维度对准确率的影响
# dimensions = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]
# accuracies = []
#
# X_meaned = train_data - np.mean(train_data, axis=0)
#
# for dim in dimensions:
#
#     X_train_pca = np.dot(X_meaned, components[:dim].T)
#     X_test_pca = np.dot(test_data - np.mean(test_data, axis=0), components[:dim].T)
#
#     knn = KNN(k=8)
#     knn.fit(X_train_pca, train_label)
#     predictions = knn.predict(X_test_pca)
#
#     accuracy = accuracy_score(test_label, predictions)
#     accuracies.append(accuracy)
#     print(f'Accuracy with {dim} dimensions: {accuracy * 100:.2f}%')
#
# # 绘制准确率随压缩维度变化的图
# plt.figure(figsize=(10, 6))
# plt.plot(dimensions, accuracies, marker='o')
# plt.title('Accuracy vs. Number of PCA Dimensions')
# plt.xlabel('Number of PCA Dimensions')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()