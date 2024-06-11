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

# 进行PCA降维，保留前256个主成分
X_pca, pca_components, explained_variance = pca(train_data, 256)
X_train_meaned = train_data - np.mean(train_data, axis=0)
X_test_meaned = test_data - np.mean(test_data, axis=0)

def reconstruct(X_pca, components, mean):
    return np.dot(X_pca, components) + mean

dims=[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]

# 展示原图
image_index = 6
original_image = test_data[image_index].reshape(64, 64)

plt.figure(figsize=(35, 20))
plt.subplot(3, 5, 1)
plt.imshow(np.rot90(original_image, -1), cmap='gray')
plt.title('Original Image')
plt.axis('off')

for dim in dims:
    X_test_pca = np.dot(X_test_meaned, pca_components[:dim].T)  # PCA降维
    X_reconstructed = reconstruct(X_test_pca, pca_components[:dim], np.mean(train_data, axis=0)) # 重构图像

    # 展示特定维度下的重构图像
    reconstructed_image = X_reconstructed[image_index].reshape(64, 64)

    plt.subplot(3, 5, dims.index(dim) + 2)
    plt.imshow(np.rot90(reconstructed_image, -1), cmap='gray')
    plt.title(f'Reconstructed Image with {dim} dimensions')
    plt.axis('off')