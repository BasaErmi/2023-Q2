from scipy import io
from func import pca, lda, KNN
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
# 进行LDA降维，保留前256个主成分
X_lda, lda_components = lda(train_data, train_label, 4096)

# 测试不同压缩维度对准确率的影响
dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]

accuracies_pca = []
accuracies_lda = []

X_train_meaned = train_data - np.mean(train_data, axis=0)
X_test_meaned = test_data - np.mean(test_data, axis=0)

for dim in dimensions:

    X_train_pca = np.dot(X_train_meaned, pca_components[:dim].T)
    X_test_pca = np.dot(X_test_meaned, pca_components[:dim].T)

    # 对PCA降维的数据进行测试
    knn_pca = KNN(k=5)
    knn_pca.fit(X_train_pca, train_label)
    predictions_pca = knn_pca.predict(X_test_pca)

    accuracy_pca = accuracy_score(test_label, predictions_pca)
    accuracies_pca.append(accuracy_pca)
    print(f'Accuracy with {dim} dimensions: {accuracy_pca * 100:.2f}%')

print('-----------------------------------')

for dim in dimensions:
    # 对LDA降维的数据进行测试
    X_train_lda = np.dot(X_train_meaned, lda_components[:, :dim])
    X_test_lda = np.dot(X_test_meaned, lda_components[:, :dim])

    knn_lda = KNN(k=5)
    knn_lda.fit(X_train_lda, train_label)
    predictions_lda = knn_lda.predict(X_test_lda)

    accuracy_lda = accuracy_score(test_label, predictions_lda)
    accuracies_lda.append(accuracy_lda)
    print(f'Accuracy with {dim} dimensions: {accuracy_lda * 100:.2f}%')

# 绘制准确率随压缩维度变化的图
plt.figure(figsize=(10, 6))
plt.plot(dimensions, accuracies_pca, marker='o', label='PCA')
plt.plot(dimensions, accuracies_lda, marker='x', label='LDA')
plt.xlabel('Number of dimensions')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of dimensions')
plt.legend()
plt.show()
plt.savefig('Accuracy vs Number of dimensions.png')