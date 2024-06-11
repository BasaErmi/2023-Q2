import numpy as np
from collections import Counter

def pca(X, n_components):
    """
    主成分分析 (PCA) 算法实现

    参数:
    X: 输入数据，形状为 (样本数, 特征数)
    n_components: 需要保留的主成分数

    输出:
    X_pca: 降维后的数据
    components: 主成分（特征向量）
    explained_variance: 主成分的方差（特征值）
    """
    # 数据标准化
    X_meaned = X - np.mean(X, axis=0)
    # 计算协方差矩阵
    covariance_matrix = np.cov(X_meaned, rowvar=False)
    # 对协方差矩阵进行SVD分解
    U, S, Vt = np.linalg.svd(covariance_matrix)
    # 选择前n_components个特征向量
    components = Vt[:n_components]
    # 计算特征值
    explained_variance = S[:n_components]
    # 将数据投影到新空间
    X_pca = np.dot(X_meaned, components.T)

    return X_pca, components, explained_variance

def lda(X, y, n_components):
    """
    LDA算法实现

    参数:
    X: 输入数据，形状为 (样本数, 特征数)
    y: 类别标签，形状为 (样本数, 1)
    n_components: 需要保留的投影维度

    输出:
    X_lda: 降维后的数据
    components: 投影矩阵
    """
    # 获取类别
    class_labels = np.unique(y)
    # 计算总体均值
    mean_overall = np.mean(X, axis=0)

    # 初始化类内散度矩阵和类间散度矩阵
    S_W = np.zeros((X.shape[1], X.shape[1]))
    S_B = np.zeros((X.shape[1], X.shape[1]))

    for c in class_labels:
        # 获取属于当前类别的样本
        X_c = X[y == c]
        # 计算当前类别的均值向量
        mean_c = np.mean(X_c, axis=0)
        # 计算类内散度矩阵
        S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
        # 计算类间散度矩阵
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_B += n_c * np.dot(mean_diff, mean_diff.T)

    # 计算 S_W^-1 * S_B
    A = np.linalg.pinv(S_W).dot(S_B)

    # 对矩阵 A 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # 按特征值从大到小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前 n_components 个特征向量
    components = eigenvectors[:, :n_components]

    # 将数据投影到新空间
    X_lda = np.dot(X, components)

    return X_lda, components



class KNN:
    """
    K-近邻算法实现

    参数:
    k: 临近样本数
    X_train: 训练数据，形状为 (样本数, 特征数)
    y_train: 训练标签，形状为 (样本数, 1)
    X_test: 测试数据，形状为 (样本数, 特征数)

    输出:
    predictions: 预测标签，形状为 (样本数, 1)
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # 计算x与所有训练样本之间的距离
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # 获取最近的k个样本的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取最近的k个样本的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 返回出现次数最多的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


