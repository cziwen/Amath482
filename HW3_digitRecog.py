import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==================== Load ====================
with open ('hw3data/train-images.idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack (">II", f.read (8))
    nrows, ncols = struct.unpack (">II", f.read (8))
    data = np.fromfile (f, dtype=np.dtype (np.uint8).newbyteorder ('>'))
    # print (data.shape)
    Xtraindata = np.transpose (data.reshape ((size, nrows * ncols)))

with open ('hw3data/train-labels.idx1-ubyte', 'rb') as f:
    magic, size = struct.unpack (">II", f.read (8))
    data = np.fromfile (f, dtype=np.dtype (np.uint8).newbyteorder ('>'))
    ytrainlabels = data.reshape ((size,))  # (Optional)

with open ('hw3data/t10k-images.idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack (">II", f.read (8))
    nrows, ncols = struct.unpack (">II", f.read (8))
    data = np.fromfile (f, dtype=np.dtype (np.uint8).newbyteorder ('>'))
    Xtestdata = np.transpose (data.reshape ((size, nrows * ncols)))

with open ('hw3data/t10k-labels.idx1-ubyte', 'rb') as f:
    magic, size = struct.unpack (">II", f.read (8))
    data = np.fromfile (f, dtype=np.dtype (np.uint8).newbyteorder ('>'))
    ytestlabels = data.reshape ((size,))  # (Optional)

traindata_imgs = np.transpose (Xtraindata).reshape ((60000, 28, 28))

print (Xtraindata.shape)  # (784, 60000)
print (ytrainlabels.shape)  # (60000, )

print (Xtestdata.shape)  # (784, 10000)
print (ytestlabels.shape)  # (10000, )


def plot_digits (XX, N, title):
    fig, ax = plt.subplots (N, N, figsize=(8, 8))

    for i in range (N):
        for j in range (N):
            ax[i, j].imshow (XX[:, (N) * i + j].reshape ((28, 28)), cmap="Greys")
            ax[i, j].axis ("off")
    fig.suptitle (title, fontsize=24)


# plot_digits (Xtraindata, 8, "First 64 Training Images") # usage


# ==================== Task 1: plot first 16 PCs ====================
# reshaped and stacked already from previous load

pca = PCA ()
pca.fit (Xtraindata.T)  # 计算主成分（未投影） (60000, 784)

# 拿到前16个pcs
# print (pca.components_.shape) # 主成分矩阵 (784, 784)
first_16_pcs = pca.components_[:16]  # (16, 784)

plot_digits (first_16_pcs.T, 4, "First 16 Principal Components")
plt.show ()

# ==================== Task 2: Compute Cumulative Energy and Determine k ====================

# 计算 singular value（explained variance）
explained_variance = pca.explained_variance_ratio_

# 计算 cumulative energy（也就是 cumsum(explained variance))
cumulative_energy = np.cumsum (explained_variance)

# 找到energy高于85%的k
k = np.argmax (cumulative_energy >= 0.85) + 1  # +1 因为index 从 0 开始
print (f"Number of principal components needed to retain 85% of the variance: {k}")

# Plot the cumulative energy
plt.figure (figsize=(8, 6))
plt.plot (np.arange (1, len (cumulative_energy) + 1), cumulative_energy, marker='o', linestyle='-')
plt.axhline (y=0.85, color='r', linestyle='--', label="85% Energy Threshold")
plt.axvline (x=k, color='g', linestyle='--', label=f"k = {k}")
plt.xlabel ("Number of Principal Components")
plt.ylabel ("Cumulative Explained Variance")
plt.title ("Cumulative Energy of Singular Values")
plt.legend ()
plt.grid (True)
plt.show ()


# 绘制 原版 和 k = 59 的 图像

# Plot function for original and reconstructed images
def plot_comparison (original, reconstructed, num_samples=10):
    fig, axes = plt.subplots (2, num_samples, figsize=(10, 4))

    for i in range (num_samples):
        # Original Image
        axes[0, i].imshow (original[:, i].reshape (28, 28), cmap="Greys")
        axes[0, i].axis ("off")

        # Reconstructed Image
        axes[1, i].imshow (reconstructed[:, i].reshape (28, 28), cmap="Greys")
        axes[1, i].axis ("off")

    axes[0, 0].set_title ("Original Images", fontsize=12)
    axes[1, 0].set_title (f"Reconstructed Images (k={k})", fontsize=12)
    plt.show ()


# Plot comparison
k = 59
X_projected = pca.transform (Xtraindata.T)[:, :k]  # (60000, 59)

X_projected_full = np.zeros ((X_projected.shape[0], 784))  # (60000, 784) # 创建补全矩阵（避免 inverse_transform 出错）
X_projected_full[:, :k] = X_projected  # 仅填充前 k 维
# 逆变换回原始维度
X_reconstructed = pca.inverse_transform (X_projected_full).T  # (784, 60000)

# 选取样本进行可视化
num_samples = 5
indices = np.random.choice (Xtraindata.shape[1], num_samples, replace=False)  # 选取随机样本

# 获取原始和重建的图像
original_images = Xtraindata[:, indices]
reconstructed_images = X_reconstructed[:, indices]

# 绘制对比
plot_comparison (original_images, reconstructed_images, num_samples)


# ==================== Task 3: Select a Subset of Particular Digits ====================

def select_digit_subset (X_train, y_train, X_test, y_test, digits):
    """
    Selects a subset of the dataset containing only the specified digits.

    Parameters:
    X_train (ndarray): Training data matrix of shape (784, num_samples)
    y_train (ndarray): Training labels of shape (num_samples,)
    X_test (ndarray): Test data matrix of shape (784, num_samples)
    y_test (ndarray): Test labels of shape (num_samples,)
    digits (list): List of digits to select (e.g., [4, 7])

    Returns:
    X_subtrain, y_subtrain, X_subtest, y_subtest
    """
    # Boolean mask for training data
    train_mask = np.isin (y_train, digits)  # 从 y_train 里获得在digit范围内的索引
    X_subtrain = X_train[:, train_mask]  # 截取该索引的 sub set
    y_subtrain = y_train[train_mask]

    # Boolean mask for test data
    test_mask = np.isin (y_test, digits)
    X_subtest = X_test[:, test_mask]
    y_subtest = y_test[test_mask]

    return X_subtrain, y_subtrain, X_subtest, y_subtest


# ==================== Task 4: Binary Classification with Ridge Classifier ====================

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Select digits 1 and 8
X_subtrain, y_subtrain, X_subtest, y_subtest = select_digit_subset (Xtraindata, ytrainlabels, Xtestdata, ytestlabels,
                                                                    [1, 8])

print("Shapes of extracted data:")
print("X_subtrain:", X_subtrain.shape, "y_subtrain:", y_subtrain.shape)
print("X_subtest:",  X_subtest.shape,  "y_subtest:",  y_subtest.shape)

# Apply PCA
pca = PCA(n_components=k)
pca.fit(Xtraindata.T) # 用原始的data计算主成分
X_train_pca = pca.transform (X_subtrain.T)
X_test_pca = pca.transform (X_subtest.T)

# Train Ridge classifier
ridge_clf = RidgeClassifier ()
ridge_clf.fit (X_train_pca, y_subtrain)

# cross validation
cv_scores = cross_val_score (ridge_clf, X_train_pca, y_subtrain, cv=5)

# Predict and evaluate
y_pred_train = ridge_clf.predict (X_train_pca)
y_pred_test = ridge_clf.predict (X_test_pca)

train_acc = accuracy_score (y_subtrain, y_pred_train)
test_acc = accuracy_score (y_subtest, y_pred_test)

print (f"Cross Validation Accuracy: {train_acc: .4f}")
print (f"Train Accuracy: {train_acc:.4f}")
print (f"Test Accuracy: {test_acc:.4f}")
