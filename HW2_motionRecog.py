# ziwen chen, zchen56
from pyexpat.model import XML_CTYPE_EMPTY

import numpy as np
import matplotlib.pyplot as plt
from networkx.classes import neighbors
from numpy import dtype
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# ================================
# ============ Task 1 ============
# ================================

def load_test_data ():
    movements = ["walking", "jumping", "running"]  # 动作类别
    num_samples_per_movement = 1  # 每个动作的样本数量
    data_list = []

    for movement in movements:
        for i in range (1, num_samples_per_movement + 1):
            file_path = f"hw2data/test/{movement}_{i}t.npy"
            data = np.load (file_path)  # data.shape = (114, 100) each
            data_list.append (data)

    # 按“时间”方向拼在一起 -> 得到 (114, 100 * num_samples_per_movement * len(movements) )
    X_test = np.concatenate (data_list, axis=1)
    return X_test


# 1. 加载训练数据
def load_train_data ():
    """
    读取训练数据，并合并到一个大矩阵 X_train。
    每个动作样本的形状是 (114, 100)，其中 114 = 38 joints * 3坐标，
    100 为时间帧数。
    最后将所有样本拼接到一起，得到形状仍为 (114, time总数)。
    """
    movements = ["walking", "jumping", "running"]  # 动作类别
    num_samples_per_movement = 5  # 每个动作的样本数量
    data_list = []

    for movement in movements:
        for i in range (1, num_samples_per_movement + 1):
            file_path = f"hw2data/train/{movement}_{i}.npy"
            data = np.load (file_path)  # data.shape = (114, 100)
            data_list.append (data)

    # 按“时间”方向拼在一起 -> 得到 (114, 1500)
    X_train = np.concatenate (data_list, axis=1)
    return X_train


# 加载数据
X_train = load_train_data ()
print (f"原始 X_train 形状: {X_train.shape}")  # (114, 1500)

# 2. 执行 PCA
pca = PCA ()
X_projected = pca.fit_transform (X_train.T)

# print(X_centered_T.shape)
# print(X_centered_T)
# pca.transform(X_centered_T)
# print(X_centered_T.shape)
# print(X_centered_T)

# 3. 计算累积能量（方差占比）
cumulative_energy = np.cumsum (pca.explained_variance_ratio_)

# 4. 找到达到 70%、80%、90%、95% 所需的主成分数
thresholds = [0.7, 0.8, 0.9, 0.95]
num_components_needed = [
    np.argmax (cumulative_energy >= thr) + 1 for thr in thresholds
]

print ("\n不同能量比例所需的主成分数：")
for thr, k in zip (thresholds, num_components_needed):
    print (f"达到 {thr * 100:.0f}% 能量需要的 PCA 模式数: {k}")

# 5. 绘制累积能量曲线
plt.figure (figsize=(8, 5))
plt.plot (cumulative_energy, marker='o', label='Cumulative Energy')
plt.axhline (y=0.7, color='r', linestyle='--', label='70% Energy')
plt.axhline (y=0.8, color='g', linestyle='--', label='80% Energy')
plt.axhline (y=0.9, color='b', linestyle='--', label='90% Energy')
plt.axhline (y=0.95, color='purple', linestyle='--', label='95% Energy')
plt.xlabel ("Number of Principal Components (k)")
plt.ylabel ("Cumulative Energy")
plt.title ("PCA Components vs. Cumulative Energy")
plt.legend ()
plt.grid ()
plt.show ()

# 6. (额外) 绘制 Scree Plot
per_var = np.round (pca.explained_variance_ratio_ * 100, decimals=1)
num_shown = 10  # 只绘制前 10  个主成分
per_var = per_var[:num_shown]  # 限制只取前 num_shown 个
labels = ['PC' + str (x) for x in range (1, num_shown + 1)]  # 生成前20个主成分标签
plt.bar (x=range (1, num_shown + 1), height=per_var, tick_label=labels)
plt.ylabel ('Percentage of Explained Variance')
plt.xlabel ('Principal Component')
plt.title (f'Scree Plot for first {num_shown} Principal Components')
plt.show ()

# ================================
# ============ Task 2 ============
# ================================
# 1. 已有的 PCA 对象和数据
#    假设我们已经执行了以下步骤:
#    X_centered_T = X_train_centered.T
#    pca = PCA()
#    X_projected = pca.fit_transform (X_centered_T)

# 3. 提取前2个主成分、前3个主成分
X_pc2 = X_projected[:, :2]  # 仅保留PC1, PC2   (1500, 2)
X_pc3 = X_projected[:, :3]  # 仅保留PC1, PC2, PC3  (1500, 3)

# 为了上色，需要知道每类运动对应的帧范围
#  - walking: [0 : 500)
#  - jumping: [500 : 1000)
#  - running: [1000 : 1500)
idx_walk = slice (0, 500)
idx_jump = slice (500, 1000)
idx_run = slice (1000, 1500)

##########################
#   2D散点图 (PC1 vs PC2)
##########################
plt.figure (figsize=(6, 5))

plt.scatter (X_pc2[idx_walk, 0], X_pc2[idx_walk, 1],
             c='red', label='walking', alpha=0.7)
plt.scatter (X_pc2[idx_jump, 0], X_pc2[idx_jump, 1],
             c='green', label='jumping', alpha=0.7)
plt.scatter (X_pc2[idx_run, 0], X_pc2[idx_run, 1],
             c='blue', label='running', alpha=0.7)

plt.xlabel ('PC1')
plt.ylabel ('PC2')
plt.title ('2D PCA (PC1 vs. PC2)')
plt.legend ()
plt.grid (True)
plt.show ()

##########################
#   3D散点图 (PC1, PC2, PC3)
##########################
fig = plt.figure (figsize=(7, 6))
ax = fig.add_subplot (111, projection='3d')

ax.scatter (X_pc3[idx_walk, 0], X_pc3[idx_walk, 1], X_pc3[idx_walk, 2],
            c='red', label='walking', alpha=0.7)
ax.scatter (X_pc3[idx_jump, 0], X_pc3[idx_jump, 1], X_pc3[idx_jump, 2],
            c='green', label='jumping', alpha=0.7)
ax.scatter (X_pc3[idx_run, 0], X_pc3[idx_run, 1], X_pc3[idx_run, 2],
            c='blue', label='running', alpha=0.7)

ax.set_xlabel ('PC1')
ax.set_ylabel ('PC2')
ax.set_zlabel ('PC3')
ax.set_title ('3D PCA (PC1, PC2, PC3)')
ax.legend ()
plt.show ()

# ================================
# ============ Task 3 ============
# ================================

# 假设之前已经做了这些：
# 1) X_train.shape = (114, 1500)  (空间=114, 时间帧=1500)
# 2) 去均值: X_train_centered = X_train - X_mean
# 3) 转置并 PCA.fit:
#      X_centered_T = X_train_centered.T  # (1500, 114)
#      pca = PCA()
#      X_projected = pca.fit_transform (X_centered_T)

# --------------- 第一步：创建 ground truth labels ---------------
# y_train 大小为 1500，每个元素是对应帧所属动作的标签(0/1/2)
y_train = np.zeros (1500, dtype=int)

# 前 500 帧对应 walking(0), 中间 500 帧对应 jumping(1), 后 500 帧对应 running(2)
y_train[0:500] = 0  # walking
y_train[500:1000] = 1  # jumping
y_train[1000:1500] = 2  # running

# --------------- 第二步：投影到 PCA 空间（k个主成分） ---------------
# X_projected = pca.transform(X_centered_T)  # shape = (1500, 114)

# 选择保留的主成分数k，这里我选择 k= 7，因为 7 已经能够达到95%的准确率了
k = 7
X_k = X_projected[:, :k]  # 只保留前 k 个主成分 -> shape = (1500, k)

# --------------- 第三步：计算各类运动在 k维空间的质心 ---------------
centroids = []
movement_classes = [0, 1, 2]
for c in movement_classes:
    # 取出属于第 c 类的所有帧在 X_k 空间的坐标
    class_data = X_k[y_train == c]  # shape = (500, k)
    # 求质心(均值)
    centroid_c = np.mean (class_data, axis=0)
    centroids.append (centroid_c)

# 现在 centroids 就是一个列表，里面有 3 个元素，每个元素是 k 维均值
# centroids[0] -> walking 的质心 (形状: (k,))
# centroids[1] -> jumping 的质心 (形状: (k,))
# centroids[2] -> running 的质心 (形状: (k,))

print ("\n=== Movement Centroids in PCA Space (k={} modes) ===".format (k))
for label, c in zip (["walking", "jumping", "running"], centroids):
    print (f"- {label} centroid: {c}")


# ================================
# ============ Task 4 ============
# ================================

# 1) 定义一个函数: 给定一个 k, 先计算质心，再用最近质心预测标签并计算准确率
def centroid_classifier_accuracy (k, X_projected, y_train):
    """
    输入:
        k: 保留的 PCA 主成分数
        X_projected: shape=(N,114)，N 为 总帧数 ,已在PCA空间
        y_train: shape=(N,), 每个样本对应的真实标签(0,1,2)，0=walking, 1=jumping, 2=running
    返回:
        accuracy: float, 该k值下用最近质心分类得到的准确率
    """
    # 截取前 k 个主成分
    X_k = X_projected[:, :k]  # (N, k)

    # --------------- 计算各类运动在 k维空间的质心 ---------------
    centroids = []
    classes = [0, 1, 2]
    for c in classes:
        class_data = X_k[y_train == c]  # 取出第 c 类
        centroid_c = np.mean (class_data, axis=0)  # shape=(k,)
        centroids.append (centroid_c)
    centroids = np.array(centroids, dtype=np.float64)  # 强制转换数据类型，确保精度一致 shape=(3,k)


    # --------------- 对每个样本做分类预测 ---------------
    #   算与 3 个质心的欧氏距离, 选最近者
    y_pred = []
    for x in X_k:  # x.shape=(k,)
        # 计算 x 与每个 centroid 的距离
        dists = np.linalg.norm (centroids - x, axis=1)  # 得到长度3的一维数组，分别对应着与每个 质心 之间的 norm 距离
        predicted_label = np.argmin (dists)  # 距离最小的质心的索引(0/1/2)，也就是距离最近的 质心 为该样本的动作
        y_pred.append (predicted_label)
    y_pred = np.array (y_pred, dtype=int)

    # --------------- 计算准确率 ---------------
    acc = accuracy_score (y_train, y_pred)
    return acc


# 2) 在多个 k 值下重复此流程
k_values = [1, 2, 3, 5, 7, 10, 14, 18, 24, 30]  # 可根据需要自行扩展

# 3) 测试
print ("\n===测试在不同k下的 centroid of train data 预测train data 自己的准确率===")
best_acc = 0.0
best_k = None
for k in k_values:
    acc = centroid_classifier_accuracy (k, X_projected, y_train)
    print (f"k={k}, accuracy={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print ("\nBest accuracy = {:.4f} at k = {}".format (best_acc, best_k))


# ================================
# ============ Task 5 ============
# ================================

def centroid_classifier_accuracy (k, X_test_projected, X_train_projected, y_test_train, y_train):
    """
    输入:
        k: 保留的 PCA 主成分数
        X_test_projected: shape=(n,114)，n 为 总帧数 ,已在PCA空间
        X_train_projected: shape = (N,114) N 为 总帧数，已在PCA空间
        y_test_train: shape=(n,), 每个样本对应的真实标签(0,1,2)，0=walking, 1=jumping, 2=running
        y_train: shape(N,), 每个样本对应的真实标签(0,1,2)，0=walking, 1=jumping, 2=running
    返回:
        accuracy: float, 该k值下用最近质心分类得到的准确率
    """
    # 截取 X_train 和 X_test 前 k 个主成分
    X_train_k = X_train_projected[:, :k]  # (N, k)
    X_test_k = X_test_projected[:, :k]  # (n, k)

    # --------------- 计算 X_train 的各类运动在 k维空间的质心 ---------------
    centroids_X_train = []
    classes = [0, 1, 2]
    for c in classes:
        class_data = X_train_k[y_train == c]  # 取出第 c 类
        centroid_c = np.mean (class_data, axis=0)  # shape=(k,)
        centroids_X_train.append (centroid_c)
    centroids_X_train = np.array (centroids_X_train, dtype=np.float64)  # shape=(3,k)

    # --------------- 对每个样本做分类预测，用 训练好的质心 来给 新样本分类 ---------------
    y_pred = []
    for x in X_test_k:  # x.shape=(k,)
        # 计算 x 与每个 centroid 的距离
        dists = np.linalg.norm (centroids_X_train - x, axis=1)  # 得到长度3的一维数组，分别对应着与每个 质心 之间的 norm 距离
        predicted_label = np.argmin (dists)  # 距离最小的质心的索引(0/1/2)，也就是距离最近的 质心 为该样本的动作
        y_pred.append (predicted_label)
    y_pred = np.array (y_pred, dtype=int)

    # --------------- 计算准确率 ---------------
    acc = accuracy_score (y_test_train, y_pred)
    return acc


# 1) 导入数据，进行pca
X_test = load_test_data ()  # (114, 300)
X_test_projected = pca.transform(X_test.T)  # 得到 shape=(300,114)

# 2） 创建label    0=walking, 1=jumping, 2=running
y_test_train = np.zeros (300, dtype=int)
y_test_train[0:100] = 0
y_test_train[100:200] = 1
y_test_train[200:300] = 2

# 3) 测试准确率
print ("\n===测试在不同k下，给新样本分类的准确率===")
best_acc = 0.0
best_k = None
for k in k_values:
    acc = centroid_classifier_accuracy (k, X_test_projected, X_projected, y_test_train, y_train)
    print (f"k={k}, accuracy={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print ("\nBest accuracy = {:.4f} at k = {}".format (best_acc, best_k))


# ================================
# ============ Task 6 ============
# ================================

def knn_classifier_accuracy (k, X_test_projected, X_train_projected, y_test_train, y_train, n_neighbors=5):
    """
    输入:
        k: 保留的 PCA 主成分数
        X_test_projected: shape=(n,114)，n 为 总帧数 ,已在PCA空间
        X_train_projected: shape = (N,114) N 为 总帧数，已在PCA空间
        y_test_train: shape=(n,), 每个样本对应的真实标签(0,1,2)，0=walking, 1=jumping, 2=running
        y_train: shape(N,), 每个样本对应的真实标签(0,1,2)，0=walking, 1=jumping, 2=running
        n_neighbors: 选择的最近邻个数 (默认 k=5)

    返回:
        accuracy: float, 该 k 值下用 k-NN 分类得到的准确率
    """
    # 截取 X_train 和 X_test 前 k 个主成分
    X_train_k = X_train_projected[:, :k]  # (N, k)
    X_test_k = X_test_projected[:, :k]  # (n, k)

    # --------------- 训练 k-NN 分类器 ---------------
    knn = KNeighborsClassifier (n_neighbors=n_neighbors)  # 选择最近邻个数
    knn.fit (X_train_k, y_train)

    # --------------- 对测试样本进行预测 ---------------
    y_pred = knn.predict (X_test_k)

    # --------------- 计算准确率 ---------------
    acc = accuracy_score (y_test_train, y_pred)
    return acc


print ("\n===测试在不同k和n_neighbor下，给knn给train Data的分类的准确率===")
print ("\n---每次打印在当前k下，准确率最高的 n_neighbor 和其准确率---")
max_n_neighbors = 10
best_acc = 0.0
best_k = None
best_n_neighbors = None
for k in k_values:

    best_acc_current = 0.0
    best_k_current = None
    best_n_neighbors_current = None

    for n_neighbors in range (1, max_n_neighbors + 1):
        acc = knn_classifier_accuracy (k, X_projected, X_projected, y_train, y_train)
        # 更新当前k下最高的准确率和neighbor
        if acc > best_acc_current:
            best_acc_current = acc
            best_k_current = k
            best_n_neighbors_current = n_neighbors

    # 更新全局最高的准确率和neighbor
    if best_acc_current > best_acc:
        best_acc = best_acc_current
        best_k = best_k_current
        best_n_neighbors = best_n_neighbors_current

    print (f"k={best_k_current}, n_neighbors={best_n_neighbors_current}, accuracy={best_acc_current:.6f}")

print ("\nBest accuracy = {:.4f} at k = {}, n_neighbors = {}".format (best_acc, best_k, best_n_neighbors))



print ("\n===测试在不同k和n_neighbor下，给knn给新样本的分类的准确率===")
print ("\n---每次打印在当前k下，准确率最高的 n_neighbor 和其准确率---")
max_n_neighbors = 10
best_acc = 0.0
best_k = None
best_n_neighbors = None
for k in k_values:

    best_acc_current = 0.0
    best_k_current = None
    best_n_neighbors_current = None

    for n_neighbors in range (1, max_n_neighbors + 1):
        acc = knn_classifier_accuracy (k, X_test_projected, X_projected, y_test_train, y_train)
        # 更新当前k下最高的准确率和neighbor
        if acc > best_acc_current:
            best_acc_current = acc
            best_k_current = k
            best_n_neighbors_current = n_neighbors

    # 更新全局最高的准确率和neighbor
    if best_acc_current > best_acc:
        best_acc = best_acc_current
        best_k = best_k_current
        best_n_neighbors = best_n_neighbors_current

    print (f"k={best_k_current}, n_neighbors={best_n_neighbors_current}, accuracy={best_acc_current:.6f}")

print ("\nBest accuracy = {:.4f} at k = {}, n_neighbors = {}".format (best_acc, best_k, best_n_neighbors))



