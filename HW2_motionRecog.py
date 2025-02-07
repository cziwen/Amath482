# ziwen chen, zchen56
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



# ================================
# ============ Task 1 ============
# ================================

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

# 2. 零均值
X_mean = np.mean (X_train, axis=1, keepdims=True)  # 每个关节的均值
X_train_centered = X_train - X_mean  # 去中心化

# 3. 转置(transpose)，让“时间”当作 sample，“空间”当作 feature
#    这样 fit PCA 时就以空间维度作为主成分方向
X_centered_T = X_train_centered.T  # (1500, 114)

# 4. 执行 PCA
pca = PCA ()
pca.fit (X_centered_T)

# print(X_centered_T.shape)
# print(X_centered_T)
# pca.transform(X_centered_T)
# print(X_centered_T.shape)
# print(X_centered_T)

# 5. 计算累积能量（方差占比）
cumulative_energy = np.cumsum (pca.explained_variance_ratio_)

# 6. 找到达到 70%、80%、90%、95% 所需的主成分数
thresholds = [0.7, 0.8, 0.9, 0.95]
num_components_needed = [
    np.argmax (cumulative_energy >= thr) + 1 for thr in thresholds
]

print ("\n不同能量比例所需的主成分数：")
for thr, k in zip (thresholds, num_components_needed):
    print (f"达到 {thr * 100:.0f}% 能量需要的 PCA 模式数: {k}")

# 7. 绘制累积能量曲线
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


# ================================
# ============ Task 2 ============
# ================================
# 1. 已有的 PCA 对象和数据
#    假设我们已经执行了以下步骤:
#    X_centered_T = X_train_centered.T
#    pca = PCA()
#    pca.fit(X_centered_T)

# 2. 获取投影系数 (1500, 114)，每一行对应时间帧，每一列对应某个主成分
X_projected = pca.transform(X_centered_T)  # (1500, 114)

# 3. 提取前2个主成分、前3个主成分
X_pc2 = X_projected[:, :2]  # 仅保留PC1, PC2
X_pc3 = X_projected[:, :3]  # 仅保留PC1, PC2, PC3

# 为了上色，需要知道每类运动对应的帧范围
#  - walking: [0 : 500)
#  - jumping: [500 : 1000)
#  - running: [1000 : 1500)
idx_walk = slice(0, 500)
idx_jump = slice(500, 1000)
idx_run  = slice(1000, 1500)

##########################
#   2D散点图 (PC1 vs PC2)
##########################
plt.figure(figsize=(6, 5))

plt.scatter(X_pc2[idx_walk, 0], X_pc2[idx_walk, 1],
            c='red',  label='walking',  alpha=0.7)
plt.scatter(X_pc2[idx_jump, 0], X_pc2[idx_jump, 1],
            c='green',label='jumping', alpha=0.7)
plt.scatter(X_pc2[idx_run, 0],  X_pc2[idx_run, 1],
            c='blue', label='running', alpha=0.7)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA (PC1 vs. PC2)')
plt.legend()
plt.grid(True)
plt.show()

##########################
#   3D散点图 (PC1, PC2, PC3)
##########################
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pc3[idx_walk, 0], X_pc3[idx_walk, 1], X_pc3[idx_walk, 2],
           c='red',   label='walking',  alpha=0.7)
ax.scatter(X_pc3[idx_jump, 0], X_pc3[idx_jump, 1], X_pc3[idx_jump, 2],
           c='green', label='jumping', alpha=0.7)
ax.scatter(X_pc3[idx_run, 0],  X_pc3[idx_run, 1],  X_pc3[idx_run, 2],
           c='blue',  label='running', alpha=0.7)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA (PC1, PC2, PC3)')
ax.legend()
plt.show()



# ================================
# ============ Task 3 ============
# ================================

# 假设之前已经做了这些：
# 1) X_train.shape = (114, 1500)  (空间=114, 时间帧=1500)
# 2) 去均值: X_train_centered = X_train - X_mean
# 3) 转置并 PCA.fit:
#      X_centered_T = X_train_centered.T  # (1500, 114)
#      pca = PCA()
#      pca.fit(X_centered_T)

# --------------- 第一步：创建 ground truth labels ---------------
# y_train 大小为 1500，每个元素是对应帧所属动作的标签(0/1/2)
y_train = np.zeros(1500, dtype=int)

# 前 500 帧对应 walking(0), 中间 500 帧对应 jumping(1), 后 500 帧对应 running(2)
y_train[0:500]   = 0  # walking
y_train[500:1000] = 1 # jumping
y_train[1000:1500] = 2 # running

# --------------- 第二步：投影到 PCA 空间（k个主成分） ---------------
X_projected = pca.transform(X_centered_T)  # shape = (1500, 114)

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
    centroid_c = np.mean(class_data, axis=0)
    centroids.append(centroid_c)

# 现在 centroids 就是一个列表，里面有 3 个元素，每个元素是 k 维均值
# centroids[0] -> walking 的质心 (形状: (k,))
# centroids[1] -> jumping 的质心 (形状: (k,))
# centroids[2] -> running 的质心 (形状: (k,))

print("=== Movement Centroids in PCA Space (k={} modes) ===".format(k))
for label, c in zip(["walking","jumping","running"], centroids):
    print(f"- {label} centroid: {c}")