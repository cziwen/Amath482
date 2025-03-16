import numpy as np
import matplotlib.pyplot as plt


# 定义递推关系
def f (p, A, B):
    return p * (A + B * p ** 2)


# 给定参数
A = 3 / 2
B = -1

# 初始条件
p0_values = [1, -1]

# 迭代次数
num_iterations = 10

# 生成 Cobweb 图
fig, axes = plt.subplots (1, 2, figsize=(12, 5))

for i, p0 in enumerate (p0_values):
    # 计算迭代序列
    p_values = [p0]
    for _ in range (num_iterations):
        p_values.append (f (p_values[-1], A, B))

    # 生成 Cobweb 线条
    p_range = np.linspace (-2, 2, 400)
    p_next = f (p_range, A, B)

    ax = axes[i]
    ax.plot (p_range, p_next, label=r'$p_{n+1} = p_n(A + Bp_n^2)$', color='blue')
    ax.plot (p_range, p_range, '--', color='gray', label=r'$p_{n+1} = p_n$')

    # 绘制 Cobweb 迭代过程
    p_cobweb = [p_values[0]]
    for j in range (1, len (p_values)):
        p_cobweb.extend ([p_values[j - 1], p_values[j]])

    ax.plot (p_cobweb[:-1], p_cobweb[1:], color='red', linestyle='-', marker='o')

    # 轴标签和标题
    ax.set_xlabel (r'$p_n$')
    ax.set_ylabel (r'$p_{n+1}$')
    ax.set_title (f'Cobweb Diagram for $p_0 = {p0}$')
    ax.legend ()

plt.tight_layout ()
plt.show ()