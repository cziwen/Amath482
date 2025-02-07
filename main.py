import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------
# 1. 定义问题参数及微分方程
# ---------------------------
T_F = 1.5  # 给定 T_F
P0 = 0.5  # 初始条件 P(0) = 0.5
t0, t_end = 0, 10  # 时间区间 [0, 10]
N = 200
dt = (t_end - t0) / N  # 每步长度


def finite_difference (e1, e2, h1, h2):
    """
    利用有限差方计算 阶 数。
    :return: 数字越大，阶数越高，方程收敛性更快
    """
    return np.abs (np.log (e2 / e1) / np.log (h2 / h1))


def P_exact (t):
    """精确解"""
    return np.exp (t) / (1 + np.exp (t))


def maxDiff (P_exact, P_numerical):
    """
    两个传进来的数组应是尺寸对应，t对应的数组
    """
    abs_diff = np.abs (P_exact - P_numerical)
    return np.max (abs_diff)


def dP_dt (t, P):
    """微分方程"""
    return P * (1 - P)


# ---------------------------
# 2. Euler 方法
# ---------------------------
t_euler = np.linspace (t0, t_end, N + 1)
P_euler = np.zeros (N + 1)
P_euler[0] = P0

for i in range (N):
    P_euler[i + 1] = P_euler[i] + dt * dP_dt (t_euler[i], P_euler[i])

# ---------------------------
# 3. Midpoint 方法
# ---------------------------
t_mid = np.linspace (t0, t_end, N + 1)
P_mid = np.zeros (N + 1)
P_mid[0] = P0

for i in range (N):
    k1 = dP_dt (t_mid[i], P_mid[i])
    P_half = P_mid[i] + 0.5 * dt * k1
    k2 = dP_dt (t_mid[i] + 0.5 * dt, P_half)
    P_mid[i + 1] = P_mid[i] + dt * k2

# ---------------------------
# 4. 经典四阶 Runge-Kutta (RK4)
# ---------------------------
t_rk4 = np.linspace (t0, t_end, N + 1)
P_rk4 = np.zeros (N + 1)
P_rk4[0] = P0

for i in range (N):
    k1 = dP_dt (t_rk4[i], P_rk4[i])
    k2 = dP_dt (t_rk4[i] + dt / 2, P_rk4[i] + dt * k1 / 2)
    k3 = dP_dt (t_rk4[i] + dt / 2, P_rk4[i] + dt * k2 / 2)
    k4 = dP_dt (t_rk4[i] + dt, P_rk4[i] + dt * k3)
    P_rk4[i + 1] = P_rk4[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------
# 5. 调用 solve_ivp (类似于MATLAB ode45)
# ---------------------------
def wrapper (t, P):
    return dP_dt (t, P)


sol_ode45 = solve_ivp (wrapper, (t0, t_end), [P0], method='RK45',
                       t_eval=np.linspace (t0, t_end, N + 1))

t_ode45 = sol_ode45.t
P_ode45 = sol_ode45.y[0]

# ---------------------------
# 6. 计算最大误差
# ---------------------------
P_exact_euler = P_exact (t_euler)
P_exact_mid = P_exact (t_mid)
P_exact_rk4 = P_exact (t_rk4)
P_exact_ode45 = P_exact (t_ode45)

error_euler = maxDiff (P_exact_euler, P_euler)
error_mid = maxDiff (P_exact_mid, P_mid)
error_rk4 = maxDiff (P_exact_rk4, P_rk4)
error_ode45 = maxDiff (P_exact_ode45, P_ode45)

print ("Maximum Errors:")
print (f"Euler Method: {error_euler}")
print (f"Midpoint Method: {error_mid}")
print (f"RK4 Method: {error_rk4}")
print (f"ode45 Method: {error_ode45}")

# ---------------------------
# 7. 可视化对比
# ---------------------------
plt.figure (figsize=(8, 6))
plt.plot (t_euler, P_euler, 'o-', label='Euler', alpha=0.7)
plt.plot (t_mid, P_mid, '^-', label='Midpoint', alpha=0.7)
plt.plot (t_rk4, P_rk4, 's-', label='Runge-Kutta (RK4)', alpha=0.7)
plt.plot(t_ode45, P_ode45, 'd-', label='ode45 (solve_ivp)', alpha=0.7)
plt.xlabel ('t')
plt.ylabel ('P(t)')
plt.title (f'Comparison of Numerical Methods N = {N}')
plt.legend ()
plt.grid (True)
plt.tight_layout ()
plt.show ()

# ---------------------------
# 8. 计算有限差方法的阶数
# ---------------------------

# N = 200
e_2s = [0.002318027852772908, 1.870775833445748e-05, 2.9289229930284932e-09]

# N = 20
e_1s = [0.024988303350432117, 0.002390970726064756, 3.656648410399477e-05]

h_2 = 200
h_1 = 20

orders = []
for i in range (len (e_1s)):
    orders.append (finite_difference (e_1s[i], e_2s[i], h_1, h_2))

for i in range (len (orders)):
    print(orders[i])


# ---------------------------
# 9. 绘制bifurcation diagram
# ---------------------------

# 绘制第一张图
# 参数范围
r_left = np.linspace(-5, -2, 1000)  # r ≤ -2
r_right = np.linspace(2, 5, 1000)   # r ≥ 2

# 计算平衡点
def equilibrium(r):
    sqrt_term = np.sqrt(r**2 - 4)
    P_plus = (r + sqrt_term) / 2
    P_minus = (r - sqrt_term) / 2
    return P_plus, P_minus

P_plus_left, P_minus_left = equilibrium(r_left)
P_plus_right, P_minus_right = equilibrium(r_right)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(r_left, P_plus_left, 'r--', label='Unstable')
plt.plot(r_left, P_minus_left, 'b-', label='Stable')
plt.plot(r_right, P_plus_right, 'r--')
plt.plot(r_right, P_minus_right, 'b-')

# 分岔点标记
plt.plot(-2, -1, 'ko', markersize=10)
plt.plot(2, 1, 'ko', markersize=10)

# 添加注释
plt.annotate("(2,1)", (2, 1), xytext=(1, 0.5),
             arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=12)
# 添加注释
plt.annotate("(-2,-1)", (-2, -1), xytext=(-1.5, -0.5),
             arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=12)

plt.xlabel('r')
plt.ylabel('P')
plt.title('Bifurcation Diagram: dP/dt = 1 - rP + P²')
plt.legend()
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()



# 绘制第二张图
r = np.linspace(-5, 5, 1000)

# 平衡点 P=0 的稳定性分段
r_stable_P0 = np.linspace(-5, 1, 500)
r_unstable_P0 = np.linspace(1, 5, 500)

# 平衡点 P=ln(r)（仅 r>0）
r_pos = r[r > 0]
P_ln = np.log(r_pos)

# 分割 r>0 的区域：0 < r <1 和 r >=1
mask_low = r_pos < 1
r_low = r_pos[mask_low]
P_ln_low = P_ln[mask_low]

r_high = r_pos[~mask_low]
P_ln_high = P_ln[~mask_low]

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(r_stable_P0, np.zeros_like(r_stable_P0), 'b-', lw=2, label='Stable')
plt.plot(r_unstable_P0, np.zeros_like(r_unstable_P0), 'r--', lw=2, label='Unstable')
plt.plot(r_low, P_ln_low, 'r--', lw=2)
plt.plot(r_high, P_ln_high, 'b-', lw=2)


plt.plot(1, 0, 'ko', markersize=8, label='Bifurcation (r=1)')
plt.annotate("(1,0)", (1, 0), xytext=(0.5, 1.5),
             arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=12)

plt.xlabel('r')
plt.ylabel('P')
plt.title('Bifurcation Diagram: dP/dt = P(r - e^P)')
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.legend()
plt.show()



