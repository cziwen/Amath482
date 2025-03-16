import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import matplotlib.pyplot as plt

# ========== 1) 准备数据 ==========
# df里必须含有 columns: ['Infected', 'Recovered', 'Susceptible']
# 以及时间点: t_data
# 这里演示
t_data = np.arange(0, 30, 1)  # 0..29天
N_data_points = len(t_data)

# 假设我们有真值(或观测值)
# (在实际里你要从CSV读，然后做 Infected=Total Cases - Recovered - Deaths 等)
I_data = 10 + 0.5*t_data  # 只是随时间线性增加的假设, 仅用来演示
R_data = 0.05*(t_data**2) # 随机编造
S_data = 1000 - I_data - R_data
df = pd.DataFrame({"t": t_data, "Infected": I_data, "Recovered": R_data, "Susceptible": S_data})


# ========== 2) 定义 SIR + 强制项 (分段常数 f,g,h) ==========
#   dS/dt = -beta*S*I + f_k
#   dI/dt =  beta*S*I - gamma*I + g_k
#   dR/dt =  gamma*I + h_k
#  with  f_k + g_k + h_k = 0  in each day k

beta_0 = 0.2   # 也可让beta_0成为待估参数
gamma_0 = 0.1  # 同上

def piecewise_forcing(t, params):
    """
    params包含 (f_0, g_0, f_1, g_1, ..., f_(N-1), g_(N-1)) for N_data_points-1 maybe
    这里简化假设: 每个整天的区间 [k, k+1) 用 f_k,g_k,h_k
    """
    # 先确定当前所在区间是 day = floor(t)
    k = int(np.floor(t))
    # 容错: 如果 t 超过数据范围, 让 k=最后一段
    if k >= N_data_points:
        k = N_data_points - 1
    # 读取 f_k, g_k
    f_k = params[2*k]
    g_k = params[2*k + 1]
    # 保证 h_k = - (f_k + g_k)
    h_k = -(f_k + g_k)
    return f_k, g_k, h_k

def sir_ode_with_forcing(t, y, params):
    S, I, R = y
    f_k, g_k, h_k = piecewise_forcing(t, params)

    dS = -beta_0*S*I + f_k
    dI =  beta_0*S*I - gamma_0*I + g_k
    dR =  gamma_0*I + h_k
    return [dS, dI, dR]

def simulate_sir_forcing(params):
    """
    给定一组分段常数强制项参数, 在 0..max(t_data) 上解方程
    返回: t_eval, (S,I,R) 的解, 用于跟数据对比
    """
    S0 = S_data[0]
    I0 = I_data[0]
    R0 = R_data[0]
    t_span = (0, t_data[-1])
    sol = solve_ivp(
        fun=lambda t,y: sir_ode_with_forcing(t,y,params),
        t_span=t_span,
        y0=[S0, I0, R0],
        t_eval=t_data,
        vectorized=False
    )
    return sol.t, sol.y  # sol.y shape=(3, len(t_data))


# ========== 3) 定义误差函数, 用于最小化 ==========
def objective(params):
    # params是 2*(N_data_points) 个分段参数(如果天数=30, 就有60个param), 也可减少一天
    # 先跑一次数值解
    t_sol, Y_sol = simulate_sir_forcing(params)
    # 取 I_sol
    I_sol = Y_sol[1,:]
    # 计算 I_sol 与 I_data 的 MSE
    mse_I = np.mean((I_sol - I_data)**2)
    # 也可加上对 R_sol vs R_data 的误差
    R_sol = Y_sol[2,:]
    mse_R = np.mean((R_sol - R_data)**2)
    # 合并
    return mse_I + mse_R

# 我们要优化的初始 guess. 需要 2*N_data_points 大小(每天 f,g)
# 为了简化, 只用 (N_data_points) 而不是 (N_data_points-1). 你可以调整
init_guess = np.zeros(2*N_data_points)

# ========== 4) 调用优化器 ==========
res = opt.minimize(objective, init_guess, method='L-BFGS-B', options={'maxiter':200})

print("Optimization success?", res.success)
print("Final MSE:", res.fun)
best_params = res.x

# ========== 5) 可视化结果 ==========
t_sol, Y_sol = simulate_sir_forcing(best_params)
S_sol, I_sol, R_sol = Y_sol

plt.figure(figsize=(9,5))
plt.plot(t_data, I_data, 'ro', label="I data")
plt.plot(t_sol, I_sol, 'b-', label="I model w forcing")
plt.plot(t_data, R_data, 'go', label="R data")
plt.plot(t_sol, R_sol, 'c--', label="R model w forcing")
plt.legend()
plt.title("SIR + piecewise forcing fit data")
plt.xlabel("Time (days)")
plt.grid(True)
plt.show()

# 还可以看看每天的 f_k,g_k,h_k
f_list = []
g_list = []
h_list = []
for k in range(N_data_points):
    f_k = best_params[2*k]
    g_k = best_params[2*k+1]
    h_k = -(f_k+g_k)
    f_list.append(f_k)
    g_list.append(g_k)
    h_list.append(h_k)

plt.figure(figsize=(9,4))
plt.plot(range(N_data_points), f_list, label="f_k")
plt.plot(range(N_data_points), g_list, label="g_k")
plt.plot(range(N_data_points), h_list, label="h_k")
plt.title("Piecewise forcing terms")
plt.legend()
plt.grid(True)
plt.show()