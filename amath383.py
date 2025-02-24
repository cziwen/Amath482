import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ===== ========== ========== ========== =====
# Define the system matrix A
# A = np.array([[0, -1],
#               [-1, 0]])
#
# # Define the ODE system
# def ode_system(t, y):
#     return A @ y
#
# # Time span and initial condition
# T = 5
# num_steps = 100
# t_eval = np.linspace(0, T, num_steps)
# init_cond = np.array([1, 0])
#
# # Solve the ODE
# sol = solve_ivp(ode_system, [0, T], init_cond, t_eval=t_eval)
#
# # Extract solutions
# r_sol, j_sol = sol.y
#
# # Create a meshgrid for vector field within [-5,5]
# r_vals = np.linspace(-5, 5, 20)
# j_vals = np.linspace(-5, 5, 20)
# r, j = np.meshgrid(r_vals, j_vals)
#
# # Compute vector field
# dr = A[0, 0] * r + A[0, 1] * j
# dj = A[1, 0] * r + A[1, 1] * j
#
# # Normalize vectors for better visualization
# magnitude = np.sqrt(dr ** 2 + dj ** 2)
# dr /= magnitude
# dj /= magnitude
#
# # Plot the vector field
# plt.figure(figsize=(8, 6))
# plt.quiver(r, j, dr, dj, color='blue', alpha=0.5)
#
# # Superimpose the trajectory
# # plt.plot(r_sol, j_sol, 'r-', linewidth=2, label="Trajectory of x(t)")
#
# # Labels and title
# plt.xlabel("U")
# plt.ylabel("C")
# plt.title("Vector Field and Trajectory")
#
# # Limit axis range to [-5, 5]
# plt.xlim([-5, 5])
# plt.ylim([-5, 5])
#
# # plt.legend()
# plt.grid()
# plt.show()


# ===== ========== ========== ========== ========== =====
# 定义 Lotka-Volterra 方程（α = β = γ = δ = 1）
def lotka_volterra(t, y):
    P, Q = y
    dP_dt = P * (1 - Q)
    dQ_dt = Q * (P - 1)
    return [dP_dt, dQ_dt]

# 定义相空间网格
P_values = np.linspace(0, 2, 20)
Q_values = np.linspace(0, 2, 20)
P, Q = np.meshgrid(P_values, Q_values)

# 计算向量场
dP, dQ = lotka_volterra(0, [P, Q])

# 时间求解设置
t_span = (0, 4 * np.pi)
t_eval = np.linspace(0, 4 * np.pi, 1000)
y0 = [1, 0.5]
solution = solve_ivp(lotka_volterra, t_span, y0, t_eval=t_eval, method='RK45')

# Solve the system using solve_ivp (Runge-Kutta method)
solution = solve_ivp(lotka_volterra, t_span, y0, t_eval=t_eval, method='RK45')






# 绘制 两个solution 在时间轴t上的变化
plt.figure(figsize=(8, 5))
plt.plot(solution.t, solution.y[0], label='Prey Population (P)')
plt.plot(solution.t, solution.y[1], label='Predator Population (Q)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Lotka-Volterra System Solution')
plt.legend()
plt.grid()
plt.show()


# 绘制向量场和解曲线
plt.figure(figsize=(8, 6))
plt.quiver(P, Q, dP, dQ, color='b', angles='xy', alpha=0.5)
plt.plot(solution.y[0], solution.y[1], 'r', label='Solution Trajectory')

# 设置标签和标题
plt.xlabel('Prey Population (P)')
plt.ylabel('Predator Population (Q)')
plt.title('Lotka-Volterra Vector Field and Solution')
plt.xlim([0, 2])
plt.ylim([0, 2])
plt.legend()
plt.grid()

# 显示图像
plt.show()



# 绘制向量场和解曲线
plt.figure(figsize=(8, 6))
plt.quiver(P, Q, dP, dQ, color='b', angles='xy', alpha=0.2)
# plt.plot(solution.y[0], solution.y[1], 'r', label='Solution Trajectory', linewidth=2)

# 画出等值曲线
E_values = [2.01, 2.05, 2.1, 2.15, 2.2, 2.25]
P_grid = np.linspace(0.1, 2, 100)
Q_grid = np.linspace(0.1, 2, 100)
P_mesh, Q_mesh = np.meshgrid(P_grid, Q_grid)
contour_function = P_mesh - np.log(P_mesh) + Q_mesh - np.log(Q_mesh)
contours = plt.contour(P_mesh, Q_mesh, contour_function, levels=E_values, colors='g', linestyles='dashed')
plt.clabel(contours, inline=True, fontsize=10, fmt={E: f"E={E}" for E in E_values})
# 设置标签和标题
plt.xlabel('Prey Population (P)')
plt.ylabel('Predator Population (Q)')
plt.title('Lotka-Volterra Vector Field, Solution, and Trajectories')
plt.xlim([0, 2])
plt.ylim([0, 2])
plt.legend()
plt.grid()
# 显示图像
plt.show()
