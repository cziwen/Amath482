import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== Exercise 1 =====
# 定义时间范围
t = np.linspace(0, 2, 100)  # 从 0 到 2，取 100 个点

# 定义已求解的 P(t) 和 Q(t) 公式
Q0 = 1
P_t = (3 * Q0 / 4) * np.exp(-2 * t) - (3 * Q0 / 4) * np.exp(-6 * t)
Q_t = (3 * Q0 / 4) * np.exp(-2 * t) + (Q0 / 4) * np.exp(-6 * t)

# 绘制曲线
plt.figure(figsize=(8, 5))
plt.plot(t, P_t, label="Bloodstream Concentration $P(t)$", linestyle='-', color='b')
plt.plot(t, Q_t, label="Target Muscle Concentration $Q(t)$", linestyle='--', color='r')

# 设置图表标题和标签
plt.title("Lidocaine Concentration Over Time")
plt.xlabel("Time (t)")
plt.ylabel("Concentration")
plt.legend()
plt.grid()

# 显示图表
plt.show()


# ===== Exercise 3 =====
# Define the system matrix A
A = np.array([[-1, 2],
              [-2, 1]])

# Define the ODE system
def ode_system(t, y):
    return A @ y

# Time span and initial condition
T = 5
num_steps = 100
t_eval = np.linspace(0, T, num_steps)
init_cond = np.array([1, 0])

# Solve the ODE
sol = solve_ivp(ode_system, [0, T], init_cond, t_eval=t_eval)

# Extract solutions
r_sol, j_sol = sol.y

# Create a meshgrid for vector field within [-5,5]
P_vals = np.linspace(-5, 5, 20)
j_vals = np.linspace(-5, 5, 20)
P, Q = np.meshgrid(P_vals, j_vals)

# Compute vector field
dP = A[0, 0] * P + A[0, 1] * Q
dQ = A[1, 0] * P + A[1, 1] * Q

# Normalize vectors for better visualization
magnitude = np.sqrt(dP**2 + dQ**2)
dP /= magnitude
dQ /= magnitude

# Plot the vector field
plt.figure(figsize=(8, 6))
plt.quiver(P, Q, dP, dQ, color='blue', alpha=0.5)

# Superimpose the trajectory
plt.plot(r_sol, j_sol, 'r-', linewidth=2, label="Trajectory of x(t)")

# Labels and title
plt.xlabel("P")
plt.ylabel("Q")
plt.title("Vector Field and Trajectory")

# Limit axis range to [-5, 5]
plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.legend()
plt.grid()
plt.show()



# ===== Exercise 4 =====
# Define the system matrix A
A = np.array([[1, 2],
              [4, -1]])

# Define the ODE system
def ode_system(t, y):
    return A @ y

# Time span and initial condition
T = 5
num_steps = 100
t_eval = np.linspace(0, T, num_steps)
init_cond = np.array([1, 0])

# Solve the ODE
sol = solve_ivp(ode_system, [0, T], init_cond, t_eval=t_eval)

# Extract solutions
r_sol, j_sol = sol.y

# Create a meshgrid for vector field within [-5,5]
r_vals = np.linspace(-5, 5, 20)
j_vals = np.linspace(-5, 5, 20)
r, j = np.meshgrid(r_vals, j_vals)

# Compute vector field
dr = A[0, 0] * r + A[0, 1] * j
dj = A[1, 0] * r + A[1, 1] * j

# Normalize vectors for better visualization
magnitude = np.sqrt(dr ** 2 + dj ** 2)
dr /= magnitude
dj /= magnitude

# Plot the vector field
plt.figure(figsize=(8, 6))
plt.quiver(r, j, dr, dj, color='blue', alpha=0.5)

# Superimpose the trajectory
# plt.plot(r_sol, j_sol, 'r-', linewidth=2, label="Trajectory of x(t)")

# Labels and title
plt.xlabel("Romeo's love")
plt.ylabel("Juliet's love")
plt.title("Vector Field and Trajectory")

# Limit axis range to [-5, 5]
plt.xlim([-5, 5])
plt.ylim([-5, 5])

# plt.legend()
plt.grid()
plt.show()