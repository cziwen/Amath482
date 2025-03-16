import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

# ========== 1. 读取数据，绘制两张初始趋势图 ==========

# 读取 CSV 文件
file_path = "amath383Finaldata/KC15March.csv"  # 替换为你的实际文件路径
df = pd.read_csv (file_path, parse_dates=['Date'])  # 解析日期列
df.set_index ("Date", inplace=True)

# 绘制趋势图 1：Total Cases, Total Deaths, Total recovered
plt.figure (figsize=(12, 6))
sns.lineplot (data=df, x=df.index, y='Total Cases', label="Total Cases")
sns.lineplot (data=df, x=df.index, y='Total Deaths', label="Total Deaths")
sns.lineplot (data=df, x=df.index, y='Total recovered', label="Total Recovered")
plt.xlabel ("Date")
plt.ylabel ("Count")
plt.title ("COVID-19 Data Trends Over Time (1)")
plt.xticks (rotation=45)
plt.legend ()
plt.grid ()
# plt.show ()

# 绘制趋势图 2：New Cases, New Deaths, Daily Tested（若有）
plt.figure (figsize=(12, 6))
sns.lineplot (data=df, x=df.index, y='New Cases', label="New Cases")
sns.lineplot (data=df, x=df.index, y='New Deaths', label="New Deaths")

# 如果你有 Daily Tested 或 Total Daily tested，可自行启用/修改
if 'Daily Tested' in df.columns:
    sns.lineplot (data=df, x=df.index, y='Daily Tested', label="Daily Tested", linestyle='dashed')

plt.xlabel ("Date")
plt.ylabel ("Count")
plt.title ("COVID-19 Data Trends Over Time (2)")
plt.xticks (rotation=45)
plt.legend ()
plt.grid ()
# plt.show ()

# ========== 2. 计算 Infected、Susceptible、Beta、Gamma、R_0 ==========

# （如无再感染，曾确诊者都不再是易感者）
N = 156607  # 替换为你的城市/地区人口数

# 易感人数 S = N - 累计确诊
df["Susceptible"] = N - df["Total Cases"] - df["Total Deaths"]

# 当前活跃感染数 (Active Cases) = 累计确诊 - 累计康复 - 累计死亡
df["Infected"] = df["Total Cases"] - df["Total recovered"] - df["Total Deaths"]

df["Recovered"] = df["Total recovered"] - df["Total Deaths"]

# 每日新增确诊 (ΔI)
df["Delta I"] = df["New Cases"].fillna (0)

# 每日新增康复 (ΔR) = 当天康复 - 前一天康复
df["Delta R"] = df["Total recovered"].diff ().fillna (0)

# 计算每日感染率和每日恢复率
# Beta_t = ΔI / (S * I)
df["Daily Beta"] = df["Delta I"] / (df["Susceptible"] * df["Infected"])

# Gamma_t = ΔR / I
df["Daily Gamma"] = df["Delta R"] / df["Infected"]

# 清理无穷大与空值
df["Daily Beta"] = df["Daily Beta"].replace ([np.inf, -np.inf], np.nan).fillna (0)
df["Daily Gamma"] = df["Daily Gamma"].replace ([np.inf, -np.inf], np.nan).fillna (0)

# df.to_csv("KC15March_modified.csv")

# print("Max Susceptible", df["Susceptible"].max())
# print("Min Susceptible", df["Susceptible"].min())
# print("mean Susceptible", df["Susceptible"].mean())
# print("median Susceptible", df["Susceptible"].median())
#
# print("Max Infected", df["Infected"].max())
# print("Min Infected", df["Infected"].min())
# print("mean Infected", df["Infected"].mean())
# print("median Infected", df["Infected"].median())
#
# print("Max Recovered", df["Recovered"].max())
# print("Min Recovered", df["Recovered"].min())
# print("mean Recovered", df["Recovered"].mean())
# print("median Recovered", df["Recovered"].median())


# 平均 Beta, 平均 Gamma
avg_beta = df["Daily Beta"].mean ()
avg_gamma = df["Daily Gamma"].mean ()
R_0 = avg_beta / avg_gamma if avg_gamma != 0 else np.nan

print (f"平均感染率 Beta: {avg_beta:.6f}")
print (f"平均恢复率 Gamma: {avg_gamma:.6f}")
print (f"基本再生数 R_0: {R_0:.6f}")

# ========== 3. 将 S、I、R 绘制在一张图上 (实际的列) ==========

plt.figure (figsize=(12, 6))
sns.lineplot (x=df.index, y=df["Susceptible"], label="Susceptible (S)")
sns.lineplot (x=df.index, y=df["Infected"], label="Infected (I)")
sns.lineplot (x=df.index, y=df["Total recovered"], label="Recovered (R)")
plt.title ("S, I, R Over Time (From Data)")
plt.xlabel ("Date")
plt.ylabel ("Population")
plt.xticks (rotation=45)
plt.legend ()
plt.grid ()
plt.show ()


# ========== 4. 使用 solve_ivp 求解 SIR，并与真实数据对比 ==========
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# 选择起始天，例如 100
start_day_index = 200
start_date = df.index[start_day_index]

# 初始状态 (S0, I0, R0) 来自 start_date 那一天
S0 = df.loc[start_date, "Susceptible"]
I0 = df.loc[start_date, "Infected"]
R0 = df.loc[start_date, "Total recovered"]  # or add Deaths if you treat them as "removed"

days = len(df) - start_day_index  # 总共要模拟的天数

t_span = (0, days - 1)
t_eval = np.linspace(0, days - 1, days)

solution = solve_ivp(
    fun=sir_model,
    t_span=t_span,
    y0=[S0, I0, R0],
    args=(avg_beta, avg_gamma),
    method='RK45',
    t_eval=t_eval
)

S_pred, I_pred, R_pred = solution.y

# 把时间映射回到真实日期 (只从 start_day_index 开始)
predict_dates = df.index[start_day_index:start_day_index+days]
df_pred = pd.DataFrame({"S": S_pred, "I": I_pred, "R": R_pred}, index=predict_dates)

# ---------- 核心：只绘制从 start_date 开始的实际数据，以保证对比一致 ----------
actual_df = df.loc[start_date:]  # 只取从 start_date 到结尾的部分

plt.figure(figsize=(12,6))

# 真实 Infected（从 start_date 开始）
# sns.lineplot(x=actual_df.index, y=actual_df["Infected"], label="Actual Infected", color="red")
# 预测 Infected
# sns.lineplot(x=df_pred.index, y=df_pred["I"], label="SIR Predicted I", color="blue")

# # S
# sns.lineplot(x=actual_df.index, y=actual_df["Susceptible"], label="Actual Susceptible(S)", color="red")
# # 预测 Infected
# sns.lineplot(x=df_pred.index, y=df_pred["S"], label="SIR Predicted S", color="blue")


# # I
# sns.lineplot(x=actual_df.index, y=actual_df["Infected"], label="Actual Infected(I)", color="green")
# # 预测 Infected
# sns.lineplot(x=df_pred.index, y=df_pred["I"], label="SIR Predicted I", color="grey")

# R
sns.lineplot(x=actual_df.index, y=actual_df["Total recovered"], label="Actual Recovered(R)", color="orange")
# 预测 Infected
sns.lineplot(x=df_pred.index, y=df_pred["R"], label="SIR Predicted R", color="purple")



plt.title("SIR Prediction vs Actual (From Specific Date)")
plt.xlabel("Date")
plt.ylabel("Population")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()
