import numpy as np
import matplotlib.pyplot as plt

# ===== 參數設定 =====
W = 50  # 重量 [k]
g = 386.1  # 重力加速度 [in/s²]
m = W / g  # 質量 [k·s²/in]
k = 100  # 彈簧剛度 [k/in]
xi = 0.12  # 阻尼比
c = 2 * m * np.sqrt(k / m) * xi  # 阻尼係數 [k·s/in]
ag = 0.25 * g  # 峰值地面加速度 [in/s²]  # 為了清晰起見，更改了變數名稱
dt = 0.01  # 時間步長 [s]
num_steps = 6  # 前 6 個時間步
t = np.arange(0, num_steps * dt, dt)  # 時間向量

# ===== 初始化陣列 =====
def init_arrays(num_steps):  # 新增 num_steps 作為引數
    return (
        np.zeros(num_steps),  # 位移 x
        np.zeros(num_steps),  # 速度 x_dot
        np.zeros(num_steps),  # 加速度 x_ddot
        np.zeros(num_steps),  # 有效力 F_eff
    )

# ===== Wilson-θ 方法 =====
theta = 1.4
omega_n = np.sqrt(k / m)
x_w, x_dot_w, x_ddot_w, F_eff_w = init_arrays(num_steps)  # 傳遞 num_steps
x_w[0] = 0
x_dot_w[0] = 0
x_ddot_w[0] = 0

for i in range(1, num_steps):
    F_eff_w[i] = -m * ag  # 已修正地面加速度
    F_dy = F_eff_w[i] - c * x_dot_w[i - 1] - k * x_w[i - 1]
    x_ddot_theta = (F_dy / m)  # 簡化加速度計算

    # Wilson-theta 方法計算
    delta_x_ddot = x_ddot_theta - x_ddot_w[i - 1]
    x_dot_w[i] = x_dot_w[i - 1] + dt / 2 * (x_ddot_w[i - 1] + x_ddot_theta)
    x_w[i] = x_w[i - 1] + dt * x_dot_w[i - 1] + dt**2 / 6 * (3 * x_ddot_w[i - 1] + x_ddot_theta)
    x_ddot_w[i] = x_ddot_theta

# ===== 中央差分法 =====
x_c, x_dot_c, x_ddot_c, F_eff_c = init_arrays(num_steps)  # 傳遞 num_steps
x_c[0] = 0
x_dot_c[0] = 0
x_ddot_c[0] = 0

for i in range(1, num_steps):
    F_eff_c[i] = -m * ag  # 已修正地面加速度
    F_dy = F_eff_c[i] - c * x_dot_c[i - 1] - k * x_c[i - 1]
    x_ddot_c[i] = F_dy / m
    if i > 1:
        x_c[i] = 2 * x_c[i - 1] - x_c[i - 2] + dt**2 * x_ddot_c[i - 1]
    if i > 0 and i < num_steps - 1:  # 已修正條件
        x_dot_c[i] = (x_c[i + 1] - x_c[i - 1]) / (2 * dt)

# ===== 平均加速度法 =====
x_avg, x_dot_avg, x_ddot_avg, F_eff_avg = init_arrays(num_steps)  # 傳遞 num_steps
x_avg[0] = 0
x_dot_avg[0] = 0
x_ddot_avg[0] = 0

for i in range(1, num_steps):
    F_eff_avg[i] = -m * ag  # 已修正地面加速度
    F_dy = F_eff_avg[i] - c * x_dot_avg[i - 1] - k * x_avg[i - 1]
    x_ddot_avg[i] = F_dy / m
    x_dot_avg[i] = x_dot_avg[i - 1] + dt * (x_ddot_avg[i] + x_ddot_avg[i - 1]) / 2
    x_avg[i] = x_avg[i - 1] + dt * x_dot_avg[i - 1] + (dt**2 / 4) * (x_ddot_avg[i - 1] + x_ddot_avg[i])  # 已修正位移計算

# ===== 線性加速度法 =====
x_lin, x_dot_lin, x_ddot_lin, F_eff_lin = init_arrays(num_steps)  # 傳遞 num_steps
x_lin[0] = 0
x_dot_lin[0] = 0
x_ddot_lin[0] = 0

for i in range(1, num_steps):
    F_eff_lin[i] = -m * ag  # 已修正地面加速度
    F_dy = F_eff_lin[i] - c * x_dot_lin[i - 1] - k * x_lin[i - 1]
    x_ddot_lin[i] = F_dy / m
    x_dot_lin[i] = x_dot_lin[i - 1] + dt / 2 * (x_ddot_lin[i] + x_ddot_lin[i - 1])
    x_lin[i] = x_lin[i - 1] + dt * x_dot_lin[i - 1] + (dt**2 / 6) * (x_ddot_lin[i] + 2 * x_ddot_lin[i - 1])

# ===== 印出前六個時間步 =====
print("First Six Time Steps:")
print(f"Time: {t}")

print("\nWilson-θ Method:")
print(f"  F_eff: {F_eff_w}")
print(f"  x:     {x_w}")
print(f"  x_dot: {x_dot_w}")
print(f"  x_ddot:{x_ddot_w}")

print("\nCentral Difference Method:")
print(f"  F_eff: {F_eff_c}")
print(f"  x:     {x_c}")
print(f"  x_dot: {x_dot_c}")
print(f"  x_ddot:{x_ddot_c}")

print("\nAverage Acceleration Method:")
print(f"  F_eff: {F_eff_avg}")
print(f"  x:     {x_avg}")
print(f"  x_dot: {x_dot_avg}")
print(f"  x_ddot:{x_ddot_avg}")

print("\nLinear Acceleration Method:")
print(f"  F_eff: {F_eff_lin}")
print(f"  x:     {x_lin}")
print(f"  x_dot: {x_dot_lin}")
print(f"  x_ddot:{x_ddot_lin}")

# ===== 繪製結果 =====
fig, axs = plt.subplots(4, 1, figsize=(10, 14))

# Displacement
axs[0].plot(t, x_w, label="Wilson θ", color="tab:green")
axs[0].plot(t, x_c, "--", label="Central Difference", color="tab:orange")
axs[0].plot(t, x_avg, "-.", label="Average Acceleration", color="tab:blue")
axs[0].plot(t, x_lin, ":", label="Linear Acceleration", color="tab:purple")
axs[0].set_ylabel("Displacement x(t) [in]")
axs[0].legend()
axs[0].grid()

# Velocity
axs[1].plot(t, x_dot_w, label="Wilson θ", color="tab:green")
axs[1].plot(t, x_dot_c, "--", label="Central Difference", color="tab:orange")
axs[1].plot(t, x_dot_avg, "-.", label="Average Acceleration", color="tab:blue")
axs[1].plot(t, x_dot_lin, ":", label="Linear Acceleration", color="tab:purple")
axs[1].set_ylabel("Velocity x_dot(t) [in/s]")
axs[1].legend()
axs[1].grid()

# Acceleration
axs[2].plot(t, x_ddot_w, label="Wilson θ", color="tab:green")
axs[2].plot(t, x_ddot_c, "--", label="Central Difference", color="tab:orange")
axs[2].plot(t, x_ddot_avg, "-.", label="Average Acceleration", color="tab:blue")
axs[2].plot(t, x_ddot_lin, ":", label="Linear Acceleration", color="tab:purple")
axs[2].set_ylabel("Acceleration x_ddot(t) [in/s²]")
axs[2].legend()
axs[2].grid()

# Effective Force
axs[3].plot(t, F_eff_w, label="Wilson θ", color="tab:green")
axs[3].plot(t, F_eff_c, "--", label="Central Difference", color="tab:orange")
axs[3].plot(t, F_eff_avg, "-.", label="Average Acceleration", color="tab:blue")
axs[3].plot(t, F_eff_lin, ":", label="Linear Acceleration", color="tab:purple")
axs[3].set_ylabel("Effective Force F_eff(t) [k]")
axs[3].set_xlabel("Time t [s]")
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()