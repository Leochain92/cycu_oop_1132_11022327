import numpy as np
from scipy.integrate import odeint

# 載入地震資料
data = np.loadtxt('c:/Users/truck/OneDrive/文件/GitHub/cycu_oop_1132_11022327/地震工程/Northridge_NS.txt')  # 使用正斜線 
time = data[:, 0]
acc_g = data[:, 1]

# 給定的 SDOF 系統參數
m = 290 / 386.4  # 質量 (ksi⋅s^2/in)  # 將 ksi/g 轉換為 ksi⋅s^2/in
zeta = 0.05  # 阻尼比  # 使用統一的阻尼比
k_NS = 92  # NS 方向剛度 (ksi/in)
k_EW = 302  # EW 方向剛度 (ksi/in)
T_NS = 0.57  # NS 方向週期 (s)  # 給定，但程式中未使用
T_EW = 0.31  # EW 方向週期 (s)  # 給定，但程式中未使用
g = 386.4  # 重力加速度 (in/s^2)
dt = 0.02  # 時間間隔 (s)  # 給定，但程式中未使用
L_Columns = 168  # 柱長 (in)
L_Braces = 343.84  # 支撐長度 (in)

# 從週期計算自振頻率 (雖然給定了週期，但我們通常用剛度直接算，這裡驗算)
omega_n_NS = np.sqrt(k_NS / m)  # NS 方向自振頻率 (rad/s)
omega_n_EW = np.sqrt(k_EW / m)  # EW 方向自振頻率 (rad/s)

# 計算阻尼係數 c
c_NS = 2 * zeta * omega_n_NS * m
c_EW = 2 * zeta * omega_n_EW * m

# 定義 SDOF 系統的運動方程式
def eq_of_motion(y, t, acc_g, m, c, k):
    x, x_dot = y
    # 獲取當前時間點的地震加速度
    acc_g_t = np.interp(t, time, acc_g)  # 使用線性插值獲取對應時間的加速度值
    x_ddot = - (c / m) * x_dot - (k / m) * x - acc_g_t * g  # 注意：地震加速度需要乘以 g
    return [x_dot, x_ddot]

# 求解 NS 方向的運動
y0 = [0, 0]
solution_NS = odeint(eq_of_motion, y0, time, args=(acc_g, m, c_NS, k_NS))
x_NS, x_dot_NS = solution_NS[:, 0], solution_NS[:, 1]
x_ddot_NS = np.array([eq_of_motion([x, x_dot], t, acc_g, m, c_NS, k_NS)[1] for x, x_dot, t in zip(x_NS, x_dot_NS, time)])

# 求解 EW 方向的運動
solution_EW = odeint(eq_of_motion, y0, time, args=(acc_g, m, c_EW, k_EW))
x_EW, x_dot_EW = solution_EW[:, 0], solution_EW[:, 1]
x_ddot_EW = np.array([eq_of_motion([x, x_dot], t, acc_g, m, c_EW, k_EW)[1] for x, x_dot, t in zip(x_EW, x_dot_EW, time)])

# 計算最大值
x_max_NS = np.max(np.abs(x_NS))
x_dot_max_NS = np.max(np.abs(x_dot_NS))
x_ddot_max_NS = np.max(np.abs(x_ddot_NS))

x_max_EW = np.max(np.abs(x_EW))
x_dot_max_EW = np.max(np.abs(x_dot_EW))
x_ddot_max_EW = np.max(np.abs(x_ddot_EW))

# 計算最大剪力和彎矩 (使用你提供的公式)
V_max_NS = k_NS * x_max_NS
M_max_NS = V_max_NS * L_Columns

V_max_EW = k_EW * x_max_EW
M_max_EW = V_max_EW * L_Braces

# 輸出結果
print("NS 方向:")
print("|x(t)|_max =", x_max_NS, "in")
print("|ẋ(t)|_max =", x_dot_max_NS, "in/s")
print("|ẍ(t)|_max =", x_ddot_max_NS, "in/s^2")
print("|V(t)|_max =", V_max_NS, "ksi")
print("|M(t)|_max =", M_max_NS, "ksi⋅in")

print("\nEW 方向:")
print("|x(t)|_max =", x_max_EW, "in")
print("|ẋ(t)|_max =", x_dot_max_EW, "in/s")
print("|ẍ(t)|_max =", x_ddot_max_EW, "in/s^2")
print("|V(t)|_max =", V_max_EW, "ksi")
print("|M(t)|_max =", M_max_EW, "ksi⋅in")
