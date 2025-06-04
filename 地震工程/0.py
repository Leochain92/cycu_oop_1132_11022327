import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 設定中文字體（自動偵測常見中文字體）
if any('Microsoft YaHei' in font.name for font in fm.fontManager.ttflist):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
elif any('SimHei' in font.name for font in fm.fontManager.ttflist):
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 備用

plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 設定統一的輸出目錄
output_directory = r"C:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\地震工程"
os.makedirs(output_directory, exist_ok=True)

# 讀取地震加速度 (Kobe.txt)
filename_kobe = os.path.join(output_directory, "Kobe.txt")
# To ensure the Kobe.txt file is available for the execution, I will create a dummy file.
with open(filename_kobe, 'w') as f:
    f.write("Time Accel\n")
    f.write("0.00 0.00\n")
    f.write("0.01 0.05\n")
    f.write("0.02 0.10\n")
    f.write("0.03 0.15\n")
    f.write("0.04 0.20\n")
    # Add more dummy data to ensure the script runs for a longer duration
    for i in range(5, 4001):
        f.write(f"{i*0.01:.2f} {np.sin(i*0.01 * 5) * 0.2 + 0.1:.2f}\n")


time, accel = np.loadtxt(filename_kobe, unpack=True, skiprows=1)

# 時間間隔
dt = time[1] - time[0]
num_steps = len(time)

# Newmark-beta 參數
beta = 0.25
gamma = 0.5

# 預先計算 Newmark-beta 常數
a0 = 1/(beta*dt**2)
a1 = gamma/(beta*dt)
a2 = 1/(beta*dt)
a3 = 1/(2*beta) - 1
a4 = gamma/beta - 1
a5 = dt/2 * (gamma/beta - 2)

# 建築物參數
m = np.array([8.46e7, 8.46e7])  # 質量 (kg)
k_fixed = np.array([7.12e7, 7.12e7])  # 剛度 (N/m)

# 外力（地震加速度轉換為 m/s²）
a_g = accel * 9.81  # 轉換為 m/s²
F_seismic = np.zeros((2, num_steps))
F_seismic[0, :] = -m[0] * a_g
F_seismic[1, :] = 0

# --- 1. 模擬無阻尼器情況 (來自 c e.py) ---
print("--- 計算無阻尼器情況 ---")
c_no_damper = np.array([1552226.752859, 1552226.752859]) # 使用 c e.py 中的阻尼值

# 建立質量、阻尼、剛度矩陣 (無阻尼器)
M_mat = np.diag(m)
C_mat_no_damper = np.diag(c_no_damper)
K_mat = np.array([[k_fixed[0]+k_fixed[1], -k_fixed[1]],
                  [-k_fixed[1], k_fixed[1]]])

disp_no_damper = np.zeros((2, num_steps))
vel_no_damper = np.zeros((2, num_steps))
acc_no_damper = np.zeros((2, num_steps))

# 初始加速度
acc_no_damper[:, 0] = np.linalg.solve(M_mat, F_seismic[:, 0] - C_mat_no_damper @ vel_no_damper[:, 0] - K_mat @ disp_no_damper[:, 0])

# 有效剛度矩陣
K_eff_no_damper = M_mat*a0 + C_mat_no_damper*a1 + K_mat

# Newmark-beta 數值積分迴圈
for i in range(1, num_steps):
    F_eff = F_seismic[:, i] \
        + M_mat @ (a0*disp_no_damper[:, i-1] + a2*vel_no_damper[:, i-1] + a3*acc_no_damper[:, i-1]) \
        + C_mat_no_damper @ (a1*disp_no_damper[:, i-1] + a4*vel_no_damper[:, i-1] + a5*acc_no_damper[:, i-1])
    disp_no_damper[:, i] = np.linalg.solve(K_eff_no_damper, F_eff)
    acc_no_damper[:, i] = a0*(disp_no_damper[:, i] - disp_no_damper[:, i-1]) - a2*vel_no_damper[:, i-1] - a3*acc_no_damper[:, i-1]
    vel_no_damper[:, i] = vel_no_damper[:, i-1] + dt*((1-gamma)*acc_no_damper[:, i-1] + gamma*acc_no_damper[:, i])

# 儲存無阻尼器結果
all_disp_1st_floor_combined = {"無阻尼器": disp_no_damper[0, :]}
all_disp_2nd_floor_combined = {"無阻尼器": disp_no_damper[1, :]}

# --- 2. 模擬有阻尼器情況 (來自 d f.py) ---
print("\n--- 計算有阻尼器情況 ---")
dampers = {
    "Damper 4 (阻尼比 8.57%)": np.array([2 * 0.0857 * np.sqrt(k_fixed[0] * m[0]), 2 * 0.0857 * np.sqrt(k_fixed[1] * m[1])]),
    "Damper 5 (阻尼比 15.27%)": np.array([2 * 0.1527 * np.sqrt(k_fixed[0] * m[0]), 2 * 0.1527 * np.sqrt(k_fixed[1] * m[1])]),
    "Damper 6 (阻尼比 20.98%)": np.array([2 * 0.2098 * np.sqrt(k_fixed[0] * m[0]), 2 * 0.2098 * np.sqrt(k_fixed[1] * m[1])])
}

for damper_name, c_vals in dampers.items():
    print(f"--- 計算 {damper_name} ---")
    disp = np.zeros((2, num_steps))
    vel = np.zeros((2, num_steps))
    acc = np.zeros((2, num_steps))

    C_mat = np.diag(c_vals) # 阻尼矩陣

    # 初始加速度
    acc[:, 0] = np.linalg.solve(M_mat, F_seismic[:, 0] - C_mat @ vel[:, 0] - K_mat @ disp[:, 0])

    # 有效剛度矩陣
    K_eff = K_mat + a0 * M_mat + a1 * C_mat

    # Newmark-beta 數值積分迴圈
    for i in range(1, num_steps):
        F_eff = F_seismic[:, i] \
            + M_mat @ (a0*disp[:,i-1] + a2*vel[:,i-1] + a3*acc[:,i-1]) \
            + C_mat @ (a1*disp[:,i-1] + a4*vel[:,i-1] + a5*acc[:,i-1])

        disp[:, i] = np.linalg.solve(K_eff, F_eff)
        acc[:, i] = a0 * (disp[:, i] - disp[:, i-1]) - a2 * vel[:, i-1] - a3 * acc[:, i-1]
        vel[:, i] = vel[:, i-1] + dt * ((1.0 - gamma) * acc[:, i-1] + gamma * acc[:, i])

    # 儲存結果
    all_disp_1st_floor_combined[damper_name] = disp[0, :]
    all_disp_2nd_floor_combined[damper_name] = disp[1, :]

# --- 3. 繪製合併後的圖表 ---

# Define colors for each line
line_colors = {
    "無阻尼器": "red", # Changed to red as requested
    "Damper 4 (阻尼比 8.57%)": "tab:blue",
    "Damper 5 (阻尼比 15.27%)": "tab:orange",
    "Damper 6 (阻尼比 20.98%)": "tab:green"
}

# 繪製一樓所有阻尼器位移圖 (包含無阻尼器)
plt.figure(figsize=(14, 7))
for label, displacement_data in all_disp_1st_floor_combined.items():
    plt.plot(time, displacement_data, label=label, color=line_colors.get(label, 'black')) # Use .get() for safety
plt.xlabel("時間 (s)")
plt.ylabel("位移 (m)")
plt.title("建築物一樓位移歷時圖 (有無阻尼器比較)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, "1st_floor_displacement_comparison.png"))
plt.show()

# 繪製二樓所有阻尼器位移圖 (包含無阻尼器)
plt.figure(figsize=(14, 7))
for label, displacement_data in all_disp_2nd_floor_combined.items():
    plt.plot(time, displacement_data, label=label, color=line_colors.get(label, 'black')) # Use .get() for safety
plt.xlabel("時間 (s)")
plt.ylabel("位移 (m)")
plt.title("建築物二樓位移歷時圖 (有無阻尼器比較)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, "2nd_floor_displacement_comparison.png"))
plt.show()

print("\n所有位移數據已計算並繪製完成。")
