import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.linalg import solve
import os # 導入 os 模組用於路徑操作

# 設定中文字體（自動偵測常見中文字體）
if any('Microsoft YaHei' in font.name for font in fm.fontManager.ttflist):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
elif any('SimHei' in font.name for font in fm.fontManager.ttflist):
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 備用

plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 建築物參數
m = np.array([8.46e7, 8.46e7])  # 質量 (kg)

# 根據新的已知參數更新剛度 (K_1=K_2=7.12*10^7 N/m)
k = np.array([7.12e7, 7.12e7])  # 剛度 (N/m)

# 根據新的已知參數更新阻尼 (C_1=C_2=1552226.752859 NS/m)
c = np.array([1552226.752859, 1552226.752859])  # 阻尼 (N·s/m)

# 建立質量、阻尼、剛度矩陣
M = np.diag(m)
C = np.diag(c)
K = np.array([[k[0]+k[1], -k[1]],
              [-k[1], k[1]]])

# 設定統一的輸出目錄
output_directory = r"C:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\地震工程"

# 確保輸出目錄存在，如果不存在則創建
os.makedirs(output_directory, exist_ok=True)

# 讀取地震加速度 (Kobe.txt)
# 將 KOBE.txt 的路徑也設定到指定的目錄下
filename = os.path.join(output_directory, "Kobe.txt")
time, accel = np.loadtxt(filename, unpack=True, skiprows=1)

# 時間間隔
dt = time[1] - time[0]
num_steps = len(time)

# Newmark-beta 參數
beta = 0.25
gamma = 0.5

# 初始化位移、速度、加速度
disp = np.zeros((2, num_steps))
vel = np.zeros((2, num_steps))
acc = np.zeros((2, num_steps))

# 外力（地震加速度轉換為 m/s²）
# 注意：這裡只對第一層施加地震力，第二層不直接受力，與你提供的原始程式碼一致
a = accel * 9.81  # 轉換為 m/s²
F = np.zeros((2, num_steps))
F[0, :] = -m[0] * a  # 只對第一層施加地震力
F[1, :] = 0          # 第二層不直接受地震力

# 初始加速度
acc[:, 0] = np.linalg.solve(M, F[:, 0] - C @ vel[:, 0] - K @ disp[:, 0])

# 預先計算 Newmark-beta 常數
a0 = 1/(beta*dt**2)
a1 = gamma/(beta*dt)
a2 = 1/(beta*dt)
a3 = 1/(2*beta) - 1
a4 = gamma/beta - 1
a5 = dt/2 * (gamma/beta - 2)

# 有效剛度矩陣
K_eff = M*a0 + C*a1 + K

# Newmark-beta 數值積分迴圈
for i in range(1, num_steps):
    # 有效外力向量
    F_eff = F[:, i] \
        + M @ (a0*disp[:, i-1] + a2*vel[:, i-1] + a3*acc[:, i-1]) \
        + C @ (a1*disp[:, i-1] + a4*vel[:, i-1] + a5*acc[:, i-1])
    # 解聯立方程式，得到當前時間步的位移
    disp[:, i] = np.linalg.solve(K_eff, F_eff)
    # 更新加速度
    acc[:, i] = a0*(disp[:, i] - disp[:, i-1]) - a2*vel[:, i-1] - a3*acc[:, i-1]
    # 更新速度
    vel[:, i] = vel[:, i-1] + dt*((1-gamma)*acc[:, i-1] + gamma*acc[:, i])

# 檢查資料內容 (印出形狀和前10個位移值)
print("time.shape:", time.shape)
print("disp.shape:", disp.shape)
print("1st floor disp (前10):", disp[0][:10])
print("2nd floor disp (前10):", disp[1][:10])

# 繪製兩層樓的位移歷時圖 (繪製在同一張圖上)
plt.figure(figsize=(12, 6))
plt.plot(time, disp[0], label="一樓位移 (1st Floor Displacement) (m)", color="tab:blue")
plt.plot(time, disp[1], label="二樓位移 (2nd Floor Displacement) (m)", color="tab:red")
plt.xlabel("時間 (s)")
plt.ylabel("位移 (m)")
plt.title("建築物樓層位移歷時圖 (無阻尼器，更新參數)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# 儲存圖片到指定目錄
plt.savefig(os.path.join(output_directory, "building_disp_both_floors_updated_params.png"))
plt.show()

# --- 計算並顯示位移指標 ---
floor_labels = ["一樓", "二樓"]

for i in range(2): # 遍歷第一層 (索引 0) 和第二層 (索引 1)
    floor_displacement = disp[i, :] # 取得當前樓層的位移數據

    print(f"\n--- {floor_labels[i]} 位移指標 ---")

    # 平均位移 (Mean Displacement)
    mean_displacement = np.mean(floor_displacement)
    print(f"平均位移 (Mean Displacement, μ): {mean_displacement:.6f} m")

    # 均方根位移 (RMS Displacement)
    rms_displacement = np.sqrt(np.mean(floor_displacement**2))
    print(f"均方根位移 (RMS Displacement): {rms_displacement:.6f} m")

    # 尖峰位移 (Peak Displacement, P)
    # 注意：PGA (Peak Ground Acceleration) 通常指地震加速度的峰值，
    # 但在此上下文，您可能是指位移的絕對值最大值。
    peak_displacement = np.max(np.abs(floor_displacement))
    print(f"尖峰位移 (Peak Displacement, P): {peak_displacement:.6f} m")


# --- 儲存位移數據到檔案 ---
# 設定輸出時間範圍
output_time = np.arange(0, 40.01, 0.01)  # 每 0.01 秒，最高至 40 秒

# 內插位移數據 (一樓和二樓位移)
disp_1st_floor_interp = np.interp(output_time, time, disp[0])
disp_2nd_floor_interp = np.interp(output_time, time, disp[1])

# 將時間與位移存成 .txt 檔案到指定目錄
output_filename_1st = os.path.join(output_directory, "1st_floor_displacement_updated_params.txt")
np.savetxt(output_filename_1st, np.column_stack((output_time, disp_1st_floor_interp)), fmt="%.2f %.6f", header="時間 (s)    位移 (m)")
print(f"一樓位移數據已儲存至 {output_filename_1st}")

output_filename_2nd = os.path.join(output_directory, "2nd_floor_displacement_updated_params.txt")
np.savetxt(output_filename_2nd, np.column_stack((output_time, disp_2nd_floor_interp)), fmt="%.2f %.6f", header="時間 (s)    位移 (m)")
print(f"二樓位移數據已儲存至 {output_filename_2nd}")
