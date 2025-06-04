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

# 建築物參數 (使用之前提供的固定值)
m = np.array([8.46e7, 8.46e7])  # 樓層質量 (kg)
k_fixed = np.array([7.12e7, 7.12e7])  # 樓層剛度 (N/m) - 使用固定值

# 阻尼器配置 (根據不同阻尼比計算阻尼係數，使用固定的 k_fixed)
# 這裡的 c 值代表了包含阻尼器後的總阻尼
dampers = {
    "Damper 4": np.array([2 * 0.0857 * np.sqrt(k_fixed[0] * m[0]), 2 * 0.0857 * np.sqrt(k_fixed[1] * m[1])]),  # 阻尼比 8.57%
    "Damper 5": np.array([2 * 0.1527 * np.sqrt(k_fixed[0] * m[0]), 2 * 0.1527 * np.sqrt(k_fixed[1] * m[1])]),  # 阻尼比 15.27%
    "Damper 6": np.array([2 * 0.2098 * np.sqrt(k_fixed[0] * m[0]), 2 * 0.2098 * np.sqrt(k_fixed[1] * m[1])])    # 阻尼比 20.98%
}

# 設定統一的輸出目錄 (使用之前提供的路徑)
output_directory = r"C:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\地震工程"

# 確保輸出目錄存在，如果不存在則創建
os.makedirs(output_directory, exist_ok=True)

# 讀取地震加速度 (Kobe.txt)
# 將 KOBE.txt 的路徑也設定到指定的目錄下
filename_kobe = os.path.join(output_directory, "Kobe.txt")
time, accel = np.loadtxt(filename_kobe, unpack=True, skiprows=1)

# 時間間隔
dt = time[1] - time[0]
num_steps = len(time)

# Newmark β 數值積分法的固定參數
beta = 0.25
gamma = 0.5

# 計算 Newmark-beta 常數 (這些常數與阻尼器類型無關)
a0 = 1.0 / (beta * dt**2)
a1 = gamma / (beta * dt)
a2 = 1.0 / (beta * dt)
a3 = 1.0 / (2 * beta) - 1.0
a4 = gamma / beta - 1.0
a5 = dt / 2.0 * (gamma / beta - 2.0)

# 初始化儲存所有阻尼器結果的字典
all_disp_1st_floor = {}
all_disp_2nd_floor = {}

# 開始計算不同阻尼器下的位移
for damper_name, c_vals in dampers.items(): # c_vals 代表當前阻尼器配置下的阻尼係數陣列
    # 初始化位移、速度、加速度 (每次新的阻尼器配置都需要重新初始化)
    disp = np.zeros((2, num_steps))
    vel = np.zeros((2, num_steps))
    acc = np.zeros((2, num_steps))

    a_g = accel * 9.81  # 轉換為 m/s² (地面加速度)
    # 外力矩陣 (作用於各樓層質量上的地震力，這裡假設地面加速度作用於所有質量)
    # 根據之前的程式碼，F[1,:] 設為 0，表示第二層不直接受地震力
    F = np.zeros((2, num_steps))
    F[0, :] = -m[0] * a_g
    F[1, :] = 0

    # 建立質量、阻尼、剛度矩陣 (阻尼矩陣 C_mat 需根據當前阻尼器配置更新)
    M_mat = np.diag(m) # 質量矩陣
    C_mat = np.diag(c_vals) # 阻尼矩陣 (使用當前阻尼器配置的 c_vals)
    K_mat = np.array([[k_fixed[0]+k_fixed[1], -k_fixed[1]],
                      [-k_fixed[1], k_fixed[1]]]) # 剛度矩陣 (使用固定的 k_fixed)

    # 初始加速度 (假設初始位移和速度為零)
    acc[:, 0] = np.linalg.solve(M_mat, F[:, 0] - C_mat @ vel[:, 0] - K_mat @ disp[:, 0])

    # 有效剛度矩陣 (K_eff 需根據當前阻尼器配置的 C_mat 更新)
    K_eff = K_mat + a0 * M_mat + a1 * C_mat

    # Newmark-beta 數值積分迴圈
    for i in range(1, num_steps):
        # 有效外力向量
        F_eff = F[:, i] \
            + M_mat @ (a0*disp[:,i-1] + a2*vel[:,i-1] + a3*acc[:,i-1]) \
            + C_mat @ (a1*disp[:,i-1] + a4*vel[:,i-1] + a5*acc[:,i-1])

        # 解聯立方程式，得到當前時間步的位移
        disp[:, i] = np.linalg.solve(K_eff, F_eff)

        # 更新加速度
        acc[:, i] = a0 * (disp[:, i] - disp[:, i-1]) - a2 * vel[:, i-1] - a3 * acc[:, i-1]
        # 更新速度
        vel[:, i] = vel[:, i-1] + dt * ((1.0 - gamma) * acc[:, i-1] + gamma * acc[:, i])

    # 儲存計算出的位移結果到字典
    all_disp_1st_floor[damper_name] = disp[0, :]
    all_disp_2nd_floor[damper_name] = disp[1, :]

    # --- 計算並顯示當前阻尼器配置下的位移指標 ---
    floor_labels = ["一樓", "二樓"]
    print(f"\n--- {damper_name} 位移指標 ---")
    for i in range(2): # 遍歷第一層 (索引 0) 和第二層 (索引 1)
        floor_displacement = disp[i, :] # 取得當前樓層的位移數據

        print(f"  {floor_labels[i]}：")

        # 平均位移 (Mean Displacement)
        mean_displacement = np.mean(floor_displacement)
        print(f"    平均位移 (Mean Displacement, μ): {mean_displacement:.6f} m")

        # 均方根位移 (RMS Displacement)
        rms_displacement = np.sqrt(np.mean(floor_displacement**2))
        print(f"    均方根位移 (RMS Displacement): {rms_displacement:.6f} m")

        # 尖峰位移 (Peak Displacement, P)
        peak_displacement = np.max(np.abs(floor_displacement))
        print(f"    尖峰位移 (Peak Displacement, P): {peak_displacement:.6f} m")

    # --- 儲存當前阻尼器配置下的位移數據到檔案 ---
    # 設定輸出時間範圍
    output_time = np.arange(0, 40.01, 0.01)  # 每 0.01 秒，最高至 40 秒

    # 內插位移數據 (一樓和二樓位移)
    disp_1st_floor_interp = np.interp(output_time, time, disp[0])
    disp_2nd_floor_interp = np.interp(output_time, time, disp[1])

    # 將時間與位移存成 .txt 檔案到指定目錄
    output_filename_1st = os.path.join(output_directory, f"{damper_name}_1st_floor_displacement.txt")
    np.savetxt(output_filename_1st, np.column_stack((output_time, disp_1st_floor_interp)), fmt="%.2f %.6f", header="時間 (s)    位移 (m)")
    print(f"位移數據已儲存至 {output_filename_1st}")

    output_filename_2nd = os.path.join(output_directory, f"{damper_name}_2nd_floor_displacement.txt")
    np.savetxt(output_filename_2nd, np.column_stack((output_time, disp_2nd_floor_interp)), fmt="%.2f %.6f", header="時間 (s)    位移 (m)")
    print(f"位移數據已儲存至 {output_filename_2nd}")


# --- 繪製一樓所有阻尼器位移圖 ---
plt.figure(figsize=(12, 6))
for damper_name, displacement_data in all_disp_1st_floor.items():
    plt.plot(time, displacement_data, label=f"{damper_name}")
plt.xlabel("時間 (s)")
plt.ylabel("位移 (m)")
plt.title("不同阻尼器下的一樓位移歷時圖")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, "1st_floor_all_dampers_displacement.png"))
plt.show()

# --- 繪製二樓所有阻尼器位移圖 ---
plt.figure(figsize=(12, 6))
for damper_name, displacement_data in all_disp_2nd_floor.items():
    plt.plot(time, displacement_data, label=f"{damper_name}")
plt.xlabel("時間 (s)")
plt.ylabel("位移 (m)")
plt.title("不同阻尼器下的二樓位移歷時圖")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, "2nd_floor_all_dampers_displacement.png"))
plt.show()
