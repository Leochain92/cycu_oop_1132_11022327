import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pandas as pd

# --- 設定Matplotlib中文顯示 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 雙線性滯後模型類 ---
class HystereticModel:
    def __init__(self, k1, k2, xy):
        self.k1 = k1  # 初始剛度
        self.k2 = k2  # 屈服後剛度
        self.xy = xy  # 屈服位移
        self.Fy = k1 * xy  # 屈服力 (正向)
        self.Fy_neg = -k1 * xy # 屈服力 (負向)

        # 滯後狀態變數
        self.current_branch = 'elastic'
        self.x_reversal_point = 0.0
        self.Fs_reversal_point = 0.0

    def get_Fs_and_Kt(self, x_trial):
        Fs_trial = 0.0
        Kt_trial = 0.0

        if self.current_branch == 'elastic':
            if x_trial >= self.xy:
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            elif x_trial <= -self.xy:
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            else: # 仍在彈性區
                Fs_trial = self.k1 * x_trial
                Kt_trial = self.k1
        elif self.current_branch == 'yield_pos':
            if x_trial >= self.x_reversal_point: # 繼續正向載入 (K2)
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else: # 從正向屈服點開始反向卸載 (K1)
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                # 檢查是否穿過負向屈服點 (根據模型邏輯，它會直接跳到骨架線上)
                # 實際的get_Fs_and_Kt會更複雜，這裡簡化為更新狀態時處理
                target_Fy_neg_on_unload = self.Fs_reversal_point + self.k1 * (-self.xy - self.x_reversal_point)
                if Fs_trial <= self.Fy_neg and x_trial <= -self.xy : # 簡易判斷是否到達負屈服區
                     Fs_trial = self.Fy_neg + self.k2 * (x_trial - (-self.xy))
                     Kt_trial = self.k2

        elif self.current_branch == 'yield_neg':
            if x_trial <= self.x_reversal_point: # 繼續負向載入 (K2)
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else: # 從負向屈服點開始反向卸載 (K1)
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                target_Fy_pos_on_unload = self.Fs_reversal_point + self.k1 * (self.xy - self.x_reversal_point)
                if Fs_trial >= self.Fy and x_trial >= self.xy: # 簡易判斷是否到達正屈服區
                    Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                    Kt_trial = self.k2

        elif self.current_branch == 'unload_pos_to_neg': # 從正向反向點卸載到負向 (K1 路徑)
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            # 檢查是否穿過負向屈服點並轉至屈服路徑
            if Fs_trial <= self.Fy_neg and x_trial <= -self.xy: # 條件應與update_state一致
                Fs_trial = self.Fy_neg + self.k2 * (x_trial - (-self.xy)) # 跳至骨架線
                Kt_trial = self.k2
        elif self.current_branch == 'unload_neg_to_pos': # 從負向反向點卸載到正向 (K1 路徑)
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            if Fs_trial >= self.Fy and x_trial >= self.xy: # 條件應與update_state一致
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy) # 跳至骨架線
                Kt_trial = self.k2
        return Fs_trial, Kt_trial

    def update_state(self, x_current_actual, Fs_current_actual, x_prev_actual, Fs_prev_actual):
        is_loading_pos = (x_current_actual >= x_prev_actual)
        is_loading_neg = (x_current_actual <= x_prev_actual)
        epsilon = 1e-6

        if self.current_branch == 'elastic':
            if x_current_actual >= self.xy - epsilon:
                self.current_branch = 'yield_pos'
                # 更新反向點為剛進入屈服的點
                self.x_reversal_point = self.xy # 或者 x_current_actual 如果允許微小超出
                self.Fs_reversal_point = self.Fy   # 對應骨架線上的力
            elif x_current_actual <= -self.xy + epsilon:
                self.current_branch = 'yield_neg'
                self.x_reversal_point = -self.xy
                self.Fs_reversal_point = self.Fy_neg
            # else: # 保持在彈性區，反向點通常在載入時不需要頻繁更新，除非速度改變
            # self.x_reversal_point = x_current_actual (不需要，因為彈性區沒有"記憶")
            # self.Fs_reversal_point = Fs_current_actual

        elif self.current_branch == 'yield_pos':
            if is_loading_neg and x_current_actual < x_prev_actual: # 開始卸載
                self.current_branch = 'unload_pos_to_neg'
                self.x_reversal_point = x_prev_actual # 卸載起始點
                self.Fs_reversal_point = Fs_prev_actual
            elif is_loading_pos and x_current_actual > self.x_reversal_point: # 繼續在屈服路徑上移動
                 # 更新反向點為當前最大點，這樣下次卸載就從這裡開始
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual


        elif self.current_branch == 'yield_neg':
            if is_loading_pos and x_current_actual > x_prev_actual: # 開始卸載
                self.current_branch = 'unload_neg_to_pos'
                self.x_reversal_point = x_prev_actual
                self.Fs_reversal_point = Fs_prev_actual
            elif is_loading_neg and x_current_actual < self.x_reversal_point: # 繼續在屈服路徑上移動
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual


        elif self.current_branch == 'unload_pos_to_neg':
            # 檢查是否達到負向屈服條件 (跳回骨架線)
            # 注意: Fs_current_actual 是基於 k1 卸載計算得到的
            # 這裡的 x_current_actual 和 Fs_current_actual 是已收斂的實際值
            if Fs_current_actual <= self.Fy_neg + self.k2 * (x_current_actual - (-self.xy)) + epsilon and x_current_actual <= -self.xy + epsilon :
                 # 上述條件判斷是否"接觸或穿過"負向骨架線
                 # 為了簡化，我們假設一旦位移 <= -xy 且力也表現出趨向或小於骨架線力，則轉換
                if x_current_actual <= -self.xy + epsilon: # 更直接的判斷
                    self.current_branch = 'yield_neg'
                    # 更新反向點為剛進入負屈服的點
                    self.x_reversal_point = -self.xy # 或者 x_current_actual
                    self.Fs_reversal_point = self.Fy_neg + self.k2 * (x_current_actual - (-self.xy)) #確保在骨架線上
            # 如果重新反向載入 (變成正向)
            elif is_loading_pos and x_current_actual > x_prev_actual :
                # 這裡的邏輯是，如果從unload_pos_to_neg路徑反向，它會創建一個新的彈性路徑
                # 但通常會繼續unload_neg_to_pos，這裡要看模型的詳細規則
                # 為了簡化，我們假設它會繼續沿著k1斜率，直到碰到正向骨架線
                # 但更常見的規則是，它會形成一個新的 k1 回彈路徑，目標是之前的 x_reversal_point
                # 這裡我們保持 unload_pos_to_neg，但反向點需要更新
                # self.current_branch = 'elastic_reloading_from_unload' # 需要更複雜的狀態
                # 根據常見的 Masing's rule 或類似規則，卸載後再反向載入，會瞄準之前的最大/最小點
                # 這裡的 self.x_reversal_point, self.Fs_reversal_point 是卸載的起始點，它們不變，除非完全反向
                 pass # 保持在 unload_pos_to_neg 但方向可能改變，這條路徑本身會處理力的計算


        elif self.current_branch == 'unload_neg_to_pos':
            if Fs_current_actual >= self.Fy + self.k2 * (x_current_actual - self.xy) - epsilon and x_current_actual >= self.xy - epsilon:
                if x_current_actual >= self.xy - epsilon:
                    self.current_branch = 'yield_pos'
                    self.x_reversal_point = self.xy
                    self.Fs_reversal_point = self.Fy + self.k2 * (x_current_actual - self.xy)
            elif is_loading_neg and x_current_actual < x_prev_actual:
                pass


# --- 參數設定 ---
m = 1.0
xi = 0.05
k1_param = 631.65  # 初始剛度 [k/in]
k2_param = 126.33  # 屈服後剛度 [k/in]
xy_param = 1.0  # 屈服位移 [in]
t = 0.0  # 初始時間 [s]
dt = 0.005
total_time = 2.0
num_steps = int(total_time / dt)

# 建立 HystereticModel 實例以獲取參數
hysteretic_model_for_params = HystereticModel(k1_param, k2_param, xy_param)
k1_val = hysteretic_model_for_params.k1
k2_val = hysteretic_model_for_params.k2
xy_val = hysteretic_model_for_params.xy
Fy_val = hysteretic_model_for_params.Fy

# --- 定義儲存路徑 ---
output_path = r'C:\\Users\\truck\\OneDrive\\文件\\GitHub\\cycu_oop_1132_11022327\\地震工程'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# --- 產生反映實際參數的迴圈路徑描述表格 ---
# 表格說明:
# - k1: 初始剛度
# - k2: 屈服後剛度
# - xy: 屈服位移
# - Fy: 屈服力 (Fy = k1 * xy)
# - x_max, Fs(x_max): 正向載入歷史中的峰值位移及對應力
# - x_min, Fs(x_min): 負向載入歷史中的峰值位移及對應力
# - x_rev, Fs_rev: 任一轉向點的位移與力

# 使用格式化後的參數值，方便閱讀
f_k1 = f"{k1_val:.2f}"
f_k2 = f"{k2_val:.2f}"
f_xy = f"{xy_val:.2f}"
f_Fy = f"{Fy_val:.2f}"

loop_path_data_bilinear_specific = [
    {
        "Hysteresis Loop Segment": "Origin (初始狀態)",
        "Displacement (位移)": "x = 0",
        "Velocity (速度)": "dx/dt = 0 (初始)",
        "Restoring Force (恢復力) Fs(x)": "Fs(x) = 0"
    },
    {
        "Hysteresis Loop Segment": f"Stage 1: Initial Elastic Loading (初始彈性加載, 剛度 {f_k1})",
        "Displacement (位移)": f"0 < |x| < {f_xy}",
        "Velocity (速度)": "dx/dt != 0",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x) = {f_k1}*x"
    },
    {
        "Hysteresis Loop Segment": f"Point a: Positive Yield (正向屈服點)",
        "Displacement (位移)": f"x = {f_xy}",
        "Velocity (速度)": "dx/dt > 0 (通常)",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x) = {f_Fy}"
    },
    {
        "Hysteresis Loop Segment": f"Stage 2: Positive Post-Yield Loading (正向屈服後加載, 剛度 {f_k2})",
        "Displacement (位移)": f"x > {f_xy} (且 x <= x_max)",
        "Velocity (速度)": "dx/dt > 0",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x) = {f_Fy} + {f_k2}*(x - {f_xy})"
    },
    {
        "Hysteresis Loop Segment": "Point b: Positive Peak Reversal (正向峰值轉向點)",
        "Displacement (位移)": "x = x_max (歷史最大正位移)",
        "Velocity (速度)": "dx/dt = 0 (在峰值點)",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x_max) = ({f_Fy} + {f_k2}*(x_max - {f_xy})) if x_max > {f_xy} else ({f_k1}*x_max)"
    },
    {
        "Hysteresis Loop Segment": f"Stage 3: Elastic Unloading from Positive Peak (從正向峰值彈性卸載, 剛度 {f_k1})",
        "Displacement (位移)": "x < x_max (卸載中)",
        "Velocity (速度)": "dx/dt < 0",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x) = Fs(x_max) + {f_k1}*(x - x_max)"
    },
    {
        "Hysteresis Loop Segment": f"Transition to Negative Yield (轉向負向屈服)",
        "Displacement (位移)": "卸載至 x <= -{f_xy} 且滿足屈服條件",
        "Velocity (速度)": "dx/dt < 0",
        "Restoring Force (恢復力) Fs(x)": f"當 x 約等於 -{f_xy} 時，Fs(x) 接近 -{f_Fy}，之後進入Stage 4"
    },
    {
        "Hysteresis Loop Segment": f"Stage 4: Negative Post-Yield Loading (負向屈服後加載, 剛度 {f_k2})",
        "Displacement (位移)": "x < -{f_xy} (且 x >= x_min, 在負向骨架線上)",
        "Velocity (速度)": "dx/dt < 0",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x) = -{f_Fy} + {f_k2}*(x - (-{f_xy}))"
    },
    {
        "Hysteresis Loop Segment": "Point d: Negative Peak Reversal (負向峰值轉向點)",
        "Displacement (位移)": "x = x_min (歷史最小負位移)",
        "Velocity (速度)": "dx/dt = 0 (在峰值點)",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x_min) = (-{f_Fy} + {f_k2}*(x_min - (-{f_xy}))) if x_min < -{f_xy} else (Fs(x_max) + {f_k1}*(x_min - x_max))"
    },
    {
        "Hysteresis Loop Segment": f"Stage 5: Elastic Reloading from Negative Peak (從負向峰值彈性再加載, 剛度 {f_k1})",
        "Displacement (位移)": "x > x_min (再加載中)",
        "Velocity (速度)": "dx/dt > 0",
        "Restoring Force (恢復力) Fs(x)": f"Fs(x) = Fs(x_min) + {f_k1}*(x - x_min)"
    },
    {
        "Hysteresis Loop Segment": f"Transition to Positive Yield (轉向正向屈服)",
        "Displacement (位移)": "再加載至 x >= {f_xy} 且滿足屈服條件",
        "Velocity (速度)": "dx/dt > 0",
        "Restoring Force (恢復力) Fs(x)": f"當 x 約等於 {f_xy} 時，Fs(x) 接近 {f_Fy}，之後回到Stage 2類型行為"
    }
]

df_hysteresis_path_specific = pd.DataFrame(loop_path_data_bilinear_specific)

print("\n--- 雙線性模型迴圈路徑描述 ---")
print(f"模型參數: k1={k1_val:.2f}, k2={k2_val:.2f}, xy={xy_val:.2f}, Fy={Fy_val:.2f}")
print(df_hysteresis_path_specific.to_string(index=False))

csv_path_specific = os.path.join(output_path, '迴圈路徑描述表格.csv')
df_hysteresis_path_specific.to_csv(csv_path_specific, index=False, encoding='utf-8-sig')
print(f"迴圈路徑描述表格 已儲存至: {csv_path_specific}")
# --- 繪製時間歷史圖 ---
plt.figure(figsize=(12, 10))

# 位移 x(t)
plt.subplot(4, 1, 1)
plt.plot(t, x, label='位移 $x(t)$')
plt.ylabel('位移 $x(t)$ [in]')
plt.title('系統動力響應時間歷史')
plt.grid(True)
plt.legend()

# 速度 ẋ(t)
plt.subplot(4, 1, 2)
plt.plot(t, x_dot, label='速度 $\\dot{x}(t)$', color='orange')
plt.ylabel('速度 $\\dot{x}(t)$ [in/s]')
plt.grid(True)
plt.legend()

# 加速度 ẍ(t)
plt.subplot(4, 1, 3)
plt.plot(t, x_ddot, label='加速度 $\\ddot{x}(t)$', color='green')
plt.ylabel('加速度 $\\ddot{x}(t)$ [in/s²]')
plt.grid(True)
plt.legend()

# 彈簧力 F_s(t)
plt.subplot(4, 1, 4)
plt.plot(t, Fs, label='彈簧力 $F_s(t)$', color='red')
plt.ylabel('彈簧力 $F_s(t)$ [k]')
plt.xlabel('時間 t [s]')
plt.grid(True)
plt.legend()

plt.tight_layout()
time_history_plot_path = os.path.join(output_path, '時間歷史圖.png')
plt.savefig(time_history_plot_path)
print(f"時間歷史圖已儲存至: {time_history_plot_path}")
plt.show()

# --- 繪製 F_s(x) 滯後迴圈 ---
plt.figure(figsize=(8, 6))
plt.plot(x, Fs, label='彈簧力 $F_s(x)$ 滯後迴圈', color='purple')
# 標記屈服點
plt.axvline(x=xy, color='gray', linestyle='--', label='正向屈服位移')
plt.axvline(x=-xy, color='gray', linestyle='--', label='負向屈服位移')
plt.axhline(y=hysteretic_model.Fy, color='gray', linestyle='-.', label='正向屈服力')
plt.axhline(y=hysteretic_model.Fy_neg, color='gray', linestyle='-.', label='負向屈服力')

# 標記特定點 a, b, c, d, e
colors = ['red', 'green', 'blue', 'cyan', 'magenta']
for j, (point_label, time_val) in enumerate(sorted_points):
    idx = int(time_val / dt)
    if idx < len(x) and idx < len(Fs):
        plt.scatter(x[idx], Fs[idx], color=colors[j], marker='o', s=100, zorder=5, label=f'點 {point_label}')
        # 調整文本位置，避免重疊
        plt.text(x[idx] + 0.05, Fs[idx] + 0.05 * hysteretic_model.Fy, point_label, fontsize=12, ha='left', va='bottom')

plt.xlabel('位移 x(t) [in]')
plt.ylabel('彈簧力 $F_s(t)$ [k]')
plt.title('彈簧力 $F_s(x)$ - 滯後迴圈')
plt.grid(True)
plt.legend()
hysteretic_loop_plot_path = os.path.join(output_path, '滯後迴圈圖.png')
plt.savefig(hysteretic_loop_plot_path)
print(f"滯後迴圈圖已儲存至: {hysteretic_loop_plot_path}")
plt.show()

def run_nonlinear_response(model, m, c, dt, num_steps, x0, x_dot0, x_ddot0, tolerance=1e-6, max_iterations=100):
    x = np.zeros(num_steps + 1)
    x_dot = np.zeros(num_steps + 1)
    x_ddot = np.zeros(num_steps + 1)
    Fs = np.zeros(num_steps + 1)
    Kt = np.zeros(num_steps + 1)
    x[0], x_dot[0], x_ddot[0] = x0, x_dot0, x_ddot0
    Fs[0], Kt[0] = model.get_Fs_and_Kt(x[0])
    for i in range(num_steps):
        x_k = x[i]
        for _ in range(max_iterations):
            Fs_k, Kt_k = model.get_Fs_and_Kt(x_k)
            R_k = m * ((4 / dt**2) * (x_k - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]) + \
                  c * ((2 / dt) * (x_k - x[i]) - x_dot[i]) + Fs_k
            R_prime_k = (4 * m) / (dt**2) + (2 * c) / dt + Kt_k
            delta_x_k = - R_k / R_prime_k
            x_k_new = x_k + delta_x_k
            if abs(delta_x_k) < tolerance:
                x[i+1] = x_k_new
                break
            x_k = x_k_new
        Fs[i+1], Kt[i+1] = model.get_Fs_and_Kt(x[i+1])
        x_ddot[i+1] = (4 / dt**2) * (x[i+1] - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]
        x_dot[i+1] = x_dot[i] + (dt / 2) * (x_ddot[i] + x_ddot[i+1])
        model.update_state(x[i+1], Fs[i+1], x[i], Fs[i])
    return x, x_dot, x_ddot, Fs, Kt