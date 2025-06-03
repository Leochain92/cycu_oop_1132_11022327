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
        # 'elastic': 彈性載入/卸載
        # 'yield_pos': 正向屈服載入
        # 'yield_neg': 負向屈服載入
        # 'unload_pos_to_neg': 從正向最大位移卸載到負向 (K1斜率)
        # 'unload_neg_to_pos': 從負向最小位移卸載到正向 (K1斜率)
        self.current_branch = 'elastic' 
        self.x_reversal_point = 0.0  # 最近一次載入方向改變時的位移
        self.Fs_reversal_point = 0.0 # 最近一次載入方向改變時的力

    def get_Fs_and_Kt(self, x_trial):
        """
        根據試探位移 (x_trial) 和當前模型的內部狀態 (current_branch, x_reversal_point, Fs_reversal_point)，
        計算試探的彈簧力 (Fs_trial) 和切線剛度 (Kt_trial)。
        此函數不改變模型的內部狀態，僅用於計算。
        """
        
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
                # 檢查是否穿過負向屈服點
                if Fs_trial <= self.Fy_neg: 
                    Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                    Kt_trial = self.k2
        
        elif self.current_branch == 'yield_neg':
            if x_trial <= self.x_reversal_point: # 繼續負向載入 (K2)
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else: # 從負向屈服點開始反向卸載 (K1)
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                # 檢查是否穿過正向屈服點
                if Fs_trial >= self.Fy: 
                    Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                    Kt_trial = self.k2

        elif self.current_branch == 'unload_pos_to_neg': # 從正向反向點卸載到負向 (K1 路徑)
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            # 檢查是否重新進入正向屈服
            if Fs_trial >= self.Fy and x_trial >= self.xy: 
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            # 檢查是否穿過負向屈服點
            elif Fs_trial <= self.Fy_neg and x_trial <= -self.xy: 
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
        
        elif self.current_branch == 'unload_neg_to_pos': # 從負向反向點卸載到正向 (K1 路徑)
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            # 檢查是否重新進入負向屈服
            if Fs_trial <= self.Fy_neg and x_trial <= -self.xy: 
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            # 檢查是否穿過正向屈服點
            elif Fs_trial >= self.Fy and x_trial >= self.xy: 
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
        
        return Fs_trial, Kt_trial

    def update_state(self, x_current_actual, Fs_current_actual, x_prev_actual, Fs_prev_actual):
        """
        在牛頓-拉夫森迭代收斂後，更新模型的內部狀態。
        這將確定下一個時間步的起始狀態。
        """
        
        # 判斷載入方向
        is_loading_pos = (x_current_actual >= x_prev_actual) # 位移增大
        is_loading_neg = (x_current_actual <= x_prev_actual) # 位移減小

        # 判斷是否處於卸載或重新載入
        epsilon = 1e-6 # 浮點數比較容忍度

        if self.current_branch == 'elastic':
            if x_current_actual >= self.xy - epsilon: # 進入正向屈服
                self.current_branch = 'yield_pos'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual 
            elif x_current_actual <= -self.xy + epsilon: # 進入負向屈服
                self.current_branch = 'yield_neg'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual 
            else: # 保持在彈性區，反向點為當前點
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
        
        elif self.current_branch == 'yield_pos':
            if is_loading_neg and x_current_actual < x_prev_actual: # 從正向屈服反向 (位移減小)
                self.current_branch = 'unload_pos_to_neg'
                self.x_reversal_point = x_prev_actual # 反向點是前一個最大點
                self.Fs_reversal_point = Fs_prev_actual
            # 如果是繼續載入，保持在 yield_pos，反向點更新為當前最大點
            elif is_loading_pos and x_current_actual > self.x_reversal_point:
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual

        elif self.current_branch == 'yield_neg':
            if is_loading_pos and x_current_actual > x_prev_actual: # 從負向屈服反向 (位移增大)
                self.current_branch = 'unload_neg_to_pos'
                self.x_reversal_point = x_prev_actual # 反向點是前一個最小點
                self.Fs_reversal_point = Fs_prev_actual
            # 如果是繼續載入，保持在 yield_neg，反向點更新為當前最小點
            elif is_loading_neg and x_current_actual < self.x_reversal_point:
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual

        elif self.current_branch == 'unload_pos_to_neg': 
            if Fs_current_actual >= self.Fy - epsilon and x_current_actual >= self.xy - epsilon: # 重新進入正向屈服
                self.current_branch = 'yield_pos'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            elif Fs_current_actual <= self.Fy_neg + epsilon and x_current_actual <= -self.xy + epsilon: # 穿過負向屈服點
                self.current_branch = 'yield_neg'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            elif abs(x_current_actual) < self.xy and abs(Fs_current_actual) < self.Fy: # 回到彈性區
                self.current_branch = 'elastic'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            else: # 繼續在卸載路徑
                pass # 反向點不變

        elif self.current_branch == 'unload_neg_to_pos':
            if Fs_current_actual <= self.Fy_neg + epsilon and x_current_actual <= -self.xy + epsilon: # 重新進入負向屈服
                self.current_branch = 'yield_neg'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            elif Fs_current_actual >= self.Fy - epsilon and x_current_actual >= self.xy - epsilon: # 穿過正向屈服點
                self.current_branch = 'yield_pos'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            elif abs(x_current_actual) < self.xy and abs(Fs_current_actual) < self.Fy: # 回到彈性區
                self.current_branch = 'elastic'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            else: # 繼續在卸載路徑
                pass # 反向點不變


# --- 參數設定 ---
m = 1.0  # 質量 [k·s²/in]
xi = 0.05  # 阻尼比
k1 = 631.65  # 初始剛度 [k/in]
k2 = 126.33  # 屈服後剛度 [k/in]
xy = 1.0  # 屈服位移 [in]
dt = 0.005  # 時間步長 [s]
total_time = 2.0 # 模擬總時間
num_steps = int(total_time / dt) 

# 導出參數
wn = np.sqrt(k1 / m)  
c = 2 * m * wn * xi  

# --- 初始條件 ---
x0 = 0.0  # 初始位移 [in]
x_dot0 = 40.0  # 初始速度 [in/s] 

x_ddot0 = -c * x_dot0 / m 

print(f"計算得出的初始阻尼係數 c = {c:.4f} k·s/in")
print(f"計算得出的初始加速度 ẍ(0) = {x_ddot0:.4f} in/s²")

F_external_input = 0.0 

# --- 初始化陣列 ---
t = np.linspace(0, total_time, num_steps + 1) 
x = np.zeros(num_steps + 1)
x_dot = np.zeros(num_steps + 1)
x_ddot = np.zeros(num_steps + 1)
Fs = np.zeros(num_steps + 1) 
Kt = np.zeros(num_steps + 1) 

# 設定初始值
x[0] = x0
x_dot[0] = x_dot0
x_ddot[0] = x_ddot0

hysteretic_model = HystereticModel(k1, k2, xy)
# 初始時，模型在彈性區，Fs[0] 應該是 0，Kt[0] 應該是 k1
# get_Fs_and_Kt 應該只傳入 x_trial
Fs[0], Kt[0] = hysteretic_model.get_Fs_and_Kt(x[0])


# --- 平均加速度法 (非線性求解) ---
tolerance = 1e-6
max_iterations = 100

points_of_interest = {} 

for i in range(num_steps):
    x_k = x[i] 

    for iter_count in range(max_iterations):
        # 計算試探狀態下的 Fs 和 Kt
        # get_Fs_and_Kt 函數現在只依賴 x_trial 和模型當前狀態 (self.current_branch, self.x_reversal_point, self.Fs_reversal_point)
        Fs_k, Kt_k = hysteretic_model.get_Fs_and_Kt(x_k) 

        R_k = m * ((4 / dt**2) * (x_k - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]) + \
              c * ((2 / dt) * (x_k - x[i]) - x_dot[i]) + \
              Fs_k - F_external_input
        
        R_prime_k = (4 * m) / (dt**2) + (2 * c) / dt + Kt_k

        delta_x_k = - R_k / R_prime_k
        x_k_new = x_k + delta_x_k

        if abs(delta_x_k) < tolerance:
            x[i+1] = x_k_new
            break
        x_k = x_k_new
    else:
        print(f"警告: 在時間步 t={i*dt:.3f}s 處，牛頓-拉夫森未收斂。")
        x[i+1] = x_k

    # --- 計算彈簧力、速度和加速度 ---
    # 在牛頓-拉夫森收斂後，計算本時間步的最終彈簧力 Fs[i+1] 和切線剛度 Kt[i+1]
    # 這裡的 get_Fs_and_Kt 使用 x[i+1] 和模型當前狀態
    Fs[i+1], Kt[i+1] = hysteretic_model.get_Fs_and_Kt(x[i+1]) 
    
    x_ddot[i+1] = (4 / dt**2) * (x[i+1] - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]
    x_dot[i+1] = x_dot[i] + (dt / 2) * (x_ddot[i] + x_ddot[i+1])

    # --- 歷史追蹤 (更新 HystereticModel 的內部狀態) ---
    # 使用當前時間步的最終收斂結果來更新模型狀態
    hysteretic_model.update_state(x[i+1], Fs[i+1], x[i], Fs[i]) 

    # --- 追蹤特定點 (a, b, c, d, e) ---
    # 為防止重複記錄同一點，只在第一次達到時記錄
    # 點 a: 首次達到正向屈服 (位移 >= xy 且力 >= Fy)
    if 'a' not in points_of_interest and x[i+1] >= xy and Fs[i+1] >= hysteretic_model.Fy * 0.9999: 
        points_of_interest['a'] = t[i+1]
        
    # 點 b: 正向最大位移點 (速度從正變負或趨近零)
    if 'a' in points_of_interest and 'b' not in points_of_interest:
        if x_dot[i] > 0 and x_dot[i+1] <= 0: 
            points_of_interest['b'] = t[i+1]

    # 點 c: 卸載並經過零點 (位移從正變負或力從正變負)
    if 'b' in points_of_interest and 'c' not in points_of_interest:
        if x[i] >= 0 and x[i+1] < 0 and Fs[i] >= 0 and Fs[i+1] < 0: # 確保位移和力都過零
             points_of_interest['c'] = t[i+1]

    # 點 d: 首次達到負向屈服 (位移 <= -xy 且力 <= -Fy)
    if 'c' in points_of_interest and 'd' not in points_of_interest:
        if x[i+1] <= -xy and Fs[i+1] <= hysteretic_model.Fy_neg * 0.9999: 
            points_of_interest['d'] = t[i+1]
            
    # 點 e: 負向最大位移點 (速度從負變正或趨近零)
    if 'd' in points_of_interest and 'e' not in points_of_interest:
        if x_dot[i] < 0 and x_dot[i+1] >= 0: 
            points_of_interest['e'] = t[i+1]

# --- 準備表格資料 ---
table_data = []

# 添加前六個時間步
for i in range(min(num_steps + 1, 6)):
    table_data.append({
        'Step': i,
        'Time (s)': t[i],
        'Displacement (in)': x[i],
        'Velocity (in/s)': x_dot[i],
        'Acceleration (in/s^2)': x_ddot[i],
        'Spring Force (k)': Fs[i],
        'Point': '-'
    })

# 添加特定點
sorted_points = sorted(points_of_interest.items(), key=lambda item: item[1])
for point_label, time_val in sorted_points:
    idx = int(time_val / dt)
    # 檢查索引是否有效，並避免添加重複的行（如果前六個時間步中已經包含了某個點）
    if idx < len(x) and idx < len(Fs) and point_label not in [row['Point'] for row in table_data]:
        table_data.append({
            'Step': idx,
            'Time (s)': t[idx],
            'Displacement (in)': x[idx],
            'Velocity (in/s)': x_dot[idx],
            'Acceleration (in/s^2)': x_ddot[idx],
            'Spring Force (k)': Fs[idx],
            'Point': point_label
        })

# 創建 DataFrame
df = pd.DataFrame(table_data)

# 對 'Time (s)' 和 'Point' 進行排序，確保點的順序正確
# 使用一個字典來定義點的優先級，非點的優先級為最低
point_order = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
df['Sort_Key'] = df['Point'].apply(lambda p: point_order.get(p, 999))
df = df.sort_values(by=['Time (s)', 'Sort_Key']).drop(columns=['Sort_Key']).reset_index(drop=True)

# 顯示表格 (使用 to_string 避免截斷)
print("\n--- 響應數據表格 ---")
print(df.to_string(index=False, float_format="%.4f"))

# --- 定義儲存路徑 ---
output_path = r"C:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\地震工程"

# 檢查並創建目錄（如果不存在）
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"創建目錄: {output_path}")
else:
    print(f"目標目錄已存在: {output_path}")

# 將表格儲存為 CSV 檔案
csv_path = os.path.join(output_path, '響應數據表格.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 以確保中文顯示
print(f"響應數據表格已儲存至: {csv_path}")

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