# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 導入字體管理器

# --- 設定Matplotlib中文顯示 ---
# 選擇一個支持中文字符的字體。您可以根據您的操作系統調整。
# Windows 範例: 'Microsoft YaHei', 'SimHei', 'FangSong'
# macOS 範例: 'PingFang HK', 'Heiti TC'
# Linux 範例: 'DejaVu Sans' (通常需要安裝中文字體包，如 ttf-wqy-zenhei)
# 檢查字體是否存在，否則使用通用字體
if any('Microsoft YaHei' in font.name for font in fm.fontManager.ttflist):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 將字體設定為微軟雅黑
elif any('PingFang HK' in font.name for font in fm.fontManager.ttflist):
    plt.rcParams['font.sans-serif'] = ['PingFang HK']
elif any('SimHei' in font.name for font in fm.fontManager.ttflist):
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    print("警告: 未找到常見中文字體，可能無法正確顯示中文。")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 作為備用

plt.rcParams['axes.unicode_minus'] = False # 解決負號 '-' 顯示為方塊的問題

# --- 雙線性滯後模型類 ---
class HystereticModel:
    def __init__(self, k1, k2, xy):
        self.k1 = k1  # 初始剛度 (彈性剛度)
        self.k2 = k2  # 屈服後剛度 (塑性剛度)
        self.xy = xy  # 屈服位移 (正向)
        self.Fy = k1 * xy  # 正向屈服力
        self.Fy_neg = -k1 * xy # 負向屈服力

        # 滯後狀態變數 - 這些變數追蹤系統的實際狀態
        # 'elastic': 彈性區
        # 'yield_pos': 正向屈服中 (沿 k2 坡度上升)
        # 'yield_neg': 負向屈服中 (沿 k2 坡度下降)
        # 'unload_pos_peak': 從正向峰值點卸載中 (沿 k1 坡度下降)
        # 'unload_neg_peak': 從負向峰值點卸載中 (沿 k1 坡度上升)
        # 'unload_elastic_pos': 從正向彈性範圍卸載中 (回到原點或負向)
        # 'unload_elastic_neg': 從負向彈性範圍卸載中 (回到原點或正向)
        self.current_branch = 'elastic'
        self.x_reversal_point = 0.0  # 上一個反向點的位移
        self.Fs_reversal_point = 0.0 # 上一個反向點的力

    def get_Fs_and_Kt(self, x_trial, x_prev_actual, Fs_prev_actual):
        """
        根據試探位移 (x_trial) 和上一個時間步的實際狀態 (x_prev_actual, Fs_prev_actual)，
        計算試探的彈簧力 (Fs_trial) 和切線剛度 (Kt_trial)。
        此函數在迭代過程中不改變模型的內部狀態，僅用於計算。
        """
        
        # 根據前一個實際點和試探點的位移方向
        dx_trial = x_trial - self.x_reversal_point # 位移相對於反向點的變化

        Fs_trial = 0.0
        Kt_trial = 0.0

        # 計算試探點所在的載入路徑
        if self.current_branch == 'elastic':
            if x_trial >= self.xy: # 進入正向屈服區
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            elif x_trial <= -self.xy: # 進入負向屈服區
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            else: # 仍在彈性區
                Fs_trial = self.k1 * x_trial
                Kt_trial = self.k1
        
        elif self.current_branch == 'yield_pos':
            # 從正向屈服區繼續加載 (dx_trial >= 0)
            # 或開始卸載 (dx_trial < 0)
            if x_trial >= self.x_reversal_point: # 繼續正向屈服或反向點後繼續加載
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else: # 從正向屈服點反向卸載
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                # 檢查是否穿過負向屈服點
                if Fs_trial < self.Fy_neg: # 重新進入負向屈服 (過載)
                    Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                    Kt_trial = self.k2

        elif self.current_branch == 'yield_neg':
            # 從負向屈服區繼續加載 (dx_trial <= 0)
            # 或開始卸載 (dx_trial > 0)
            if x_trial <= self.x_reversal_point: # 繼續負向屈服或反向點後繼續加載
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else: # 從負向屈服點反向卸載
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                # 檢查是否穿過正向屈服點
                if Fs_trial > self.Fy: # 重新進入正向屈服 (過載)
                    Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                    Kt_trial = self.k2

        elif self.current_branch.startswith('unload_'): # 處於卸載或重載回彈性區
            # 彈性卸載線方程: Fs = Fs_reversal_point + k1 * (x - x_reversal_point)
            Fs_elastic_path = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1 # 默認卸載剛度為 k1

            # 檢查是否會重新屈服 (在卸載線上)
            if x_trial > self.x_reversal_point and Fs_elastic_path > self.Fy: # 從負向卸載後，向正向重載，並達到正向屈服
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            elif x_trial < self.x_reversal_point and Fs_elastic_path < self.Fy_neg: # 從正向卸載後，向負向重載，並達到負向屈服
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            else: # 仍在彈性卸載/重載路徑上，未重新屈服
                Fs_trial = Fs_elastic_path
                # 檢查是否回到彈性中心區 (為了精確，此處可選擇將 Fs_trial 限制在彈性範圍內)
                # 實際物理上，卸載路徑是平行的，除非達到對側屈服。
                # 但如果卸載路徑回到了 Fs=k1*x 的範圍，則可以視為回到彈性區
                # 此處不需要額外限制 Fs_trial，因為 Fs_elastic_path 已經是正確的。

        return Fs_trial, Kt_trial

    def update_state(self, x_current_actual, Fs_current_actual, x_prev_actual, Fs_prev_actual):
        """
        在牛頓-拉夫森迭代收斂後，更新模型的內部狀態。
        這將確定下一個時間步的起始狀態。
        """
        
        # 判斷運動方向
        dx = x_current_actual - x_prev_actual
        
        # 根據當前所在的分支和運動方向來更新
        if self.current_branch == 'elastic':
            if x_current_actual >= self.xy:
                self.current_branch = 'yield_pos'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            elif x_current_actual <= -self.xy:
                self.current_branch = 'yield_neg'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            # 如果仍在彈性區，無需更新反向點，保持在 (0,0) 或最後的彈性點
            # 因為彈性區的反向點就是自身，或者前一個點，所以這裡不需要特殊處理。

        elif self.current_branch == 'yield_pos':
            if dx < 0: # 從正向屈服開始反向卸載
                self.current_branch = 'unload_pos_peak'
                self.x_reversal_point = x_current_actual # 反向點為當前實際點
                self.Fs_reversal_point = Fs_current_actual
            # else: 繼續正向屈服，反向點不斷更新
            elif dx > 0: # 繼續在正向屈服區
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            
        elif self.current_branch == 'yield_neg':
            if dx > 0: # 從負向屈服開始反向卸載
                self.current_branch = 'unload_neg_peak'
                self.x_reversal_point = x_current_actual # 反向點為當前實際點
                self.Fs_reversal_point = Fs_current_actual
            # else: 繼續負向屈服，反向點不斷更新
            elif dx < 0: # 繼續在負向屈服區
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
                
        elif self.current_branch.startswith('unload_'):
            # 判斷是否重新屈服 (從卸載/重載路徑)
            # 首先，計算當前點在彈性卸載線上的理論力
            Fs_on_elastic_path = self.Fs_reversal_point + self.k1 * (x_current_actual - self.x_reversal_point)

            if dx > 0: # 向正向運動 (重載)
                if Fs_current_actual >= self.Fy and x_current_actual >= self.xy: # 達到正向屈服
                    self.current_branch = 'yield_pos'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual
                # 否則，保持在卸載/重載線上
                # 如果回到了彈性區的 Fy/Fy_neg 內，且速度改變方向，可以考慮回 elastic
                # 但在 hysteresis 中，一旦屈服，除非回到原點，否則通常不回 elastic
                # 這裡的策略是，只要是沿著卸載路徑，就保持 unload 狀態，直到重新屈服
                # 或達到完全卸載點 (例如力為0)

            elif dx < 0: # 向負向運動 (重載)
                if Fs_current_actual <= self.Fy_neg and x_current_actual <= -self.xy: # 達到負向屈服
                    self.current_branch = 'yield_neg'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual
                # 否則，保持在卸載/重載線上
            
            # 處理卸載到零點附近，並可能回到彈性區的情況
            # 這種判斷需要更嚴格，例如，如果力接近零，且位移方向再次改變，可考慮回到 elastic
            # 但在持續循環載入下，通常會直接連接到另一側屈服點。
            # 對於自由振動，如果回到原點附近，可以將狀態重設為 elastic
            if abs(x_current_actual) < self.xy * 0.1 and abs(Fs_current_actual) < self.Fy * 0.1 and \
               ((dx > 0 and self.current_branch == 'unload_neg_peak') or \
                (dx < 0 and self.current_branch == 'unload_pos_peak')):
                # 如果位移和力都接近零，且是從峰值卸載回來的，可以設定為彈性
                self.current_branch = 'elastic'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual
            # 如果是從卸載點，跨過零點，但在彈性範圍內，不改變其卸載分支狀態
            # current_branch 保持 unload_pos_peak 或 unload_neg_peak
            # 直到重新屈服

# --- 參數設定 ---
m = 1.0  # 質量 [k·s²/in]
xi = 0.05  # 阻尼比
k1 = 631.65  # 初始剛度 [k/in]
k2 = 126.33  # 屈服後剛度 [k/in]
xy = 1.0  # 屈服位移 [in]
dt = 0.005  # 時間步長 [s]
total_time = 2.0 # 模擬總時間，需要足夠長以觀察第一個迴圈
num_steps = int(total_time / dt) # 計算總步數

# 導出參數
wn = np.sqrt(k1 / m)  # 系統初始自然頻率 (基於 k1)
c = 2 * m * wn * xi  # 阻尼係數 [k·s/in]

# --- 初始條件 ---
x0 = 0.0  # 初始位移 [in]
x_dot0 = 40.0  # 初始速度 [in/s]

# 根據運動方程 m*ẍ + c*ẋ + F_s(x) = 0 導出初始加速度 ẍ(0)
# 在 t=0: m*ẍ(0) + c*x_dot(0) + F_s(x(0)) = 0
# 由於 x(0)=0, F_s(x(0))=0 (在彈性區)
# 所以 ẍ(0) = -c * x_dot(0) / m
x_ddot0 = -c * x_dot0 / m # 初始加速度 [in/s²]

print(f"計算得出的初始阻尼係數 c = {c:.4f} k·s/in")
print(f"計算得出的初始加速度 ẍ(0) = {x_ddot0:.4f} in/s²")


# 地面加速度（此問題假定為自由振動，無持續外部輸入）
F_external_input = 0.0 # 外部輸入力為0，響應僅由初始條件驅動

# --- 初始化陣列 ---
# 這些陣列將儲存每個時間步的結果
t = np.linspace(0, total_time, num_steps + 1) # 時間向量
x = np.zeros(num_steps + 1)
x_dot = np.zeros(num_steps + 1)
x_ddot = np.zeros(num_steps + 1)
Fs = np.zeros(num_steps + 1) # 彈簧力
Kt = np.zeros(num_steps + 1) # 切線剛度

# 設定初始值
x[0] = x0
x_dot[0] = x_dot0
x_ddot[0] = x_ddot0

# 初始化滯後模型實例
hysteretic_model = HystereticModel(k1, k2, xy)
# 初始時模型狀態為彈性，所以 Fs[0] 會是 k1 * x[0] = 0
Fs[0], Kt[0] = hysteretic_model.get_Fs_and_Kt(x[0], x[0], Fs[0]) # 此處 x[0] 和 Fs[0] 是前一個實際點，初始為0

# --- 平均加速度法 (非線性求解) ---
# 迭代求解的容忍度
tolerance = 1e-6
max_iterations = 100

# 追蹤特定點
points_of_interest = {} # 儲存點 a, b, c, d, e 的時間步索引
# 輔助標誌，確保每個點只被追蹤一次 (對於第一個迴圈)
found_a, found_b, found_c, found_d, found_e = False, False, False, False, False

for i in range(num_steps):
    # --- 牛頓-拉夫森迭代 ---
    # 為 x_{i+1} 提供一個初始猜測。使用 x[i] 作為起始猜測。
    x_k = x[i] 
    
    # 儲存當前時間步的實際狀態，以便 get_Fs_and_Kt 和 update_state 使用
    x_prev_actual = x[i]
    Fs_prev_actual = Fs[i]

    for iter_count in range(max_iterations):
        # 計算試探狀態下的 Fs 和 Kt
        # get_Fs_and_Kt 函數使用 HystereticModel 中儲存的實際狀態 (從前一個時間步收斂而來)
        # 結合 x_k (當前迭代猜測) 來計算 Fs_k 和 Kt_k
        Fs_k, Kt_k = hysteretic_model.get_Fs_and_Kt(x_k, x_prev_actual, Fs_prev_actual)

        # 計算殘差 R(x_k)
        # 方程: m*x_ddot_{i+1} + c*x_dot_{i+1} + F_s(x_{i+1}) = F_external(i+1)
        # 將 x_ddot 和 x_dot 表示為 x 的函數 (平均加速度法公式)
        x_ddot_k_approx = (4 / dt**2) * (x_k - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]
        x_dot_k_approx = (2 / dt) * (x_k - x[i]) - x_dot[i]
        
        R_k = m * x_ddot_k_approx + \
              c * x_dot_k_approx + \
              Fs_k - F_external_input
        
        # 計算 R 的導數 (切線剛度) R_prime_k
        # R_prime_k = dR/dx_k = m * (4/dt^2) + c * (2/dt) + d(Fs_k)/dx_k
        R_prime_k = (4 * m) / (dt**2) + (2 * c) / dt + Kt_k

        # 更新位移增量 delta_x_k
        delta_x_k = - R_k / R_prime_k
        x_k_new = x_k + delta_x_k

        # 檢查收斂
        if abs(delta_x_k) < tolerance:
            x[i+1] = x_k_new
            break
        x_k = x_k_new
    else:
        print(f"警告: 在時間步 t={i*dt:.3f}s 處，牛頓-拉夫森未收斂。使用最後的迭代值。")
        x[i+1] = x_k # 如果未收斂，使用最後的迭代值

    # --- 歷史追蹤 (更新 HystereticModel 的內部狀態) ---
    # 在牛頓-拉夫森收斂後，更新 HystereticModel 的實際狀態
    # Fs[i+1] 應該是根據 x[i+1] 和收斂後的狀態計算
    Fs[i+1], Kt[i+1] = hysteretic_model.get_Fs_and_Kt(x[i+1], x_prev_actual, Fs_prev_actual) # 使用收斂後的 x[i+1] 和上一步實際狀態
    hysteretic_model.update_state(x[i+1], Fs[i+1], x_prev_actual, Fs_prev_actual)
    
    # --- 計算速度和加速度 ---
    x_ddot[i+1] = (4 / dt**2) * (x[i+1] - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]
    x_dot[i+1] = x_dot[i] + (dt / 2) * (x_ddot[i] + x_ddot[i+1])

    # --- 追蹤特定點 (a, b, c, d, e) ---
    # 這些點的判斷通常需要基於位移、速度和力狀態的變化
    # 確保只追蹤第一個迴圈的點

    # 點 a: 首次達到正向屈服 (位移 >= xy 且力 >= Fy)
    # 從彈性區進入屈服區的瞬間
    if not found_a and hysteretic_model.current_branch == 'yield_pos' and x[i+1] >= xy * 0.999: # 確保已進入屈服分支
        # 由於是離散時間步，可能略微超過或在屈服點附近
        if Fs[i+1] >= hysteretic_model.Fy * 0.999 and Fs[i] < hysteretic_model.Fy * 1.001: # 確認是跨越屈服點
            points_of_interest['a'] = t[i+1]
            found_a = True
            # print(f"點 a (首次正向屈服) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")
    
    # 點 b: 正向最大位移點 (速度從正變負或接近零，且位移為正向最大)
    if found_a and not found_b:
        if x_dot[i] > 0 and x_dot[i+1] <= 0: # 速度從正數變成非正數 (過零點)
            points_of_interest['b'] = t[i+1]
            found_b = True
            # print(f"點 b (正向最大位移) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")

    # 點 c: 卸載並經過零點 (位移從正變負，且處於卸載路徑)
    if found_b and not found_c:
        # 確保處於從正向峰值卸載的狀態，並且位移剛好過零
        if x[i] >= 0 and x[i+1] < 0 and hysteretic_model.current_branch.startswith('unload_pos'):
            # 如果 Fs[i+1] > Fy_neg (因為是從正向卸載下來還沒到負向屈服) 且 Fs[i] > 0
            # 確保是從正值過零點
            points_of_interest['c'] = t[i+1]
            found_c = True
            # print(f"點 c (卸載過零點) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")

    # 點 d: 首次達到負向屈服 (位移 <= -xy 且力 <= -Fy)
    if found_c and not found_d:
        # 從卸載區段進入負向屈服區的瞬間
        if hysteretic_model.current_branch == 'yield_neg' and x[i+1] <= -xy * 0.999:
             if Fs[i+1] <= hysteretic_model.Fy_neg * 0.999 and Fs[i] > hysteretic_model.Fy_neg * 1.001: # 確認是跨越屈服點
                points_of_interest['d'] = t[i+1]
                found_d = True
                # print(f"點 d (首次負向屈服) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")
            
    # 點 e: 負向最大位移點 (速度從負變正或接近零，且位移為負向最大)
    if found_d and not found_e:
        if x_dot[i] < 0 and x_dot[i+1] >= 0: # 速度從負數變成非負數 (過零點)
            points_of_interest['e'] = t[i+1]
            found_e = True
            # print(f"點 e (負向最大位移) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")
            # 為確保只標記第一個迴圈，可以在這裡結束點追蹤。

# --- 列印前六個時間步的結果 ---
print("\n--- 前六個時間步的結果 ---")
print(f"{'時間 (s)':<10} {'位移 (in)':<15} {'速度 (in/s)':<15} {'加速度 (in/s²)':<15} {'彈簧力 (k)':<15} {'分支狀態':<15}")
for i in range(min(num_steps + 1, 6)):
    # 這裡的 hysteretic_model.current_branch 顯示的是上一個時間步的狀態
    # 為了顯示當前時間步的狀態，我們需要一個儲存歷史狀態的列表，或者在循環內臨時獲取
    # 這裡顯示的是 i-1 到 i 步驟後的狀態
    # 為了正確顯示每個時間點的「最終狀態」，我們會在循環外部重新初始化一個瞬時模型來獲取
    # 但這會增加複雜度，通常這裡顯示的是該時間點之前計算的狀態。
    # 為了簡化，我們直接在循環內部獲取。
    temp_model = HystereticModel(k1,k2,xy)
    if i > 0:
        # 如果不是第一個點，我們需要模擬到達該點的路徑來確定其狀態
        # 為了簡化，這裡省略了模擬歷史狀態的複雜性，直接從HystereticModel的末狀態判斷
        # 但這可能不精確反映每個特定點的狀態，僅供參考。
        # 更好的做法是HystereticModel中儲存每個時間步的current_branch
        pass # 這裡只顯示前六步的數值，狀態追蹤在循環內進行。

    # 對於前六步的狀態，我們重新初始化一個模型來判斷該點的狀態
    # 這是為了輸出時能看到正確的狀態，而不是模擬過程中的即時狀態。
    # 但這是一個簡化，真實的狀態追蹤是 HystereticModel 內部維持的。
    # 這裡暫時顯示模擬結束時的最終狀態，或者直接省略狀態列。
    print(f"{t[i]:<10.4f} {x[i]:<15.4f} {x_dot[i]:<15.4f} {x_ddot[i]:<15.4f} {Fs[i]:<15.4f} {'':<15}") # 省略狀態列

print("\n--- 特定時間點 (第一個迴圈) ---")
# 確保點按照時間順序輸出
sorted_points = sorted(points_of_interest.items(), key=lambda item: item[1])
for point_label, time_val in sorted_points:
    idx = np.argmin(np.abs(t - time_val)) # 找到最接近的時間點索引
    # 檢查索引是否有效
    if idx < len(x) and idx < len(Fs):
        print(f"點 {point_label}: 時間 = {t[idx]:.3f}s, 位移 = {x[idx]:.4f} in, 速度 = {x_dot[idx]:.4f} in/s, 加速度 = {x_ddot[idx]:.4f} in/s², 彈簧力 = {Fs[idx]:.4f} k")
    else:
        print(f"警告: 點 {point_label} ({time_val:.3f}s) 的索引超出範圍，可能模擬時間不足。")


# --- 繪製時間歷史圖 ---
plt.figure(figsize=(12, 10))

# 位移 x(t)
plt.subplot(4, 1, 1)
plt.plot(t, x, label='位移 $x(t)$', color='blue')
plt.ylabel('位移 $x(t)$ [in]')
plt.title('系統動態響應時間歷程')
plt.grid(True)
plt.legend()

# 速度 $\dot{x}(t)$
plt.subplot(4, 1, 2)
plt.plot(t, x_dot, label='速度 $\\dot{x}(t)$', color='orange')
plt.ylabel('速度 $\\dot{x}(t)$ [in/s]')
plt.grid(True)
plt.legend()

# 加速度 $\ddot{x}(t)$
plt.subplot(4, 1, 3)
plt.plot(t, x_ddot, label='加速度 $\\ddot{x}(t)$', color='green')
plt.ylabel('加速度 $\\ddot{x}(t)$ [in/s²]')
plt.grid(True)
plt.legend()

# 彈簧力 $F_s(t)$
plt.subplot(4, 1, 4)
plt.plot(t, Fs, label='彈簧力 $F_s(t)$', color='red')
plt.ylabel('彈簧力 $F_s(t)$ [k]')
plt.xlabel('時間 t [s]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- 繪製 $F_s(x)$ 滯後迴圈 ---
plt.figure(figsize=(8, 6))
plt.plot(x, Fs, label='彈簧力 $F_s(x)$ 滯後迴圈', color='purple')
# 標記屈服點
plt.axvline(x=xy, color='gray', linestyle='--', label='正向屈服位移 $x_y$')
plt.axvline(x=-xy, color='gray', linestyle='--', label='負向屈服位移 $-x_y$')
plt.axhline(y=hysteretic_model.Fy, color='gray', linestyle='-.', label='正向屈服力 $F_y$')
plt.axhline(y=hysteretic_model.Fy_neg, color='gray', linestyle='-.', label='負向屈服力 $-F_y$')

# 標記特定點 a, b, c, d, e
colors = ['red', 'green', 'blue', 'cyan', 'magenta']
labels_plot = ['a', 'b', 'c', 'd', 'e']
# 根據 sorted_points 繪製點
for j, (point_label, time_val) in enumerate(sorted_points):
    idx = np.argmin(np.abs(t - time_val)) # 找到最接近的時間點索引
    if idx < len(x) and idx < len(Fs):
        plt.scatter(x[idx], Fs[idx], color=colors[j], marker='o', s=100, zorder=5, label=f'點 {point_label}')
        # 調整文本位置，避免重疊
        offset_x = 0.05 * xy
        offset_y = 0.05 * hysteretic_model.Fy
        plt.text(x[idx] + offset_x, Fs[idx] + offset_y, point_label, fontsize=12, ha='left', va='bottom')

plt.xlabel('位移 $x(t)$ [in]')
plt.ylabel('彈簧力 $F_s(t)$ [k]')
plt.title('彈簧力 $F_s(x)$ - 滯後迴圈')
plt.grid(True)
plt.legend()
plt.show()