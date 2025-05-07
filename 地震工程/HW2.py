import numpy as np
import matplotlib.pyplot as plt
# 設定字型以支援中文
rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 使用微軟正黑體
rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示

def avg_acceleration_method(ground_acceleration_data, m, k, dt, L_col):
    """
    使用平均加速度法計算 SDOF 系統的位移、速度、加速度、剪力和彎矩。

    Args:
        ground_acceleration_data (numpy.ndarray): 地面加速度時間序列。
        m (float): 質量。
        k (float): 剛度。
        dt (float): 時間步長。
        L_col (float): 柱長度，用於計算彎矩。

    Returns:
        tuple: 包含時間、位移、速度、加速度、剪力和彎矩的 NumPy 陣列。
    """

    time = ground_acceleration_data[:, 0]
    ground_acceleration = ground_acceleration_data[:, 1]

    num_steps = len(time)
    x = np.zeros(num_steps)
    x_dot = np.zeros(num_steps)
    x_ddot = np.zeros(num_steps)
    V = np.zeros(num_steps)  # 剪力
    M = np.zeros(num_steps)  # 彎矩

    # 初始條件
    x[0] = 0.0
    x_dot[0] = 0.0
    x_ddot[0] = -ground_acceleration[0]

    # 阻尼係數
    xi = 0.05
    c = 2 * xi * np.sqrt(m * k)

    # 時間步進
    for i in range(num_steps - 1):
        # 預測
        x_pred = x[i] + x_dot[i] * dt + 0.5 * x_ddot[i] * dt**2
        x_dot_pred = x_dot[i] + 0.5 * x_ddot[i] * dt

        # 有效剛度和力
        k_eff = m + 0.5 * c * dt + 0.25 * k * dt**2
        f_eff = -m * ground_acceleration[i+1] - c * x_dot_pred - k * x_pred

        # 加速度
        x_ddot[i+1] = f_eff / k_eff

        # 更新
        x[i+1] = x_pred + 0.25 * x_ddot[i+1] * dt**2
        x_dot[i+1] = x_dot_pred + 0.5 * x_ddot[i+1] * dt

        # 計算剪力和彎矩
        V[i+1] = m * x_ddot[i+1] + c * x_dot[i+1] # 剪力 = 慣性力 + 阻尼力
        M[i+1] = V[i+1] * L_col                  # 彎矩 = 剪力 * 柱長度

    return time, x, x_dot, x_ddot, V, M

if __name__ == "__main__":
    # 載入地面加速度資料
    file_path = r"c:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\地震工程\Northridge_NS.txt"
    ground_acceleration_data = np.loadtxt(file_path)

    # 系統參數
    m = 0.7505  # ksi*s^2/in
    k_NS = 92.0  # ksi/in
    k_EW = 302.0 # ksi/in
    dt = 0.02
    L_col_NS = 168.0 # in
    L_brace_EW = 343.84 # in

    # 執行分析
    time, x_NS, x_dot_NS, x_ddot_NS, V_NS, M_NS = avg_acceleration_method(
        ground_acceleration_data, m, k_NS, dt, L_col_NS
    )
    time, x_EW, x_dot_EW, x_ddot_EW, V_EW, M_EW = avg_acceleration_method(
        ground_acceleration_data, m, k_EW, dt, L_brace_EW # 注意這裡用 L_brace_EW
    )

    # 計算最大值
    max_x_NS = np.max(np.abs(x_NS))
    max_x_dot_NS = np.max(np.abs(x_dot_NS))
    max_x_ddot_NS = np.max(np.abs(x_ddot_NS))
    max_V_NS = np.max(np.abs(V_NS))
    max_M_NS = np.max(np.abs(M_NS))

    max_x_EW = np.max(np.abs(x_EW))
    max_x_dot_EW = np.max(np.abs(x_dot_EW))
    max_x_ddot_EW = np.max(np.abs(x_ddot_EW))
    max_V_EW = np.max(np.abs(V_EW))
    max_M_EW = np.max(np.abs(M_EW))

    # 輸出結果
    print("N-S 方向最大值:")
    print(f"|x|_max: {max_x_NS:.4f} in")
    print(f"|x_dot|_max: {max_x_dot_NS:.4f} in/s")
    print(f"|x_ddot|_max: {max_x_ddot_NS:.4f} in/s^2")
    print(f"|V|_max: {max_V_NS:.4f} ksi")
    print(f"|M|_max: {max_M_NS:.4f} ksi*in")

    print("\nE-W 方向最大值:")
    print(f"|x|_max: {max_x_EW:.4f} in")
    print(f"|x_dot|_max: {max_x_dot_EW:.4f} in/s")
    print(f"|x_ddot|_max: {max_x_ddot_EW:.4f} in/s^2")
    print(f"|V|_max: {max_V_EW:.4f} ksi")
    print(f"|M|_max: {max_M_EW:.4f} ksi*in")

    # 繪製結果 (與之前類似，但包含剪力和彎矩)
    plt.figure(figsize=(12, 12))

    # 位移
    plt.subplot(4, 1, 1)
    plt.plot(time, x_NS, label="位移 (k_NS)")
    plt.plot(time, x_EW, label="位移 (k_EW)")
    plt.ylabel("位移 (in)")
    plt.legend()

    # 速度
    plt.subplot(4, 1, 2)
    plt.plot(time, x_dot_NS, label="速度 (k_NS)")
    plt.plot(time, x_dot_EW, label="速度 (k_EW)")
    plt.ylabel("速度 (in/s)")
    plt.legend()

    # 加速度
    plt.subplot(4, 1, 3)
    plt.plot(time, x_ddot_NS, label="加速度 (k_NS)")
    plt.plot(time, x_ddot_EW, label="加速度 (k_EW)")
    plt.ylabel("加速度 (in/s^2)")
    plt.legend()

    # 剪力
    plt.subplot(4, 1, 4)
    plt.plot(time, V_NS, label="剪力 (k_NS)")
    plt.plot(time, V_EW, label="剪力 (k_EW)")
    plt.ylabel("剪力 (ksi)")
    plt.xlabel("時間 (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()