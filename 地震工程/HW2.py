import numpy as np
import matplotlib.pyplot as plt
import os

# 檔案路徑
file_path = "C:\\Users\\truck\\Downloads\\Northridge_NS.txt"

try:
    # 讀取 Northridge_NS.txt 檔案中的資料
    data = np.loadtxt(file_path)
    time = data[:, 0]  # 時間 (秒)
    acceleration = data[:, 1]  # 加速度 (g)

    # 將加速度單位從 g 轉換為 m/s^2
    acceleration_ms2 = acceleration * 9.81

    # 繪製加速度-時間關係圖
    plt.figure(figsize=(10, 6))
    plt.plot(time, acceleration_ms2)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Northridge Earthquake Acceleration")
    plt.grid(True)

    # 找出最大加速度及其時間
    max_acceleration = np.max(np.abs(acceleration_ms2))
    max_acceleration_time = time[np.argmax(np.abs(acceleration_ms2))]

    # 標示最大加速度
    plt.plot(max_acceleration_time, max_acceleration, 'ro')  # 標示最大值
    plt.text(max_acceleration_time, max_acceleration, f'Max: {max_acceleration:.2f} m/s^2', verticalalignment='bottom', horizontalalignment='right')

    plt.show()

    # 輸出最大加速度值
    print(f"Maximum acceleration: {max_acceleration:.2f} m/s^2")

except FileNotFoundError:
    print(f"錯誤：找不到檔案 {file_path}")
except Exception as e:
    print(f"發生錯誤：{e}")
