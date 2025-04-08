import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def plot_lognormal_cdf(mu, sigma, x_range, output_file):
    """
    繪製對數常態分布的累積分布函數 (CDF) 並儲存為 JPG 檔案。

    :param mu: 對數常態分布的 μ
    :param sigma: 對數常態分布的 σ
    :param x_range: x 軸範圍 (tuple，格式為 (start, end, num_points))
    :param output_file: 輸出的 JPG 檔案名稱
    """
    # 計算對數常態分布的 s 和 scale
    s = sigma
    scale = np.exp(mu)

    # 定義 x 軸範圍
    x = np.linspace(*x_range)

    # 計算累積分布函數 (CDF)
    cdf = lognorm.cdf(x, s, scale=scale)

    # 繪製圖形
    plt.figure(figsize=(8, 6))
    plt.plot(x, cdf, label='Lognormal CDF', color='blue')
    plt.title('Lognormal Cumulative Distribution Function')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()

    # 儲存為 JPG 檔案
    plt.savefig(output_file, format='jpg')
    plt.show()

# 主程式
if __name__ == "__main__":
    # 定義參數
    mu = 1.5  # μ
    sigma = 0.4  # σ
    x_range = (0.01, 10, 1000)  # x 軸範圍 (起始, 結束, 點數)
    output_file = 'lognormal_cdf.jpg'  # 輸出的檔案名稱

    # 呼叫函式繪製圖形
    plot_lognormal_cdf(mu, sigma, x_range, output_file)