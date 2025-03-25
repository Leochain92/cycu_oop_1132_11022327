import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
df = pd.read_csv(r'C:\Users\User\Documents\GitHub\cycu_oop_1132_11022327\20250325\ExchangeRate@202503251832.csv')

# 將 '資料日期' 欄位轉換為日期格式
df['資料日期'] = pd.to_datetime(df['資料日期'], format='%Y%m%d')

# 繪製線圖
plt.plot(df['資料日期'], df['現金1'], label='buy', color='blue')
plt.plot(df['資料日期'], df['現金2'], label='sell', color='red')

# 設定圖表標題和標籤
plt.title('USD Exchange Rates')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()

# 顯示圖表
plt.show()