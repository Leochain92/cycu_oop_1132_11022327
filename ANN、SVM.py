import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os
import os

folder_path = r"C:/done"
print(os.listdir(folder_path))  # 列出該目錄下的所有檔案


# 1. 載入數據
data_path = r"C:/sand5200.csv" 
#在這邊更改檔案路徑、EXCEL存檔格式為UTF-8
data = pd.read_csv(data_path)
# data = pd.read_excel(data_path)  # 讀取 Excel 文件

# 檢查文件是否存在
if not os.path.exists(data_path):
    raise FileNotFoundError(f"檔案不存在：{data_path}")

# 讀取 CSV 文件
try:
    data = pd.read_csv(data_path)
    print("CSV 文件讀取成功。")
except pd.errors.ParserError as e:
    raise ValueError(f"解析 CSV 文件時出錯：{e}")

# 顯示數據前幾行以確認
print("數據預覽：")
print(data.head())

# 2. 確認列名並提取特徵和目標值
expected_columns = ["Avg_R", "Avg_G", "Avg_B", "Avg_H", "Avg_S", "Avg_V", "Water content"]

# 檢查是否所有預期的列都存在
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"CSV 文件中缺少列：{missing_columns}")

# 提取特徵和目標值
X = data[["Avg_R", "Avg_G", "Avg_B", "Avg_H", "Avg_S", "Avg_V"]]
y = data["Water content"]

# 3. 數據預處理
# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 模型訓練與預測
# ANN 模型
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", max_iter=500, random_state=42)

ann_model.fit(X_train, y_train)
ann_pred = ann_model.predict(X_test)

# SVM 模型
svm_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# 5. 模型評估
# ANN 評估
ann_mse = mean_squared_error(y_test, ann_pred)
ann_r2 = r2_score(y_test, ann_pred)
print(f"ANN MSE: {ann_mse:.4f}, R²: {ann_r2:.4f}")

# SVM 評估
svm_mse = mean_squared_error(y_test, svm_pred)
svm_r2 = r2_score(y_test, svm_pred)
print(f"SVM MSE: {svm_mse:.4f}, R²: {svm_r2:.4f}")

# 6. 結果可視化
plt.figure(figsize=(12, 6))

# ANN 模型結果
plt.subplot(1, 2, 1)
plt.scatter(y_test, ann_pred, alpha=0.7, label="Testing data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="1:1 line")
plt.xlabel("Measured water content")
plt.ylabel("Predicted water content by ANN")
plt.title("ANN model")
plt.legend()

# SVM 模型結果
plt.subplot(1, 2, 2)
plt.scatter(y_test, svm_pred, alpha=0.7, label="Testing data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="1:1 line")
plt.xlabel("Measured water content")
plt.ylabel("Predicted water content by SVM")
plt.title("SVM model")
plt.legend()

plt.tight_layout()
plt.show()