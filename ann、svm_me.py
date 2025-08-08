import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 載入數據
data_path = r"C:/soil15.csv"  # 更新路徑為你的 CSV 文件位置
if not os.path.exists(data_path):
    raise FileNotFoundError(f"檔案不存在：{data_path}")

try:
    data = pd.read_csv(data_path)
    print("CSV 文件讀取成功。")
except pd.errors.ParserError as e:
    raise ValueError(f"解析 CSV 文件時出錯：{e}")

print("數據預覽：")
print(data.head())

# 確認列名
expected_columns = ["Avg_R", "Avg_G", "Avg_B", "Avg_H", "Avg_S", "Avg_V", "Water content"]
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"CSV 文件中缺少列：{missing_columns}")

# 提取特徵和目標值
X = data[["Avg_R", "Avg_G", "Avg_B", "Avg_H", "Avg_S", "Avg_V"]]
y = data["Water content"]

# 2. 數據預處理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#test_size=0.2，表示測試集佔全部資料的20%。剩下的80%用作訓練集。
#random_state=42，確保每次分割資料的結果都一致（重現性）。若不指定 random_state，每次分割資料可能會隨機改變。
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 模型訓練與預測
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", max_iter=500, random_state=42)
ann_model.fit(X_train, y_train)
ann_pred = ann_model.predict(X_test)

svm_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# 4. 模型評估
ann_mse = mean_squared_error(y_test, ann_pred)
ann_r2 = r2_score(y_test, ann_pred)
print(f"ANN MSE: {ann_mse:.4f}, R²: {ann_r2:.4f}")

svm_mse = mean_squared_error(y_test, svm_pred)
svm_r2 = r2_score(y_test, svm_pred)
print(f"SVM MSE: {svm_mse:.4f}, R²: {svm_r2:.4f}")

# 5. 結果可視化
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, ann_pred, alpha=0.7, label="Testing data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="1:1 line")
plt.xlabel("Measured water content")
plt.ylabel("Predicted water content by ANN")
plt.title("ANN model")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, svm_pred, alpha=0.7, label="Testing data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="1:1 line")
plt.xlabel("Measured water content")
plt.ylabel("Predicted water content by SVM")
plt.title("SVM model")
plt.legend()

plt.tight_layout()
plt.show()
