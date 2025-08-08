import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

# 1. 設定資料夾路徑，批量讀取所有圖片
folder_path = r'C:/soil50side'  # 替換成你的資料夾路徑
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 2. 設定模型輸入的圖像大小
image_height, image_width = 256, 256  # 模型需要的圖像大小改為 256x256

# 3. 構建 TensorFlow 模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 編譯模型
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# 初始化一個列表來存儲每張圖片的 RGB 和 HSV 值
data_list = []

# 4. 遍歷資料夾中的每張圖片
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)  # 獲取圖片的完整路徑
    image = cv2.imread(image_path)  # 使用 OpenCV 讀取圖片

    if image is None:
        print(f"Error: 無法讀取圖像 {image_file}，請確認文件是否正確。")
        continue  # 如果圖像無法讀取，跳過該圖像

    # 5. 計算 RGB 值
    avg_rgb = cv2.mean(image)[:3]  # 獲取平均 RGB 值（BGR 格式，取前三個通道）
    avg_rgb = (avg_rgb[2], avg_rgb[1], avg_rgb[0])  # 轉換為 RGB 格式

    # 6. 轉換為 HSV 並計算 HSV 值
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 將 BGR 轉換為 HSV
    avg_hsv = cv2.mean(hsv_image)[:3]  # 獲取平均 HSV 值

    # 7. 預處理圖像
    image_resized = cv2.resize(image, (image_height, image_width))  # 調整大小至 256x256
    image_normalized = image_resized / 255.0  # 將像素值標準化到 [0, 1]
    
    # 添加一個維度，使其符合模型的輸入格式 (1, 256, 256, 3)
    image_input = np.expand_dims(image_normalized, axis=0)

    # 8. 使用模型進行預測
    predictions = model.predict(image_input)

    # 9. 收集每張圖片的數據
    data_list.append({
        'Image': image_file,
        'Prediction': predictions[0][0],  # 假設模型返回一個值
        'Avg_R': avg_rgb[0],
        'Avg_G': avg_rgb[1],
        'Avg_B': avg_rgb[2],
        'Avg_H': avg_hsv[0],
        'Avg_S': avg_hsv[1],
        'Avg_V': avg_hsv[2]
    })

# 10. 使用 pandas 將數據寫入 Excel，並指定文件保存路徑
output_path = r'C:/image_data.xlsx'  ## 替換成你的保存路徑##
sheet_name = os.path.basename(folder_path)  # 使用資料夾名稱作為工作表名稱

with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df = pd.DataFrame(data_list)
    df.to_excel(writer, sheet_name=sheet_name, index=False) 
    #if_sheet_exists='replace' 表示如果工作表已經存在，會覆蓋該工作表。選擇 'overlay'，這樣會在原有的工作表上追加數據而不是覆蓋。

print(f"RGB 和 HSV 值已成功寫入 {output_path} 的工作表 '{sheet_name}'")
