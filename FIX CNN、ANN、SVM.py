import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import random
import re
from collections import defaultdict
import shutil
from tensorflow.keras.layers import LeakyReLU
# === 基本設定 ===
picture_folder = r"C:\Users\User\Desktop\2025-experimental testing images-red soil\picture"
true_folder = r"C:\Users\User\Desktop\2025-experimental testing images-red soil\true"
output_path = r"C:\Users\User\Desktop\sand_final_result.xlsx"
image_height, image_width = 256, 256
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# === 擷取真實 water content ===
data = {}
for subfolder in os.listdir(true_folder):
    match = re.search(r"water content ([\d\.]+)%", subfolder)
    if not match:
        continue
    wc = float(match.group(1))
    for file in os.listdir(os.path.join(true_folder, subfolder)):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            data[file.replace(" ", "")] = wc

# === 分類與切分資料 ===
all_images = [f for f in os.listdir(picture_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
wc_to_files = defaultdict(list)
for fname in all_images:
    clean_name = fname.replace(" ", "")
    if clean_name in data:
        wc_to_files[data[clean_name]].append(fname)
print('所有 water content 類別:', sorted(wc_to_files.keys()))
for wc in sorted(wc_to_files.keys()):
    print(f"water content {wc}%: {len(wc_to_files[wc])} 張圖片")
train_files, test_files = [], []
for wc, files in wc_to_files.items():
    random.shuffle(files)
    split = max(int(0.8 * len(files)), 10)
    train_files += files[:split]
    test_files += files[split:]

# === 載入影像特徵
def load_image_features(file):
    path = os.path.join(picture_folder, file)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    resized = cv2.resize(image, (image_height, image_width)) / 255.0
    avg_rgb = cv2.mean(image)[:3][::-1]
    avg_hsv = cv2.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[:3]
    wc = data.get(file.replace(" ", ""), None)
    return resized, avg_rgb, avg_hsv, wc
records = []
for file in train_files + test_files:
    result = load_image_features(file)
    if result and result[3] is not None:
        img, rgb, hsv, wc = result
        set_type = "Train" if file in train_files else "Test"
        records.append({
            "Image": file,
            "Set": set_type,
            "Water content": wc,
            "Avg_R": rgb[0], "Avg_G": rgb[1], "Avg_B": rgb[2],
            "Avg_H": hsv[0], "Avg_S": hsv[1], "Avg_V": hsv[2],
            "img_array": img })
df = pd.DataFrame(records)
X_img = np.stack(df['img_array'].values)
y_all = df['Water content'].values
X_train = X_img[df['Set'] == "Train"]
y_train = y_all[df['Set'] == "Train"]
X_test = X_img[df['Set'] == "Test"]
y_test = y_all[df['Set'] == "Test"]

# === CNN 模型
inputs = Input(shape=(image_height, image_width, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
cnn_features = Dense(64, activation='relu', name='cnn_feat')(x)
outputs = Dense(1)(cnn_features)
cnn_model = Model(inputs=inputs, outputs=outputs)
cnn_model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
cnn_model.fit(X_train, y_train, epochs=50, batch_size=8, callbacks=[early_stop], verbose=1)
cnn_pred_all = cnn_model.predict(X_img, verbose=0).flatten()

# === CNN 特徵萃取 → ANN/SVM 用
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("cnn_feat").output)
cnn_feat_all = feature_extractor.predict(X_img)
scaler = StandardScaler()
cnn_feat_train = scaler.fit_transform(cnn_feat_all[df['Set'] == "Train"])
cnn_feat_test = scaler.transform(cnn_feat_all[df['Set'] == "Test"])
cnn_feat_all_scaled = scaler.transform(cnn_feat_all)

# === ANN 三層架構 + Dropout
ann_model = Sequential([
    Dense(128, input_shape=(cnn_feat_train.shape[1],)),
    LeakyReLU(alpha=0.01),
    Dense(64),
    LeakyReLU(alpha=0.01),
    Dense(32),
    LeakyReLU(alpha=0.01),
    Dense(16),
    LeakyReLU(alpha=0.01),
    Dense(1)
])
ann_model.compile(optimizer='adam', loss='mse')

early_stop_ann = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
ann_model.fit(
    cnn_feat_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop_ann],
    verbose=1)
ann_pred_all = ann_model.predict(cnn_feat_all_scaled).flatten()

# === SVM GridSearch 最佳參數
svm_param_grid = {
    'C': [10, 100],
    'gamma': ['scale', 0.1],
    'epsilon': [0.01, 0.1]
}
svm_search = GridSearchCV(SVR(kernel='rbf'), svm_param_grid, cv=3, n_jobs=-1)
svm_search.fit(cnn_feat_train, y_train)
svm_best = svm_search.best_estimator_
svm_pred_all = svm_best.predict(cnn_feat_all_scaled)

# === 統整結果與指標
df['CNN_Pred'] = cnn_pred_all
df['ANN_Pred'] = ann_pred_all
df['SVM_Pred'] = svm_pred_all
df = df.drop(columns=["img_array"])
df_train = df[df['Set'] == 'Train'].reset_index(drop=True)
df_test = df[df['Set'] == 'Test'].reset_index(drop=True)
metrics = []
for name, true_train, pred_train, true_test, pred_test in [
    ("CNN", df_train['Water content'], df_train['CNN_Pred'], df_test['Water content'], df_test['CNN_Pred']),
    ("ANN", df_train['Water content'], df_train['ANN_Pred'], df_test['Water content'], df_test['ANN_Pred']),
    ("SVM", df_train['Water content'], df_train['SVM_Pred'], df_test['Water content'], df_test['SVM_Pred'])
]:
    for set_name, true, pred in [("Train", true_train, pred_train), ("Test", true_test, pred_test)]:
        metrics.append({
            "Model": name,
            "Set": set_name,
            "MAE": mean_absolute_error(true, pred),
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
            "MAPE": mean_absolute_percentage_error(true, pred),
            "R2": r2_score(true, pred)
        })
df_metrics = pd.DataFrame(metrics)

# === 計算誤差欄位
df['CNN_Error'] = abs(df['CNN_Pred'] - df['Water content'])
df['ANN_Error'] = abs(df['ANN_Pred'] - df['Water content'])
df['SVM_Error'] = abs(df['SVM_Pred'] - df['Water content'])

# === Top N 與誤差 >3 輸出圖片函式
def top_errors(df, model_col, err_col, set_type="Test", top_n=10):
    subset = df[df['Set'] == set_type].copy()
    subset = subset[['Image', 'Set', 'Water content', model_col, err_col]]
    subset.columns = ['Image', 'Set', 'True', 'Predicted', 'Error']
    return subset.sort_values(by='Error', ascending=False).head(top_n).reset_index(drop=True)
def copy_top_error_images(df_errors, model_name, set_type):
    folder_name = f"{model_name}_{set_type}TopErrors"
    error_dir = os.path.join(os.path.dirname(output_path), folder_name)
    os.makedirs(error_dir, exist_ok=True)
    for _, row in df_errors.iterrows():
        img_name = row['Image']
        src_path = os.path.join(picture_folder, img_name)
        dst_name = f"{row['True']:.1f}_to_{row['Predicted']:.1f}_{img_name}"
        dst_path = os.path.join(error_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
def copy_errors_above_threshold(df, model_name, pred_col, error_col, threshold=3):
    folder_name = f"{model_name}_ErrorAbove{threshold}"
    error_dir = os.path.join(os.path.dirname(output_path), folder_name)
    os.makedirs(error_dir, exist_ok=True)
    df_above = df[(df['Set'] == 'Test') & (df[error_col] > threshold)].copy()
    for _, row in df_above.iterrows():
        img_name = row['Image']
        src_path = os.path.join(picture_folder, img_name)
        dst_name = f"{row['Water content']:.1f}_to_{row[pred_col]:.1f}_{img_name}"
        dst_path = os.path.join(error_dir, dst_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

# === 寫入 Excel 與輸出誤差圖表與圖片
models = [('CNN_Pred', 'CNN_Error'), ('ANN_Pred', 'ANN_Error'), ('SVM_Pred', 'SVM_Error')]
with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
    df_train.to_excel(writer, sheet_name="Train", index=False)
    df_test.to_excel(writer, sheet_name="Test", index=False)
    df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    for model, err in models:
        name = model.split('_')[0]
        for set_type in ['Train', 'Test']:
            df_errors = top_errors(df, model, err, set_type)
            df_errors.to_excel(writer, sheet_name=f"{name}_{set_type}_Errors", index=False)
            copy_top_error_images(df_errors, name, set_type)
        copy_errors_above_threshold(df, name, model, err, threshold=3)
        print("\n✅ 預測、錯誤圖像、誤差 >3 圖片皆已完整輸出！")

# === 預測圖
plt.figure(figsize=(18, 10))
def plot_pred_vs_true(y_true, pred, title):
    plt.scatter(y_true, pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Water Content')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)

plt.subplot(2, 3, 1); plot_pred_vs_true(df_train['Water content'], df_train['CNN_Pred'], 'CNN Train')
plt.subplot(2, 3, 2); plot_pred_vs_true(df_train['Water content'], df_train['ANN_Pred'], 'ANN Train')
plt.subplot(2, 3, 3); plot_pred_vs_true(df_train['Water content'], df_train['SVM_Pred'], 'SVM Train')
plt.subplot(2, 3, 4); plot_pred_vs_true(df_test['Water content'], df_test['CNN_Pred'], 'CNN Test')
plt.subplot(2, 3, 5); plot_pred_vs_true(df_test['Water content'], df_test['ANN_Pred'], 'ANN Test')
plt.subplot(2, 3, 6); plot_pred_vs_true(df_test['Water content'], df_test['SVM_Pred'], 'SVM Test')
plt.tight_layout()
plt.show()









