import os, cv2, re, random, numpy as np, pandas as pd
from collections import defaultdict
import tensorflow as tf

# === 基本設定 ===
picture_folder = r"D:\專題\照片\0-experimental testing images-red soil\picture"
true_folder = r"D:\專題\照片\0-experimental testing images-red soil\true"
image_height, image_width = 256, 256
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# === 擷取真實 water content ===
data = {}
for subfolder in os.listdir(true_folder):
    match = re.search(r"water content ([\d\.]+)%", subfolder)
    if not match: continue
    wc = float(match.group(1))
    for file in os.listdir(os.path.join(true_folder, subfolder)):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            data[file.replace(" ", "")] = wc

# === 分類與切分資料 ===
all_images = [f for f in os.listdir(picture_folder) if f.lower().endswith(('jpg','jpeg','png'))]
wc_to_files = defaultdict(list)
for fname in all_images:
    clean_name = fname.replace(" ", "")
    if clean_name in data:
        wc_to_files[data[clean_name]].append(fname)

train_files, test_files = [], []
for wc, files in wc_to_files.items():
    random.shuffle(files)
    split = max(int(0.8*len(files)), 10)
    train_files += files[:split]
    test_files += files[split:]

def load_image_features(file):
    path = os.path.join(picture_folder, file)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None: return None
    resized = cv2.resize(image, (image_height, image_width)) / 255.0
    wc = data.get(file.replace(" ", ""), None)
    return resized, wc

def get_dataset():
    records = []
    for file in train_files + test_files:
        result = load_image_features(file)
        if result and result[1] is not None:
            img, wc = result
            set_type = "Train" if file in train_files else "Test"
            records.append({"Image": file, "Set": set_type, "Water content": wc, "img_array": img})
    df = pd.DataFrame(records)
    X_img = np.stack(df['img_array'].values)
    y_all = df['Water content'].values
    return df, X_img, y_all
