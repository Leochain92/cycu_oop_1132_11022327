import requests
import os
import time

# ----- 設定你的 Unsplash Access Key -----
# 請將這行替換成你自己的金鑰
UNSPLASH_ACCESS_KEY = "ea87LsKFtxhy3pf3u0wGiLL_dpt2eqJQMphp_0SLdrI"

# ----- 設定搜尋參數 -----
# 搜尋關鍵字，可根據需求調整
query = "asian engineer safety helmet"

# 目標下載圖片數量
target_image_count = 1000

# 建立圖片下載資料夾
folder_name = "安全帽"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"資料夾 '{folder_name}' 已創建。")

# ----- 下載邏輯 -----
total_downloaded = 0
page = 1
skipped_count = 0

while total_downloaded < target_image_count:
    try:
        # Unsplash API 搜尋接口
        url = "https://api.unsplash.com/search/photos"
        
        # 設定請求參數
        params = {
            "query": query,
            "per_page": 30,  # Unsplash API 每頁最多 30 張
            "page": page,
        }
        
        # 設定請求頭，包含 Access Key
        headers = {
            "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
        }
        
        print(f"正在搜尋 Unsplash 第 {page} 頁圖片...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        photos = data.get("results", [])
        
        if not photos:
            print("沒有找到更多圖片了。")
            break
        
        for photo in photos:
            if total_downloaded >= target_image_count:
                break
            
            # 取得圖片的原始連結
            image_url = photo["urls"]["full"]
            
            # 根據圖片ID命名檔案，確保唯一性
            file_name = f"unsplash_{photo['id']}.jpg"
            file_path = os.path.join(folder_name, file_name)

            # --- 修改部分：檢查檔案是否已存在 ---
            if os.path.exists(file_path):
                print(f"⏩ 圖片已存在，跳過下載：{file_name}")
                skipped_count += 1
                continue
            # ------------------------------------
            
            try:
                # 下載圖片
                image_response = requests.get(image_url, stream=True)
                image_response.raise_for_status()

                with open(file_path, "wb") as f:
                    # 將圖片內容寫入檔案
                    for chunk in image_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                total_downloaded += 1
                print(f"✅ 第 {total_downloaded} 張圖片下載成功：{file_name}")

            except requests.exceptions.RequestException as e:
                print(f"❌ 下載圖片失敗，URL: {image_url}，錯誤: {e}")

        page += 1
        time.sleep(1) # 延遲 1 秒，符合 API 規範

    except requests.exceptions.RequestException as e:
        print(f"API 請求失敗，錯誤: {e}")
        break

print(f"\n--- 所有搜尋與下載流程已完成 ---")
print(f"總共下載了 {total_downloaded} 張新圖片到 '{folder_name}' 資料夾中。")
if skipped_count > 0:
    print(f"共跳過 {skipped_count} 張重複的圖片。")