import folium
import random
import time
import webbrowser
import re
import csv

# --- 引入 Selenium 相關的庫 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- 獲取公車路線的站牌名稱和真實經緯度函式 ---
def get_bus_route_stops_from_ebus(route_id, bus_name, driver_instance):
    """
    從台北市公車動態資訊系統抓取指定路線的站牌名稱、真實經緯度及預估到站時間。
    返回一個站牌列表，每個元素是字典，包含 'name', 'lat', 'lon', 'stop_id', 'direction', 'estimated_time'。
    """
    print(f"\n正在從 ebus.gov.taipei 獲取路線 '{bus_name}' ({route_id}) 的站牌數據和預估到站時間...")

    url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}'
    wait = WebDriverWait(driver_instance, 20)

    all_stops_data = [] # 用於存放所有站牌數據，包括去程和返程

    try:
        driver_instance.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.auto-list-stationlist-place')))
        time.sleep(2) # 額外延遲確保渲染和JavaScript執行

        # --- 處理去程站牌 ---
        try:
            go_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.stationlist-go")))
            go_button.click()
            print("已點擊 '去程' 按鈕。")
            time.sleep(3) # 給予足夠時間載入去程資料
            
            # 使用更精確的選擇器來匹配站牌列表項，並從中提取所需信息
            go_elements = driver_instance.find_elements(By.CSS_SELECTOR, "#GoDirectionRoute li")
            for element in go_elements:
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, ".auto-list-stationlist-place")
                    name = name_elem.text.strip()
                    
                    stop_id_input = element.find_element(By.CSS_SELECTOR, "input[name='item.UniStopId']")
                    stop_id = stop_id_input.get_attribute("value")
                    
                    lat_input = element.find_element(By.CSS_SELECTOR, "input[name='item.Latitude']")
                    lat = float(lat_input.get_attribute("value"))
                    
                    lon_input = element.find_element(By.CSS_SELECTOR, "input[name='item.Longitude']")
                    lon = float(lon_input.get_attribute("value"))

                    # 抓取預估到站時間
                    status_elem = element.find_element(By.CSS_SELECTOR, ".auto-list-stationlist-position")
                    estimated_time = status_elem.text.strip() if status_elem.text.strip() else "無資料"

                    all_stops_data.append({
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "stop_id": stop_id,
                        "direction": "去程",
                        "estimated_time": estimated_time
                    })
                except Exception as e_stop:
                    # 某些li元素可能不是標準站牌，或者缺少部分資訊，跳過
                    # print(f"處理去程站牌時發生錯誤：{e_stop}，可能為非標準站牌，已跳過。")
                    pass # 不印出太多雜訊
            print(f"已獲取去程 {len([s for s in all_stops_data if s['direction'] == '去程'])} 個站牌數據。")

        except Exception as e_go:
            print(f"處理去程路線時發生錯誤或無去程資料：{e_go}")

        # --- 處理返程站牌 ---
        try:
            return_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.stationlist-come")))
            return_button.click()
            print("已點擊 '返程' 按鈕。")
            time.sleep(3) # 給予足夠時間載入返程資料

            # 使用更精確的選擇器來匹配站牌列表項，並從中提取所需信息
            return_elements = driver_instance.find_elements(By.CSS_SELECTOR, "#BackDirectionRoute li")
            for element in return_elements:
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, ".auto-list-stationlist-place")
                    name = name_elem.text.strip()
                    
                    stop_id_input = element.find_element(By.CSS_SELECTOR, "input[name='item.UniStopId']")
                    stop_id = stop_id_input.get_attribute("value")
                    
                    lat_input = element.find_element(By.CSS_SELECTOR, "input[name='item.Latitude']")
                    lat = float(lat_input.get_attribute("value"))
                    
                    lon_input = element.find_element(By.CSS_SELECTOR, "input[name='item.Longitude']")
                    lon = float(lon_input.get_attribute("value"))

                    # 抓取預估到站時間
                    status_elem = element.find_element(By.CSS_SELECTOR, ".auto-list-stationlist-position")
                    estimated_time = status_elem.text.strip() if status_elem.text.strip() else "無資料"

                    all_stops_data.append({
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "stop_id": stop_id,
                        "direction": "返程",
                        "estimated_time": estimated_time
                    })
                except Exception as e_stop:
                    # 某些li元素可能不是標準站牌，或者缺少部分資訊，跳過
                    # print(f"處理返程站牌時發生錯誤：{e_stop}，可能為非標準站牌，已跳過。")
                    pass # 不印出太多雜訊
            print(f"已獲取返程 {len([s for s in all_stops_data if s['direction'] == '返程'])} 個站牌數據。")

        except Exception as e_return:
            print(f"處理返程路線時發生錯誤或無返程資料：{e_return}")

    except Exception as e:
        print(f"[錯誤] 獲取路線 {bus_name} 站牌數據失敗：{e}")
        all_stops_data = []

    print(f"路線 '{bus_name}' 的站牌數據獲取完成。共 {len(all_stops_data)} 站 (包含去返程)。")
    return all_stops_data

# --- 顯示地圖函式 (更新以使用實際預估時間) ---
def display_bus_route_on_map(route_name, stops_data, bus_location=None):
    """
    將公車路線、站牌和預估時間顯示在地圖上。
    stops_data: 列表，每個元素是一個字典，包含 'name', 'lat', 'lon', 'estimated_time', 'direction'
    bus_location: 字典，包含 'lat', 'lon'，可選 (此版本未使用，因為公車即時位置需額外抓取)
    """
    if not stops_data:
        print(f"沒有路線 '{route_name}' 的站牌數據可顯示。")
        return

    print(f"\n正在為路線 '{route_name}' 生成地圖...")

    # 以所有站牌的中心點為地圖中心
    avg_lat = sum(s["lat"] for s in stops_data) / len(stops_data)
    avg_lon = sum(s["lon"] for s in stops_data) / len(stops_data)
    map_center = [avg_lat, avg_lon]
    m = folium.Map(location=map_center, zoom_start=14)

    # 分離去程和返程，以便繪製兩條線
    go_direction_stops = [s for s in stops_data if s["direction"] == "去程"]
    return_direction_stops = [s for s in stops_data if s["direction"] == "返程"]

    # 添加站牌標記和彈出視窗
    for stop in stops_data:
        stop_name = stop["name"]
        coords = [stop["lat"], stop["lon"]]
        est_time_text = stop.get("estimated_time", "未知")
        direction_text = stop.get("direction", "")

        popup_html = f"<b>{stop_name}</b><br>方向: {direction_text}<br>預估時間: {est_time_text}"
        
        # 根據方向給予不同顏色
        icon_color = "blue" if direction_text == "去程" else "purple"

        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=icon_color, icon="info-sign")
        ).add_to(m)

    # 繪製去程路線路徑
    if len(go_direction_stops) > 1:
        go_route_coords_list = [[s["lat"], s["lon"]] for s in go_direction_stops]
        folium.PolyLine(
            locations=go_route_coords_list,
            color='green',
            weight=5,
            opacity=0.7,
            tooltip=f"路線: {route_name} (去程)"
        ).add_to(m)

    # 繪製返程路線路徑
    if len(return_direction_stops) > 1:
        return_route_coords_list = [[s["lat"], s["lon"]] for s in return_direction_stops]
        folium.PolyLine(
            locations=return_route_coords_list,
            color='orange',
            weight=5,
            opacity=0.7,
            tooltip=f"路線: {route_name} (返程)"
        ).add_to(m)

    # 添加公車當前位置標記 (如果提供) - 這裡保留，但本範例未實際抓取公車即時位置
    if bus_location:
        folium.Marker(
            location=[bus_location["lat"], bus_location["lon"]],
            popup=folium.Popup(f"<b>公車位置</b><br>路線: {route_name}", max_width=200),
            icon=folium.Icon(color="red", icon="bus", prefix="fa")
        ).add_to(m)

    # 將地圖保存為HTML文件並自動打開
    map_filename = f"bus_route_{route_name}_map.html"
    m.save(map_filename)
    print(f"地圖已保存到 '{map_filename}'。")
    print("正在嘗試在瀏覽器中打開地圖...")
    webbrowser.open(map_filename)
    print("✅ 完成！")

# --- 將站牌數據輸出為 CSV 檔案的函式 (更新欄位) ---
def export_stops_to_csv(route_name, stops_data):
    """
    將公車路線的站牌數據輸出為 CSV 檔案。
    stops_data: 列表，每個元素是一個字典，包含 'name', 'lat', 'lon', 'stop_id', 'direction', 'estimated_time'
    """
    if not stops_data:
        print(f"沒有路線 '{route_name}' 的站牌數據可輸出到 CSV。")
        return

    csv_filename = f"bus_route_{route_name}_stops_with_time.csv" # 更改檔案名稱
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # 定義 CSV 檔頭，新增 '方向' 和 '預估到站時間'
            fieldnames = ['方向', '站牌名稱', '預估到站時間', '緯度', '經度', '站牌ID']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader() # 寫入標題行
            for stop in stops_data:
                writer.writerow({
                    '方向': stop.get('direction', ''),
                    '站牌名稱': stop.get('name', ''),
                    '預估到站時間': stop.get('estimated_time', ''),
                    '緯度': stop.get('lat', ''),
                    '經度': stop.get('lon', ''),
                    '站牌ID': stop.get('stop_id', '')
                })
        print(f"站牌數據（含預估時間）已成功輸出到 '{csv_filename}'。")
    except Exception as e:
        print(f"錯誤：輸出 '{csv_filename}' 時發生問題：{e}")

# --- 主程式 (保持不變，因為函式接口已更新) ---
if __name__ == "__main__":
    print("歡迎使用台北市公車路線查詢與地圖顯示工具！")
    print("-----------------------------------")

    # 設置 Selenium WebDriver
    print("正在啟動 Chrome WebDriver...")
    chrome_options = Options()
    # 這裡可以根據需要調整是否無頭模式，若要看瀏覽器操作請設為 False
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    chrome_options.page_load_strategy = 'normal' # 正常載入頁面，等所有資源載入完成

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        print("WebDriver 已啟動。")

        # 預先抓取所有公車路線的名稱和其對應的 route_id
        print("正在獲取所有公車路線列表，請稍候...")
        all_bus_routes_data = []

        driver.get("https://ebus.gov.taipei/ebus")
        wait_initial = WebDriverWait(driver, 30)

        # 1. 等待頁面載入，確保摺疊面板的連結已存在
        wait_initial.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-toggle='collapse'][href*='#collapse']")))
        time.sleep(2) # 給予頁面一些額外時間渲染

        # 2. 展開所有摺疊區塊 (從 collapse1 到 collapse22)
        for i in range(1, 23): # 假設從 collapse1 到 collapse22
            try:
                collapse_link_selector = f"a[href='#collapse{i}']"
                collapse_link = driver.find_element(By.CSS_SELECTOR, collapse_link_selector)

                if collapse_link.get_attribute("aria-expanded") == "false" or "collapse" in collapse_link.get_attribute("class"):
                    driver.execute_script("arguments[0].click();", collapse_link)
                    # print(f"已點擊展開 #collapse{i}...") # 減少雜訊
                    time.sleep(0.3) # 每次點擊後稍微等待，讓內容載入

            except Exception as e:
                # print(f"點擊 #collapse{i} 失敗或該元素不存在: {e}") # 減少雜訊
                pass # 忽略點擊失敗的collapse，繼續下一個

        time.sleep(3) # 在所有區塊點擊完畢後，給予足夠的時間讓所有內容載入到 DOM 中

        # 3. 重新抓取所有包含 'javascript:go' 的連結
        bus_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='javascript:go']")
        for link in bus_links:
            href = link.get_attribute("href")
            name = link.text.strip()
            if href and name:
                try:
                    route_id_match = re.search(r"go\('([^']+)'\)", href)
                    if route_id_match:
                        route_id = route_id_match.group(1)
                        all_bus_routes_data.append({"name": name, "route_id": route_id})
                except Exception as e:
                    print(f"處理連結 {href} 時發生錯誤：{e}，跳過此連結。")
        print(f"已獲取 {len(all_bus_routes_data)} 條公車路線。")

    except Exception as e:
        print(f"錯誤：無法獲取公車路線列表或啟動 WebDriver。原因：{e}")
        print("請檢查您的網路連接或稍後再試。程式將退出。")
        if driver:
            driver.quit()
        exit()

    # --- 顯示所有可讀取的路線 ---
    if all_bus_routes_data:
        print("\n--- 可查詢的公車路線列表 ---")
        display_count = 20
        if len(all_bus_routes_data) > 2 * display_count:
            print(f"部分路線列表 (共 {len(all_bus_routes_data)} 條):")
            for i in range(display_count):
                print(f"- {all_bus_routes_data[i]['name']}")
            print("...")
            for i in range(len(all_bus_routes_data) - display_count, len(all_bus_routes_data)):
                print(f"- {all_bus_routes_data[i]['name']}")
        else:
            for route in all_bus_routes_data:
                print(f"- {route['name']}")
        print("----------------------------")
    else:
        print("\n警告：未獲取到任何公車路線資訊。")

    while True:
        route_name_input = input("\n請輸入您想查詢的公車路線號碼 (請輸入完整的名稱，例如: 299, 0東)，或輸入 'exit' 退出: ").strip()

        if route_name_input.lower() == 'exit':
            print("感謝使用，再見！")
            break

        if not route_name_input:
            print("輸入不能為空，請重試。")
            continue

        selected_route = None
        for route in all_bus_routes_data:
            if route["name"] == route_name_input:
                selected_route = route
                break

        if not selected_route:
            print(f"錯誤：找不到路線 '{route_name_input}' 的相關資料。請確認路線號碼是否正確。")
            continue

        route_id = selected_route["route_id"]
        bus_name = selected_route["name"]

        # 1. 從 ebus.gov.taipei 獲取路線的站牌數據 (現在包含預估時間和方向)
        stops_data = get_bus_route_stops_from_ebus(route_id, bus_name, driver)

        # 2. 顯示路線的站牌和預估時間
        if stops_data:
            print(f"\n路線 '{bus_name}' 的站牌數據：")
            for stop in stops_data:
                print(f"- 方向: {stop['direction']}, 站牌: {stop['name']} (ID: {stop['stop_id']}, 經度: {stop['lon']}, 緯度: {stop['lat']}), 預估到站時間: {stop['estimated_time']}")
        else:
            print(f"未能獲取路線 '{bus_name}' 的站牌數據。")

        # 3. 在地圖上顯示路線和站牌 (不再需要額外的 estimated_times 字典)
        display_bus_route_on_map(bus_name, stops_data)

        # 4. 將站牌數據輸出為 CSV 檔案
        export_stops_to_csv(bus_name, stops_data)

    # 關閉 WebDriver
    if driver:
        driver.quit()
        print("WebDriver 已關閉。")