import requests
import html
import pandas as pd
from bs4 import BeautifulSoup

# 設定 Pandas 顯示所有列
pd.set_option('display.max_rows', None)  # 顯示所有列
pd.set_option('display.max_columns', None)  # 顯示所有欄位
pd.set_option('display.width', 1000)  # 設定輸出寬度

url = '''https://pda5284.gov.taipei/MQS/route.jsp?rid=10417'''

# 發送 GET 請求
response = requests.get(url)

# 確保請求成功
if response.status_code == 200:
    # 將內容寫入 bus1.html
    with open("bus1.html", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("網頁已成功下載並儲存為 bus1.html")

    # 重新讀取並解碼 HTML
    with open("bus1.html", "r", encoding="utf-8") as file:
        content = file.read()
        decoded_content = html.unescape(content)  # 解碼 HTML 實體
        print(decoded_content)  # 顯示解碼後的內容

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(content, "html.parser")

    # 找到所有表格
    tables = soup.find_all("table")

    # 初始化資料列表
    all_rows_1 = []  # 用於存放 ttego1 和 tteback1 的資料
    all_rows_2 = []  # 用於存放 ttego2 和 tteback2 的資料

    # 遍歷表格
    for table in tables:
        # 找到所有符合條件的 tr 標籤
        for tr in table.find_all("tr", class_=["ttego1", "tteback1", "ttego2", "tteback2"]):
            # 提取站點名稱和連結
            td = tr.find("td")
            if td:
                stop_name = html.unescape(td.text.strip())  # 解碼站點名稱
                stop_link = td.find("a")["href"] if td.find("a") else None
                # 根據 class 區分去程與回程
                if "ttego1" in tr["class"] or "tteback1" in tr["class"]:
                    stop_type = "去程站點名稱" if "ttego1" in tr["class"] else "回程站點名稱"
                    all_rows_1.append({"類型": stop_type, "站點名稱": stop_name, "連結": stop_link})
                elif "ttego2" in tr["class"] or "tteback2" in tr["class"]:
                    stop_type = "去程站點名稱" if "ttego2" in tr["class"] else "回程站點名稱"
                    all_rows_2.append({"類型": stop_type, "站點名稱": stop_name, "連結": stop_link})

    # 將資料分成兩個 DataFrame
    df_1 = pd.DataFrame(all_rows_1)  # 包含 ttego1 和 tteback1 的資料
    df_2 = pd.DataFrame(all_rows_2)  # 包含 ttego2 和 tteback2 的資料

    # 輸出結果
    print("DataFrame 1 (包含去程與回程):")
    print(df_1)
    print("\nDataFrame 2 (包含去程與回程):")
    print(df_2)
else:
    print(f"無法下載網頁，HTTP 狀態碼: {response.status_code}")
