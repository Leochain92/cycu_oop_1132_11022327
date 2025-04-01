from requests_html import HTMLSession
import requests
import html
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import os

def get_stop_info(stop_link: str) -> dict:
    """下載並儲存指定站點的 HTML 頁面"""
    stop_id = stop_link.split("=")[1]
    url = f'https://pda5284.gov.taipei/MQS/{stop_link}'

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            content = page.content()

            # 確保 bus_stops 資料夾存在
            os.makedirs("bus_stops", exist_ok=True)

            # 儲存 HTML 檔案
            file_path = f"bus_stops/bus_stop_{stop_id}.html"
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
                print(f"✅ 網頁已成功下載並儲存為 {file_path}")

            browser.close()

        return {"stop_id": stop_id, "html_file": file_path}

    except Exception as e:
        print(f"❌ 無法下載站點 {stop_link}: {e}")
        return {"stop_id": stop_id, "html_file": None}


def get_bus_route(rid: str):
    """根據公車路線 ID 取得去程與回程的站點資訊，回傳 DataFrame"""
    url = f'https://pda5284.gov.taipei/MQS/route.jsp?rid={rid}'

    try:
        response = requests.get(url)
        response.raise_for_status()

        # 儲存 HTML 檔案
        with open(f"bus_route_{rid}.html", "w", encoding="utf-8") as file:
            file.write(response.text)

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # 找到所有表格
        tables = soup.find_all("table")

        if not tables:
            raise ValueError("❌ 未找到任何表格資料")

        # 初始化資料列表
        go_stops = []
        return_stops = []

        # 遍歷表格
        for table in tables:
            # 去程站點 (ttego1, ttego2)
            for tr in table.find_all("tr", class_=["ttego1", "ttego2"]):
                td = tr.find("td")
                if td:
                    stop_name = html.unescape(td.text.strip())
                    stop_link = td.find("a")["href"] if td.find("a") else None
                    go_stops.append({"stop_name": stop_name, "stop_link": stop_link})

            # 回程站點 (tteback1, tteback2)
            for tr in table.find_all("tr", class_=["tteback1", "tteback2"]):
                td = tr.find("td")
                if td:
                    stop_name = html.unescape(td.text.strip())
                    stop_link = td.find("a")["href"] if td.find("a") else None
                    return_stops.append({"stop_name": stop_name, "stop_link": stop_link})

        # 轉換為 DataFrame
        df_go = pd.DataFrame(go_stops)
        df_return = pd.DataFrame(return_stops)

        # 下載所有站點的 HTML 頁面
        print("\n🚀 開始下載去程站點詳細資訊...")
        for _, row in df_go.iterrows():
            if row["stop_link"]:
                get_stop_info(row["stop_link"])

        print("\n🚀 開始下載回程站點詳細資訊...")
        for _, row in df_return.iterrows():
            if row["stop_link"]:
                get_stop_info(row["stop_link"])

        return df_go, df_return

    except requests.exceptions.RequestException as e:
        raise ValueError(f"❌ 無法下載網頁: {e}")


# 測試函數
if __name__ == "__main__":
    rid = "10417"  # 測試公車路線 ID
    try:
        df_go, df_return = get_bus_route(rid)

        print("\n🚏 去程站點 DataFrame:")
        print(df_go)

        print("\n🚏 回程站點 DataFrame:")
        print(df_return)

    except ValueError as e:
        print(f"Error: {e}")
