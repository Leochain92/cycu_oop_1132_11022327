import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Point, LineString
import os

# --- 設定中文字型 ---
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 載入公車路線資料的輔助函式 ---
def read_route_csv(csv_path):
    """讀取公車路線CSV檔案並轉換為GeoDataFrame"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    # 確保經緯度欄位名稱正確，通常是 'longitude', 'latitude'
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf

# --- 主繪圖函式 ---
def plot_region_and_bus_routes(shp_dir: str, bus_route_files: list, output_path: str):
    """
    繪製北北基桃區界圖，並疊加公車路線。

    Args:
        shp_dir (str): 包含區界Shapefile的目錄路徑。
        bus_route_files (list): 包含公車路線CSV檔案路徑的列表。
        output_path (str): 輸出圖片的儲存路徑。
    """
    # 找出指定目錄下的第一個 .shp 檔案 (區界圖)
    shp_file = None
    for fname in os.listdir(shp_dir):
        if fname.endswith(".shp"):
            shp_file = os.path.join(shp_dir, fname)
            break

    if shp_file is None:
        print(f"在 {shp_dir} 中找不到 Shapefile。")
        return

    # 讀取區界 Shapefile
    gdf_regions = gpd.read_file(shp_file)

    # 篩選出北北基桃的區界
    target_counties = ['臺北市', '新北市', '基隆市', '桃園市']
    # 假設縣市名稱欄位是 'COUNTYNAME'。如果不是，請檢查並修改此處
    if 'COUNTYNAME' in gdf_regions.columns:
        gdf_filtered_regions = gdf_regions[gdf_regions['COUNTYNAME'].isin(target_counties)]
    else:
        print("警告: Shapefile中找不到'COUNTYNAME'欄位，無法篩選縣市。將繪製所有區域。")
        gdf_filtered_regions = gdf_regions

    # --- 繪圖設定 ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 14)) # 調整 figsize 以容納更多內容

    # 繪製北北基桃區界
    gdf_filtered_regions.plot(
        column='COUNTYNAME',
        ax=ax,
        legend=True,
        cmap='tab20',  # 使用更多樣的顏色映射
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7 # 讓區域半透明，以便看清下面的公車路線
    )
    # 調整圖例位置
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1.05, 1)) # 將圖例移到圖外右側

    # 疊加公車路線
    bus_route_colors = ['red', 'blue', 'green', 'purple', 'orange'] # 預備多條線用不同顏色
    for idx, file in enumerate(bus_route_files):
        gdf_route = read_route_csv(file)
        route_color = bus_route_colors[idx % len(bus_route_colors)]
        route_name = os.path.basename(file).replace('.csv', '') # 從檔案名取得路線名稱

        # 繪製公車路線點 (仍然保留點，但可以縮小或修改樣式)
        gdf_route.plot(
            ax=ax,
            color=route_color,
            marker='o',
            markersize=8, # 稍微縮小點的大小
            label=f"{route_name} - 站點",
            zorder=5 # 確保點在區界圖之上
        )

        # 繪製公車路線線條
        line_geometry = LineString(gdf_route.geometry.tolist())
        line_gdf = gpd.GeoDataFrame([1], geometry=[line_geometry], crs=gdf_route.crs)
        line_gdf.plot(
            ax=ax,
            color=route_color,
            linewidth=2.5, # 加粗線條使其更明顯
            linestyle='-', # 使用實線
            label=f"{route_name} - 路線",
            zorder=4 # 確保線條在區界圖之上
        )

        # 移除站名顯示的迴圈
        # for x, y, name in zip(gdf_route.geometry.x, gdf_route.geometry.y, gdf_route["車站名稱"]):
        #     ax.text(x, y, name, fontsize=8, ha='left', va='center', color='darkslategray', weight='bold', zorder=6)


    # 設定標題與圖例
    ax.set_title("北北基桃區界圖與公車路線疊加", fontsize=20, pad=20)
    ax.set_xlabel("經度", fontsize=12)
    ax.set_ylabel("緯度", fontsize=12)
    ax.set_aspect('equal') # 保持地圖比例不變形
    ax.grid(True, linestyle=':', alpha=0.6) # 增加格線

    # 確保圖例包含所有元素
    ax.legend(title="圖例", loc='upper left', bbox_to_anchor=(1.05, 1))


    # 儲存圖檔
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' 確保所有元素都被保存
    print(f"圖表已儲存至 {output_path}")

    # 顯示圖表
    plt.show()

# --- 執行主程式 ---
if __name__ == "__main__":
    # 區界圖的 Shapefile 目錄 (保持不變，因為你的 .shp 檔在這裡)
    region_shp_directory = r"C:\Users\User\Documents\GitHub\cycu_oop_1132_11022327\20250520"

    # 更改為新的公車路線 CSV 檔案名稱
    bus_route_csv_files = [
        os.path.join(region_shp_directory, "bus_route_0161000900.csv"),
        os.path.join(region_shp_directory, "bus_route_0161001500.csv")
    ]

    # 輸出圖片的路徑 (可以更改檔名)
    output_image_path = os.path.join(region_shp_directory, "北北基桃_Bus_Routes_Overlay_NoStationNames.png")

    plot_region_and_bus_routes(
        shp_dir=region_shp_directory,
        bus_route_files=bus_route_csv_files,
        output_path=output_image_path
    )