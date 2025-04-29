import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Point
import os

# 加這段以支援中文字
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def draw_csv_to_png(inputfile: str, outputfile: str):
    df = pd.read_csv(inputfile, encoding='utf-8')
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='blue', markersize=5)

    for x, y, name in zip(gdf.geometry.x, gdf.geometry.y, gdf["車站名稱"]):
        ax.text(x, y, name, fontsize=6, ha='left', va='center')

    ax.set_title("Bus Stops")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    plt.savefig(outputfile, dpi=300)
    plt.close()

if __name__ == "__main__":
    inputfile = "20250429/bus_route_0161000900.csv"
    outputfile = "20250429/0161000900_bus_stops.png"
    draw_csv_to_png(inputfile, outputfile)