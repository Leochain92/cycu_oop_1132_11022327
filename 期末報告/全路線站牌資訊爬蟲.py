# -*- coding: utf-8 -*-
import pandas as pd
import re
from playwright.sync_api import sync_playwright
import os
import time # Import time for potential delays

# Define the base path for input and output files
BASE_PATH = r'C:\Users\User\Documents\GitHub\cycu_oop_1132_11022327\期末報告'
OUTPUT_DIR = os.path.join(BASE_PATH, 'bus_stop_data') # New directory for bus stop CSVs

class BusRouteInfo:
    def __init__(self, routeid: str, direction: str = 'go'):
        self.rid = routeid
        self.content = None
        self.url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={routeid}'
        self.dataframe = None # To store parsed bus stop data

        if direction not in ['go', 'come']:
            raise ValueError("Direction must be 'go' or 'come'")

        self.direction = direction

        self._fetch_content()
    
    def _fetch_content(self):
        """
        Fetches the webpage content using Playwright.
        No longer saves the rendered HTML to a local file.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(self.url, timeout=60000) # Increased timeout for page loading
                
                if self.direction == 'come':
                    # Wait for the button to be visible and enabled before clicking
                    page.wait_for_selector('a.stationlist-come-go-gray.stationlist-come', state='visible', timeout=10000)
                    page.click('a.stationlist-come-go-gray.stationlist-come')
                    time.sleep(2) # Give a short delay after clicking to ensure content updates

                # Wait for a specific part of the content to be loaded, e.g., the first station name
                # This is more robust than a fixed timeout
                page.wait_for_selector('span.auto-list-stationlist-place', timeout=10000)
                
                self.content = page.content()
            except Exception as e:
                print(f"Error fetching content for route {self.rid}, direction {self.direction}: {e}")
                self.content = None # Set content to None if fetching fails
            finally:
                browser.close()

        # REMOVED: No longer saving the rendered HTML to a file
        # with open(f"data/ebus_taipei_{self.rid}.html", "w", encoding="utf-8") as file:
        #     file.write(self.content)

    def parse_route_info(self) -> pd.DataFrame:
        """
        Parses the fetched HTML content to extract bus stop data.
        """
        if self.content is None:
            print(f"No content to parse for route {self.rid}, direction {self.direction}.")
            return pd.DataFrame() # Return empty DataFrame if content is not available

        pattern = re.compile(
            r'<li>.*?<span class="auto-list-stationlist-position.*?">(.*?)</span>\s*'
            r'<span class="auto-list-stationlist-number">\s*(\d+)</span>\s*'
            r'<span class="auto-list-stationlist-place">(.*?)</span>.*?'
            r'<input[^>]+name="item\.UniStopId"[^>]+value="(\d+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Latitude"[^>]+value="([\d\.]+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Longitude"[^>]+value="([\d\.]+)"[^>]*>',
            re.DOTALL
        )

        matches = pattern.findall(self.content)
        if not matches:
            print(f"No stop data found with regex for route ID {self.rid}, direction {self.direction}. "
                  f"Might be an empty route or pattern mismatch.")
            return pd.DataFrame() # Return empty DataFrame if no matches

        bus_stops_data = [m for m in matches]
        self.dataframe = pd.DataFrame(
            bus_stops_data,
            columns=["arrival_info", "stop_number", "stop_name", "stop_id", "latitude", "longitude"]
        )

        # Convert appropriate columns to numeric types
        # Using errors='coerce' will turn unparseable values into NaN
        self.dataframe["stop_number"] = pd.to_numeric(self.dataframe["stop_number"], errors='coerce')
        self.dataframe["stop_id"] = pd.to_numeric(self.dataframe["stop_id"], errors='coerce')
        self.dataframe["latitude"] = pd.to_numeric(self.dataframe["latitude"], errors='coerce')
        self.dataframe["longitude"] = pd.to_numeric(self.dataframe["longitude"], errors='coerce')

        self.dataframe["direction"] = self.direction
        self.dataframe["route_id"] = self.rid

        return self.dataframe

    def export_to_csv(self, output_filepath: str):
        """
        Exports the parsed bus stop data for a single route and direction to a CSV file.
        """
        if self.dataframe is None or self.dataframe.empty:
            print(f"No data to export for route {self.rid}, direction {self.direction}.")
            return

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        self.dataframe.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"Successfully exported route {self.rid} ({self.direction}) stops to: {output_filepath}")


if __name__ == "__main__":
    # 1. 確保輸出目錄存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. 讀取所有路線代碼
    route_list_csv_path = os.path.join(BASE_PATH, 'taipei_bus_routes.csv')
    try:
        df_routes = pd.read_csv(route_list_csv_path, encoding='utf-8-sig')
        # Filter out potential problematic routes or ensure 'route_id' column exists
        if 'route_id' not in df_routes.columns:
            raise ValueError(f"'{route_list_csv_path}' must contain a 'route_id' column.")
        
        all_route_ids = df_routes['route_id'].tolist()
        print(f"Successfully read {len(all_route_ids)} route IDs from {route_list_csv_path}.")
    except FileNotFoundError:
        print(f"Error: '{route_list_csv_path}' not found. Please run the previous script to generate it.")
        exit() # Exit if the input CSV is missing
    except Exception as e:
        print(f"Error reading CSV file '{route_list_csv_path}': {e}")
        exit()

    # 3. 遍歷每個路線代碼，抓取站點資訊並儲存為 CSV
    processed_count = 0
    for route_id in all_route_ids:
        print(f"\n--- Processing Route: {route_id} ---")
        
        # Process 'go' direction
        try:
            print(f"  Fetching 'go' direction for {route_id}...")
            route_info_go = BusRouteInfo(route_id, direction="go")
            df_stops_go = route_info_go.parse_route_info()
            
            if not df_stops_go.empty:
                output_filename_go = f"bus_stops_{route_id}_go.csv"
                output_filepath_go = os.path.join(OUTPUT_DIR, output_filename_go)
                route_info_go.export_to_csv(output_filepath_go)
            else:
                print(f"  No stop data found for route {route_id} (go direction). Skipping CSV export.")
            
            time.sleep(1) # Add a small delay between requests to avoid overwhelming the server

        except Exception as e:
            print(f"  Error processing 'go' direction for route {route_id}: {e}")
            
        # Process 'come' direction
        try:
            print(f"  Fetching 'come' direction for {route_id}...")
            route_info_come = BusRouteInfo(route_id, direction="come")
            df_stops_come = route_info_come.parse_route_info()

            if not df_stops_come.empty:
                output_filename_come = f"bus_stops_{route_id}_come.csv"
                output_filepath_come = os.path.join(OUTPUT_DIR, output_filename_come)
                route_info_come.export_to_csv(output_filepath_come)
            else:
                print(f"  No stop data found for route {route_id} (come direction). Skipping CSV export.")

            time.sleep(1) # Add a small delay between requests

        except Exception as e:
            print(f"  Error processing 'come' direction for route {route_id}: {e}")
        
        processed_count += 1
        print(f"--- Finished processing route {route_id}. ({processed_count}/{len(all_route_ids)}) ---")
        time.sleep(3) # Add a slightly longer delay between routes

    print(f"\nAll {processed_count} routes processed. Bus stop data saved to {OUTPUT_DIR}")