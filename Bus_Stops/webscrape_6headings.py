import os
import time
import math
import requests
import pandas as pd

# ---------------------------------------------------------
# 1. GLOBAL SETUP: Load data once for efficiency
# ---------------------------------------------------------
stops_df = pd.read_csv("Altered 2026 GoDurham Bus Stop List.csv")

with open(".api/api_key.txt", "r") as f:
    api_key = f.read().strip()

os.makedirs("images_metadata_6headings", exist_ok=True)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_filename(text):
    text = str(text).strip()
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']
    for char in bad_chars:
        text = text.replace(char, "")
    text = text.replace(" ", "_")
    return text

def get_metadata(lat, lon):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "radius": 25,
        "key": api_key
    }
    response = requests.get(url, params=params)
    return response.json()

def calculate_heading(from_lat, from_lon, to_lat, to_lon):
    """Calculates compass direction from the Street View camera to the bus stop."""
    from_lat = math.radians(from_lat)
    from_lon = math.radians(from_lon)
    to_lat = math.radians(to_lat)
    to_lon = math.radians(to_lon)

    d_lon = to_lon - from_lon

    x = math.sin(d_lon) * math.cos(to_lat)
    y = (
        math.cos(from_lat) * math.sin(to_lat)
        - math.sin(from_lat) * math.cos(to_lat) * math.cos(d_lon)
    )

    heading = math.degrees(math.atan2(x, y))
    return (heading + 360) % 360

# ---------------------------------------------------------
# 3. MAIN FUNCTION
# ---------------------------------------------------------
def fetch_stop_images(target_stop_code):
    """
    Fetches a 6-image panorama for a specific bus stop code.
    """
    # Find the specific stop in the dataframe
    stop_data = stops_df[stops_df["Stop Code"] == target_stop_code]
    
    if stop_data.empty:
        print(f"Error: Stop code {target_stop_code} not found in the CSV.")
        return

    # Grab the data from the first matching row
    row = stop_data.iloc[0]
    stop_name = row["Stop Name"]
    bus_lat = row["Latitude"]
    bus_lon = row["Longitude"]

    print(f"Processing Stop {target_stop_code}: {stop_name}")

    metadata = get_metadata(bus_lat, bus_lon)

    if metadata.get("status") != "OK":
        print(f"  -> Skipping. Google says: {metadata}")
        return

    pano_lat = metadata["location"]["lat"]
    pano_lon = metadata["location"]["lng"]
    image_date = metadata.get("date", "unknown-date")
    print(f"  -> Street View date = {image_date}")

    heading = calculate_heading(
        from_lat=pano_lat,
        from_lon=pano_lon,
        to_lat=bus_lat,
        to_lon=bus_lon
    )

    safe_stop_name = clean_filename(stop_name)

    sweep_offsets = {
        "far_left": -75,
        "mid_left": -45,
        "slight_left": -15,
        "slight_right": 15,
        "mid_right": 45,
        "far_right": 75
    }

    for view_name, offset in sweep_offsets.items():
        sweep_heading = (heading + offset) % 360

        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": "640x640",
            "pano": metadata["pano_id"],
            "heading": sweep_heading,
            "pitch": 0,
            "fov": 60,
            "key": api_key
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"  -> Failed {view_name}: image request error")
            continue

        filename = (
            f"images_metadata_6headings/"
            f"{target_stop_code}_{safe_stop_name}_{image_date}_"
            f"{view_name}_heading-{round(sweep_heading)}.jpg"
        )

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"  -> Saved {view_name} (Heading {round(sweep_heading, 1)})")
        
        # Keep the small delay to avoid hitting API rate limits
        time.sleep(0.1) 

    print(f"Finished processing stop {target_stop_code}.\n")

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
# You can now call the function with any stop code from your CSV:
fetch_stop_images(5005)
