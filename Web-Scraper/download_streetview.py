import os
import time
import requests
import pandas as pd

stops = pd.read_csv("Altered 2026 GoDurham Bus Stop List.csv")
stops = stops.head(10)
with open(".api/api_key.txt", "r") as f:
    api_key = f.read().strip()

os.makedirs("images", exist_ok=True)

headings = [0, 90, 180, 270]

for _, row in stops.iterrows():
    stop_code = row["Stop Code"]
    stop_name = row["Stop Name"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    for heading in headings:
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": "640x640",
            "location": f"{lat},{lon}",
            "radius": 25,
            "heading": heading,
            "pitch": 0,
            "fov": 90,
            "key": api_key
        }

        response = requests.get(url, params=params)

        filename = f"images/{stop_code}_{heading}.jpg"

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"Saved {filename} - {stop_name}")

        time.sleep(0.1)

print("Done.")

    
