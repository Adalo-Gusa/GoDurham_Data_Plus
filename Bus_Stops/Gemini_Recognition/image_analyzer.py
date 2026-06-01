import os
import re
import csv
import json
import time
import math
import shutil
import mimetypes
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict

from google import genai
from google.genai import types

# ============================================================
# CONFIG & SETUP
# ============================================================

INPUT_DIR = Path("images_metadata")          
FALLBACK_DIR = Path("images_metadata_6headings")
FINAL_IMAGES_DIR = Path("final_images")
OUTPUT_JSON = Path("bus_stop_results.json")
OUTPUT_CSV = Path("bus_stop_results.csv")

# Ensure this matches your actual CSV file name
STOPS_CSV = "../Altered 2026 GoDurham Bus Stop List.csv" 

MODEL = "gemini-3.5-flash"
GEMINI_PROJECT = "dataplus-godurham" # Your Vertex Project

# Ensure directories exist
FINAL_IMAGES_DIR.mkdir(exist_ok=True)
FALLBACK_DIR.mkdir(exist_ok=True)

def load_api_key(path: str) -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"Could not find API key file: {key_path}")
    return key_path.read_text(encoding="utf-8").strip()

# Initialize API Keys (Adjust the path here if needed)
GOOGLE_MAPS_API_KEY = load_api_key("../.api/api_key.txt") 

client = genai.Client(
    vertexai=True,
    project=GEMINI_PROJECT,
    location="global",
)

# Load the bus stop dataframe once globally
stops_df = pd.read_csv(STOPS_CSV)


# ============================================================
# 6-HEADING WEB SCRAPER (PASS 2)
# ============================================================

def clean_filename(text):
    text = str(text).strip()
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']
    for char in bad_chars:
        text = text.replace(char, "")
    return text.replace(" ", "_")

def get_metadata(lat, lon):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "radius": 25, "key": GOOGLE_MAPS_API_KEY}
    return requests.get(url, params=params).json()

def calculate_heading(from_lat, from_lon, to_lat, to_lon):
    from_lat, from_lon = math.radians(from_lat), math.radians(from_lon)
    to_lat, to_lon = math.radians(to_lat), math.radians(to_lon)
    d_lon = to_lon - from_lon
    x = math.sin(d_lon) * math.cos(to_lat)
    y = math.cos(from_lat) * math.sin(to_lat) - math.sin(from_lat) * math.cos(to_lat) * math.cos(d_lon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def fetch_stop_images(target_stop_code) -> dict:
    """Fetches 6 sweep images and returns a dictionary of {view_name: Path}"""
    stop_data = stops_df[stops_df["Stop Code"] == int(target_stop_code)]
    
    if stop_data.empty:
        print(f"    Error: Stop code {target_stop_code} not found in {STOPS_CSV}.")
        return {}

    row = stop_data.iloc[0]
    stop_name, bus_lat, bus_lon = row["Stop Name"], row["Latitude"], row["Longitude"]

    metadata = get_metadata(bus_lat, bus_lon)
    if metadata.get("status") != "OK":
        return {}

    pano_lat, pano_lon = metadata["location"]["lat"], metadata["location"]["lng"]
    image_date = metadata.get("date", "unknown-date")
    heading = calculate_heading(pano_lat, pano_lon, bus_lat, bus_lon)
    safe_stop_name = clean_filename(stop_name)

    sweep_offsets = {
        "far_left": -75, "mid_left": -45, "slight_left": -15,
        "slight_right": 15, "mid_right": 45, "far_right": 75
    }

    downloaded_views = {}

    for view_name, offset in sweep_offsets.items():
        sweep_heading = (heading + offset) % 360
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": "640x640", "pano": metadata["pano_id"], 
            "heading": sweep_heading, "pitch": 0, "fov": 60, "key": GOOGLE_MAPS_API_KEY
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            filename = FALLBACK_DIR / f"{target_stop_code}_{safe_stop_name}_{image_date}_{view_name}_heading-{round(sweep_heading)}.jpg"
            with open(filename, "wb") as f:
                f.write(response.content)
            downloaded_views[view_name] = filename
        time.sleep(0.1)

    return downloaded_views


# ============================================================
# PARSER & IMAGE GROUPING (PASS 1)
# ============================================================

def parse_stop_id_and_view(path: Path):
    filename = path.name.lower()
    match = re.match(r"^(\d{4})", filename)
    if not match:
        raise ValueError(f"Filename must start with a four-digit stop code: {path.name}")
    
    stop_id = match.group(1)
    view_match = re.search(r"_(left|center|centre|right|far_left|mid_left|slight_left|slight_right|mid_right|far_right)_heading", filename)

    if not view_match:
        raise ValueError(f"Could not determine view from filename: {path.name}.")

    view = view_match.group(1).replace("centre", "center")
    return stop_id, view

def group_images_by_stop(input_dir: Path):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(input_dir.rglob(ext))

    grouped = defaultdict(dict)
    for path in image_paths:
        stop_id, view = parse_stop_id_and_view(path)
        grouped[stop_id][view] = path

    # Only grab groups that have the basic left/center/right trio for Pass 1
    complete_groups = {s: v for s, v in grouped.items() if all(k in v for k in ["left", "center", "right"])}
    return complete_groups


# ============================================================
# GEMINI PROMPT & SCHEMA
# ============================================================

PROMPT = """
You are analyzing bus stop images for a transit stop accessibility inventory.

First, determine whether the provided images appear to show a bus stop or bus stop area.
Set bus_stop_visible:
- "Yes" if at least one image clearly shows a bus stop sign, boarding area, shelter, or obvious bus stop zone.
- "No" if no images appear to show a bus stop.
- "Unclear"  If a large vehicle (bus, truck, car), heavy foliage, or shadows are completely blocking the view of the curb where a stop should be, making it impossible to evaluate.


Then choose which image gives the best overall view of the bus stop (the boarding area, curb, and amenities).

If bus_stop_visible is "Yes", classify the features using ONLY the best selected image.
If bus_stop_visible is "No" or "Unclear", classify what you can, but primarily explain the issue in the notes.

1. stop_surface: "Grass" or "Concrete"
2. landing_type: "Paved", "Unpaved", or "Unpaved_Grass_Strip_And_Sidewalk"
3. sidewalk_connection: "Yes", "No", or "NA"
4. landing_pad: "Two_doors", "One_door", or "NA"
5. shelter_number: Integer count
6. bench_number: Integer count
7. trash_can_number: Integer count
8. street_lighting: "Yes" or "No"

Return only JSON matching the schema.
"""

response_schema = {
    "type": "object",
    "properties": {
        "stop_id": {"type": "string"},
        "best_view": {
            "type": "string",
            "enum": ["left", "center", "right", "far_left", "mid_left", "slight_left", "slight_right", "mid_right", "far_right"],
        },
        "selected_image_filename": {"type": "string"},
        "bus_stop_visible": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
        "bus_stop_visibility_confidence": {"type": "number"},
        "stop_surface": {"type": "string", "enum": ["Grass", "Concrete"]},
        "landing_type": {"type": "string", "enum": ["Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"]},
        "sidewalk_connection": {"type": "string", "enum": ["Yes", "No", "NA"]},
        "landing_pad": {"type": "string", "enum": ["Two_doors", "One_door", "NA"]},
        "shelter_number": {"type": "integer"},
        "bench_number": {"type": "integer"},
        "trash_can_number": {"type": "integer"},
        "street_lighting": {"type": "string", "enum": ["Yes", "No"]},
        "notes": {"type": "string"},
    },
    "required": ["stop_id", "best_view", "selected_image_filename", "bus_stop_visible", "stop_surface", "landing_type", "sidewalk_connection", "landing_pad", "shelter_number", "bench_number", "trash_can_number", "street_lighting", "notes"],
}


# ============================================================
# LOGIC HELPERS
# ============================================================

def make_image_part(path: Path):
    mime_type, _ = mimetypes.guess_type(path)
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type or "image/jpeg")

def enforce_logical_consistency(result: dict) -> dict:
    if result.get("sidewalk_connection") in ["No", "NA"]:
        result["landing_pad"] = "NA"
    if result.get("landing_type") == "Unpaved" and result.get("sidewalk_connection") != "Yes":
        result["landing_pad"] = "NA"
    return result

def force_na_attributes(result: dict) -> dict:
    """Forces all classification fields to NA/0 when a stop is completely invisible after Pass 2."""
    result["stop_surface"] = "Grass" 
    result["landing_type"] = "Unpaved"
    result["sidewalk_connection"] = "NA"
    result["landing_pad"] = "NA"
    result["shelter_number"] = 0
    result["bench_number"] = 0
    result["trash_can_number"] = 0
    result["street_lighting"] = "No"
    
    current_notes = result.get("notes", "No stop visible after 6-heading fallback.")
    if "MANUAL REVIEW REQUIRED" not in current_notes:
        result["notes"] = "MANUAL REVIEW REQUIRED: " + current_notes
        
    return result


# ============================================================
# GEMINI CALL (Dynamic Image Injection + Deletion)
# ============================================================

def analyze_stop(stop_id: str, views: dict, pass_number: int = 1, previous_image_path: str = None) -> dict:
    contents = [PROMPT, f"\nStop ID: {stop_id}\nReview the following {len(views)} images for this bus stop:\n"]
    
    for view_name, path in views.items():
        contents.extend([
            f"{view_name.upper()} VIEW:",
            make_image_part(path),
            f"Filename: {path.name}"
        ])

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    result = json.loads(response.text)
    result = enforce_logical_consistency(result)

    # Safely select the best view returned by the model
    selected_view = result.get("best_view")
    if selected_view not in views:
        selected_view = list(views.keys())[0] 
        
    selected_path = views[selected_view]

    result["stop_id"] = stop_id
    result["selected_image_filename"] = selected_path.name
    destination = FINAL_IMAGES_DIR / selected_path.name

    # Check if we need to delete an old image from Pass 1
    if pass_number == 2 and previous_image_path:
        old_path = Path(previous_image_path)
        if old_path.exists() and old_path.name != destination.name:
            old_path.unlink()
            print(f"    [Deleted previous Pass 1 image: {old_path.name}]")

    shutil.copy2(selected_path, destination)
    result["final_image_path"] = str(destination)

    return result

# ============================================================
# SAVE OUTPUTS
# ============================================================

def write_csv(results):
    if not results: return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def save_json(results):
    if not results: return
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    complete_groups = group_images_by_stop(INPUT_DIR)
    results = []
    
    for stop_id, views in sorted(complete_groups.items()):
        print(f"\nProcessing stop {stop_id} (Pass 1)...")
        
        # PASS 1: The 3-Heading Check
        result = analyze_stop(stop_id, views, pass_number=1)
        
        # Branch Logic
        if result["bus_stop_visible"] in ["No", "Unclear"]:
            print(f"  -> Stop {stop_id} unclear. Triggering 6-heading fallback (Pass 2)...")
            
            fallback_views = fetch_stop_images(stop_id)
            
            if fallback_views:
                # PASS 2: Pass previous image path so it can be deleted if a better one is found
                previous_img = result.get("final_image_path")
                result = analyze_stop(stop_id, fallback_views, pass_number=2, previous_image_path=previous_img)
                
                # Final check
                if result["bus_stop_visible"] in ["No", "Unclear"]:
                    print(f"  -> Stop {stop_id} STILL unclear. Flagging for manual review.")
                    result = force_na_attributes(result)
            else:
                print(f"  -> Scraper failed to find data for {stop_id}. Flagging.")
                result = force_na_attributes(result)
                
        results.append(result)
        
        # Save both CSV and JSON live
        write_csv(results)
        save_json(results)

    print("\nPipeline Finished successfully.")

if __name__ == "__main__":
    main()