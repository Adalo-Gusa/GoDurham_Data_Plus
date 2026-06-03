import os
import re
import json
import time
import math
import shutil
import mimetypes
import requests
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types

# ============================================================
# CONFIGURATION & SETUP
# ============================================================

INPUT_DIR = Path("../images_metadata")          
FALLBACK_DIR = Path("tuning_images_metadata_8headings") # Updated for 8 headings
FINAL_IMAGES_DIR = Path("final_images_sandbox")  # Sandbox specific folder
OUTPUT_JSON = Path("sandbox_results.json")       # Sandbox specific JSON

# Ensure this matches your actual CSV file name path
STOPS_CSV = "../Altered 2026 GoDurham Bus Stop List.csv" 

MODEL = "gemini-3-flash-preview"
GEMINI_PROJECT = "dataplus-godurham" 

# Ensure directories exist
FINAL_IMAGES_DIR.mkdir(exist_ok=True)
FALLBACK_DIR.mkdir(exist_ok=True)

# Initialize API Keys
def load_api_key(path: str) -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"Could not find API key file: {key_path}")
    return key_path.read_text(encoding="utf-8").strip()

GOOGLE_MAPS_API_KEY = load_api_key("../../.api/api_key.txt") 

client = genai.Client(
    vertexai=True,
    project=GEMINI_PROJECT,
    location="global",
)

# Load the bus stop dataframe once globally for the scraper
stops_df = pd.read_csv(STOPS_CSV)

# ============================================================
# 8-HEADING WEB SCRAPER (PASS 2 FALLBACK)
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
    """Fetches 8 sweep images and returns a dictionary of {view_name: Path}"""
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

    # 8-Heading Sweep (45-degree increments)
    sweep_offsets = {
        "n": 0, "ne": 45, "e": 90, "se": 135,
        "s": 180, "sw": 225, "w": 270, "nw": 315
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
# PROMPT & SCHEMA
# ============================================================

PROMPT = """
You are an expert visual analyst evaluating a sequence of Street View images for a transit stop accessibility inventory. 

CRITICAL INSTRUCTION: You must follow a strict TWO-PATH logical workflow. Attempt Path A first. Only proceed to Path B if Path A fails.

=== PATH A: SINGLE-IMAGE FAST-TRACK ===
Scan all provided images for transit infrastructure. Look for: 
1. A bus stop sign (Note: These are often attached to standard wooden utility poles, street light poles, or metal U-channel posts).
2. A bus shelter.
3. A public bench situated directly at the road's shoulder/curb.
(Note: Parked buses, bike racks, fire hydrants, and BARE utility poles WITHOUT signs DO NOT count as bus stops).

If you find a clear view of the transit infrastructure AND the boarding area in ONE single image:
1. Set bus_stop_visible to "Yes".
2. Set best_view to the name of that specific winning image.
3. Classify ALL features (stop_surface, landing_type, amenities) using ONLY that single image. Stop here and output your JSON.

=== PATH B: PANORAMIC SYNTHESIS ===
If no single image provides a perfect view, you must synthesize the visual evidence from ALL images combined to evaluate the continuous environment.

Evaluate the synthesized environment and choose ONE of the following outcomes:

Outcome 1: Synthesized "Yes" (Stop Exists)
- Condition: The combined panorama proves a bus stop exists (e.g., the transit sign is visible in one image, and the concrete landing pad or bench is in another). 
- Action: Set bus_stop_visible to "Yes". Set best_view to the image containing the transit sign, bench, or clearest part of the boarding pad. Synthesize the environment to count all features accurately. DO NOT flag for manual review.

Outcome 2: Definitively "No" (Stop is Missing)
- Condition: You have viewed the entire 360/panoramic area. There is absolutely no transit infrastructure (no bus sign, no shelter, no bench near the curb). You only see general street features.
- Action: Set bus_stop_visible to "No". Classify whatever generic features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: Definitively no bus stop infrastructure visible in any image."

Outcome 3: "Unclear" (Blocked or Obscured)
- Condition: A parked vehicle (bus, car, truck), heavy foliage, or active construction completely blocks the view of the curb. Do NOT assume a stop exists just because a bus is parked there.
- Action: Set bus_stop_visible to "Unclear". Classify whatever background features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: View of the curb is blocked by a vehicle/object."

=== DEFINITIONS ===
1. stop_surface: "Grass" or "Concrete"
2. landing_type: "Paved", "Unpaved", or "Unpaved_Grass_Strip_And_Sidewalk"
3. sidewalk_connection: "Yes" (paved path connects stop to curb), "No" (must cross grass/dirt to reach curb), or "NA"
4. landing_pad: "Two_doors", "One_door", or "NA"
5. shelter_number: Total integer count across the stop area.
6. bench_number: Total integer count across the stop area.
7. trash_can_number: Total integer count across the stop area.
8. street_lighting: "Yes" (dedicated streetlight visible near stop) or "No"

Return only JSON matching the schema.
"""

response_schema = {
    "type": "object",
    "properties": {
        "stop_id": {"type": "string"},
        "best_view": {
            "type": "string",
            "enum": ["left", "center", "right", "n", "ne", "e", "se", "s", "sw", "w", "nw"],
        },
        "selected_image_filename": {"type": "string"},
        "bus_stop_visible": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
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

def fetch_local_views(stop_id: str, directory: Path) -> dict:
    """Finds all images in a directory matching the stop_id."""
    views = {}
    for path in directory.glob(f"{stop_id}_*.jpg"):
        match = re.search(r"_(left|center|right|n|ne|e|se|s|sw|w|nw)_heading", path.name.lower())
        if match:
            views[match.group(1)] = path
    return views

def load_sandbox_json():
    """Loads existing sandbox results to append to."""
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_sandbox_json(results):
    """Saves the combined list back to the sandbox JSON file."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

# ============================================================
# GEMINI CALL (Handles image copying/deletion for the Sandbox)
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
    
    selected_view = result.get("best_view")
    if selected_view not in views:
        selected_view = list(views.keys())[0] 
        
    selected_path = views[selected_view]

    result["stop_id"] = stop_id
    result["selected_image_filename"] = selected_path.name
    destination = FINAL_IMAGES_DIR / selected_path.name

    # Delete previous image if Fallback found a better one
    if pass_number == 2 and previous_image_path:
        old_path = Path(previous_image_path)
        if old_path.exists() and old_path.name != destination.name:
            old_path.unlink()
            print(f"    [Deleted previous Pass 1 image: {old_path.name}]")

    # Save new image
    shutil.copy2(selected_path, destination)
    result["final_image_path"] = str(destination)

    return result

# ============================================================
# INTERACTIVE TERMINAL LOOP
# ============================================================

def main():
    print("=== Bus Stop Prompt Testing Sandbox ===")
    print(f"Model: {MODEL}")
    
    # Load historical tests
    sandbox_results = load_sandbox_json()
    
    #ENTER STOP ID HERE
    #Problem Stops
        #1406
        #5023*
        #6711
        #6631*
    stop_id = 6631
        
    print(f"\nSearching for Pass 1 images for Stop {stop_id} in {INPUT_DIR}...")
    views = fetch_local_views(stop_id, INPUT_DIR)
    
    if not views:
        print(f"Could not find any base images for {stop_id}.")
    else:
        print(f"Found {len(views)} images. Running Gemini (Pass 1)...")
        result = analyze_stop(stop_id, views, pass_number=1)
    
    # Check if we need to trigger Pass 2
    if result.get("bus_stop_visible") in ["No", "Unclear"]:
        print(f"\n[Triggered] Stop {stop_id} is {result['bus_stop_visible']}. Initiating 8-Heading Fallback...")
        
        fallback_views = fetch_local_views(stop_id, FALLBACK_DIR)
        
        # Scrape if we haven't already downloaded them for this stop
        if len(fallback_views) < 8:
            print("Scraping 8 headings from Google Maps API...")
            fallback_views = fetch_stop_images(stop_id)
        
        if fallback_views:
            print(f"Found {len(fallback_views)} fallback images. Running Gemini (Pass 2)...")
            previous_img = result.get("final_image_path")
            result = analyze_stop(stop_id, fallback_views, pass_number=2, previous_image_path=previous_img)
        else:
            print("No fallback images could be scraped.")
    
    # Save to JSON
    sandbox_results.append(result)
    save_sandbox_json(sandbox_results)
    print(f"\nData saved to {OUTPUT_JSON} and image saved to {FINAL_IMAGES_DIR}")

if __name__ == "__main__":
    main()