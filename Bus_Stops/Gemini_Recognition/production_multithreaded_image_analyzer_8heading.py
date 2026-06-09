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
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

# ============================================================
# CONFIGURATION & SETUP
# ============================================================


INPUT_DIR = Path("images_metadata")          
FALLBACK_DIR = Path("images_metadata_8headings") 
FINAL_IMAGES_DIR = Path("final_images")  
OUTPUT_JSON = Path("bus_stop_results.json")
OUTPUT_CSV = Path("bus_stop_results.csv") 

# Ensure this matches your actual CSV file name path
STOPS_CSV = "Final 2026 GoDurham Bus Stop List_updated_filtered.csv" 

MODEL = "gemini-3.5-flash"
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

GOOGLE_MAPS_API_KEY = load_api_key("../.api/api_key.txt") 

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
# PROMPTS & SCHEMA
# ============================================================

PROMPT_PASS_1 = """
You are an expert visual analyst evaluating a sequence of Street View images for a transit stop accessibility inventory.

CRITICAL INSTRUCTION: You must follow a strict TWO-PATH logical workflow. Attempt Path A first. Only proceed to Path B if Path A fails.

=== PATH A: SINGLE-IMAGE FAST-TRACK ===
Scan all provided images for clear, explicit transit infrastructure (a dedicated GoDurham bus stop sign, a bus shelter, or a dedicated transit bench).
(Note: Parked buses, generic yellow poles, utility poles, and bike racks DO NOT count).

If you find a clear view of the transit infrastructure AND the boarding area in ONE single image:
1. Set bus_stop_visible to "Yes".
2. Set best_view to the name of that specific winning image.
3. Classify ALL features (stop_surface, landing_type, amenities) using ONLY that single image. 
STILL: Synthesize all of the images to search for a cross walk. 
Stop here and output your JSON.

=== PATH B: PANORAMIC SYNTHESIS ===
If no single image provides a perfect view, you must synthesize the visual evidence from ALL images combined to evaluate the continuous environment.

Evaluate the synthesized environment and choose ONE of the following outcomes:

Outcome 1: Synthesized "Yes" (Stop Exists)
- Condition: The combined panorama proves a bus stop exists (e.g., the transit sign is in the far-left image, but the concrete landing pad and bench are in the mid-left image).
- Action: Set bus_stop_visible to "Yes". Set best_view to the image containing the transit sign or the clearest part of the boarding pad. Synthesize the environment to count all features accurately. DO NOT flag for manual review.

Outcome 2: Definitively "No" (Stop is Missing)
- Condition: You have viewed the entire 360/panoramic area. There is absolutely no transit infrastructure (no bus sign, no shelter). You only see general street features (sidewalks, grass, utility poles).
- Action: Set bus_stop_visible to "No". Classify whatever generic features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: Definitively no bus stop infrastructure visible in any image."

Outcome 3: "Unclear" (Blocked or Obscured)
- Condition: A parked vehicle (bus, car, truck), heavy foliage, or active construction completely blocks the view of the curb. Do NOT assume a stop exists just because a bus is parked there.
- Action: Set bus_stop_visible to "Unclear". Classify whatever background features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: View of the curb is blocked by a vehicle/object."

Important: 
Confidence Scores: bus_stop_visibility, shelter_present, bench_present. trash_can_present
These attributes are extremely important. They will show up as decimal values between 0.0 and 1.0 that will accuratly
Represent your confidence in the presence of each of these attributes. It is important that give your honest rating and actually include
decimal values in between 0.0 and 1.0 in a reproducible way as we will experiment around with different thresholds for a final classifcation.
Remember this when assinging your confidence scores. 

=== DEFINITIONS ===
stop_name: EXACT value saved in variable: "name"
lattidude: EXACT saved in variable: "lattitude"
longitude: EXACT saved in variable: "longitude"
1. shelter_number: Total integer count across ALL images.
2. bench_number: Total integer count across ALL images. If there is a shelter present there is at least one.
3. trash_can_number: Total integer count of ANY visible public trash cans. 
4. stop_surface: "Grass" or "Concrete"
5. landing_type: "Paved", "Unpaved", or "Unpaved_Grass_Strip_And_Sidewalk"
6. sidewalk_connection: "Yes", "No", or "NA"
7. landing_pad: "Two_doors", "One_door", or "NA"
8. cross_walk: "Yes", or "No"
9. street_lighting: "Yes" or "No"

Return only JSON matching the schema.
"""

PROMPT_PASS_2 = """
You are an expert visual analyst evaluating a 360-degree panoramic sequence of a transit stop location. 
YOU ARE A PRECISE MACHINE THAT PRIORITIZES REPRODUCIBILITY. FOLLOW EVER LINE AS IT COMES UP, THE PROMPT IS MENT TO BE FOLLOWED IN ORDER.

INSTRUCTIONS:
You must SYNTHESIZE the visual evidence from ALL images combined to evaluate the continuous environment.

Classification Refresher:
Scan the provided images for clear, explicit transit infrastructure. If ANY of these components are found then the stop EXISTS.
1. A GoDurham bus stop sign (often attached to wooden utility poles or metal posts). Depending on the view you may be able to make out "Bus Stop" written on the stop if close enough and have a appropriate angle.
2. A bus shelter, will likely include a bench situated inside of it.
3. A public bench either included with the bus shelter, or in close proximity to a bus sign or bus shelter. 
4. A public trash can adjacent to the road that is near where pedestrians would walk.
(Note: Parked buses will obstruct view, bike racks, fire hydrants, and BARE utility poles or WITHOUT the up-right rectangle GoDurham bus sign DO NOT count).

Outcome 1: Synthesized "Yes" (Stop Exists)
- Condition: The combined panorama has ANY of the previously listed classification refreshers(1-4) are present throughout the entire synthesis. THEY DO NOT NEED TO BE IN CLOSE PROXIMITY TO EACH OTHER.
- Action: Set bus_stop_visible to "Yes". Set best_view to the EXACT string identifier of the image containing any of the components or the best representation of a component. (e.g., "n", "sw", "right"). Synthesize the environment across all images to count features.
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification, be sure to include the bus stop components in this note.

Outcome 2: Definitively "No" (Stop is Missing)
- Condition: You have viewed the entire 360 area and found ZERO of the components. (There is absolutely no sign, no shelter, no bench, AND no trash can).
- Action: Set bus_stop_visible to "No". Classify features you can see (sidewalks, grass). You MUST begin your notes with "MANUAL REVIEW REQUIRED: Definitively no bus stop infrastructure visible in any image."
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification, be sure to include the bus stop components in this note.

Outcome 3: "Unclear" (Blocked or Obscured)
- Condition: A parked vehicle (bus, car, truck), heavy foliage, or active construction blocks the view of the curb across ALL relevant images.
- Action: Set bus_stop_visible to "Unclear". Classify whatever background features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: View of the curb is blocked by a vehicle/object."
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification

IMPORTANT: 
Confidence Scores: bus_stop_visibility, shelter_present, bench_present. trash_can_present
These attributes are extremely important. They will show up as decimal values between 0.0 and 1.0 that will accuratly
Represent your confidence in the presence of each of these attributes. It is important that give your HONEST rating and actually include
decimal values in between 0.0 and 1.0 in a reproducible way as we will experiment around with different thresholds for a final classifcation.
Remember this when assinging your confidence scores. 

DEFINITIONS:
stop_name: EXACT value saved in variable: "name"
lattidude: EXACT saved in variable: "lattitude"
longitude: EXACT saved in variable: "longitude"
1. shelter_number: Total integer count across ALL images.
2. bench_number: Total integer count across ALL images, if there is a shelter present there is at least one.
3. trash_can_number: Total integer count of ANY visible public trash cans. 
4. stop_surface: "Grass" or "Concrete"
5. landing_type: "Paved", "Unpaved", or "Unpaved_Grass_Strip_And_Sidewalk"
6. sidewalk_connection: "Yes", "No", or "NA"
7. landing_pad: "Two_doors", "One_door", or "NA"
8. cross_walk: "Yes", or "No"
9. street_lighting: "Yes" or "No"

Return only JSON matching the schema.
"""

response_schema = {
    "type": "object",
    "properties": {
        "stop_id": {"type": "string"},
        "stop_name": {"type": "string"},
        "selected_image_filename": {"type": "string"},
        "lattitude": {"type": "number"},
        "longitude": {"type": "number"},
        "best_view": {
            "type": "string",
            "description": "The exact view name (e.g., left, right, center, front, front_right) that best shows the stop."
        },
        "bus_stop_visible": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
        "bus_stop_visibility_confidence": {
            "type": "number",
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a Bus stop. "
            "1.0 means with ALL certainty a Bus Stop EXISTS "
            "0.0 means there is no possibility of a Bus Stop AT ALL"
            
        },
        
        "shelter_present": {
            "type": "number", 
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a shelter. "
            "1.0 means with ALL certainty a shelter EXISTS "
            "0.0 means there is no possibility of a shelter AT ALL"
        },
        "shelter_number": {"type": "integer"},

        "bench_present": {
            "type": "number", 
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a bench. "
            "1.0 means with ALL certainty a bench EXISTS "
            "0.0 means there is no possibility of a bench AT ALL"
        },
        "bench_number": {"type": "integer"},

        "trash_can_present": {
            "type": "number", 
            "description": "A decimal value between 0.0 and 1.0 representing confidence that there IS a trash can. "
            "1.0 means with ALL certainty a trash can EXISTS "
            "0.0 means there is no possibility of a trash can AT ALL"
        },
        "trash_can_number": {"type": "integer"},
        
        "stop_surface": {"type": "string", "enum": ["Grass", "Concrete"]},
        "landing_type": {"type": "string", "enum": ["Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"]},
        "sidewalk_connection": {"type": "string", "enum": ["Yes", "No", "NA"]},
        "landing_pad": {"type": "string", "enum": ["Two_doors", "One_door", "NA"]},
        "cross_walk": {"type": "string", "enum": ["Yes", "No"]},
        "street_lighting": {"type": "string", "enum": ["Yes", "No"]},
        "date": {
            "type": "string",
            "description": "The exact year and month the image was taken (e.g., 2026-1, 2025-3) explicitly stated in the file name"
        },
        "notes": {"type": "string"},
    },
    "required": [
        "stop_id", "stop_name", "selected_image_filename", "lattitude", "longitude", "best_view", "bus_stop_visible", "bus_stop_visibility_confidence", 
        "stop_surface", "landing_type", "sidewalk_connection", "landing_pad", 
        "shelter_number", "shelter_present",
        "bench_number", "bench_present",  
        "trash_can_number", "trash_can_present", "date",
        "street_lighting", "notes"
    ],
    "additionalProperties": False
}

# ============================================================
# LOGIC HELPERS
# ============================================================

def make_image_part(path: Path):
    mime_type, _ = mimetypes.guess_type(path)
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type or "image/jpeg")

def fetch_local_views(stop_id: str, directory: Path) -> dict:
    views = {}
    for path in directory.glob(f"{stop_id}_*.jpg"):
        match = re.search(r"_([a-z_]+)_heading", path.name.lower())
        if match:
            views[match.group(1)] = path
    return views

def group_images_by_stop(directory: Path) -> dict:
    """Groups all image files in a directory by their stop_id."""
    groups = defaultdict(dict)
    for path in directory.glob("*.jpg"):
        parts = path.name.split("_")
        if parts and parts[0].isdigit():
            stop_id = parts[0]
            match = re.search(r"_([a-z_]+)_heading", path.name.lower())
            if match:
                groups[stop_id][match.group(1)] = path
    return dict(groups)

def load_existing_results():
    """Loads already processed stops to prevent duplicate API calls."""
    if not OUTPUT_CSV.exists():
        return [], set()
    
    existing_results = []
    processed_ids = set()
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_results.append(row)
            processed_ids.add(row["stop_id"])
            
    return existing_results, processed_ids


def write_csv(results):
    if not results: return
    keys = list(results[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def save_json(results):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

def get_name_lat_long(stop_id):
    # 1. Filter the dataframe where the "Stop Code" column matches your ID
    stop_row = stops_df[stops_df["Stop Code"] == int(stop_id)]
    
    # 2. Safety check: Did we find the stop?
    if stop_row.empty:
        print(f"Error: Stop ID {stop_id} not found in CSV.")
        return None, None
        
    # 3. Extract the exact values using .iloc[0] (which grabs the first matching row)
    lat = stop_row.iloc[0]["Latitude"]
    lon = stop_row.iloc[0]["Longitude"]
    name = stop_row.iloc[0]["Stop Name"]
    
    return name, lat, lon


# ============================================================
# GEMINI CALL 
# ============================================================

def analyze_stop(stop_id: str, name: str, lat: float, lon: float, views: dict, pass_number: int = 1, previous_image_path: str = None) -> dict:
    
    # 1. Dynamically select the correct prompt based on the pass number!
    active_prompt = PROMPT_PASS_1 if pass_number == 1 else PROMPT_PASS_2

    text_context = (
        f"\nStop ID: {stop_id}"
        f"\nStop Name: {name}"
        f"\nLatitude: {lat}"
        f"\nLongitude: {lon}"
    )
    
    contents = [active_prompt, f"\nStop ID: {stop_id}\nReview the following {len(views)} images for this bus stop:\n", text_context]
    
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

    if pass_number == 2 and previous_image_path:
        old_path = Path(previous_image_path)
        if old_path.exists() and old_path.name != destination.name:
            old_path.unlink()
            print(f"    [Deleted previous Pass 1 image: {old_path.name}]")

    shutil.copy2(selected_path, destination)
    result["final_image_path"] = str(destination)

    return result

# ============================================================
# AUTOMATIC MULTI-THREADED PIPELINE
# ============================================================
def main():
    print("=== Bus Stop Automatic Multi-Threaded Pipeline ===")
    print(f"Model: {MODEL}")
    
    # Load and group all available baseline images
    complete_groups = group_images_by_stop(INPUT_DIR)
    results, processed_ids = load_existing_results()
    
    # Filter out stops we've already done
    stops_to_process = {
        stop_id: views for stop_id, views in complete_groups.items() 
        if stop_id not in processed_ids
    }
    
    if not stops_to_process:
        print("All stops have already been processed! Nothing to do.")
        return
        
    print(f"Found {len(processed_ids)} already processed stops.")
    print(f"Starting multi-threaded processing for {len(stops_to_process)} remaining stops (10 threads)...")
    
    # Sort the stop IDs numerically
    sorted_stop_ids = sorted(stops_to_process.keys(), key=lambda x: int(x))

    # Worker function to handle API calls for a single stop
    def process_single_stop(stop_id):
        name, lattitude, longitude = get_name_lat_long(stop_id)
        views = stops_to_process[stop_id]
        print(f"[{stop_id}] Starting Fast Track...")
        
        try:
            result = analyze_stop(stop_id, name, lattitude, longitude, views, pass_number=1)
            
            # Trigger fallback if needed
            if result.get("bus_stop_visible") in ["No", "Unclear"]:
                print(f"[{stop_id}] Triggered 8-Heading Fallback...")
                fallback_views = fetch_local_views(stop_id, FALLBACK_DIR)
                
                if len(fallback_views) < 8:
                    print(f"[{stop_id}] Scraping API for fallback...")
                    fallback_views = fetch_stop_images(stop_id)
                
                if fallback_views:
                    previous_img = result.get("final_image_path")
                    result = analyze_stop(stop_id, name, lattitude, longitude, fallback_views, pass_number=2, previous_image_path=previous_img)
                else:
                    print(f"[{stop_id}] No fallback images found.")
            
            # Enforce "MANUAL REVIEW" text if Pass 2 still fails
            current_notes = result.get("notes", "")
            if result.get("bus_stop_visible") in ["No", "Unclear"] and "MANUAL REVIEW REQUIRED" not in current_notes.upper():
                result["notes"] = "MANUAL REVIEW REQUIRED: " + current_notes

            return result
            
        except Exception as e:
            print(f"[{stop_id}] Error: {e}")
            return None

    # Execute API calls concurrently using 10 threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks to the thread pool
        future_to_stop = {
            executor.submit(process_single_stop, stop_id): stop_id 
            for stop_id in sorted_stop_ids
        }
        
        # Process them as they finish (safely saving in the main thread)
        for future in as_completed(future_to_stop):
            stop_id = future_to_stop[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    
                    # Sort and save live data
                    results_sorted = sorted(results, key=lambda x: int(x["stop_id"]))
                    write_csv(results_sorted)
                    save_json(results_sorted)
                    
                    print(f"[{stop_id}] Finished. Final Classification: {result.get('bus_stop_visible')}")
            except Exception as e:
                print(f"[{stop_id}] Thread resulted in an exception: {e}")

    print("\nPipeline Finished successfully.")


if __name__ == "__main__":
    main()