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
FALLBACK_DIR = Path("images_metadata_8headings")
FINAL_IMAGES_DIR = Path("final_images")
OUTPUT_JSON = Path("bus_stop_results.json")
OUTPUT_CSV = Path("bus_stop_results.csv")

# Ensure this matches your actual CSV file name
STOPS_CSV = "../Altered 2026 GoDurham Bus Stop List.csv"

MODEL = "gemini-3.5-flash"
GEMINI_PROJECT = "dataplus-godurham"

# Ensure directories exist
FINAL_IMAGES_DIR.mkdir(exist_ok=True)
FALLBACK_DIR.mkdir(exist_ok=True)


def load_api_key(path: str) -> str:
    key_path = Path(path)

    if not key_path.exists():
        raise FileNotFoundError(f"Could not find API key file: {key_path}")

    api_key = key_path.read_text(encoding="utf-8").strip()

    if not api_key:
        raise ValueError(f"API key file is empty: {key_path}")

    return api_key


# Initialize API Keys
GOOGLE_MAPS_API_KEY = load_api_key("../.api/api_key.txt")

client = genai.Client(
    vertexai=True,
    project=GEMINI_PROJECT,
    location="global",
)

# Load the bus stop dataframe once globally
stops_df = pd.read_csv(STOPS_CSV)


# ============================================================
# 8-HEADING WEB SCRAPER (PASS 2)
# ============================================================

def clean_filename(text):
    text = str(text).strip()
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']

    for char in bad_chars:
        text = text.replace(char, "")

    return text.replace(" ", "_")


def normalize_stop_code(stop_code):
    return str(stop_code).strip().zfill(4)


def get_metadata(lat, lon):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"

    params = {
        "location": f"{lat},{lon}",
        "radius": 25,
        "key": GOOGLE_MAPS_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def calculate_heading(from_lat, from_lon, to_lat, to_lon):
    from_lat, from_lon = math.radians(from_lat), math.radians(from_lon)
    to_lat, to_lon = math.radians(to_lat), math.radians(to_lon)

    d_lon = to_lon - from_lon

    x = math.sin(d_lon) * math.cos(to_lat)
    y = (
        math.cos(from_lat) * math.sin(to_lat)
        - math.sin(from_lat) * math.cos(to_lat) * math.cos(d_lon)
    )

    return (math.degrees(math.atan2(x, y)) + 360) % 360


def fetch_stop_images(target_stop_code) -> dict:
    """
    Fetches 8 sweep images for a comprehensive 360-degree view.

    The "front" image points toward the bus stop.
    The remaining images rotate around the panorama every 45 degrees.
    Returns a dictionary of {view_name: Path}.
    """

    target_stop_code = normalize_stop_code(target_stop_code)

    stop_codes_normalized = (
        stops_df["Stop Code"]
        .astype(str)
        .str.strip()
        .str.zfill(4)
    )

    stop_data = stops_df[stop_codes_normalized == target_stop_code]

    if stop_data.empty:
        print(f"    Error: Stop code {target_stop_code} not found in {STOPS_CSV}.")
        return {}

    row = stop_data.iloc[0]

    stop_name = row["Stop Name"]
    bus_lat = row["Latitude"]
    bus_lon = row["Longitude"]

    metadata = get_metadata(bus_lat, bus_lon)

    if metadata.get("status") != "OK":
        print(f"    Google metadata not OK for stop {target_stop_code}: {metadata}")
        return {}

    pano_lat = metadata["location"]["lat"]
    pano_lon = metadata["location"]["lng"]
    pano_id = metadata["pano_id"]
    image_date = metadata.get("date", "unknown-date")

    heading = calculate_heading(
        from_lat=pano_lat,
        from_lon=pano_lon,
        to_lat=bus_lat,
        to_lon=bus_lon,
    )

    safe_stop_name = clean_filename(stop_name)

    # 8 headings, every 45 degrees, full 360-degree sweep.
    sweep_offsets = {
        "front": 0,
        "front_right": 45,
        "right": 90,
        "back_right": 135,
        "back": 180,
        "back_left": 225,
        "left": 270,
        "front_left": 315,
    }

    downloaded_views = {}

    for view_name, offset in sweep_offsets.items():
        sweep_heading = (heading + offset) % 360
        rounded_heading = round(sweep_heading)

        url = "https://maps.googleapis.com/maps/api/streetview"

        params = {
            "size": "640x640",
            "pano": pano_id,
            "heading": sweep_heading,
            "pitch": 0,
            "fov": 60,
            "key": GOOGLE_MAPS_API_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")

            if "image" not in content_type.lower():
                print(f"    Failed {view_name}: response was not an image")
                continue

        except requests.RequestException as e:
            print(f"    Failed {view_name}: {e}")
            continue

        filename = (
            FALLBACK_DIR
            / f"{target_stop_code}_{safe_stop_name}_{image_date}_{view_name}_heading-{rounded_heading}.jpg"
        )

        with open(filename, "wb") as f:
            f.write(response.content)

        downloaded_views[view_name] = filename

        print(f"    Saved {view_name} heading {round(sweep_heading, 1)}")

        time.sleep(0.1)

    return downloaded_views


# ============================================================
# PARSER & IMAGE GROUPING
# ============================================================

def parse_stop_id_and_view(path: Path):
    filename = path.name.lower()

    match = re.match(r"^(\d{4})", filename)

    if not match:
        raise ValueError(f"Filename must start with a four-digit stop code: {path.name}")

    stop_id = match.group(1)

    view_match = re.search(
        r"_(left|center|centre|right|front|front_right|back_right|back|back_left|front_left|far_left|mid_left|slight_left|slight_right|mid_right|far_right)_heading",
        filename,
    )

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

        if view in grouped[stop_id]:
            raise ValueError(
                f"Duplicate {view} image for stop {stop_id}: "
                f"{grouped[stop_id][view].name} and {path.name}"
            )

        grouped[stop_id][view] = path

    # Only grab groups that have the basic left/center/right trio for Pass 1.
    complete_groups = {
        stop_id: views
        for stop_id, views in grouped.items()
        if all(k in views for k in ["left", "center", "right"])
    }

    return complete_groups


# ============================================================
# GEMINI PROMPT & SCHEMA
# ============================================================

PROMPT = """
You are an observant visual analyst evaluating Street View images for a transit stop accessibility inventory.

First, determine whether the provided images clearly show a usable bus stop area.

Set bus_stop_visible:
- "Yes": If you clearly see dedicated transit infrastructure, such as a GoDurham bus stop sign on a pole, a bus shelter, or a transit bench.
- "No": If you only see general street features like sidewalks, grass, utility poles, or pedestrian crosswalk signs. Please note that generic yellow poles, fire hydrants, or utility poles do not confirm a bus stop.
- "Unclear": If the view of the curb is blocked by a vehicle, heavy foliage, active construction fencing, distance, blur, or a poor camera angle. Important: A bus stopped in the road does not guarantee the actual stop infrastructure is visible; if the bus blocks the curb, please mark it "Unclear".

Then choose the image that gives the best overall view of the boarding area.

Classify the features using ONLY the best selected image.

If bus_stop_visible is "No" or "Unclear", please still classify the environment and any visible objects as accurately as possible. However, you MUST begin your notes with "MANUAL REVIEW REQUIRED: " and briefly explain the issue.

Definitions:
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
        "stop_id": {
            "type": "string",
        },
        "best_view": {
            "type": "string",
            "enum": [
                "left",
                "center",
                "right",
                "front",
                "front_right",
                "back_right",
                "back",
                "back_left",
                "front_left",
                "far_left",
                "mid_left",
                "slight_left",
                "slight_right",
                "mid_right",
                "far_right",
            ],
        },
        "selected_image_filename": {
            "type": "string",
        },
        "bus_stop_visible": {
            "type": "string",
            "enum": ["Yes", "No", "Unclear"],
        },
        "bus_stop_visibility_confidence": {
            "type": "number",
        },
        "stop_surface": {
            "type": "string",
            "enum": ["Grass", "Concrete"],
        },
        "landing_type": {
            "type": "string",
            "enum": ["Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"],
        },
        "sidewalk_connection": {
            "type": "string",
            "enum": ["Yes", "No", "NA"],
        },
        "landing_pad": {
            "type": "string",
            "enum": ["Two_doors", "One_door", "NA"],
        },
        "shelter_number": {
            "type": "integer",
        },
        "bench_number": {
            "type": "integer",
        },
        "trash_can_number": {
            "type": "integer",
        },
        "street_lighting": {
            "type": "string",
            "enum": ["Yes", "No"],
        },
        "notes": {
            "type": "string",
        },
    },
    "required": [
        "stop_id",
        "best_view",
        "selected_image_filename",
        "bus_stop_visible",
        "bus_stop_visibility_confidence",
        "stop_surface",
        "landing_type",
        "sidewalk_connection",
        "landing_pad",
        "shelter_number",
        "bench_number",
        "trash_can_number",
        "street_lighting",
        "notes",
    ],
}


# ============================================================
# LOGIC HELPERS
# ============================================================

def make_image_part(path: Path):
    mime_type, _ = mimetypes.guess_type(path)

    return types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=mime_type or "image/jpeg",
    )


def enforce_logical_consistency(result: dict) -> dict:
    if result.get("sidewalk_connection") in ["No", "NA"]:
        result["landing_pad"] = "NA"

    if result.get("landing_type") == "Unpaved" and result.get("sidewalk_connection") != "Yes":
        result["landing_pad"] = "NA"

    return result


def force_na_attributes(result: dict) -> dict:
    """
    Flags for manual review WITHOUT erasing the model's feature counts.
    """

    current_notes = result.get("notes", "No stop visible after 8-heading fallback.")

    if "MANUAL REVIEW REQUIRED" not in current_notes.upper():
        result["notes"] = "MANUAL REVIEW REQUIRED: " + current_notes

    return result


# ============================================================
# GEMINI CALL
# ============================================================

def analyze_stop(
    stop_id: str,
    views: dict,
    pass_number: int = 1,
    previous_image_path: str = None,
) -> dict:
    contents = [
        PROMPT,
        f"\nStop ID: {stop_id}\nReview the following {len(views)} images for this bus stop:\n",
    ]

    for view_name, path in views.items():
        contents.extend([
            f"{view_name.upper()} VIEW:",
            make_image_part(path),
            f"Filename: {path.name}",
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

    selected_view = result.get("best_view")

    if selected_view not in views:
        selected_view = list(views.keys())[0]
        result["best_view"] = selected_view

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
# SAVE OUTPUTS
# ============================================================

def write_csv(results):
    if not results:
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def save_json(results):
    if not results:
        return

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    complete_groups = group_images_by_stop(INPUT_DIR)

    results = []
    already_done = set()

    # Resume if previous results exist.
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)

                already_done = {
                    str(r.get("stop_id")).strip().zfill(4)
                    for r in results
                    if r.get("stop_id")
                }

            except json.JSONDecodeError:
                print(f"Warning: {OUTPUT_JSON} is malformed. Starting fresh.")
                results = []
                already_done = set()

    for stop_id, views in sorted(complete_groups.items()):
        normalized_stop_id = str(stop_id).strip().zfill(4)

        if normalized_stop_id in already_done:
            print(f"Skipping already processed stop {stop_id}")
            continue

        print(f"\nProcessing stop {stop_id} (Pass 1)...")

        # PASS 1: The original 3-heading check
        result = analyze_stop(stop_id, views, pass_number=1)

        # Branch Logic
        if result["bus_stop_visible"] in ["No", "Unclear"]:
            print(f"  -> Stop {stop_id} unclear/not visible. Triggering 8-heading fallback (Pass 2)...")

            fallback_views = fetch_stop_images(stop_id)

            if fallback_views:
                previous_img = result.get("final_image_path")

                # Pass 2: analyze original 3 images + 8 fallback images together.
                combined_views = {**views, **fallback_views}

                result = analyze_stop(
                    stop_id,
                    combined_views,
                    pass_number=2,
                    previous_image_path=previous_img,
                )

                if result["bus_stop_visible"] in ["No", "Unclear"]:
                    print(f"  -> Stop {stop_id} STILL unclear/not visible. Flagging for manual review.")
                    result = force_na_attributes(result)

            else:
                print(f"  -> Scraper failed to find data for {stop_id}. Flagging.")
                result = force_na_attributes(result)

        results.append(result)

        # Save both CSV and JSON live after each stop.
        write_csv(results)
        save_json(results)

    print("\nPipeline Finished successfully.")


if __name__ == "__main__":
    main()