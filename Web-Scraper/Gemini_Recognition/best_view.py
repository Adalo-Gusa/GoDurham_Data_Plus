import os
import json
import shutil
import mimetypes
from pathlib import Path
from collections import defaultdict

from google import genai
from google.genai import types # type: ignore


# -----------------------------
# CONFIG
# -----------------------------

INPUT_DIR = Path("images")
FINAL_IMAGES_DIR = Path("final_images")
OUTPUT_JSON = Path("bus_stop_results.json")
OUTPUT_CSV = Path("bus_stop_results.csv")

MODEL = "gemini-3.5-flash"

client = genai.Client(
    enterprise=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location="global",
)

FINAL_IMAGES_DIR.mkdir(exist_ok=True)


# -----------------------------
# PROMPT
# -----------------------------

PROMPT = """
You are analyzing bus stop images for transit stop accessibility inventory.

Each stop has three views:
- left view
- center view
- right view

First, determine which of the three images gives the best overall view of the bus stop. The best view should show the boarding/landing area, road edge/curb, sidewalk if present, and nearby amenities such as shelter, bench, trash can, and lighting.

Then classify the bus stop using the best selected image.

Use only visible evidence. Do not guess hidden features.
If a field is not applicable, use "NA".
If uncertain, choose the most visually supported option and lower the confidence.
Count only clearly visible objects at or immediately around the bus stop.

Field definitions:

1. stop_surface
Allowed values: "Grass", "Concrete"

Choose "Grass" when the surface immediately next to the road/curb where a rider would stand or board is mostly grass, dirt, or unpaved ground.

Choose "Concrete" when that surface is mostly concrete, pavement, asphalt, or another hard paved surface.

2. landing_type
Allowed values: "Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"

Choose "Paved" when the bus stop landing/standing area next to the road is paved, usually concrete, asphalt, or a paved sidewalk/road shoulder.

Choose "Unpaved" when the landing/standing area next to the road is grass, dirt, gravel, or otherwise unpaved and there is no nearby sidewalk forming part of the stop area.

Choose "Unpaved_Grass_Strip_And_Sidewalk" when the area next to the road is grass/unpaved but there is a sidewalk nearby or behind it, creating a grass strip between the road and sidewalk.

3. sidewalk_connection
Allowed values: "Yes", "No", "NA"

Choose "Yes" if there is a paved path, curb cut, concrete pad, sidewalk, or continuous paved surface connecting the pedestrian area to the road/curb.

Also choose "Yes" when the stop area is concrete/paved in its entirety from the sidewalk or standing area to the curb.

Choose "No" if a sidewalk is visible but the rider would have to cross grass, dirt, gravel, or another unpaved surface to reach the road/curb.

Choose "NA" if there is no sidewalk or pedestrian path visible.

4. landing_pad
Allowed values: "Two_doors", "One_door", "NA"

Only classify this when there is a usable paved boarding area with sidewalk_connection = "Yes".

Choose "Two_doors" if the paved landing area appears long enough and positioned to serve both the front and rear bus doors.

Choose "One_door" if the paved landing area appears to serve only one bus door.

Choose "NA" if there is no usable paved landing area, sidewalk_connection is "No" or "NA", or the landing pad is not visible enough to decide.

5. shelter_number
Integer count.

0 means no visible shelter.
Count the number of bus shelters visible at or immediately around the stop.

6. bench_number
Integer count.

0 means no visible bench.
Count the number of benches visible at or immediately around the stop.

7. trash_can_number
Integer count.

0 means no visible trash can.
Count the number of trash cans visible at or immediately around the stop.

8. street_lighting
Allowed values: "Yes", "No"

Choose "Yes" if a streetlight, lamp post, or dedicated lighting fixture is visible near the stop.

Choose "No" if no lighting is visible near the stop.

9. year_of_collection
Use the year provided in the filename, folder name, or input metadata.
Only allowed values: 2025 or 2026.
Do not infer year from the visual image unless the year is visibly printed in the image.

10. best_view
Allowed values: "left", "center", "right"

Choose the image that gives the clearest and most complete view of the bus stop and boarding area.

Prefer the view that best shows:
- road edge or curb
- stop surface
- landing area
- sidewalk or lack of sidewalk
- shelter, bench, trash can, and lighting if present

Do not choose based only on image sharpness. Choose based on usefulness for stop classification.
"""


# -----------------------------
# RESPONSE SCHEMA
# -----------------------------

response_schema = {
    "type": "object",
    "properties": {
        "stop_id": {"type": "string"},
        "year_of_collection": {
            "type": "integer",
            "enum": [2025, 2026],
        },
        "best_view": {
            "type": "string",
            "enum": ["left", "center", "right"],
        },
        "selected_image_filename": {"type": "string"},
        "stop_surface": {
            "type": "string",
            "enum": ["Grass", "Concrete"],
        },
        "landing_type": {
            "type": "string",
            "enum": [
                "Paved",
                "Unpaved",
                "Unpaved_Grass_Strip_And_Sidewalk",
            ],
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
            "minimum": 0,
        },
        "bench_number": {
            "type": "integer",
            "minimum": 0,
        },
        "trash_can_number": {
            "type": "integer",
            "minimum": 0,
        },
        "street_lighting": {
            "type": "string",
            "enum": ["Yes", "No"],
        },
        "confidence": {
            "type": "object",
            "properties": {
                "best_view": {"type": "number"},
                "stop_surface": {"type": "number"},
                "landing_type": {"type": "number"},
                "sidewalk_connection": {"type": "number"},
                "landing_pad": {"type": "number"},
                "shelter_number": {"type": "number"},
                "bench_number": {"type": "number"},
                "trash_can_number": {"type": "number"},
                "street_lighting": {"type": "number"},
            },
            "required": [
                "best_view",
                "stop_surface",
                "landing_type",
                "sidewalk_connection",
                "landing_pad",
                "shelter_number",
                "bench_number",
                "trash_can_number",
                "street_lighting",
            ],
        },
        "notes": {"type": "string"},
    },
    "required": [
        "stop_id",
        "year_of_collection",
        "best_view",
        "selected_image_filename",
        "stop_surface",
        "landing_type",
        "sidewalk_connection",
        "landing_pad",
        "shelter_number",
        "bench_number",
        "trash_can_number",
        "street_lighting",
        "confidence",
        "notes",
    ],
}


# -----------------------------
# HELPERS
# -----------------------------

def get_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {path}")
    return mime_type


def extract_year_from_path(path: Path) -> int | None:
    text = str(path)
    if "2025" in text:
        return 2025
    if "2026" in text:
        return 2026
    return None


def parse_stop_id_and_view(path: Path):
    """
    Assumes filenames contain one of:
    _left, _center, _right

    Examples:
    stop_001_left.jpg
    stop_001_center.jpg
    stop_001_right.jpg

    Returns:
    stop_id, view
    """
    stem = path.stem.lower()

    for view in ["left", "center", "right"]:
        marker = f"_{view}"
        if marker in stem:
            stop_id = stem.replace(marker, "")
            return stop_id, view

    raise ValueError(f"Could not find left/center/right view in filename: {path.name}")


def group_images_by_stop(input_dir: Path):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(input_dir.rglob(ext))

    grouped = defaultdict(dict)

    for path in image_paths:
        stop_id, view = parse_stop_id_and_view(path)
        grouped[stop_id][view] = path

    complete_groups = {}
    incomplete_groups = {}

    for stop_id, views in grouped.items():
        if all(v in views for v in ["left", "center", "right"]):
            complete_groups[stop_id] = views
        else:
            incomplete_groups[stop_id] = views

    return complete_groups, incomplete_groups


def make_image_part(path: Path):
    return types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=get_mime_type(path),
    )


def enforce_logical_consistency(result: dict) -> dict:
    """
    Post-processing guardrails.
    This fixes obvious contradictions after Gemini returns JSON.
    """

    # If no sidewalk/pedestrian path connection, no landing pad.
    if result["sidewalk_connection"] in ["No", "NA"]:
        result["landing_pad"] = "NA"

    # If there is no usable paved connection, landing pad should be NA.
    if result["landing_type"] == "Unpaved" and result["sidewalk_connection"] != "Yes":
        result["landing_pad"] = "NA"

    # Concrete + Paved should usually imply a usable paved connection.
    # This reflects your corrected rule: concrete throughout = connection.
    if (
        result["stop_surface"] == "Concrete"
        and result["landing_type"] == "Paved"
        and result["sidewalk_connection"] == "NA"
    ):
        result["sidewalk_connection"] = "Yes"

    return result


def analyze_stop(stop_id: str, views: dict) -> dict:
    year = None
    for path in views.values():
        year = extract_year_from_path(path)
        if year:
            break

    if year is None:
        raise ValueError(
            f"Could not determine year for {stop_id}. "
            "Put 2025/2026 in filename/folder or pass metadata separately."
        )

    contents = [
        PROMPT,
        f"""
Stop ID: {stop_id}
Year of collection: {year}

The next three images are the left, center, and right views of the same bus stop.
Choose the best_view from these three, classify the stop using that selected view,
and set selected_image_filename to the filename of the chosen image.
""",
        "LEFT VIEW:",
        make_image_part(views["left"]),
        f"Left filename: {views['left'].name}",
        "CENTER VIEW:",
        make_image_part(views["center"]),
        f"Center filename: {views['center'].name}",
        "RIGHT VIEW:",
        make_image_part(views["right"]),
        f"Right filename: {views['right'].name}",
    ]

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=types.ThinkingConfig(
                thinking_level="LOW"
            ),
            media_resolution="MEDIA_RESOLUTION_LOW",
        ),
    )

    result = json.loads(response.text)
    result = enforce_logical_consistency(result)

    selected_view = result["best_view"]
    selected_path = views[selected_view]

    # Make sure filename matches the actually selected file.
    result["selected_image_filename"] = selected_path.name

    # Copy best image to final_images folder.
    destination = FINAL_IMAGES_DIR / selected_path.name
    shutil.copy2(selected_path, destination)

    result["final_image_path"] = str(destination)

    return result


# -----------------------------
# MAIN
# -----------------------------

def main():
    complete_groups, incomplete_groups = group_images_by_stop(INPUT_DIR)

    print(f"Complete stops found: {len(complete_groups)}")
    print(f"Incomplete stops found: {len(incomplete_groups)}")

    if incomplete_groups:
        print("\nIncomplete groups skipped:")
        for stop_id, views in incomplete_groups.items():
            print(stop_id, list(views.keys()))

    results = []

    # Resume if results already exist.
    already_done = set()
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r") as f:
            existing = json.load(f)
            results = existing
            already_done = {r["stop_id"] for r in existing}

    for stop_id, views in complete_groups.items():
        if stop_id in already_done:
            print(f"Skipping already processed stop: {stop_id}")
            continue

        try:
            print(f"Processing {stop_id}...")
            result = analyze_stop(stop_id, views)
            results.append(result)

            # Save after every stop so progress is not lost.
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Done {stop_id}: selected {result['best_view']}")

        except Exception as e:
            print(f"Failed {stop_id}: {e}")

    write_csv(results)
    print(f"\nSaved JSON to {OUTPUT_JSON}")
    print(f"Saved CSV to {OUTPUT_CSV}")
    print(f"Copied selected images to {FINAL_IMAGES_DIR}/")


def write_csv(results):
    import csv

    fields = [
        "stop_id",
        "year_of_collection",
        "best_view",
        "selected_image_filename",
        "final_image_path",
        "stop_surface",
        "landing_type",
        "sidewalk_connection",
        "landing_pad",
        "shelter_number",
        "bench_number",
        "trash_can_number",
        "street_lighting",
        "notes",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            row = {field: r.get(field, "") for field in fields}
            writer.writerow(row)


if __name__ == "__main__":
    main()