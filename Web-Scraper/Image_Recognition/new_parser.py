import json
import glob
import re
import pandas as pd
from pathlib import Path


OUTPUT_FOLDER = "vision_output"
FINAL_CSV = "bus_stop_results.csv"


def detect_bus_stop(labels, objects, text):
    labels_lower = [x.lower() for x in labels if x]
    objects_lower = [x.lower() for x in objects if x]
    text_lower = text.lower() if text else ""

    label_text = " ".join(labels_lower)
    object_text = " ".join(objects_lower)
    combined_text = f"{label_text} {object_text} {text_lower}"

    score = 0
    reasons = []

    # 1. Explicit bus-stop evidence: strongest
    explicit_terms = [
        "bus stop",
        "bus-stop",
        "bus shelter",
        "bus station",
        "transit stop",
        "coach stop",
        "stop id",
        "stop no",
        "stop number",
        "bus bay",
        "bus zone",
        "bus lane",
        "bus only",
        "bus stand",
        "bus terminal"
    ]

    for term in explicit_terms:
        if term in combined_text:
            score += 7
            reasons.append(f"Explicit bus-stop term: {term}")

    # 2. OCR route/schedule clues
    route_patterns = [
        r"\broute\s*[0-9a-z]+\b",
        r"\brt\.?\s*[0-9a-z]+\b",
        r"\bline\s*[0-9a-z]+\b",
        r"\bbus\s*[0-9a-z]+\b",
        r"\bstop\s*[0-9]{2,}\b",
        r"\b[0-9]{1,3}\s+bus\b",
        r"\bnext\s+bus\b",
        r"\bbus\s+route\b",
        r"\broute\s+map\b",
    ]

    for pattern in route_patterns:
        if re.search(pattern, text_lower):
            score += 4
            reasons.append(f"Route/stop OCR pattern: {pattern}")

    # 3. Transit words in OCR
    transit_ocr_terms = [
        "route",
        "routes",
        "schedule",
        "timetable",
        "arrival",
        "arrivals",
        "departure",
        "departures",
        "transit",
        "metro",
        "bus",
        "buses",
        "public transport",
        "public transportation",
        "passenger",
        "fare",
        "boarding",
        "alighting",
        "stop",
        "next bus"
    ]

    for term in transit_ocr_terms:
        if term in text_lower:
            score += 2
            reasons.append(f"OCR transit clue: {term}")

    # 4. Transit agency clues
    # Add local agencies here if needed.
    agency_terms = [
        "mta",
        "mbta",
        "cta",
        "septa",
        "wmata",
        "muni",
        "metrobus",
        "metrolink",
        "translink",
        "go transit",
        "greyhound",
        "megabus",
        "stagecoach",
        "first bus",
        "arriva",
        "keolis",
        "rta",
        "vta",
        "trimet",
        "nj transit",
        "dart",
        "pace",
        "ac transit"
    ]

    for term in agency_terms:
        if term in text_lower:
            score += 5
            reasons.append(f"Transit agency clue: {term}")

    # 5. Google labels suggesting transit/street context
    transit_label_terms = [
        "public transport",
        "public transportation",
        "transport",
        "transit",
        "bus",
        "bus station",
        "vehicle",
        "road",
        "road surface",
        "street",
        "sidewalk",
        "urban area",
        "metropolitan area",
        "lane",
        "traffic",
        "signage",
        "traffic sign",
        "street sign",
        "infrastructure",
        "thoroughfare",
        "public space",
        "asphalt",
        "parking"
    ]

    for term in transit_label_terms:
        if term in label_text:
            score += 1
            reasons.append(f"Label clue: {term}")

    # 6. Object clues
    object_terms = [
        "bus",
        "sign",
        "traffic sign",
        "bench",
        "person",
        "car",
        "truck",
        "bicycle",
        "vehicle"
    ]

    for term in object_terms:
        if term in object_text:
            score += 1
            reasons.append(f"Object clue: {term}")

    # 7. Infrastructure clues
    infrastructure_terms = [
        "shelter",
        "bench",
        "pole",
        "sign",
        "signage",
        "timetable",
        "schedule",
        "platform",
        "bay",
        "curb",
        "kerb",
        "sidewalk",
        "pavement",
        "public space"
    ]

    for term in infrastructure_terms:
        if term in combined_text:
            score += 1
            reasons.append(f"Infrastructure clue: {term}")

    # 8. Useful combinations
    has_bus = "bus" in combined_text or "buses" in combined_text
    has_stop = "stop" in text_lower
    has_route = "route" in text_lower or "line" in text_lower
    has_schedule = "schedule" in text_lower or "timetable" in text_lower
    has_sign = "sign" in combined_text or "signage" in combined_text
    has_street = any(x in combined_text for x in [
        "road",
        "road surface",
        "street",
        "sidewalk",
        "lane",
        "curb",
        "kerb",
        "thoroughfare",
        "asphalt",
        "public space"
    ])
    has_shelter_or_bench = "shelter" in combined_text or "bench" in combined_text

    if has_bus and has_sign:
        score += 4
        reasons.append("Combination: bus + sign")

    if has_bus and has_street:
        score += 3
        reasons.append("Combination: bus + street context")

    if has_stop and has_street:
        score += 3
        reasons.append("Combination: stop text + street context")

    if has_route and has_sign:
        score += 4
        reasons.append("Combination: route + sign")

    if has_schedule and has_sign:
        score += 4
        reasons.append("Combination: schedule/timetable + sign")

    if has_shelter_or_bench and has_street:
        score += 3
        reasons.append("Combination: shelter/bench + street context")

    if has_bus and has_route:
        score += 5
        reasons.append("Combination: bus + route")

    # 9. Broad bus-stop environment clues
    # This catches cases where Google misses the actual bus-stop sign,
    # but the scene looks like a plausible roadside/sidewalk stop.
    has_road_context = any(x in combined_text for x in [
        "road",
        "road surface",
        "street",
        "thoroughfare",
        "lane",
        "asphalt"
    ])

    has_pedestrian_context = any(x in combined_text for x in [
        "sidewalk",
        "public space",
        "pavement",
        "curb",
        "kerb"
    ])

    has_vehicle_context = any(x in combined_text for x in [
        "car",
        "family car",
        "vehicle",
        "traffic",
        "parking"
    ])

    if has_road_context and has_pedestrian_context:
        score += 3
        reasons.append("Broad context: road + sidewalk/public pedestrian area")

    if has_road_context and has_pedestrian_context and has_vehicle_context:
        score += 1
        reasons.append("Broad context: road + sidewalk + vehicle context")

    # 10. Weak false-positive protection
    # Only subtract if there is not much evidence already.
    negative_terms = [
        "foodmart",
        "atm",
        "gas",
        "fuel",
        "parking lot",
        "drive thru",
        "drive-through",
        "car wash",
        "dealership"
    ]

    if score < 6:
        for term in negative_terms:
            if term in combined_text:
                score -= 1
                reasons.append(f"Weak negative clue: {term}")

    # Final high-recall classification.
    # IMPORTANT: maybes are marked as True.
    if score >= 7:
        status = "yes"
        bus_stop_present = True
    elif score >= 3:
        status = "maybe_review"
        bus_stop_present = True
    else:
        status = "no"
        bus_stop_present = False

    return bus_stop_present, score, status, "; ".join(reasons)


def parse_google_vision_results():
    rows = []

    json_files = glob.glob(f"{OUTPUT_FOLDER}/**/*.json", recursive=True)

    if not json_files:
        raise RuntimeError("No JSON files found in vision_output. Check your download step.")

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        for response in data.get("responses", []):
            image_uri = response.get("context", {}).get("uri", "")
            filename = Path(image_uri).name if image_uri else ""

            labels = [
                item.get("description", "")
                for item in response.get("labelAnnotations", [])
            ]

            objects = [
                item.get("name", "")
                for item in response.get("localizedObjectAnnotations", [])
            ]

            text = ""
            if "fullTextAnnotation" in response:
                text = response["fullTextAnnotation"].get("text", "")
            elif response.get("textAnnotations"):
                text = response["textAnnotations"][0].get("description", "")

            bus_stop_present, score, status, reason = detect_bus_stop(labels, objects, text)

            rows.append({
                "filename": filename,
                "gcs_uri": image_uri,
                "bus_stop_present": bus_stop_present,
                "bus_stop_status": status,
                "bus_stop_score": score,
                "reason": reason,
                "labels": " | ".join(labels),
                "objects": " | ".join(objects),
                "ocr_text": text.replace("\n", " ")[:500]
            })

    return pd.DataFrame(rows)


df = parse_google_vision_results()

df = df.sort_values(
    by=["bus_stop_present", "bus_stop_score"],
    ascending=[False, False]
)

df.to_csv(FINAL_CSV, index=False)

print(df[["filename", "bus_stop_present", "bus_stop_status", "bus_stop_score", "reason"]].head(30))
print()
print(f"Saved results to {FINAL_CSV}")
print(f"Total images processed: {len(df)}")
print(f"Flagged as bus stop: {df['bus_stop_present'].sum()}")
print(f"Not flagged as bus stop: {(~df['bus_stop_present']).sum()}")