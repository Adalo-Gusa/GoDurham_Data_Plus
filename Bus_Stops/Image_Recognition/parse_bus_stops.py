import json
import glob
import re
import pandas as pd
from pathlib import Path


OUTPUT_FOLDER = "vision_output"
FINAL_CSV = "bus_stop_results.csv"


def detect_bus_stop(labels, objects, text):
    labels_lower = [x.lower() for x in labels]
    objects_lower = [x.lower() for x in objects]
    text_lower = text.lower() if text else ""

    score = 0
    reasons = []

    label_text = " ".join(labels_lower)
    object_text = " ".join(objects_lower)

    # Strong signals
    if re.search(r"\bbus\s*stop\b", text_lower):
        score += 5
        reasons.append("OCR contains 'bus stop'")

    if "bus stop" in label_text:
        score += 5
        reasons.append("Label contains 'bus stop'")

    # Medium signals
    if any(term in label_text for term in [
        "public transport",
        "public transportation",
        "transport",
        "transit",
        "bus station",
        "bus"
    ]):
        score += 2
        reasons.append("Transit-related label")

    if re.search(r"\b(bus|route|metro|transit|stop)\b", text_lower):
        score += 2
        reasons.append("Transit-related OCR text")

    # Weak object/context signals
    if "bus" in object_text:
        score += 1
        reasons.append("Object: bus")

    if "sign" in object_text or "traffic sign" in object_text:
        score += 1
        reasons.append("Object: sign")

    if "bench" in object_text:
        score += 1
        reasons.append("Object: bench")

    if any(term in label_text for term in ["road", "street", "sidewalk", "urban area"]):
        score += 1
        reasons.append("Street context")

  
    bus_stop_present = score >= 2

    return bus_stop_present, score, "; ".join(reasons)


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
            filename = Path(image_uri).name

            labels = [
                item.get("description", "")
                for item in response.get("labelAnnotations", [])
            ]

            objects = [
                item.get("name", "")
                for item in response.get("localizedObjectAnnotations", [])
            ]

            # OCR text usually lives in fullTextAnnotation.
            text = ""
            if "fullTextAnnotation" in response:
                text = response["fullTextAnnotation"].get("text", "")
            elif response.get("textAnnotations"):
                text = response["textAnnotations"][0].get("description", "")

            bus_stop_present, score, reason = detect_bus_stop(labels, objects, text)

            rows.append({
                "filename": filename,
                "gcs_uri": image_uri,
                "bus_stop_present": bus_stop_present,
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

print(df[["filename", "bus_stop_present", "bus_stop_score", "reason"]].head(20))
print()
print(f"Saved results to {FINAL_CSV}")
print(f"Total images processed: {len(df)}")
print(f"Flagged as bus stop: {df['bus_stop_present'].sum()}")