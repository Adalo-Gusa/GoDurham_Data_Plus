"""
Task 2.2 Cleaning Priority Calculator
-------------------------------------
Usage:
    python task_2_2_cleaning_priority.py v0_2022_2026_inventories.csv

Outputs written to a folder named task_2_2_outputs:
    1. scored_cleaning_priority.csv
    2. tier_summary.csv
    3. frequency_summary.csv
    4. top_high_priority_stops.csv
    5. abc_comparison_summary.csv, if a current A/B/C category column exists

Optional:
    python task_2_2_cleaning_priority.py input.csv output_folder

Notes:
    - This script uses only Python standard libraries.
    - It is designed so you can paste/use any updated GoDurham inventory CSV with similar column names.
    - If the dataset has a current A/B/C cleaning category column, the script compares it to the recommended tier.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ============================================================
# EDITABLE SCORING SETTINGS
# ============================================================

RIDERSHIP_THRESHOLDS = [
    (66.9, 4),
    (35.4, 3),
    (14.5, 2),
    (0.000001, 1),
]

COMPLAINT_THRESHOLDS = [
    (4, 4),
    (2, 3),
    (1, 2),
]

AADT_THRESHOLDS = [
    (22000, 2),
    (15500, 1),
]

# Score cutoff rules.
HIGH_CUTOFF = 12
MEDIUM_CUTOFF = 6

# Recommended cleaning schedule by tier.
FREQUENCY_BY_TIER = {
    "High": "Weekly",
    "Medium": "Every 2 weeks",
    "Low": "Monthly",
    "Manual Review": "Verify first",
}

CLEANING_LEVEL_BY_TIER = {
    "High": "Enhanced",
    "Medium": "Basic, enhanced as needed",
    "Low": "Basic",
    "Manual Review": "Manual review before assignment",
}

# Assumption for current A/B/C categories if present.
# Change this if the City defines A/B/C differently.
CURRENT_CATEGORY_TO_TIER = {
    "A": "High",
    "B": "Medium",
    "C": "Low",
}

CONTEXT_KEYWORDS = {
    # keyword: points
    "station": 3,
    "major transfer": 3,
    "transfer": 3,
    "mall": 3,
    "hospital": 3,
    "school": 2,
    "hs": 2,
    "university": 2,
}

VISIBLE_STOP_YES_VALUES = {"yes", "y", "true", "1"}
VISIBLE_STOP_NO_REVIEW_VALUES = {"no", "n", "false", "0", "unclear", "manual review"}


# ============================================================
# HELPERS
# ============================================================

def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def lower(value: object) -> str:
    return clean_text(value).lower()


def to_float(value: object, default: float = 0.0) -> float:
    text = clean_text(value).replace(",", "")
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: object, default: int = 0) -> int:
    return int(round(to_float(value, default)))


def get_first(row: Dict[str, str], possible_names: List[str], default: str = "") -> str:
    for name in possible_names:
        if name in row and clean_text(row[name]) != "":
            return row[name]
    return default


def find_column(fieldnames: List[str], possible_names: List[str]) -> Optional[str]:
    exact = {name.lower(): name for name in fieldnames}
    for candidate in possible_names:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    return None


def threshold_score(value: float, thresholds: List[Tuple[float, int]]) -> int:
    for threshold, points in thresholds:
        if value >= threshold:
            return points
    return 0


def present_count(row: Dict[str, str], possible_names: List[str]) -> int:
    return max(to_int(row.get(name, 0)) for name in possible_names if name in row) if any(name in row for name in possible_names) else 0


def has_present(row: Dict[str, str], possible_names: List[str]) -> bool:
    return present_count(row, possible_names) > 0


def determine_manual_review(row: Dict[str, str]) -> bool:
    visible = lower(get_first(row, ["bus_stop_visible_2026", "bus_stop_visible"], ""))
    if visible in VISIBLE_STOP_NO_REVIEW_VALUES:
        return True
    return False


def score_ridership(row: Dict[str, str]) -> int:
    activity = to_float(get_first(row, ["Avg Daily Total Activity", "Avg_Daily_Total_Activity", "avg_daily_total_activity"], 0))
    return threshold_score(activity, RIDERSHIP_THRESHOLDS)


def score_complaints(row: Dict[str, str]) -> int:
    complaints = to_float(get_first(row, ["Complaints_n_2026", "Complaints", "complaints"], 0))
    return threshold_score(complaints, COMPLAINT_THRESHOLDS)


def score_amenities(row: Dict[str, str]) -> int:
    shelter = has_present(row, ["Shelter_QT", "shelter_n_2026", "shelter_number"])
    seating = has_present(row, ["Seating_QT", "bench_n_2026", "bench_number"])
    trash = has_present(row, ["Trash_Can_", "trash_can_n_2026", "trash_can_number"])

    score = 0
    if shelter:
        score += 3
    if seating:
        score += 1
    if trash:
        score += 3
    return score


def score_condition(row: Dict[str, str]) -> int:
    score = 0

    stop_surface = lower(get_first(row, ["stop_surface_2026", "stop_surface"], ""))
    landing_type = lower(get_first(row, ["landing_type_2026", "Landing_Ty", "landing_type"], ""))
    sidewalk_connection = lower(get_first(row, ["sidewalk_connection_2026", "Sidewalk_C", "sidewalk_connection"], ""))

    if "grass" in stop_surface:
        score += 1

    if "unpaved" in landing_type:
        if "grass_strip" in landing_type or "grass strip" in landing_type or "sidewalk" in landing_type:
            score += 1
        else:
            score += 2

    if sidewalk_connection in {"no", "n", "false", "0"}:
        score += 1

    return score


def score_roadway(row: Dict[str, str]) -> int:
    aadt = to_float(get_first(row, ["AADT", "aadt"], 0))
    return threshold_score(aadt, AADT_THRESHOLDS)


def score_context(row: Dict[str, str]) -> int:
    text_parts = [
        get_first(row, ["Stop Name_x", "Stop Name_y", "stop_name", "Stop Name"], ""),
        get_first(row, ["notes_2026", "notes"], ""),
    ]
    text = lower(" ".join(text_parts))
    points = 0
    for keyword, keyword_points in CONTEXT_KEYWORDS.items():
        if keyword in text:
            points = max(points, keyword_points)
    return points


def assign_tier(total_score: int, manual_review: bool) -> str:
    if manual_review:
        return "Manual Review"
    if total_score >= HIGH_CUTOFF:
        return "High"
    if total_score >= MEDIUM_CUTOFF:
        return "Medium"
    return "Low"


def compare_current_to_recommended(row: Dict[str, str], current_category_col: Optional[str], recommended_tier: str) -> str:
    if not current_category_col or recommended_tier == "Manual Review":
        return "Needs current A/B/C category" if not current_category_col else "Manual Review"

    current_raw = clean_text(row.get(current_category_col, "")).upper()
    current_tier = CURRENT_CATEGORY_TO_TIER.get(current_raw, "")
    if current_tier == "":
        return "Missing/unknown current category"

    rank = {"Low": 1, "Medium": 2, "High": 3}
    current_rank = rank[current_tier]
    recommended_rank = rank[recommended_tier]

    if recommended_rank > current_rank:
        return "More frequent cleaning recommended"
    if recommended_rank < current_rank:
        return "Less frequent cleaning possible"
    return "Same/general match"


def score_row(row: Dict[str, str], current_category_col: Optional[str]) -> Dict[str, str]:
    manual_review = determine_manual_review(row)

    ridership_score = score_ridership(row)
    complaint_score = score_complaints(row)
    amenity_score = score_amenities(row)
    condition_score = score_condition(row)
    roadway_score = score_roadway(row)
    context_score = score_context(row)

    total_score = ridership_score + complaint_score + amenity_score + condition_score + roadway_score + context_score
    tier = assign_tier(total_score, manual_review)

    return {
        "ridership_score": str(ridership_score),
        "complaint_score": str(complaint_score),
        "amenity_score": str(amenity_score),
        "condition_score": str(condition_score),
        "roadway_score": str(roadway_score),
        "context_score": str(context_score),
        "cleaning_priority_score": str(total_score),
        "recommended_tier": tier,
        "recommended_frequency": FREQUENCY_BY_TIER[tier],
        "recommended_cleaning_level": CLEANING_LEVEL_BY_TIER[tier],
        "review_flag": "Manual review before tier assignment" if manual_review else "",
        "abc_comparison": compare_current_to_recommended(row, current_category_col, tier),
    }


def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_counts(rows: List[Dict[str, str]], column: str) -> List[Dict[str, str]]:
    counts = Counter(row.get(column, "") for row in rows)
    return [{column: key, "count": str(counts[key])} for key in sorted(counts.keys())]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python task_2_2_cleaning_priority.py input_inventory.csv [output_folder]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else "task_2_2_outputs"
    os.makedirs(output_dir, exist_ok=True)

    rows, original_fields = read_csv(input_path)

    current_category_col = find_column(
        original_fields,
        [
            "Current Cleaning Category",
            "current_cleaning_category",
            "Cleaning Category",
            "cleaning_category",
            "ABC Category",
            "A/B/C Category",
            "Current_Category",
        ],
    )

    score_fields = [
        "ridership_score",
        "complaint_score",
        "amenity_score",
        "condition_score",
        "roadway_score",
        "context_score",
        "cleaning_priority_score",
        "recommended_tier",
        "recommended_frequency",
        "recommended_cleaning_level",
        "review_flag",
        "abc_comparison",
    ]

    scored_rows: List[Dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        new_row.update(score_row(row, current_category_col))
        scored_rows.append(new_row)

    scored_fields = original_fields + [f for f in score_fields if f not in original_fields]
    write_csv(os.path.join(output_dir, "scored_cleaning_priority.csv"), scored_rows, scored_fields)

    tier_summary = summarize_counts(scored_rows, "recommended_tier")
    write_csv(os.path.join(output_dir, "tier_summary.csv"), tier_summary, ["recommended_tier", "count"])

    frequency_summary = summarize_counts(scored_rows, "recommended_frequency")
    write_csv(os.path.join(output_dir, "frequency_summary.csv"), frequency_summary, ["recommended_frequency", "count"])

    abc_summary = summarize_counts(scored_rows, "abc_comparison")
    write_csv(os.path.join(output_dir, "abc_comparison_summary.csv"), abc_summary, ["abc_comparison", "count"])

    top_high = [r for r in scored_rows if r.get("recommended_tier") == "High"]
    top_high.sort(key=lambda r: to_float(r.get("cleaning_priority_score")), reverse=True)
    top_fields = [
        f for f in [
            "Stop Code",
            "Stop Name_x",
            "stop_name",
            "AADT",
            "Avg Daily Total Activity",
            "Complaints_n_2026",
            "Shelter_QT",
            "Seating_QT",
            "Trash_Can_",
            "landing_type_2026",
            "stop_surface_2026",
            "sidewalk_connection_2026",
            "notes_2026",
            "cleaning_priority_score",
            "recommended_tier",
            "recommended_frequency",
            "recommended_cleaning_level",
        ]
        if f in scored_fields
    ]
    write_csv(os.path.join(output_dir, "top_high_priority_stops.csv"), top_high[:50], top_fields)

    print("Done.")
    print(f"Input rows: {len(rows)}")
    print(f"Output folder: {output_dir}")
    print("Tier summary:")
    for item in tier_summary:
        print(f"  {item['recommended_tier']}: {item['count']}")
    if current_category_col:
        print(f"Current A/B/C category column used: {current_category_col}")
    else:
        print("No current A/B/C category column found. Add one if you want the comparison to work.")


if __name__ == "__main__":
    main()
