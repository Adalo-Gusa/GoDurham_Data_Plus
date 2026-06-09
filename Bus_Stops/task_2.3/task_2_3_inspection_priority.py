#!/usr/bin/env python3
"""
Task 2.3: Inspection Priority Score calculator

Usage:
    python task_2_3_inspection_priority.py inventory.csv
    python task_2_3_inspection_priority.py inventory.csv output_folder

Outputs:
    scored_inspection_priority.csv
    inspection_tier_summary.csv
    inspection_frequency_summary.csv
    top_high_priority_inspection_stops.csv
    inspection_focus_area_summary.csv
    manual_review_stops.csv
    inspection_framework_report_text.txt

Notes:
    This script uses the available inventory fields to create a draft inspection priority score.
    If repair records with issue type/date are later added, they should be merged into the input
    and added to the issue_history_score.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def to_num(s, default=0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def norm_text(s):
    return s.fillna("").astype(str).str.strip()


def yes_like(series):
    txt = norm_text(series).str.lower()
    return txt.isin(["yes", "y", "true", "1", "present"])


def positive_num(series):
    return to_num(series, 0) > 0


def contains_any(series, words):
    txt = norm_text(series).str.lower()
    pattern = "|".join([w.lower() for w in words])
    return txt.str.contains(pattern, regex=True, na=False)


def add_missing_output_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Keep helpful fields even if missing from a different inventory file.
    base_cols = [
        "Stop Code", "Stop Name_x", "stop_name", "Latitude", "Longitude", "AADT",
        "Avg Daily Total Activity", "Complaints_n_2026", "Shelter_QT", "Seating_QT",
        "Trash_Can_", "Area_Light", "Real_Time_", "Bike_Parki", "Bus_Loadin", "ADA",
        "Landing_Ty", "Sidewalk_C", "bus_stop_visible_2026", "stop_surface_2026",
        "landing_type_2026", "sidewalk_connection_2026", "shelter_n_2026",
        "bench_n_2026", "trash_can_n_2026", "street_lighting_2026", "notes_2026"
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_missing_output_cols(out)

    activity_col = first_existing_col(out, ["Avg Daily Total Activity", "Avg_Daily_Total_Activity", "Activity"])
    complaints_col = first_existing_col(out, ["Complaints_n_2026", "Complaints", "Complaint Count", "complaint_count"])
    aadt_col = first_existing_col(out, ["AADT"])

    shelter_cols = [c for c in [
        first_existing_col(out, ["Shelter_QT", "Shelter Count", "shelter_count"]),
        first_existing_col(out, ["shelter_n_2026", "shelter_number_2026"]),
    ] if c]
    seating_cols = [c for c in [
        first_existing_col(out, ["Seating_QT", "Bench_QT", "bench_count"]),
        first_existing_col(out, ["bench_n_2026", "bench_number_2026"]),
    ] if c]
    trash_cols = [c for c in [
        first_existing_col(out, ["Trash_Can_", "Trash_Can_QT", "trash_can_count"]),
        first_existing_col(out, ["trash_can_n_2026", "trash_can_number_2026"]),
    ] if c]

    area_light_col = first_existing_col(out, ["Area_Light", "Area Light", "Lighting"])
    street_light_2026_col = first_existing_col(out, ["street_lighting_2026", "Street_Lighting_2026"])
    real_time_col = first_existing_col(out, ["Real_Time_", "Real Time", "Real_Time"])
    bike_col = first_existing_col(out, ["Bike_Parki", "Bike_Parking", "Bike_Par_1"])
    bus_loading_col = first_existing_col(out, ["Bus_Loadin", "Bus_Loading"])
    ada_col = first_existing_col(out, ["ADA"])
    landing_col = first_existing_col(out, ["Landing_Ty", "Landing_Type", "landing_type_2026"])
    landing_2026_col = first_existing_col(out, ["landing_type_2026"])
    sidewalk_col = first_existing_col(out, ["Sidewalk_C", "Sidewalk_Connection", "sidewalk_connection_2026"])
    sidewalk_2026_col = first_existing_col(out, ["sidewalk_connection_2026"])
    visible_col = first_existing_col(out, ["bus_stop_visible_2026", "bus_stop_visible"])
    notes_col = first_existing_col(out, ["notes_2026", "notes"])

    activity = to_num(out[activity_col], 0) if activity_col else pd.Series(0, index=out.index)
    complaints = to_num(out[complaints_col], 0) if complaints_col else pd.Series(0, index=out.index)
    aadt = to_num(out[aadt_col], 0) if aadt_col else pd.Series(0, index=out.index)

    # Use whichever source indicates the amenity exists.
    shelter_present = pd.Series(False, index=out.index)
    for c in shelter_cols:
        shelter_present = shelter_present | positive_num(out[c])

    seating_present = pd.Series(False, index=out.index)
    for c in seating_cols:
        seating_present = seating_present | positive_num(out[c])

    trash_present = pd.Series(False, index=out.index)
    for c in trash_cols:
        trash_present = trash_present | positive_num(out[c])

    light_present = pd.Series(False, index=out.index)
    if area_light_col:
        light_present = light_present | yes_like(out[area_light_col]) | contains_any(out[area_light_col], ["good", "fair", "pole", "light"])
    if street_light_2026_col:
        light_present = light_present | yes_like(out[street_light_2026_col])

    realtime_present = pd.Series(False, index=out.index)
    if real_time_col:
        realtime_present = yes_like(out[real_time_col]) | positive_num(out[real_time_col])

    bike_present = pd.Series(False, index=out.index)
    if bike_col:
        bike_present = yes_like(out[bike_col]) | positive_num(out[bike_col]) | contains_any(out[bike_col], ["rack", "parking", "yes"])

    traffic_lane_loading = pd.Series(False, index=out.index)
    if bus_loading_col:
        traffic_lane_loading = contains_any(out[bus_loading_col], ["traffic_lane", "traffic lane"])

    non_ada_or_unknown = pd.Series(False, index=out.index)
    if ada_col:
        ada_txt = norm_text(out[ada_col]).str.lower()
        non_ada_or_unknown = ada_txt.isin(["no", "not ada", "non-ada", "unknown", ""])

    unpaved_or_grass_landing = pd.Series(False, index=out.index)
    if landing_col:
        unpaved_or_grass_landing = unpaved_or_grass_landing | contains_any(out[landing_col], ["unpaved", "grass"])
    if landing_2026_col:
        unpaved_or_grass_landing = unpaved_or_grass_landing | contains_any(out[landing_2026_col], ["unpaved", "grass"])

    no_sidewalk = pd.Series(False, index=out.index)
    if sidewalk_col:
        no_sidewalk = no_sidewalk | contains_any(out[sidewalk_col], ["no", "none", "missing"])
    if sidewalk_2026_col:
        no_sidewalk = no_sidewalk | norm_text(out[sidewalk_2026_col]).str.lower().eq("no")

    notes_issue = pd.Series(False, index=out.index)
    if notes_col:
        notes_issue = contains_any(out[notes_col], [
            "construction", "fenced", "blocked", "leaning", "brush", "missing", "damaged",
            "broken", "manual review", "no bus stop", "not visible", "panel", "burnt", "loose"
        ])

    visible_txt = norm_text(out[visible_col]).str.lower() if visible_col else pd.Series("yes", index=out.index)
    manual_review = visible_txt.isin(["no", "unclear", "unknown"]) | (notes_issue & contains_any(out[notes_col], ["manual review", "no bus stop", "not visible"]))

    # Score components.
    out["activity_score"] = np.select(
        [activity >= 66.9, activity >= 35.4, activity >= 14.5, activity > 0],
        [3, 2, 1, 0],
        default=0,
    )
    out["issue_history_score"] = np.select(
        [complaints >= 4, complaints >= 2, complaints >= 1],
        [4, 3, 2],
        default=0,
    )
    out["equipment_score"] = (
        shelter_present.astype(int) * 4
        + seating_present.astype(int) * 2
        + trash_present.astype(int) * 1
        + light_present.astype(int) * 2
        + realtime_present.astype(int) * 2
        + bike_present.astype(int) * 1
    )
    out["condition_safety_score"] = (
        traffic_lane_loading.astype(int) * 1
        + non_ada_or_unknown.astype(int) * 1
        + unpaved_or_grass_landing.astype(int) * 1
        + no_sidewalk.astype(int) * 1
        + notes_issue.astype(int) * 2
    )
    out["roadway_score"] = np.select([aadt >= 22000, aadt >= 15500], [2, 1], default=0)

    out["inspection_priority_score"] = (
        out["activity_score"]
        + out["issue_history_score"]
        + out["equipment_score"]
        + out["condition_safety_score"]
        + out["roadway_score"]
    )

    out["inspection_tier"] = np.select(
        [manual_review, out["inspection_priority_score"] >= 12, out["inspection_priority_score"] >= 6],
        ["Manual Review", "High", "Medium"],
        default="Low",
    )
    out["recommended_inspection_frequency"] = out["inspection_tier"].map({
        "High": "Monthly",
        "Medium": "Quarterly",
        "Low": "Semiannual",
        "Manual Review": "Field verify before assigning schedule",
    })
    out["inspection_level"] = out["inspection_tier"].map({
        "High": "Full equipment + safety inspection",
        "Medium": "Standard equipment inspection",
        "Low": "Basic condition check",
        "Manual Review": "Confirm stop/location first",
    })
    out["review_flag"] = np.where(manual_review, "Manual review needed", "")

    # Amenity-specific focus areas.
    focus = []
    for i in out.index:
        items = ["Sign/post condition"]
        if shelter_present.loc[i]:
            items += ["Shelter roof/frame/panels", "Shelter cleanliness/damage", "Loose screws/anchors"]
        if seating_present.loc[i]:
            items += ["Bench/seating condition", "Bench anchors/surface damage"]
        if trash_present.loc[i]:
            items += ["Trash can condition/liner/lid", "Trash overflow or pests"]
        if light_present.loc[i]:
            items += ["Lighting/bulb outage", "Electrical/pole condition"]
        if realtime_present.loc[i]:
            items += ["Real-time information sign/display"]
        if bike_present.loc[i]:
            items += ["Bike rack/parking condition"]
        if unpaved_or_grass_landing.loc[i] or no_sidewalk.loc[i] or traffic_lane_loading.loc[i]:
            items += ["Landing/sidewalk/access condition", "ADA/safety access issues"]
        if notes_issue.loc[i]:
            items += ["Known visual issue from notes"]
        # Deduplicate while preserving order.
        dedup = []
        for item in items:
            if item not in dedup:
                dedup.append(item)
        focus.append("; ".join(dedup))
    out["inspection_focus_areas"] = focus

    return out


def save_outputs(scored: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    scored.to_csv(output_dir / "scored_inspection_priority.csv", index=False)

    tier_order = ["High", "Medium", "Low", "Manual Review"]
    tier_summary = (
        scored["inspection_tier"].value_counts().reindex(tier_order, fill_value=0)
        .rename_axis("inspection_tier").reset_index(name="stop_count")
    )
    tier_summary.to_csv(output_dir / "inspection_tier_summary.csv", index=False)

    frequency_summary = (
        scored.groupby(["inspection_tier", "recommended_inspection_frequency"], dropna=False)
        .size().reset_index(name="stop_count")
        .sort_values(["inspection_tier", "recommended_inspection_frequency"])
    )
    frequency_summary.to_csv(output_dir / "inspection_frequency_summary.csv", index=False)

    top_cols = [c for c in [
        "Stop Code", "Stop Name_x", "stop_name", "inspection_priority_score", "inspection_tier",
        "recommended_inspection_frequency", "inspection_level", "activity_score", "issue_history_score",
        "equipment_score", "condition_safety_score", "roadway_score", "inspection_focus_areas",
        "Avg Daily Total Activity", "Complaints_n_2026", "Shelter_QT", "Seating_QT", "Trash_Can_",
        "Area_Light", "Real_Time_", "AADT"
    ] if c in scored.columns]
    top = scored[scored["inspection_tier"].eq("High")].sort_values(
        "inspection_priority_score", ascending=False
    ).head(50)
    top[top_cols].to_csv(output_dir / "top_high_priority_inspection_stops.csv", index=False)

    manual = scored[scored["inspection_tier"].eq("Manual Review")]
    manual[top_cols].to_csv(output_dir / "manual_review_stops.csv", index=False)

    focus_counts = []
    focus_terms = [
        "Shelter roof/frame/panels", "Bench/seating condition", "Trash can condition/liner/lid",
        "Lighting/bulb outage", "Real-time information sign/display", "Bike rack/parking condition",
        "Landing/sidewalk/access condition", "ADA/safety access issues", "Known visual issue from notes",
        "Sign/post condition"
    ]
    for term in focus_terms:
        focus_counts.append({
            "inspection_focus_area": term,
            "stop_count": scored["inspection_focus_areas"].str.contains(term, regex=False, na=False).sum()
        })
    pd.DataFrame(focus_counts).to_csv(output_dir / "inspection_focus_area_summary.csv", index=False)

    report = f"""Task 2.3 Inspection Framework - Draft Report Text

Inspection Priority Score explanation:
The Inspection Priority Score groups stops by the likelihood that they need routine maintenance monitoring. Unlike the Cleaning Priority Score, this score emphasizes equipment and safety risk. Stops receive points for amenities and assets that require inspection, including shelters, benches, trash cans, lighting, real-time information signs, and bike parking. Stops also receive points for complaint history, passenger activity, difficult boarding/access conditions, high-traffic roadway context, and known visual issues identified in the inventory notes.

Tier definitions and recommended frequencies:
- High: score 12 or higher. Inspect monthly. These stops should receive full equipment and safety inspections because they have multiple inspectable assets, higher activity, complaint history, or known condition concerns.
- Medium: score 6 to 11. Inspect quarterly. These stops have some equipment or risk factors and should receive standard equipment inspections.
- Low: score 0 to 5. Inspect semiannually. These stops generally have fewer assets and lower inspection risk, so a basic condition check is appropriate.
- Manual Review: stops where the bus stop was not visible, unclear, or flagged for verification. These should be field-verified before receiving a regular inspection schedule.

Current results from this dataset:
{tier_summary.to_string(index=False)}

Inspection focus areas:
Inspection checklists should be customized by stop amenities. All stops should include sign/post condition. Sheltered stops should include shelter roof, frame, panels, anchors, and loose screws. Stops with seating should include bench surface condition and anchors. Stops with trash cans should include can condition, liner/lid condition, overflow, and pests. Stops with lighting should include bulb outages and pole/electrical condition. Stops with real-time signage should include display condition and functionality. Stops with unpaved landings, missing sidewalk connections, or traffic-lane loading should include landing, sidewalk, ADA, and safety access checks.

City/contractor oversight recommendations:
GoDurham should require photo documentation for completed inspections, especially for High-tier stops and stops with shelters, lighting, or real-time information signs. City staff should review a random sample of inspection photos each month, conduct spot checks on a sample of High and Medium stops, and track KPIs such as inspections completed on time, issues found, issues resolved, repeat issues, photo compliance, and time from issue identification to repair. Stops with repeated failed inspections or repeat complaints should be escalated for follow-up.

Repair records caveat:
This draft uses complaints as the available issue-history field. If detailed repair records are provided later, they should be merged into the dataset and added as a separate repair-history score, especially for recurring issues such as lighting outages, damaged shelters, missing panels, loose bench anchors, or repeated trash can damage.
"""
    (output_dir / "inspection_framework_report_text.txt").write_text(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python task_2_3_inspection_priority.py inventory.csv [output_folder]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("task_2_3_inspection_outputs")

    df = pd.read_csv(input_path)
    scored = compute_scores(df)
    save_outputs(scored, output_dir)

    print(f"Wrote inspection outputs to: {output_dir}")
    print(scored["inspection_tier"].value_counts().to_string())
    print("\nNote: If detailed repair records become available, merge them into the input and add them to issue_history_score.")


if __name__ == "__main__":
    main()
