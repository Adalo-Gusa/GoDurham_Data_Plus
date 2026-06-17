from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_CSV = Path("output/step_2_2_cleaning_priority_score.csv")
IMAGE_RESULTS_CSV = Path("bus_stop_results.csv")

OUTPUT_CSV = Path("output/step_2_3_inspection_priority_score.csv")


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def find_column(df, possible_names):
    normalized_columns = {
        col.lower().strip().replace(" ", "_"): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower().strip().replace(" ", "_")
        if key in normalized_columns:
            return normalized_columns[key]

    return None


def yes_no_score(value, yes_score=10, no_score=0, unclear_score=5):
    value = normalize_text(value)

    if value == "YES":
        return yes_score
    elif value == "NO":
        return no_score
    elif value == "UNCLEAR":
        return unclear_score
    elif value in {"NA", "N/A", ""}:
        return 0
    else:
        return unclear_score


def inverse_yes_no_score(value, yes_score=0, no_score=10, unclear_score=5):
    """
    For fields where NO is worse than YES.
    Example: sidewalk_connection = No should increase inspection priority.
    """

    value = normalize_text(value)

    if value == "YES":
        return yes_score
    elif value == "NO":
        return no_score
    elif value == "UNCLEAR":
        return unclear_score
    elif value in {"NA", "N/A", ""}:
        return 0
    else:
        return unclear_score


def numeric_count_score(value, points_per_item=3, max_score=15):
    try:
        count = int(float(value))
    except:
        return 0

    return min(count * points_per_item, max_score)


def inspection_level(score):
    if score >= 80:
        return "Very High"
    elif score >= 60:
        return "High"
    elif score >= 35:
        return "Medium"
    else:
        return "Low"


# ============================================================
# MAIN
# ============================================================

def main():
    base_df = pd.read_csv(BASE_CSV)
    image_df = pd.read_csv(IMAGE_RESULTS_CSV)

    base_stop_col = find_column(base_df, ["Stop Code", "stop_code", "stop_id"])
    image_stop_col = find_column(image_df, ["stop_id", "Stop Code", "stop_code"])

    if base_stop_col is None:
        raise ValueError(f"Could not find stop code column in base file: {list(base_df.columns)}")

    if image_stop_col is None:
        raise ValueError(f"Could not find stop id column in image file: {list(image_df.columns)}")

    base_df[base_stop_col] = base_df[base_stop_col].astype(str).str.strip()
    image_df[image_stop_col] = image_df[image_stop_col].astype(str).str.strip()

    df = base_df.merge(
        image_df,
        left_on=base_stop_col,
        right_on=image_stop_col,
        how="left",
        suffixes=("", "_image")
    )

    # Try to find useful Gemini/image-analysis columns
    visible_col = find_column(df, ["bus_stop_visible"])
    confidence_col = find_column(df, ["confidence"])
    landing_type_col = find_column(df, ["landing_type"])
    sidewalk_col = find_column(df, ["sidewalk_connection"])
    landing_pad_col = find_column(df, ["landing_pad"])
    shelter_col = find_column(df, ["shelter_count", "shelter"])
    bench_col = find_column(df, ["bench_count", "bench"])
    trash_col = find_column(df, ["trash_count", "trash"])
    lighting_col = find_column(df, ["street_lighting"])

    df["inspection_priority_score"] = 0

    # Start with some cleaning priority weight
    if "cleaning_priority_score" in df.columns:
        df["inspection_priority_score"] += df["cleaning_priority_score"] * 0.35

    # If stop is not clearly visible, inspect it
    if visible_col:
        df["inspection_priority_score"] += df[visible_col].apply(
            lambda x: yes_no_score(x, yes_score=0, no_score=25, unclear_score=15)
        )

    # Low confidence means inspect
    if confidence_col:
        df["confidence_numeric"] = pd.to_numeric(df[confidence_col], errors="coerce")
        df["inspection_priority_score"] += df["confidence_numeric"].apply(
            lambda x: 15 if pd.isna(x) else max(0, 15 - (x * 15 if x <= 1 else x / 100 * 15))
        )

    # Missing or weak pedestrian infrastructure increases inspection priority
    if sidewalk_col:
        df["inspection_priority_score"] += df[sidewalk_col].apply(
            lambda x: inverse_yes_no_score(x, yes_score=0, no_score=20, unclear_score=10)
        )

    if landing_pad_col:
        df["inspection_priority_score"] += df[landing_pad_col].apply(
            lambda x: inverse_yes_no_score(x, yes_score=0, no_score=15, unclear_score=8)
        )

    # Unpaved / unclear landing type increases inspection priority
    if landing_type_col:
        df["inspection_priority_score"] += df[landing_type_col].apply(
            lambda x: 15 if normalize_text(x) in {"UNPAVED", "DIRT", "GRASS", "GRAVEL", "UNCLEAR"} else 0
        )

    # Amenities add inspection value because there is more to check/maintain
    if shelter_col:
        df["inspection_priority_score"] += df[shelter_col].apply(
            lambda x: numeric_count_score(x, points_per_item=5, max_score=15)
        )

    if bench_col:
        df["inspection_priority_score"] += df[bench_col].apply(
            lambda x: numeric_count_score(x, points_per_item=3, max_score=9)
        )

    if trash_col:
        df["inspection_priority_score"] += df[trash_col].apply(
            lambda x: numeric_count_score(x, points_per_item=3, max_score=9)
        )

    # No lighting increases inspection priority
    if lighting_col:
        df["inspection_priority_score"] += df[lighting_col].apply(
            lambda x: inverse_yes_no_score(x, yes_score=0, no_score=8, unclear_score=4)
        )

    df["inspection_priority_score"] = df["inspection_priority_score"].round(2)
    df["inspection_priority_level"] = df["inspection_priority_score"].apply(inspection_level)

    df = df.sort_values(
        by=["inspection_priority_score", base_stop_col],
        ascending=[False, True]
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")
    print(f"Base file: {BASE_CSV}")
    print(f"Image results file: {IMAGE_RESULTS_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    print()
    print("Columns used:")
    print(f"visible_col: {visible_col}")
    print(f"confidence_col: {confidence_col}")
    print(f"landing_type_col: {landing_type_col}")
    print(f"sidewalk_col: {sidewalk_col}")
    print(f"landing_pad_col: {landing_pad_col}")
    print(f"shelter_col: {shelter_col}")
    print(f"bench_col: {bench_col}")
    print(f"trash_col: {trash_col}")
    print(f"lighting_col: {lighting_col}")
    print()
    print(df["inspection_priority_level"].value_counts())


if __name__ == "__main__":
    main()