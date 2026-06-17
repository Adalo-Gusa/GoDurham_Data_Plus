from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path("output/Altered 2026 GoDurham Bus Stop List_updated_filtered_with_abc.csv")
OUTPUT_CSV = Path("output/step_2_2_cleaning_priority_score.csv")


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def find_column(df, possible_names):
    """
    Finds a column even if the name is slightly different.
    """

    normalized_columns = {
        col.lower().strip().replace(" ", "_"): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower().strip().replace(" ", "_")
        if key in normalized_columns:
            return normalized_columns[key]

    return None


def abc_to_score(category):
    """
    A = highest cleaning priority
    B = medium
    C = lower
    """

    category = normalize_text(category)

    if category == "A":
        return 100
    elif category == "B":
        return 70
    elif category == "C":
        return 40
    else:
        return 0


def score_to_priority(score):
    if score >= 90:
        return "Very High"
    elif score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


# ============================================================
# MAIN
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)

    category_col = find_column(df, [
        "ABC",
        "Category",
        "Cleaning Category",
        "Maintenance Category",
        "ABC Category",
        "cleaning_category"
    ])

    if category_col is None:
        raise ValueError(
            f"Could not find ABC/category column. Available columns: {list(df.columns)}"
        )

    df["cleaning_priority_score"] = df[category_col].apply(abc_to_score)
    df["cleaning_priority_level"] = df["cleaning_priority_score"].apply(score_to_priority)

    df = df.sort_values(
        by=["cleaning_priority_score", "Stop Code"],
        ascending=[False, True]
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")
    print(f"Input file: {INPUT_CSV}")
    print(f"ABC/category column used: {category_col}")
    print(f"Output file: {OUTPUT_CSV}")
    print()
    print(df["cleaning_priority_level"].value_counts())


if __name__ == "__main__":
    main()