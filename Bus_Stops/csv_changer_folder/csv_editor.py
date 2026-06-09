from pathlib import Path
import csv

# ============================================================
# CONFIG
# ============================================================

MAIN_CSV = Path("Altered 2026 GoDurham Bus Stop List.csv")
COORDS_CSV = Path("newestcoordinates.csv")
STOPS_FILTER_FILE = Path("stops.txt")

OUTPUT_CSV = Path("Altered 2026 GoDurham Bus Stop List_updated_filtered.csv")


# ============================================================
# HELPERS
# ============================================================

def parse_stop_row(row):
    """
    Handles rows where Stop Name may contain commas.

    Expected:
    Stop Code, Stop Name, Latitude, Longitude

    Uses:
    - first column = Stop Code
    - last 2 columns = Latitude / Longitude
    - everything between = Stop Name
    """

    if len(row) < 4:
        raise ValueError(f"Bad row, fewer than 4 columns: {row}")

    stop_code = row[0].strip()
    stop_name = ",".join(row[1:-2]).strip()
    lat = row[-2].strip()
    lon = row[-1].strip()

    return stop_code, stop_name, lat, lon


def load_updated_coords(coords_csv):
    updated = {}

    with coords_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if not row or not row[0].strip():
                continue

            stop_code, _, lat, lon = parse_stop_row(row)
            updated[stop_code] = (lat, lon)

    return updated


def load_allowed_stop_codes(stops_file):
    """
    Reads stops.txt GTFS-style file.

    Uses stop_code column if present.
    Falls back to stop_id if stop_code is missing.
    """

    allowed = set()

    with stops_file.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            stop_code = (row.get("stop_code") or row.get("stop_id") or "").strip()

            if stop_code:
                allowed.add(stop_code)

    return allowed


# ============================================================
# MAIN
# ============================================================

def replace_and_filter_coordinates():
    updated_coords = load_updated_coords(COORDS_CSV)
    allowed_stop_codes = load_allowed_stop_codes(STOPS_FILTER_FILE)

    total_count = 0
    kept_count = 0
    filtered_out_count = 0
    replaced_count = 0

    with MAIN_CSV.open("r", newline="", encoding="utf-8-sig") as infile, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        next(reader)  # skip old header

        writer.writerow(["Stop Code", "Stop Name", "Latitude", "Longitude"])

        for row in reader:
            if not row or not row[0].strip():
                continue

            total_count += 1

            stop_code, stop_name, lat, lon = parse_stop_row(row)

            # Filter out stops not in stops.txt
            if stop_code not in allowed_stop_codes:
                filtered_out_count += 1
                continue

            # Replace coordinates only if stop code exists in newestedcoordinates.csv
            if stop_code in updated_coords:
                lat, lon = updated_coords[stop_code]
                replaced_count += 1

            writer.writerow([stop_code, stop_name, lat, lon])
            kept_count += 1

    print("Done.")
    print(f"Total stops in old file: {total_count}")
    print(f"Stops kept from stops.txt: {kept_count}")
    print(f"Stops filtered out: {filtered_out_count}")
    print(f"Coordinates replaced: {replaced_count}")
    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    replace_and_filter_coordinates()