from pathlib import Path
import csv

MAIN_CSV = Path("Altered 2026 GoDurham Bus Stop List.csv")
COORDS_CSV = Path("newestedcoordinates.csv")
OUTPUT_CSV = Path("Altered 2026 GoDurham Bus Stop List_updated_coords.csv")


def parse_stop_row(row):
    if len(row) < 4:
        raise ValueError(f"Bad row, fewer than 4 columns: {row}")

    stop_code = row[0].strip()
    lat = row[-2].strip()
    lon = row[-1].strip()
    stop_name = ",".join(row[1:-2]).strip()

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


def replace_coordinates():
    updated_coords = load_updated_coords(COORDS_CSV)

    replaced_count = 0
    total_count = 0

    with MAIN_CSV.open("r", newline="", encoding="utf-8-sig") as infile, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        writer.writerow(["Stop Code", "Stop Name", "Latitude", "Longitude"])

        next(reader)

        for row in reader:
            if not row or not row[0].strip():
                continue

            total_count += 1
            stop_code, stop_name, lat, lon = parse_stop_row(row)

            if stop_code in updated_coords:
                lat, lon = updated_coords[stop_code]
                replaced_count += 1

            writer.writerow([stop_code, stop_name, lat, lon])

    print("Done.")
    print(f"Total stops processed: {total_count}")
    print(f"Coordinates replaced: {replaced_count}")
    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    replace_coordinates()