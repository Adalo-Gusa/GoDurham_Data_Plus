import re
import shutil
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path("images_metadata")
OUTPUT_DIR = Path("images_metadata_current_only")

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

REPLACEMENT_MESSAGE = "no current image detected\n"


# ============================================================
# HELPERS
# ============================================================

def extract_year_from_filename(path: Path) -> int | None:
    """
    Finds a 4-digit year in the filename.

    Example:
    1072_Old_Chapel_Hill_Rd_2025-01_center_heading-102.jpg -> 2025
    1072_Old_Chapel_Hill_Rd_2024-09_center_heading-102.jpg -> 2024
    """

    match = re.search(r"(19\d{2}|20\d{2})", path.name)

    if not match:
        return None

    return int(match.group(1))


def output_path_for(source_path: Path) -> Path:
    """
    Keeps the same subfolder structure in the new output folder.
    """

    relative_path = source_path.relative_to(SOURCE_DIR)
    return OUTPUT_DIR / relative_path


# ============================================================
# MAIN
# ============================================================

def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        path for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    checked = 0
    copied_current = 0
    replaced_old = 0
    skipped_no_year = 0

    for source_path in image_files:
        checked += 1

        year = extract_year_from_filename(source_path)

        destination_path = output_path_for(source_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if year is None:
            skipped_no_year += 1
            print(f"Skipped no year found: {source_path.name}")
            continue

        if year < 2025:
            txt_path = destination_path.with_suffix(".txt")
            txt_path.write_text(REPLACEMENT_MESSAGE, encoding="utf-8")
            replaced_old += 1
            print(f"Old image replaced with txt in output folder: {txt_path.name}")

        else:
            shutil.copy2(source_path, destination_path)
            copied_current += 1
            print(f"Copied current image: {source_path.name}")

    print("\nFinished.")
    print(f"Images checked: {checked}")
    print(f"Current images copied: {copied_current}")
    print(f"Old images replaced with txt: {replaced_old}")
    print(f"Skipped because no year found: {skipped_no_year}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()