import re
import shutil
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path("images_metadata")
OUTPUT_DIR = Path("images_metadata_current_only")
NO_TXT_OUTPUT_DIR = Path("images_metadata_current_only_no_txt")
TXT_ONLY_OUTPUT_DIR = Path("images_metadata_current_only_txt_only")
NON_CURRENT_IMAGES_DIR = Path("non_current_iimages")

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

def extract_year_from_filename(path: Path):
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


def non_current_output_path_for(source_path: Path) -> Path:
    """
    Keeps the same subfolder structure in the non-current images folder.
    """

    relative_path = source_path.relative_to(SOURCE_DIR)
    return NON_CURRENT_IMAGES_DIR / relative_path


def create_no_txt_folder():
    """
    Creates a second output folder that copies everything from OUTPUT_DIR
    except .txt files.
    """

    if NO_TXT_OUTPUT_DIR.exists():
        shutil.rmtree(NO_TXT_OUTPUT_DIR)

    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".txt":
            relative_path = path.relative_to(OUTPUT_DIR)
            destination_path = NO_TXT_OUTPUT_DIR / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination_path)

    print(f"No-txt folder created: {NO_TXT_OUTPUT_DIR.resolve()}")


def create_txt_only_folder():
    """
    Creates a third output folder that copies only .txt files from OUTPUT_DIR.
    """

    if TXT_ONLY_OUTPUT_DIR.exists():
        shutil.rmtree(TXT_ONLY_OUTPUT_DIR)

    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".txt":
            relative_path = path.relative_to(OUTPUT_DIR)
            destination_path = TXT_ONLY_OUTPUT_DIR / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination_path)

    print(f"Txt-only folder created: {TXT_ONLY_OUTPUT_DIR.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR.resolve()}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    if NON_CURRENT_IMAGES_DIR.exists():
        shutil.rmtree(NON_CURRENT_IMAGES_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NON_CURRENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        path for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    checked = 0
    copied_current = 0
    replaced_old = 0
    copied_non_current = 0
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
            # Create txt replacement in OUTPUT_DIR
            txt_path = destination_path.with_suffix(".txt")
            txt_path.write_text(REPLACEMENT_MESSAGE, encoding="utf-8")
            replaced_old += 1
            print(f"Old image replaced with txt in output folder: {txt_path.name}")

            # Copy original non-current image into separate folder
            non_current_destination_path = non_current_output_path_for(source_path)
            non_current_destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, non_current_destination_path)
            copied_non_current += 1
            print(f"Copied non-current image: {source_path.name}")

        else:
            shutil.copy2(source_path, destination_path)
            copied_current += 1
            print(f"Copied current image: {source_path.name}")

    create_no_txt_folder()
    create_txt_only_folder()

    print("\nFinished.")
    print(f"Images checked: {checked}")
    print(f"Current images copied: {copied_current}")
    print(f"Old images replaced with txt: {replaced_old}")
    print(f"Non-current images copied: {copied_non_current}")
    print(f"Skipped because no year found: {skipped_no_year}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print(f"Non-current images folder: {NON_CURRENT_IMAGES_DIR.resolve()}")


if __name__ == "__main__":
    main()