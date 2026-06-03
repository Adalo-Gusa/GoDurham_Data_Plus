import cv2
import shutil
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("images_metadata_current_only_no_txt")
OUTPUT_DIR = Path("images_metadata_current_only_no_txt_upscaled")

MODEL_PATH = Path("FSRCNN_x2.pb")
MODEL_NAME = "fsrcnn"
SCALE = 2

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def output_path_for(source_path: Path) -> Path:
    """
    Keeps the same subfolder structure in the upscaled output folder.
    """

    relative_path = source_path.relative_to(INPUT_DIR)
    return OUTPUT_DIR / relative_path


def load_super_resolution_model():
    """
    Loads OpenCV's deep-learning super-resolution model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Super-resolution model not found: {MODEL_PATH.resolve()}\n"
            "Download FSRCNN_x2.pb and put it in the same folder as this script."
        )

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODEL_PATH))
    sr.setModel(MODEL_NAME, SCALE)

    return sr


def upscale_image(sr, image_path: Path):
    """
    Reads and upscales one image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    upscaled = sr.upsample(image)

    return upscaled


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR.resolve()}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sr = load_super_resolution_model()

    image_files = [
        path for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    print(f"Found {len(image_files)} images to upscale.")

    processed = 0
    failed = 0

    for source_path in image_files:
        destination_path = output_path_for(source_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            upscaled = upscale_image(sr, source_path)
        except Exception as error:
            failed += 1
            print(f"Failed to upscale {source_path}: {error}")
            continue

        success = cv2.imwrite(str(destination_path), upscaled)

        if success:
            processed += 1
            print(f"Upscaled: {source_path.name}")
        else:
            failed += 1
            print(f"Failed to save: {destination_path}")

    print("\nFinished.")
    print(f"Images found: {len(image_files)}")
    print(f"Images upscaled: {processed}")
    print(f"Failed: {failed}")
    print(f"Input folder: {INPUT_DIR.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()