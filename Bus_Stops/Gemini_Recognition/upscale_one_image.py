import cv2
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGE = Path("/Users/sebastiansanchez121/data/Bus_Stops/Gemini_Recognition/images_metadata_8headings/6631_TW_Alexander_Dr_at_Page_Rd_WB_2025-01_front_left_heading-297.jpg")

OUTPUT_DIR = Path("single_image_upscaled")
OUTPUT_IMAGE = OUTPUT_DIR / f"{INPUT_IMAGE.stem}_upscaled_x4{INPUT_IMAGE.suffix}"

MODEL_PATH = Path("FSRCNN_x4.pb")
MODEL_NAME = "fsrcnn"
SCALE = 4


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE.resolve()}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Super-resolution model not found: {MODEL_PATH.resolve()}\n"
            "Download FSRCNN_x4.pb and put it in the same folder as this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(INPUT_IMAGE))

    if image is None:
        raise ValueError(f"Could not read image: {INPUT_IMAGE}")

    print(f"Original size: {image.shape[1]} x {image.shape[0]}")

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODEL_PATH))
    sr.setModel(MODEL_NAME, SCALE)

    upscaled = sr.upsample(image)

    print(f"Upscaled size: {upscaled.shape[1]} x {upscaled.shape[0]}")

    success = cv2.imwrite(str(OUTPUT_IMAGE), upscaled)

    if not success:
        raise ValueError(f"Could not save image: {OUTPUT_IMAGE}")

    print(f"Saved upscaled image to: {OUTPUT_IMAGE.resolve()}")


if __name__ == "__main__":
    main()