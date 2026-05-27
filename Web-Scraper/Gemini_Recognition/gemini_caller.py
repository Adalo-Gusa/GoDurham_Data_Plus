import os
import json
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client(
    enterprise=True,
    project=os.environ["DataPlus-GoDurham"],
    location="global",
)

MODEL = "gemini-3.5-flash"

CRITERIA = """
Criteria:
1. has_person: at least one human is visible
2. has_text: readable or partially readable text is visible
3. has_food: food or drink is visible
4. has_logo: a brand/logo/mark is visible
5. is_blurry: image is noticeably blurry or low-quality
"""

PROMPT = f"""
Analyze the image and determine whether each listed criterion is present.

Rules:
- Only mark present=true if the criterion is clearly visible.
- If unclear, mark present=false and set confidence below 0.60.
- Do not infer from context unless visually supported.
- Return only JSON matching the schema.

{CRITERIA}
"""

response_schema = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string"},
        "criteria": {
            "type": "object",
            "properties": {
                "has_person": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["present", "confidence", "evidence"],
                },
                "has_text": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["present", "confidence", "evidence"],
                },
                "has_food": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["present", "confidence", "evidence"],
                },
                "has_logo": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["present", "confidence", "evidence"],
                },
                "is_blurry": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["present", "confidence", "evidence"],
                },
            },
            "required": [
                "has_person",
                "has_text",
                "has_food",
                "has_logo",
                "is_blurry",
            ],
        },
    },
    "required": ["image_id", "criteria"],
}


def analyze_image(image_path: str) -> dict:
    image_path = Path(image_path)

    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type="image/jpeg",  # change to image/png if needed
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            PROMPT + f'\nImage ID: "{image_path.name}"',
            image_part,
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=types.ThinkingConfig(
                thinking_level="LOW"
            ),
            media_resolution="MEDIA_RESOLUTION_LOW",
        ),
    )

    return json.loads(response.text)


image_files = list(Path("images").glob("*.jpg"))

results = []
for img in image_files:
    try:
        result = analyze_image(str(img))
        results.append(result)
        print(f"Done: {img.name}")
    except Exception as e:
        print(f"Failed: {img.name}: {e}")

with open("image_criteria_results.json", "w") as f:
    json.dump(results, f, indent=2)