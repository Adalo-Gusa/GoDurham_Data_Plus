import os
from inference_sdk import InferenceHTTPClient

with open("/Users/sebastiansanchez121/data/roboflow_api.txt", "r") as f:
    api_key = f.read().strip()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

result = CLIENT.infer(
    "/Users/sebastiansanchez121/data/Web-Scraper/images_metadata/1072_Old_Chapel_Hill_Rd_at_Garrett_Rd_EB_2025-01_center_heading-102.jpg",
    model_id="bus-stop-afloz/1"
)

print(result)

