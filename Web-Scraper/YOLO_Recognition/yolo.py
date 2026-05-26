pip install inference-sdk

# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workspace
client = InferenceHTTPClient(
  api_url="https://serverless.roboflow.com",
  api_key="VT3lDj8yZcgP7G23yPB2"
)

# 3. Run your workflow on an image
result = client.run_workflow(
  workspace_name="sebastian-muvf7",
  workflow_id="general-segmentation-api",
  images={
    "image": "YOUR_IMAGE.jpg"  # Path to your image file
  },
  parameters={
    "classes": "sign, shelter"
  },
  use_cache=True  # cache workflow definition for 15 minutes
)

# 4. Get your results
print(result)