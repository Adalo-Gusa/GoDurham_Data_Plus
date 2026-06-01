import cv2
import os  # <-- New tool to help us create folders

# 1. Initialize the Super Resolution object
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# 2. Path to your downloaded pre-trained model file
model_path = "/Users/sebastiansanchez121/data/Bus_Stops/Gemini_Recognition/EDSR_x2.pb"

# 3. Read the model and load it into memory
sr.readModel(model_path)
sr.setModel("edsr", 2)

# 4. Load your low-resolution input image
input_image = cv2.imread("/Users/sebastiansanchez121/data/Bus_Stops/Gemini_Recognition/images_metadata/1072_Old_Chapel_Hill_Rd_at_Garrett_Rd_EB_2025-01_center_heading-102.jpg")

# 5. Upsample the image
print("Processing image resolution upgrade...")
high_res_image = sr.upsample(input_image)

# --- NEW FOLDER SAVING LOGIC ---

# 6. Define the exact path for your new folder
output_folder = "/Users/sebastiansanchez121/data/Bus_Stops/Gemini_Recognition/upscaled_images"

# 7. Create the folder automatically (if it doesn't already exist)
os.makedirs(output_folder, exist_ok=True)

# 8. Combine the folder path and the new file name
output_file_path = os.path.join(output_folder, "bus_stop_high_res.jpg")

# 9. Save the enhanced result into the new folder
cv2.imwrite(output_file_path, high_res_image)
print(f"Done! Enhanced image saved to: {output_file_path}")