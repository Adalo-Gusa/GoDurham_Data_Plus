import cv2

# 1. Initialize the Super Resolution object
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# 2. Path to your downloaded pre-trained model file
model_path = "Bus_Stops/images_metadata/EDSR_x2.pb"

# 3. Read the model and load it into memory
sr.readModel(model_path)
sr.setModel("edsr", 2)

# 4. Load your low-resolution input image
# Make sure your actual image is named 'bus_stop.jpg' or change this line!
# Update the path to look inside the folder
input_image = cv2.imread("Bus_Stops/images_metadata/1072_Old_Chapel_Hill_Rd_at_Garrett_Rd_EB_2025-01_center_heading-102.jpg")

# 5. Upsample the image
print("Processing image resolution upgrade...")
high_res_image = sr.upsample(input_image)

# 6. Save and view the enhanced result
cv2.imwrite("bus_stop_high_res.jpg", high_res_image)
print("Done! Enhanced image saved as 'bus_stop_high_res.jpg'")