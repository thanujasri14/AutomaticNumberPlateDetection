import cv2
import os
import logging
from flask import Flask, send_file

logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

# Path to the dataset containing vehicle images
dataset_path = r"C:\Assignment-Sohga\dataset"  

# Load pre-trained Cascade Classifier for license plate detection
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize counters
total_plate_count = 0
yellow_plate_count = 0

# Create a directory to store processed images
output_dir = os.path.join(dataset_path, "processed_images")
os.makedirs(output_dir, exist_ok=True)

# Iterate through each image in the dataset
for image_filename in os.listdir(dataset_path):
    if image_filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(dataset_path, image_filename)
        img = cv2.imread(image_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect license plates
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
        
        for (x, y, w, h) in plates:
            total_plate_count += 1
            
            roi = img[y:y+h, x:x+w]
            
            # Convert the ROI to HSV for color analysis
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define the lower and upper bounds of yellow color in HSV
            lower_yellow = (20, 100, 100)
            upper_yellow = (40, 255, 255)

            # Define the lower and upper bounds for white color in HSV
            lower_white = (0, 0, 150)
            upper_white = (180, 50, 255)

            # Define the lower and upper bounds for green color in HSV
            lower_green = (35, 50, 50)
            upper_green = (85, 255, 255)
            
            # Create masks to filter regions of different colors
            yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
            white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
            green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

            # Count non-zero pixels in each mask
            yellow_pixel_count = cv2.countNonZero(yellow_mask)
            white_pixel_count = cv2.countNonZero(white_mask)
            green_pixel_count = cv2.countNonZero(green_mask)
            
            # If a significant number of pixels are detected, draw a bounding box
            if yellow_pixel_count > 500 or white_pixel_count > 500 or green_pixel_count > 500:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            if yellow_pixel_count > 500:
                yellow_plate_count += 1
        
        # Save the processed image
        processed_image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(processed_image_path, img)

print(f"Total number of license plates: {total_plate_count}")
print(f"Total number of yellow plates: {yellow_plate_count}")

download_response = input("Do you want to download the processed image dataset? (yes/no): ")
if download_response.lower() == "yes":
    import shutil
    shutil.make_archive(output_dir, 'zip', output_dir)
    print("Processed image dataset has been zipped and is ready for download. You can now open a web browser and go to http://127.0.0.1:5000/download_processed_images to trigger the download of the processed images zip file.")

@app.route('/download_processed_images')
def download_processed_images():
    shutil.make_archive(output_dir, 'zip', output_dir)
    zip_path = f"{output_dir}.zip"
    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run()


#You can now open a web browser and go to "http://127.0.0.1:5000/download_processed_images" to trigger the download of the processed images zip file. remember to run the program before browsing.













