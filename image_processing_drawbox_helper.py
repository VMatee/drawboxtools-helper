import requests
from PIL import Image, ImageDraw, ImageFont
import os
import json
import base64
import shutil

# ==================== Configuration ====================
BASE_URL = ""  # Update if needed (do not include trailing slash)
INPUT_FOLDER = "input_images"      # Folder containing images to process
FONT_PATH = "AI_Model/Fonts/THSarabunNew BoldItalic.ttf"  # Update if needed
OUTPUT_FOLDER = "labelme_output"   # Folder to store the output images and JSON annotations

username = ""
password = ""

# Dictionary to store a unique color for each class
class_colors = {}

def get_color_for_class(class_name):
    """Return a fixed color for the given class name."""
    if class_name in class_colors:
        return class_colors[class_name]
    else:
        # A fixed palette of colors (RGB tuples)
        color_palette = [
            (255, 0, 0),     # red
            (0, 255, 0),     # green
            (0, 0, 255),     # blue
            (255, 255, 0),   # yellow
            (255, 0, 255),   # magenta
            (0, 255, 255),   # cyan
            (128, 0, 0),     # maroon
            (0, 128, 0),     # dark green
            (0, 0, 128),     # navy
            (128, 128, 0),   # olive
            (128, 0, 128),   # purple
            (0, 128, 128),   # teal
        ]
        # Cycle through the palette if more classes appear
        assigned_color = color_palette[len(class_colors) % len(color_palette)]
        class_colors[class_name] = assigned_color
        return assigned_color

def generate_labelme_json(image_path, shapes, image_width, image_height):
    """
    Create a LabelMe-style JSON annotation.
    
    The image data is embedded as a base64 string.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
        
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageWidth": image_width,
        "imageHeight": image_height
    }
    return labelme_json

def process_image(image_path, prediction_url, headers, font):
    """
    Process a single image: send it for prediction, draw annotations,
    generate LabelMe JSON, and save outputs.
    """
    print(f"\nProcessing image: {image_path}")
    
    # === Send image for prediction ===
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            resp = requests.post(prediction_url, files=files, headers=headers)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    
    if resp.status_code != 201:
        print(f"Prediction failed for {image_path}: {resp.text}")
        return

    predictions = resp.json().get("result", [])
    print(f"Received {len(predictions)} predictions for {os.path.basename(image_path)}")
    image_info = resp.json().get("image_info", [])
    print(image_info)
    # === Open image and prepare drawing ===
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    shapes = []  # To store shapes for LabelMe JSON

    # === Process each prediction ===
    for pred in predictions:
        x, y = pred["x"], pred["y"]
        w, h = pred["w"], pred["h"]
        class_name = pred["class_name"]
        confidence = pred["confidence"]

        # Get a fixed color for this class
        color = get_color_for_class(class_name)

        # Draw the bounding box
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
        
        # Draw the label text (above the bounding box)
        label_text = f"{class_name}: {confidence:.2f}"
        draw.text((x, y - 20), label_text, font=font, fill=color)
        
        # Prepare the shape for LabelMe JSON
        shape = {
            "label": class_name,
            "points": [[x, y], [x + w, y + h]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(shape)
    
    # === Save the image with drawn predictions (preview) ===
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(OUTPUT_FOLDER, base_name + "_prediction.jpg")
    image.save(output_image_path)
    print(f"Saved preview image with predictions to: {output_image_path}")
    
    # === Generate and save LabelMe JSON annotation ===
    width, height = image.size
    labelme_json = generate_labelme_json(image_path, shapes, width, height)
    json_filename = os.path.join(OUTPUT_FOLDER, base_name + ".json")
    with open(json_filename, "w") as jf:
        json.dump(labelme_json, jf, indent=4)
    print(f"Saved LabelMe annotation JSON to: {json_filename}")

    # === Copy the original image into the output folder ===
    image_copy_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    shutil.copy(image_path, image_copy_path)
    print(f"Copied original image to: {image_copy_path}")

def main():
    # === 1. Login and get token ===
    login_url = f"{BASE_URL}/login"
    credentials = {"username": username, "password": password}
    
    print("=== Logging in ===")
    resp = requests.post(login_url, json=credentials)
    if resp.status_code != 200:
        print(f"Login failed: {resp.text}")
        return
    access_token = resp.json().get("access_token")
    if not access_token:
        print("Access token not found in response.")
        return
    print("Login successful!\n")
    
    # === 2. Prepare output folder and prediction URL ===
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    prediction_url = f"{BASE_URL}/predictionbox"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Load the custom font (or fallback to default)
    try:
        font = ImageFont.truetype(FONT_PATH, size=16)
    except IOError:
        print("Warning: Custom font not found, using default.")
        font = ImageFont.load_default()
    
    # === 3. Process all images in the input folder ===
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' does not exist.")
        return

    # Only process files with image extensions
    allowed_extensions = (".png", ".jpg", ".jpeg")
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(allowed_extensions)]
    
    if not image_files:
        print(f"No image files found in folder '{INPUT_FOLDER}'.")
        return

    for image_file in image_files:
        image_path = os.path.join(INPUT_FOLDER, image_file)
        process_image(image_path, prediction_url, headers, font)

if __name__ == "__main__":
    main()


