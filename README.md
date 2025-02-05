# drawboxtools-helper
# Image Processing and Annotation Tool

This project provides a Python script designed to process images for object detection, annotate detected objects with bounding boxes, and generate LabelMe-compatible JSON files. The script interacts with an external prediction API to obtain prediction results which are then used to draw annotations and create JSON files for further labeling work in LabelMe.

## Features

- **Batch Processing:** Automatically processes all images within a specified directory.
- **API Integration:** Sends images to a configured prediction API and retrieves bounding box data.
- **Annotation Drawing:** Draws bounding boxes and labels on images based on API responses.
- **JSON Generation:** Creates LabelMe-compatible JSON annotations for each processed image.

## Requirements

- Python 3.x
- Pillow (Python Imaging Library Fork)
- Requests

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/VMatee/drawboxtools-helper.git
   cd your-repository-directory
2. **Install Required Python Packages:**
Install the required Python packages using the provided requirements.txt:
   ```bash
    pip install -r requirements.txt

**Configuration**
Before running the script, configure the following settings in the script file:

- BASE_URL: The URL of the prediction API (without a trailing slash).
- INPUT_FOLDER: Directory containing the images to be processed.
- FONT_PATH: Path to a TrueType font file used for annotation labels.
- OUTPUT_FOLDER: Directory where annotated images and JSON files will be stored.
- username: Username for API authentication.
- password: Password for API authentication.

**Usage**
To run the script, execute the following command in your terminal:
- window
  
  ```bash
  py image_processing_drawbox_helper.py

- Linux/macOS:

  ```bash
  python3 image_processing_drawbox_helper.py

**What Happens When You Run the Script?**

- The script logs into the prediction API using the provided credentials.
- It iterates through all images in the INPUT_FOLDER.
- Each image is sent to the prediction API, and the script receives data about the locations and classes of objects detected in the images.
- The script draws bounding boxes around each detected object on the images and labels them with the class name and prediction confidence.
- A LabelMe-compatible JSON file is generated for each image, containing the annotations.
- Both the annotated image and the JSON file are saved in the OUTPUT_FOLDER.

**Output**
The output includes:

- Annotated Images: Images with drawn bounding boxes and class labels saved in the specified output folder.
- JSON Files: LabelMe-compatible JSON files for each image, allowing for further annotation or review in the LabelMe tool.

**Troubleshooting**
If you encounter issues with the prediction API:

- Ensure that the BASE_URL is correct and the API server is running.
- Check your network connection.
- Verify that your API credentials (username and password) are correct.

If there are errors related to file paths, such as INPUT_FOLDER or FONT_PATH:

- Confirm that the paths are correctly specified and accessible.
- Ensure that the font file exists at the specified FONT_PATH.
