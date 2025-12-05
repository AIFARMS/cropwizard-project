"""
Pest Detection Tool for CropWizard
Based on custom-trained YOLOv8n model for pest detection and classification

To deploy: beam deploy pest_detection.py
For testing: beam serve pest_detection.py

This tool accepts image URLs and returns base64-encoded annotated images with bounding boxes and class labels.

Model trained by Aditya Sengupta
Google Colab Implementation: https://colab.research.google.com/drive/1GO-lw2PJtVewlA-xhBfgBLId8v4v-BE2?usp=sharing
"""

import base64
import inspect
import io
import json
import os
import time
import traceback
from typing import Any, Dict, List

from beam import App, Runtime, Volume
from PIL import Image
from ultralytics import YOLO
import requests
import beam

# Define Python package requirements for the Beam environment
# Updated versions to address security vulnerabilities
requirements = [
    "torch>=2.6.0",
    "ultralytics>=8.3.0",
    "torchvision>=0.19.0",
    "opencv-python>=4.10.0",
    "pillow>=10.4.0",
]

# Define volume path for model storage
volume_path = "./models"

# Create Beam app with runtime configuration
app = App(
    "pest_detection",
    runtime=Runtime(
        cpu=1,
        memory="3Gi",
        image=beam.Image(  # type: ignore
            python_version="python3.10",
            python_packages=requirements,
        ),
    ),
    volumes=[Volume(
        name="pest_detection_models",
        path=volume_path,
    )],
)


def loader():
    """
    Load the YOLOv8n pest detection model.
    
    Model weights are downloaded from a hosted URL and cached in a volume for faster subsequent loads.
    The model is trained on various pest species and can detect and classify them in images.
    
    Returns:
        YOLO: Loaded YOLO model ready for inference
    """
    model_url = "https://assets.kastan.ai/pest_detection_model_weights.pt"
    model_path = f'{volume_path}/pest_detection_model_weights.pt'
    start_time = time.monotonic()

    # Check if model exists in volume, otherwise download it
    if not os.path.exists(model_path):
        print("Downloading pest detection model from the internet...")
        response = requests.get(model_url, timeout=60)
        
        # Ensure directory exists
        os.makedirs(volume_path, exist_ok=True)
        
        with open(model_path, "wb") as f:
            f.write(response.content)
        model = YOLO(model_path)
        print("Model downloaded and loaded successfully")
    else:
        print("Loading pest detection model from volume...")
        model = YOLO(model_path)
        print("Model loaded successfully from cache")

    print(f"⏰ Runtime to load model: {(time.monotonic() - start_time):.2f} seconds")
    return model


@app.rest_api(
    workers=1,
    max_pending_tasks=10,
    max_retries=1,
    timeout=60,
    keep_warm_seconds=60 * 3,
    loader=loader
)
def predict(**inputs: Dict[str, Any]):
    """
    REST API endpoint for pest detection.
    
    Accepts a list of image URLs, runs YOLO inference on each image,
    and returns base64-encoded PNG images with bounding boxes and class labels.
    
    Args:
        **inputs: Dictionary containing:
            - context: The loaded YOLO model (injected by Beam)
            - image_urls: List of image URLs to process
    
    Returns:
        List[str]: List of base64-encoded PNG images with pest detection annotations,
                   or error message string if processing fails
    
    Example request:
        {
            "image_urls": [
                "https://www.arborday.org/trees/health/pests/images/figure-whiteflies-1.jpg",
                "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg"
            ]
        }
    """
    print("🔍 Pest detection endpoint called")
    
    # Get the pre-loaded model from context
    model = inputs["context"]
    
    # Get image URLs from input
    image_urls: List[str] = inputs.get('image_urls', [])  # type: ignore
    print(f"📸 Processing {len(image_urls)} image(s)")

    # Handle JSON string input
    if image_urls and isinstance(image_urls, str):
        image_urls = json.loads(image_urls)
    
    if not image_urls:
        return "❌ No input image URLs provided."

    try:
        # Run pest detection on all images
        annotated_images = _detect_pests(model, image_urls)
        print(f"✅ Successfully processed {len(annotated_images)} image(s)")
        
        # Convert images to base64-encoded blobs
        blob_results = []
        for image in annotated_images:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Encode to base64 string
            base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
            blob_results.append(base64_encoded)

        return blob_results
        
    except Exception as e:
        err = f"❌ Error in pest_detection.predict(): {e}\nTraceback:\n{traceback.format_exc()}"
        print(err)
        return err


def _detect_pests(model, image_paths: List[str]) -> List[Image.Image]:
    """
    Internal function to run pest detection on images.
    
    Uses the YOLOv8n model to detect and classify pests in the provided images.
    Each image is annotated with bounding boxes around detected pests and class labels.
    
    Args:
        model: YOLO model instance
        image_paths: List of image URLs or local file paths
    
    Returns:
        List[Image.Image]: List of PIL Images with pest detection annotations
    
    The YOLO model can detect various pest species including:
    - Japanese beetles
    - Whiteflies
    - Aphids
    - And other common agricultural pests
    """
    # Run YOLO inference on all images
    results = model(image_paths)
    
    annotated_images = []
    
    # Extract and process results for each image
    # The results object contains detection data for each input image
    for result_set in results:
        for r in result_set:
            # Plot generates a BGR numpy array with bounding boxes and labels
            im_array = r.plot()
            
            # Convert BGR to RGB and create PIL Image
            im = Image.fromarray(im_array[..., ::-1])
            annotated_images.append(im)
    
    return annotated_images


# For local testing
if __name__ == "__main__":
    print("""
    Pest Detection Tool for CropWizard
    ===================================
    
    To deploy this tool:
        beam deploy pest_detection.py
    
    To test locally:
        beam serve pest_detection.py
    
    Then send POST requests to the endpoint with:
        {
            "image_urls": [
                "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg"
            ]
        }
    """)
