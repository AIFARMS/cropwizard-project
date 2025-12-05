# Pest Detection Tool

A computer vision-based tool for detecting and classifying agricultural pests in crop images using a custom-trained YOLOv8n deep learning model.

## Overview

This tool provides automated pest detection capabilities for the CropWizard platform. It accepts images of crops and returns annotated images with bounding boxes around detected pests, along with classification labels.

## Features

- **Multi-pest Detection**: Detects various common agricultural pests including:
  - Japanese beetles
  - Whiteflies
  - Aphids
  - And other pest species

- **Visual Annotations**: Returns images with:
  - Bounding boxes around detected pests
  - Classification labels
  - Confidence scores

- **Scalable Deployment**: Built on Beam.cloud for serverless deployment with:
  - Auto-scaling based on demand
  - GPU/CPU support
  - Model caching for fast cold starts

## Model Information

- **Architecture**: YOLOv8n (nano variant for efficiency)
- **Training**: Custom-trained on agricultural pest dataset
- **Model Author**: Aditya Sengupta
- **Training Notebook**: [Google Colab](https://colab.research.google.com/drive/1GO-lw2PJtVewlA-xhBfgBLId8v4v-BE2?usp=sharing)

## Installation

### Requirements

```bash
torch==2.2.0
ultralytics==8.1.0
torchvision==0.17.0
opencv-python
pillow
```

### Deployment

Deploy to Beam.cloud:

```bash
beam deploy pest_detection.py
```

For local testing:

```bash
beam serve pest_detection.py
```

## Usage

### API Endpoint

Send a POST request to the deployed Beam endpoint with the following JSON payload:

```json
{
    "image_urls": [
        "https://example.com/crop-image-1.jpg",
        "https://example.com/crop-image-2.jpg"
    ]
}
```

### Response Format

The endpoint returns a list of base64-encoded PNG images:

```json
[
    "iVBORw0KGgoAAAANSUhEUgAA...",
    "iVBORw0KGgoAAAANSUhEUgAA..."
]
```

### Example Usage

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# API endpoint (replace with your deployed endpoint)
endpoint = "https://your-beam-endpoint.run/"

# Prepare request
payload = {
    "image_urls": [
        "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg"
    ]
}

# Send request
response = requests.post(endpoint, json=payload)
base64_images = response.json()

# Decode and display first image
img_data = base64.b64decode(base64_images[0])
img = Image.open(BytesIO(img_data))
img.show()
```

## Technical Details

### Model Loading

The model is automatically downloaded on first deployment and cached in a Beam volume for subsequent fast cold starts. The model weights are hosted at:
```
https://assets.kastan.ai/pest_detection_model_weights.pt
```

### Resource Configuration

- **CPU**: 1 core
- **Memory**: 3Gi RAM
- **Workers**: 1
- **Max Pending Tasks**: 10
- **Timeout**: 60 seconds
- **Keep Warm**: 3 minutes

### Processing Flow

1. **Input Validation**: Accepts list of image URLs
2. **Model Inference**: Runs YOLOv8n detection on each image
3. **Annotation**: Draws bounding boxes and labels on detected pests
4. **Encoding**: Converts annotated images to base64-encoded PNG
5. **Response**: Returns list of encoded images

## Integration with CropWizard

This tool can be integrated into the CropWizard platform to provide:

- Real-time pest detection in uploaded crop images
- Automated pest monitoring from field cameras
- Historical pest tracking and analysis
- Integration with other agricultural tools (weather, soil analysis, etc.)

## Related Pull Requests

This implementation is based on work from the ai-ta-backend repository:

- [PR #210](https://github.com/Center-for-AI-Innovation/ai-ta-backend/pull/210) - Initial pest detection plugin
- [PR #211](https://github.com/Center-for-AI-Innovation/ai-ta-backend/pull/211) - Add pest detection plugin
- [PR #216](https://github.com/Center-for-AI-Innovation/ai-ta-backend/pull/216) - Implement pest detection using custom YOLOv8n model
- [PR #279](https://github.com/Center-for-AI-Innovation/ai-ta-backend/pull/279) - Refactor to return image data as blobs

## Error Handling

The tool includes comprehensive error handling:

- Invalid image URLs
- Network timeouts
- Model inference failures
- Memory constraints

All errors are logged with detailed stack traces for debugging.

## Performance

- **Cold Start**: ~5-10 seconds (first request after deployment)
- **Warm Start**: <1 second (subsequent requests within keep-warm window)
- **Processing Time**: ~1-3 seconds per image (depending on image size and pest count)

## Future Enhancements

Potential improvements for future versions:

- Support for additional pest species
- Confidence threshold filtering
- Batch processing optimization
- GPU support for faster inference
- Disease detection capabilities
- Integration with pest control recommendations

## License

See the main repository LICENSE file for details.

## Credits

- **Model Training**: Aditya Sengupta (@adityasngpta)
- **Implementation**: Based on ai-ta-backend pest detection work
- **YOLOv8**: Ultralytics YOLO framework
