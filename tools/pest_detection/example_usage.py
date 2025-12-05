"""
Example usage of the Pest Detection Tool

This script demonstrates how to use the pest detection tool in different scenarios.
"""

import requests
import base64
from PIL import Image
from io import BytesIO
import os


class PestDetectionClient:
    """
    Client for interacting with the Pest Detection Tool endpoint.
    """
    
    def __init__(self, endpoint_url):
        """
        Initialize the client with an endpoint URL.
        
        Args:
            endpoint_url: URL of the deployed Beam endpoint
        """
        self.endpoint_url = endpoint_url
    
    def detect_pests(self, image_urls, save_output=False, output_dir="output"):
        """
        Send images to the pest detection endpoint and get annotated results.
        
        Args:
            image_urls: List of image URLs to analyze
            save_output: Whether to save the annotated images locally
            output_dir: Directory to save output images (if save_output=True)
        
        Returns:
            List of PIL Image objects with pest detection annotations
        """
        payload = {
            "image_urls": image_urls
        }
        
        print(f"🔍 Analyzing {len(image_urls)} image(s) for pests...")
        
        try:
            response = requests.post(self.endpoint_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                base64_images = response.json()
                
                if isinstance(base64_images, str):
                    # Error message
                    print(f"❌ Error from API: {base64_images}")
                    return []
                
                print(f"✅ Successfully processed {len(base64_images)} image(s)")
                
                # Decode base64 images
                images = []
                for i, base64_img in enumerate(base64_images):
                    img_data = base64.b64decode(base64_img)
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                    
                    if save_output:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"pest_detection_{i+1}.png")
                        img.save(output_path)
                        print(f"💾 Saved annotated image to: {output_path}")
                
                return images
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return []


def example_single_image():
    """
    Example: Analyze a single image for pests.
    """
    print("\n" + "="*60)
    print("Example 1: Single Image Analysis")
    print("="*60)
    
    # Replace with your deployed endpoint URL
    endpoint = "http://localhost:8000/"  # Local testing endpoint
    
    client = PestDetectionClient(endpoint)
    
    # Analyze a single image
    image_url = "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg"
    
    images = client.detect_pests([image_url], save_output=True)
    
    if images:
        print(f"\n📊 Results:")
        print(f"   - Processed {len(images)} image")
        print(f"   - Output saved to: output/pest_detection_1.png")


def example_multiple_images():
    """
    Example: Analyze multiple images in a batch.
    """
    print("\n" + "="*60)
    print("Example 2: Batch Image Analysis")
    print("="*60)
    
    endpoint = "http://localhost:8000/"
    client = PestDetectionClient(endpoint)
    
    # Multiple test images
    image_urls = [
        "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg",
        "https://www.arborday.org/trees/health/pests/images/figure-whiteflies-1.jpg",
    ]
    
    images = client.detect_pests(image_urls, save_output=True)
    
    if images:
        print(f"\n📊 Results:")
        print(f"   - Processed {len(images)} images")
        for i in range(len(images)):
            print(f"   - Image {i+1}: {images[i].size} pixels")


def example_field_monitoring():
    """
    Example: Simulating field monitoring with periodic pest detection.
    """
    print("\n" + "="*60)
    print("Example 3: Field Monitoring Simulation")
    print("="*60)
    
    endpoint = "http://localhost:8000/"
    client = PestDetectionClient(endpoint)
    
    # Simulate field camera images
    field_images = {
        "North Field": "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg",
        "South Field": "https://www.arborday.org/trees/health/pests/images/figure-whiteflies-1.jpg",
    }
    
    print("\n🌾 Monitoring field images for pest activity...\n")
    
    for field_name, image_url in field_images.items():
        print(f"📍 {field_name}:")
        images = client.detect_pests([image_url])
        
        if images:
            print(f"   ✅ Analysis complete")
        else:
            print(f"   ❌ Analysis failed")
    
    print("\n✅ Field monitoring complete")


def example_integration_workflow():
    """
    Example: Complete workflow integrating pest detection with other CropWizard tools.
    """
    print("\n" + "="*60)
    print("Example 4: Integrated Workflow")
    print("="*60)
    
    print("""
    Complete CropWizard Workflow:
    
    1. 📸 Capture field images (from camera or upload)
    2. 🔍 Pest Detection (this tool)
    3. 🌤️  Weather Analysis (weather_agent tool)
    4. 📊 Geographic Analysis (csv_geo_agent tool)
    5. 🔎 Web Search for pest control (google_web_search tool)
    6. 📝 Generate recommendations
    
    This integrated approach provides comprehensive crop management insights.
    """)
    
    endpoint = "http://localhost:8000/"
    client = PestDetectionClient(endpoint)
    
    # Step 1: Pest Detection
    print("Step 1: Detecting pests in field images...")
    image_url = "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg"
    images = client.detect_pests([image_url], save_output=True)
    
    if images:
        print("   ✅ Pests detected: Japanese beetle")
        
        # Step 2: Could trigger weather analysis
        print("\nStep 2: Analyzing weather conditions...")
        print("   (Would call weather_agent here)")
        
        # Step 3: Could search for pest control methods
        print("\nStep 3: Searching for control methods...")
        print("   (Would call google_web_search here)")
        
        # Step 4: Generate recommendations
        print("\nStep 4: Generating recommendations...")
        print("   ✅ Recommendation: Apply pest control treatment")
        print("   ✅ Optimal spray time: Early morning or late evening")
        print("   ✅ Consider integrated pest management strategies")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║   Pest Detection Tool - Example Usage Scripts    ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    print("\nNote: Make sure to update the endpoint URL with your deployed Beam endpoint.")
    print("For local testing, run: beam serve pest_detection.py")
    print()
    
    # Run examples
    try:
        example_single_image()
        example_multiple_images()
        example_field_monitoring()
        example_integration_workflow()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
