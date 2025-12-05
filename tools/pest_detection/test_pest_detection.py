"""
Test script for the Pest Detection Tool

This script provides basic testing functionality for the pest detection endpoint.
"""

import requests
import base64
from PIL import Image
from io import BytesIO
import json


def test_pest_detection_local():
    """
    Test the pest detection tool with sample images.
    This assumes the endpoint is running locally via 'beam serve pest_detection.py'
    """
    # Sample test images (Japanese beetle and whiteflies)
    test_images = [
        "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg",
        "https://www.arborday.org/trees/health/pests/images/figure-whiteflies-1.jpg"
    ]
    
    # Local Beam endpoint (adjust port if needed)
    endpoint = "http://localhost:8000/"
    
    print("🧪 Testing Pest Detection Tool")
    print("=" * 50)
    print(f"Test images: {len(test_images)}")
    print(f"Endpoint: {endpoint}")
    print("=" * 50)
    
    payload = {
        "image_urls": test_images
    }
    
    try:
        print("\n📤 Sending request to endpoint...")
        response = requests.post(endpoint, json=payload, timeout=120)
        
        if response.status_code == 200:
            print("✅ Request successful!")
            
            result = response.json()
            
            if isinstance(result, list):
                print(f"\n📊 Received {len(result)} annotated images")
                
                # Optionally save the first image
                if result:
                    print("\n💾 Saving first annotated image...")
                    img_data = base64.b64decode(result[0])
                    img = Image.open(BytesIO(img_data))
                    
                    output_path = "test_output_annotated.png"
                    img.save(output_path)
                    print(f"✅ Saved to: {output_path}")
                    
                    # Display image info
                    print(f"   Image size: {img.size}")
                    print(f"   Image mode: {img.mode}")
            else:
                print(f"❌ Unexpected response format: {type(result)}")
                print(f"Response: {result}")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The model might still be loading.")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the endpoint running?")
        print("   Run: beam serve pest_detection.py")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_pest_detection_deployed(endpoint_url):
    """
    Test the pest detection tool on a deployed Beam endpoint.
    
    Args:
        endpoint_url: The deployed Beam endpoint URL
    """
    test_images = [
        "https://www.arborday.org/trees/health/pests/images/figure-japanese-beetle-3.jpg"
    ]
    
    print("🧪 Testing Deployed Pest Detection Tool")
    print("=" * 50)
    print(f"Endpoint: {endpoint_url}")
    print("=" * 50)
    
    payload = {
        "image_urls": test_images
    }
    
    try:
        print("\n📤 Sending request to deployed endpoint...")
        response = requests.post(endpoint_url, json=payload, timeout=120)
        
        if response.status_code == 200:
            print("✅ Deployment working correctly!")
            result = response.json()
            print(f"📊 Received {len(result)} annotated images")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    print("""
    Pest Detection Tool - Test Script
    ==================================
    
    Options:
    1. Test local endpoint: python test_pest_detection.py local
    2. Test deployed endpoint: python test_pest_detection.py deployed <endpoint_url>
    3. Interactive mode: python test_pest_detection.py
    """)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "local":
            test_pest_detection_local()
        elif sys.argv[1] == "deployed" and len(sys.argv) > 2:
            test_pest_detection_deployed(sys.argv[2])
        else:
            print("Invalid arguments. See usage above.")
    else:
        # Default: test local
        test_pest_detection_local()
