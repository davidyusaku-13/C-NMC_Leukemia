"""
Test script for Leukemia Classification API
Tests all endpoints with sample images
"""

import requests
import json
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8000"


def test_root_endpoint():
    """Test the root endpoint"""
    print("=" * 60)
    print("Testing Root Endpoint (GET /)")
    print("=" * 60)

    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_health_endpoint():
    """Test the health check endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint (GET /health)")
    print("=" * 60)

    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_predict_endpoint(image_path: str):
    """Test prediction endpoint with an image"""
    print("=" * 60)
    print(f"Testing Prediction Endpoint (POST /predict)")
    print(f"Image: {image_path}")
    print("=" * 60)

    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        print("Please provide a valid image path from your dataset")
        return

    # Open and send image
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/bmp")}
        response = requests.post(f"{API_URL}/predict", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction Results:")
        print(f"  - Filename: {result['filename']}")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Confidence: {result['confidence']}%")
        print(f"  - Probabilities:")
        for class_name, prob in result["probabilities"].items():
            print(f"      • {class_name}: {prob}%")
    else:
        print(f"❌ Error: {response.text}")
    print()


def run_all_tests():
    """Run all API tests"""
    print("\n" + "🔬" * 30)
    print("LEUKEMIA CLASSIFICATION API - TEST SUITE")
    print("🔬" * 30 + "\n")

    try:
        # Basic endpoint tests
        test_root_endpoint()
        test_health_endpoint()

        # Prediction tests
        print("Testing Prediction Endpoint with Sample Images")
        print("-" * 60)

        # Test with sample images from training data
        all_sample = "training_data/all/UID_33_1_1_all.bmp"
        hem_sample = "training_data/hem/UID_H10_1_1_hem.bmp"

        print("Test 1: Leukemia Cell (ALL)")
        test_predict_endpoint(all_sample)

        print("Test 2: Normal Cell (HEM)")
        test_predict_endpoint(hem_sample)

        print("=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("Make sure the server is running:")
        print("  python main.py")
        print()


if __name__ == "__main__":
    run_all_tests()
