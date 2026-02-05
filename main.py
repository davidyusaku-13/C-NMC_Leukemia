"""
FastAPI Server for Leukemia Cell Classification
Serves MobileNetV3-Large model trained on C-NMC dataset
"""

import io
from typing import Dict

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms

from model import create_model

# Initialize FastAPI app
app = FastAPI(
    title="Leukemia Classification API",
    description="Classify leukemia cells (ALL vs HEM) from microscopy images",
    version="1.0.0",
)

# Add CORS middleware to allow requests from mobile devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None
transform = None
class_names = {0: "HEM (Normal)", 1: "ALL (Leukemia)"}


def load_model():
    """Load the trained MobileNetV3-Large model"""
    global model, device, transform

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model architecture using shared create_model function
    model = create_model(device)

    # Load trained weights
    try:
        model.load_state_dict(
            torch.load("best_leukemia_model_weights.pth", map_location=device)
        )
        model.eval()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        raise RuntimeError(
            "Model weights file 'best_leukemia_model_weights.pth' not found!"
        )

    # Define the same transforms used during validation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Leukemia Classification API",
        "status": "running",
        "model": "MobileNetV3-Large",
        "classes": class_names,
        "endpoints": {
            "/mobile": "GET - Mobile web interface",
            "/predict": "POST - Upload image for classification",
            "/health": "GET - Check API health",
        },
    }


@app.get("/mobile")
async def mobile_interface():
    """Serve the mobile web interface"""
    return FileResponse("mobile_interface.html", media_type="text/html")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Predict leukemia classification from uploaded image

    Args:
        file: Image file (BMP, PNG, JPG supported)

    Returns:
        Dict with prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail=f"File must be an image, got {file.content_type}"
        )

    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Prepare response
        result = {
            "prediction": class_names[predicted_class],
            "class_id": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "HEM (Normal)": round(probabilities[0][0].item() * 100, 2),
                "ALL (Leukemia)": round(probabilities[0][1].item() * 100, 2),
            },
            "filename": file.filename,
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Accessible from network
        port=8000,
        reload=True,  # Auto-reload on code changes
    )
