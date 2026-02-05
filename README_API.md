# 🔬 Leukemia Classification API

FastAPI server for classifying leukemia cells (ALL vs HEM) from microscopy images using a trained MobileNetV3-Large model.

## 📋 Overview

This API serves a deep learning model trained on the C-NMC (Cancer-NMC) Leukemia dataset. It can classify microscopy images of blood cells as either:

- **ALL (Acute Lymphoblastic Leukemia)** - Cancerous cells
- **HEM (Normal/Healthy)** - Normal blood cells

## 🚀 Quick Start

### Option 1: Run Locally

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Start the server:**

```bash
python main.py
```

3. **Access the API:**
   - API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

### Option 2: Run with Docker

1. **Build and run with Docker Compose:**

```bash
docker-compose up --build
```

2. **Access the API at http://localhost:8000**

3. **Stop the server:**

```bash
docker-compose down
```

## 📡 API Endpoints

### 1. Root Endpoint

```http
GET /
```

Returns API information and available endpoints.

**Response:**

```json
{
  "message": "Leukemia Classification API",
  "status": "running",
  "model": "MobileNetV3-Large",
  "classes": {
    "0": "HEM (Normal)",
    "1": "ALL (Leukemia)"
  },
  "endpoints": {
    "/predict": "POST - Upload image for classification",
    "/health": "GET - Check API health"
  }
}
```

### 2. Health Check

```http
GET /health
```

Check if the API and model are working properly.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 3. Predict Classification

```http
POST /predict
```

Upload an image to get classification results.

**Request:**

- Content-Type: `multipart/form-data`
- Body: Image file (BMP, PNG, JPG supported)

**Response:**

```json
{
  "prediction": "ALL (Leukemia)",
  "class_id": 1,
  "confidence": 98.45,
  "probabilities": {
    "HEM (Normal)": 1.55,
    "ALL (Leukemia)": 98.45
  },
  "filename": "sample_cell.bmp"
}
```

## 📱 Using from Your Phone

### Method 1: Mobile App (Recommended)

Create a simple mobile app or use API testing apps like:

- **Postman** (iOS/Android)
- **HTTP Request Manager** (Android)
- Build a custom React Native/Flutter app

### Method 2: Web Interface

Access the interactive API documentation from your phone's browser:

```
http://YOUR_COMPUTER_IP:8000/docs
```

To find your computer's IP:

- **Windows:** `ipconfig` (look for IPv4 Address)
- **Mac/Linux:** `ifconfig` or `ip addr`

### Method 3: Python Client (from Mobile Terminus/Terminal)

```python
import requests

# Upload image for prediction
with open("cell_image.bmp", "rb") as f:
    files = {"file": f}
    response = requests.post("http://YOUR_IP:8000/predict", files=files)
    print(response.json())
```

### Method 4: cURL Command

```bash
curl -X POST "http://YOUR_IP:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@cell_image.bmp"
```

## 🧪 Testing the API

Run the included test script:

```bash
python test_api.py
```

This will test all endpoints with sample images from your dataset.

## 🔧 Configuration

### Change Port

Edit `main.py` or run with custom port:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Enable HTTPS

For production, use a reverse proxy like nginx with SSL certificates.

### CORS Settings

The API allows all origins by default. For production, edit `main.py`:

```python
allow_origins=["https://your-mobile-app.com"]
```

## 📊 Model Details

- **Architecture:** MobileNetV3-Large
- **Input Size:** 224x224 pixels
- **Training Dataset:** C-NMC Leukemia
- **Classes:** 2 (ALL, HEM)
- **Weights File:** `best_leukemia_model_weights.pth`

## 🛡️ Security Considerations

**For Production Deployment:**

1. ✅ Add authentication (API keys, OAuth)
2. ✅ Implement rate limiting
3. ✅ Use HTTPS/SSL certificates
4. ✅ Restrict CORS to specific domains
5. ✅ Add input validation and file size limits
6. ✅ Use environment variables for sensitive data
7. ✅ Deploy behind a reverse proxy (nginx, Caddy)

## 📦 Example Mobile App Integration

### React Native Example

```javascript
const uploadImage = async (imageUri) => {
  const formData = new FormData();
  formData.append("file", {
    uri: imageUri,
    type: "image/jpeg",
    name: "cell_image.jpg",
  });

  try {
    const response = await fetch("http://YOUR_IP:8000/predict", {
      method: "POST",
      body: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    const result = await response.json();
    console.log("Prediction:", result.prediction);
    console.log("Confidence:", result.confidence);
  } catch (error) {
    console.error("Error:", error);
  }
};
```

### Flutter Example

```dart
import 'package:http/http.dart' as http;
import 'dart:io';

Future<void> uploadImage(File image) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://YOUR_IP:8000/predict')
  );

  request.files.add(
    await http.MultipartFile.fromPath('file', image.path)
  );

  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  print(responseData);
}
```

## 🐛 Troubleshooting

### Model Not Loading

- Ensure `best_leukemia_model_weights.pth` exists in the same directory
- Check file permissions

### Connection Refused from Phone

- Verify your computer's firewall allows port 8000
- Ensure your phone and computer are on the same network
- Use your computer's local IP, not `localhost`

### Slow Predictions

- Consider deploying on a server with GPU support
- Reduce image size before uploading
- Use model quantization for faster inference

## 📝 License

This API serves a model trained on the C-NMC dataset. Please cite the original dataset if used for research.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!

## 📞 Support

For issues or questions, please create an issue in the repository.

---

**Made with ❤️ using FastAPI and PyTorch**
