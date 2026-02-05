# Leukemia Classification (C-NMC Dataset)

## 🔬 Overview

This project implements a deep learning model to classify leukemia cells from the **C-NMC (Cancer-NMC) dataset**. The model distinguishes between:
- **ALL (Acute Lymphoblastic Leukemia)** - Cancerous cells (Label 1)
- **HEM (Normal/Healthy)** - Normal cells (Label 0)

The implementation uses **MobileNetV3-Large** with a custom classifier head, achieving ~95% accuracy on validation and test sets.

---

## 📁 File Structure

```
C-NMC_Leukemia/
├── training_data/              # Dataset directory
│   ├── all/                    # Leukemia cell images (*.bmp)
│   └── hem/                    # Normal cell images (*.bmp)
│
├── data.py                     # Data loading, preprocessing, Dataset class
├── model.py                    # Model architecture (MobileNetV3-Large)
├── training.py                 # Training loop, callbacks, history plotting
├── run_training.py             # Main training script (entry point)
├── evaluate.py                 # Evaluation script (metrics & confusion matrices)
│
├── main.py                     # FastAPI inference server
├── test_api.py                 # API testing utilities
├── mobile_interface.html       # Web interface for mobile devices
│
├── leukemia.ipynb              # Original Jupyter notebook
├── best_leukemia_model_weights.pth  # Trained model weights (generated after training)
└── README.md                   # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| **`data.py`** | Data loading from `training_data/`, stratified splitting (70/15/15), `LeukemiaDataset` class, transforms, visualization helpers |
| **`model.py`** | Creates MobileNetV3-Large with custom classifier head (BN → Dense(256) → ReLU → Dropout(0.45) → Dense(2)) |
| **`training.py`** | `CustomCallback` class for dual-phase monitoring (accuracy → val_loss), LR scheduling, early stopping, `train_model()` function |
| **`run_training.py`** | **Training entry point** - Orchestrates data loading, model creation, training, and history plotting |
| **`evaluate.py`** | **Evaluation script** - Loads trained model, evaluates on validation/test sets, displays confusion matrices |
| **`main.py`** | FastAPI server for inference (serves predictions via REST API) |

---

## 🛠️ Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ (with CUDA/XPU support if available)
- Additional dependencies listed below

### Install Dependencies

```bash
pip install torch torchvision tqdm numpy pandas pillow matplotlib seaborn scikit-learn fastapi uvicorn
```

#### Optional: Intel GPU Support (XPU)

If you have an Intel Arc GPU (like Arc B580):

```bash
pip install intel-extension-for-pytorch
```

---

## 📊 Dataset Structure

The `training_data/` directory must follow this structure:

```
training_data/
├── all/                    # Leukemia cells (ALL)
│   ├── UID_1_1_1_all.bmp
│   ├── UID_1_2_1_all.bmp
│   └── ...
└── hem/                    # Normal cells (HEM)
    ├── UID_H1_1_1_hem.bmp
    ├── UID_H1_2_1_hem.bmp
    └── ...
```

**Key Points:**
- Images must be in `.bmp` format (or modify `glob("*.bmp")` in `data.py`)
- The directory names `all/` and `hem/` are case-sensitive
- Total expected images: ~10,661 (7,272 ALL, 3,389 HEM)

**Dataset Split:**
- **Training:** 70% (~7,462 images)
- **Validation:** 15% (~1,599 images)
- **Test:** 15% (~1,600 images)

Split is stratified to maintain class balance across all sets.

---

## 🚀 Usage

### 1️⃣ Training the Model

Run the training script:

```bash
python run_training.py
```

**What happens:**
1. Loads data from `training_data/`
2. Creates stratified train/val/test splits
3. Builds MobileNetV3-Large with custom classifier
4. Trains for up to 40 epochs with:
   - Dual-phase monitoring (accuracy → val_loss)
   - Dynamic LR reduction (halved when plateau detected)
   - Early stopping (stops after 3 LR reductions)
5. Saves best weights to `best_leukemia_model_weights.pth`
6. Displays training/validation curves

**Expected Duration:** ~40-50 minutes (depends on hardware)

**Console Output Example:**
```
============================================================
Leukemia Classification Training
Model: MobileNetV3-Large
============================================================
Using device: xpu
GPU Name: Intel(R) Arc(TM) B580 Graphics

Loading data from training_data/...
------------------------------------------------------------
Total Images: 10661
------------------------------------------------------------
Training Set:    7462 images | 5090 ALL (1) | 2372 HEM (0)
Validation Set:  1599 images | 1091 ALL (1) |  508 HEM (0)
Test Set:        1600 images | 1091 ALL (1) |  509 HEM (0)
------------------------------------------------------------

...training progress...

✓ Best weights saved to: best_leukemia_model_weights.pth
```

---

### 2️⃣ Evaluating the Model

After training, evaluate on validation and test sets:

```bash
python evaluate.py
```

**What happens:**
1. Automatically finds the most recent `.pth` weights file
2. Loads trained model
3. Evaluates on validation set (displays classification report + confusion matrix)
4. Evaluates on test set (displays classification report + confusion matrix)

**Expected Metrics:**
- **Validation Accuracy:** ~94-95%
- **Test Accuracy:** ~94-95%
- **Precision/Recall:** Balanced for both classes

---

### 3️⃣ Running the Inference Server

Start the FastAPI server for real-time predictions:

```bash
python main.py
```

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /mobile` - Mobile web interface
- `POST /predict` - Upload image for classification

**Example API Usage:**

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/cell_image.bmp"
```

**Response:**
```json
{
  "prediction": "ALL (Leukemia)",
  "class_id": 1,
  "confidence": 96.42,
  "probabilities": {
    "HEM (Normal)": 3.58,
    "ALL (Leukemia)": 96.42
  },
  "filename": "cell_image.bmp"
}
```

**Mobile Interface:**
Visit `http://localhost:8000/mobile` in a browser to use the web-based uploader.

---

## 🏗️ Model Architecture

### Backbone: MobileNetV3-Large
- Pre-trained on ImageNet (1000 classes)
- Efficient for mobile/edge deployment
- Input size: 224×224 RGB images

### Custom Classifier Head
```
Input (960 features from backbone)
    ↓
BatchNorm1d
    ↓
Linear(960 → 256)
    ↓
ReLU
    ↓
Dropout(p=0.45)  # High dropout for better generalization
    ↓
Linear(256 → 2)  # Output: HEM (0) or ALL (1)
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adamax |
| Learning Rate | 0.001 (initial) |
| Weight Decay | 0.001 (L2 regularization) |
| Batch Size | 40 |
| Max Epochs | 40 (early stopping active) |
| Loss Function | CrossEntropyLoss |
| LR Reduction Factor | 0.5 |
| Patience | 1 epoch |
| Early Stopping | After 3 LR reductions |

### Data Augmentation (Training Only)
- Random horizontal flip
- Random vertical flip
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation, hue)

### Dual-Phase Monitoring
1. **Phase 1:** Monitor training accuracy until it reaches 90%
2. **Phase 2:** Switch to monitoring validation loss for generalization

---

## 📈 Performance

### Expected Results

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | 94-95% | 94-95% |
| Precision (HEM) | ~93% | ~93% |
| Precision (ALL) | ~96% | ~96% |
| Recall (HEM) | ~91% | ~92% |
| Recall (ALL) | ~97% | ~97% |
| F1-Score (HEM) | ~92% | ~92% |
| F1-Score (ALL) | ~96% | ~96% |

### Training Time
- **Intel Arc B580:** ~40-50 minutes (40 epochs)
- **NVIDIA RTX 3080:** ~30-40 minutes
- **CPU (16 cores):** ~3-4 hours

---

## 🧪 Testing the API

Use the included test script:

```bash
python test_api.py
```

Or manually test with:

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("training_data/all/UID_1_1_1_all.bmp", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 🔧 Customization

### Modify Hyperparameters

Edit `run_training.py`:

```python
# Configuration section
IMG_SIZE = (224, 224)       # Image size
BATCH_SIZE = 40             # Batch size
NUM_EPOCHS = 40             # Max epochs
LEARNING_RATE = 0.001       # Initial LR
WEIGHT_DECAY = 1e-3         # L2 regularization
```

### Change Data Augmentation

Edit `data.py` → `get_transforms()`:

```python
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    # Add/modify augmentations here
    ...
])
```

### Use Different Model

Edit `model.py` → `create_model()`:

```python
# Replace MobileNetV3 with another backbone
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Modify classifier accordingly
```

---

## 📚 References

- **Dataset:** [C-NMC Challenge Dataset](https://wiki.cancerimagingarchive.net/display/Public/C-NMC)
- **Paper:** Gupta & Gupta (2019), "ALL Challenge Dataset of ISBI 2019"
- **Model:** [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)

---

## 🤝 Contributing

To improve this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is for educational purposes. The C-NMC dataset has its own usage terms - please review before using in production.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 🎯 Quick Start Checklist

- [ ] Install dependencies: `pip install torch torchvision tqdm ...`
- [ ] Verify `training_data/all/` and `training_data/hem/` exist
- [ ] Run training: `python run_training.py`
- [ ] Wait for `best_leukemia_model_weights.pth` to be created
- [ ] Evaluate: `python evaluate.py`
- [ ] (Optional) Start server: `python main.py`

---

**Happy Training! 🚀**
