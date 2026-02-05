"""
Evaluation Script for Leukemia Classification

Loads trained model and evaluates on validation and test sets.
Displays:
- Classification reports (precision, recall, f1-score)
- Confusion matrices

Usage:
    python evaluate.py
"""

import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import (
    LeukemiaDataset,
    create_splits,
    get_transforms,
    load_all_training_data,
)
from model import create_model

# ============ Configuration ============
BATCH_SIZE = 40


def load_trained_model(weights_path: str, device):
    """
    Load trained model with saved weights.

    Args:
        weights_path: str - Path to saved weights file
        device: torch.device - Device to load model on

    Returns:
        model: nn.Module - Loaded model in eval mode
    """
    print(f"Loading model from: {weights_path}")
    model = create_model(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    return model


def evaluate_model(model, dataloader, device, title="Evaluation"):
    """
    Evaluate model on a dataset and display results.

    Args:
        model: nn.Module - Model to evaluate
        dataloader: DataLoader - Data to evaluate on
        device: torch.device - Device to use
        title: str - Title for the evaluation (e.g., "Validation Set")
    """
    model.eval()
    all_preds = []
    all_labels = []

    print(f"\nEvaluating on {title}...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=title):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print(f"\nClassification Report ({title}):")
    print(
        classification_report(
            all_labels, all_preds, target_names=["HEM (0)", "ALL (1)"]
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["HEM (0)", "ALL (1)"],
        yticklabels=["HEM (0)", "ALL (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({title})")
    plt.tight_layout()
    plt.show()


def find_latest_weights() -> str:
    """
    Find the most recently modified weights file.

    Returns:
        str - Path to the most recent weights file
    """
    # Look for common weight file patterns
    weight_patterns = ["best_leukemia_model_weights.pth", "*leukemia*.pth", "*.pth"]

    for pattern in weight_patterns:
        files = glob.glob(pattern)
        if files:
            # Return the most recently modified file
            latest = max(files, key=os.path.getmtime)
            return latest

    raise FileNotFoundError(
        "No model weights file found! Please train the model first."
    )


# ============ Main Execution ============
if __name__ == "__main__":
    print("=" * 60)
    print("Leukemia Classification Evaluation")
    print("=" * 60)

    # 1. Setup Device
    device = (
        torch.device("xpu")
        if torch.xpu.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    print()

    # 2. Find and Load Model Weights
    try:
        weights_path = find_latest_weights()
        print(f"Found weights: {weights_path}")
        model = load_trained_model(weights_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python run_training.py' first to train the model.")
        exit(1)
    print()

    # 3. Load Data
    print("Loading data...")
    all_paths, all_labels = load_all_training_data()
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        create_splits(all_paths, all_labels)
    )

    _, val_test_transforms = get_transforms()

    val_dataset = LeukemiaDataset(val_paths, val_labels, transform=val_test_transforms)
    test_dataset = LeukemiaDataset(
        test_paths, test_labels, transform=val_test_transforms
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    print(f"✓ Loaded {len(val_dataset)} validation samples")
    print(f"✓ Loaded {len(test_dataset)} test samples")

    # 4. Evaluate on Validation Set
    print("\n" + "=" * 60)
    print("VALIDATION SET EVALUATION")
    print("=" * 60)
    evaluate_model(model, val_loader, device, title="Validation Set")

    # 5. Evaluate on Test Set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    evaluate_model(model, test_loader, device, title="Test Set")

    # 6. Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("✓ Confusion matrices displayed")
    print("✓ Classification reports printed")
    print("=" * 60)
