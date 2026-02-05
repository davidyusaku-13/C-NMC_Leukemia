"""
Training Script for Leukemia Classification

Executes the full training pipeline:
1. Load and split data
2. Create DataLoaders
3. Build model
4. Train with dual-phase monitoring
5. Save best weights
6. Plot training history

Usage:
    python run_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import (
    LeukemiaDataset,
    create_splits,
    get_transforms,
    load_all_training_data,
)
from model import create_model
from training import plot_history, train_model

# ============ Configuration ============
IMG_SIZE = (224, 224)
BATCH_SIZE = 40
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
WEIGHTS_SAVE_PATH = "best_leukemia_model_weights.pth"

# ============ Main Execution ============
if __name__ == "__main__":
    print("=" * 60)
    print("Leukemia Classification Training")
    print("Model: MobileNetV3-Large")
    print("=" * 60)

    # 1. Setup Device
    device = (
        torch.device("xpu")
        if torch.xpu.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    if torch.xpu.is_available():
        print(f"GPU Name: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print()

    # 2. Load Data
    print("Loading data from training_data/...")
    all_paths, all_labels = load_all_training_data()
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        create_splits(all_paths, all_labels)
    )

    # Print dataset statistics
    print("-" * 60)
    print(f"Total Images: {len(all_paths)}")
    print("-" * 60)
    print(
        f"Training Set:   {len(train_paths):5d} images | {train_labels.count(1):4d} ALL (1) | {train_labels.count(0):4d} HEM (0)"
    )
    print(
        f"Validation Set: {len(val_paths):5d} images | {val_labels.count(1):4d} ALL (1) | {val_labels.count(0):4d} HEM (0)"
    )
    print(
        f"Test Set:       {len(test_paths):5d} images | {test_labels.count(1):4d} ALL (1) | {test_labels.count(0):4d} HEM (0)"
    )
    print("-" * 60)
    print()

    # 3. Create DataLoaders
    print("Creating DataLoaders...")
    train_transforms, val_test_transforms = get_transforms()

    train_dataset = LeukemiaDataset(
        train_paths, train_labels, transform=train_transforms
    )
    val_dataset = LeukemiaDataset(val_paths, val_labels, transform=val_test_transforms)
    test_dataset = LeukemiaDataset(
        test_paths, test_labels, transform=val_test_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    print(f"✓ DataLoaders ready (Batch Size: {BATCH_SIZE})")
    print()

    # 4. Create Model
    print("Building model...")
    model = create_model(device)
    print("✓ MobileNetV3-Large created with custom classifier")
    print(f"  Classifier Head:")
    print(f"    - BatchNorm1d")
    print(f"    - Linear(960 → 256)")
    print(f"    - ReLU")
    print(f"    - Dropout(0.45)")
    print(f"    - Linear(256 → 2)")
    print()

    # 5. Setup Training Components
    print("Setting up optimizer and loss function...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    print(f"✓ Optimizer: Adamax (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"✓ Loss: CrossEntropyLoss")
    print()

    # 6. Train Model
    print("=" * 60)
    print("TRAINING START")
    print("=" * 60)
    trained_model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
    )
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print()

    # 7. Plot Training History
    print("Displaying training history...")
    plot_history(history)

    # 8. Final Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Training completed successfully")
    print(f"✓ Best weights saved to: {WEIGHTS_SAVE_PATH}")
    print(f"✓ Total epochs: {len(history['train_loss'])}")
    print(f"✓ Best Training Accuracy: {max(history['train_acc']):.4f}")
    print(f"✓ Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print()
    print("Next steps:")
    print("  1. Run 'python evaluate.py' to evaluate on test set")
    print("  2. Run 'python main.py' to start inference server")
    print("=" * 60)
