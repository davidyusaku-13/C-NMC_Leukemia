"""
Model Architecture for Leukemia Classification

MobileNetV3-Large with custom classifier head:
- Batch Normalization
- Dense layer (256 units)
- ReLU activation
- Dropout (0.45)
- Output layer (2 classes: HEM, ALL)
"""

import torch.nn as nn
from torchvision import models


def create_model(device):
    """
    Creates MobileNetV3-Large model with custom classifier head.

    Args:
        device: torch.device - Device to move model to (cpu/cuda/xpu)

    Returns:
        model: nn.Module - Model ready for training or inference
    """
    # Load pre-trained MobileNetV3-Large
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )

    # Modify classifier head
    # Original classifier[3] is a Linear layer
    num_ftrs = model.classifier[3].in_features

    # Replace with custom head matching notebook architecture
    model.classifier[3] = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.45),
        nn.Linear(256, 2),  # 2 classes: HEM (0), ALL (1)
    )

    # Move to device
    model = model.to(device)

    return model
