"""
Data Loading and Preprocessing for Leukemia Classification

Handles:
- Loading image paths and labels from training_data/
- Creating stratified train/val/test splits (70/15/15)
- Custom PyTorch Dataset class
- Data transforms (augmentation for training, normalization for val/test)
- Visualization helpers
"""

import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


def load_all_training_data(base_dir="training_data") -> Tuple[List[str], List[int]]:
    """
    Loads all image paths and labels from the training directory.

    Args:
        base_dir: str - Path to training data directory

    Returns:
        image_paths: List[str] - Absolute paths to all images
        labels: List[int] - Corresponding labels (0=HEM, 1=ALL)
    """
    image_paths = []
    labels = []

    # Class mapping: ALL (Leukemia) = 1, HEM (Normal) = 0
    classes = {"all": 1, "hem": 0}

    base_path = Path(base_dir)
    for class_name, label in classes.items():
        class_dir = base_path / class_name
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} not found.")
            continue

        # Find all .bmp files in class directory
        for img_file in class_dir.glob("*.bmp"):
            image_paths.append(str(img_file.resolve()))
            labels.append(label)

    return image_paths, labels


def create_splits(
    image_paths: List[str], labels: List[int]
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Creates stratified train/val/test splits (70%/15%/15%).

    Args:
        image_paths: List[str] - All image paths
        labels: List[int] - All labels

    Returns:
        train_data: Tuple[List[str], List[int]] - Training paths and labels
        val_data: Tuple[List[str], List[int]] - Validation paths and labels
        test_data: Tuple[List[str], List[int]] - Test paths and labels
    """
    # First split: 70% train, 30% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=123
    )

    # Second split: Split temp into 50/50 -> 15% val, 15% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=123
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and validation/test transforms.

    Returns:
        train_transforms: Augmentation + normalization for training
        val_test_transforms: Only resize + normalization for val/test
    """
    IMG_SIZE = (224, 224)

    # Training: Augmentation to prevent overfitting
    train_transforms = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Validation/Test: No augmentation
    val_test_transforms = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, val_test_transforms


class LeukemiaDataset(Dataset):
    """
    Custom PyTorch Dataset for Leukemia Cell Images.

    Args:
        image_paths: List[str] - Paths to images
        labels: List[int] - Corresponding labels
        transform: Optional[transforms.Compose] - Transforms to apply
    """

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def display_balanced_samples(
    paths: List[str], labels: List[int], title: str, num_per_class=2
):
    """
    Displays balanced random samples from each class.

    Args:
        paths: List[str] - Image paths
        labels: List[int] - Image labels
        title: str - Plot title
        num_per_class: int - Number of samples per class to display
    """
    # Group indices by label
    label_groups = {}
    for i, label in enumerate(labels):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(i)

    selected_indices = []
    # Sort labels to ensure consistent order (0=HEM, 1=ALL)
    for label in sorted(label_groups.keys()):
        indices = label_groups[label]
        selected_indices.extend(
            random.sample(indices, min(len(indices), num_per_class))
        )

    if not selected_indices:
        print(f"No samples found to display for {title}.")
        return

    plt.figure(figsize=(16, 5))
    for i, idx in enumerate(selected_indices):
        plt.subplot(1, len(selected_indices), i + 1)
        img = Image.open(paths[idx])
        plt.imshow(img)

        # Get relative filename for display
        filename = Path(paths[idx]).name
        label_text = "ALL (1)" if labels[idx] == 1 else "HEM (0)"

        plt.title(f"{label_text}\n{filename}", fontsize=8)
        plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def imshow(img, title=None):
    """
    Helper function to display a preprocessed tensor image.

    Args:
        img: torch.Tensor - Image tensor (C, H, W)
        title: Optional[str] - Title for the plot
    """
    img = img.numpy().transpose((1, 2, 0))

    # Un-normalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
