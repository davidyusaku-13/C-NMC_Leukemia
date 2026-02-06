"""
Training Loop and Callback System for Leukemia Classification

Features:
- CustomCallback: Dual-phase monitoring (accuracy → val_loss)
- Dynamic learning rate reduction
- Early stopping
- Automatic best weight restoration
- Training history visualization
"""

import copy
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class CustomCallback:
    """
    Manages the training lifecycle with dual-phase monitoring.

    Phase 1: Monitor training accuracy until threshold (0.9)
    Phase 2: Monitor validation loss for generalization

    Features:
    - Learning rate reduction on plateau (factor=0.5)
    - Early stopping after multiple LR reductions
    - Automatic best weight saving

    Args:
        model: nn.Module - The model being trained
        optimizer: torch.optim.Optimizer - The optimizer
        patience: int - Epochs to wait before reducing LR
        stop_patience: int - Number of LR reductions before stopping
        threshold: float - Accuracy threshold to switch from accuracy to val_loss monitoring
        factor: float - LR reduction factor
    """

    def __init__(
        self, model, optimizer, patience=1, stop_patience=3, threshold=0.9, factor=0.5
    ):
        self.model = model
        self.optimizer = optimizer
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor

        self.count = 0
        self.stop_count = 0
        self.best_epoch = 1
        self.highest_tracc = 0.0
        self.lowest_vloss = float("inf")
        self.best_weights = copy.deepcopy(model.state_dict())
        self.stop_training = False

    def on_train_begin(self):
        """Initialize training session."""
        print("Starting training...")
        print(
            f"{'Epoch':^8s}{'Train_Loss':^12s}{'Train_Acc':^12s}{'Val_Loss':^12s}{'Val_Acc':^12s}{'LR':^10s}{'Monitor':^12s}{'Duration':^10s}"
        )
        self.start_time = time.time()

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_duration: float,
    ):
        """
        Called at the end of each epoch to update metrics and adjust training.

        Args:
            epoch: int - Current epoch number (0-indexed)
            train_loss: float - Average training loss
            train_acc: float - Training accuracy
            val_loss: float - Validation loss
            val_acc: float - Validation accuracy
            epoch_duration: float - Time taken for epoch (seconds)
        """
        lr = self.optimizer.param_groups[0]["lr"]
        monitor = "val_loss"

        # Dual monitoring logic
        if train_acc < self.threshold:
            # Phase 1: Monitor training accuracy
            monitor = "train_acc"
            if train_acc > self.highest_tracc:
                self.highest_tracc = train_acc
                self.best_weights = copy.deepcopy(self.model.state_dict())
                self.count = 0
                self.stop_count = 0
                self.best_epoch = epoch + 1
            else:
                self.count += 1
        else:
            # Phase 2: Monitor validation loss
            monitor = "val_loss"
            if val_loss < self.lowest_vloss:
                self.lowest_vloss = val_loss
                self.best_weights = copy.deepcopy(self.model.state_dict())
                self.count = 0
                self.stop_count = 0
                self.best_epoch = epoch + 1
            else:
                self.count += 1
                # Still track highest train acc for logging
                if train_acc > self.highest_tracc:
                    self.highest_tracc = train_acc

        # Adjust learning rate if patience exceeded
        if self.count >= self.patience:
            next_lr = lr * self.factor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = next_lr
            self.count = 0
            self.stop_count += 1
            print(f"\n   >> LR reduced to {next_lr:.6f} (Patience met)")

        # Print epoch metrics
        print(
            f"{epoch + 1:^8d}{train_loss:^12.4f}{train_acc:^12.4f}{val_loss:^12.4f}{val_acc:^12.4f}{lr:^10.5f}{monitor:^12s}{epoch_duration:^10.2f}"
        )

        # Check early stopping condition
        if self.stop_count > self.stop_patience - 1:
            print(
                f"\n   >> Stopping training: {self.stop_patience} LR reductions with no improvement."
            )
            self.stop_training = True

    def on_train_end(self):
        """Restore best weights and save to disk."""
        print(f"Restoring best weights from epoch {self.best_epoch}...")
        self.model.load_state_dict(self.best_weights)
        # Save to disk
        torch.save(self.model.state_dict(), "best_leukemia_model_weights.pth")
        print("Best model weights saved to 'best_leukemia_model_weights.pth'")


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    val_loader,
    num_epochs=40,
    device=None,
    threshold=0.9,
) -> Tuple[nn.Module, Dict]:
    """
    Main training loop with callback system.

    Args:
        model: nn.Module - Model to train
        criterion: nn.Module - Loss function
        optimizer: torch.optim.Optimizer - Optimizer
        train_loader: DataLoader - Training data
        val_loader: DataLoader - Validation data
        num_epochs: int - Maximum number of epochs
        device: torch.device - Device to train on
        threshold: float - Accuracy threshold before switching to val_loss monitoring

    Returns:
        model: nn.Module - Trained model with best weights
        history: Dict - Training history (losses and accuracies)
    """
    callback = CustomCallback(
        model, optimizer, patience=1, stop_patience=3, threshold=threshold, factor=0.5
    )
    callback.on_train_begin()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ============ Training Phase ============
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # ============ Validation Phase ============
        model.eval()
        val_loss_run = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_run += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss_run / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        # Callback step
        callback.on_epoch_end(
            epoch,
            train_loss,
            train_acc.item(),
            val_loss,
            val_acc.item(),
            time.time() - epoch_start,
        )

        # Save to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.item())

        # Check early stopping
        if callback.stop_training:
            break

    callback.on_train_end()
    return model, history


def plot_history(history: Dict):
    """
    Plots training and validation loss/accuracy curves.

    Args:
        history: Dict - Training history with keys: train_loss, train_acc, val_loss, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Training Acc")
    plt.plot(epochs, history["val_acc"], "r-", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
