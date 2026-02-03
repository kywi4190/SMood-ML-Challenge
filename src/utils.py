"""
Utility functions for the Beetles Drought Prediction project.

This module contains helper functions used across the project,
including device selection, directory creation, and data formatting.

For beginners: Utility functions are "helper" code that gets reused
in multiple places. Putting them in one file keeps code organized.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np


def ensure_dir(directory):
    """
    Create a directory if it doesn't exist.

    Args:
        directory: Path to the directory (str or Path object)

    Returns:
        Path: The directory path as a Path object

    Example:
        ensure_dir("checkpoints")  # Creates 'checkpoints' folder if needed
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp():
    """
    Get current timestamp as a string for naming experiments.

    Returns:
        str: Timestamp in format "YYYYMMDD_HHMMSS"

    Example:
        >>> get_timestamp()
        '20250815_143022'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds):
    """
    Convert seconds to a human-readable time string.

    Args:
        seconds: Number of seconds (float or int)

    Returns:
        str: Formatted time string like "2h 15m 30s"

    Example:
        >>> format_time(3661)
        '1h 1m 1s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model (nn.Module)

    Returns:
        int: Total number of trainable parameters

    For beginners: Parameters are the "learnable weights" of a neural network.
    More parameters = more capacity to learn, but also more risk of overfitting.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, name="Model"):
    """
    Print a summary of the model architecture and parameters.

    Args:
        model: PyTorch model (nn.Module)
        name: Name to display for the model

    Example output:
        Model Summary: RegressionHead
        Total trainable parameters: 215,814
    """
    num_params = count_parameters(model)
    print(f"\n{name} Summary")
    print("-" * 40)
    print(f"Total trainable parameters: {num_params:,}")
    print("-" * 40)


def move_to_device(data, device):
    """
    Move data (tensor, dict, or list) to the specified device.

    Args:
        data: Can be a tensor, dict of tensors, or list of tensors
        device: Target device (e.g., torch.device('cuda'))

    Returns:
        Data moved to the specified device

    For beginners: Data needs to be on the same device (CPU or GPU) as the model.
    This helper handles different data structures automatically.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(val, device) for key, val in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def save_checkpoint(model, optimizer, epoch, loss, path, extra_info=None):
    """
    Save a training checkpoint.

    A checkpoint contains everything needed to resume training later:
    - Model weights
    - Optimizer state (learning rates, momentum, etc.)
    - Current epoch
    - Loss value

    Args:
        model: The PyTorch model
        optimizer: The optimizer being used
        epoch: Current epoch number
        loss: Current loss value
        path: Where to save the checkpoint
        extra_info: Optional dict of additional info to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """
    Load a training checkpoint.

    Args:
        model: The PyTorch model to load weights into
        optimizer: The optimizer to restore state to (can be None)
        path: Path to the checkpoint file
        device: Device to load the model to

    Returns:
        dict: The full checkpoint dictionary with epoch, loss, etc.
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return checkpoint


def format_predictions(mu_values, sigma_values, target_names=None):
    """
    Format model outputs into the competition submission format.

    The competition expects predictions in this structure:
    {
        "SPEI_30d": {"mu": float, "sigma": float},
        "SPEI_1y": {"mu": float, "sigma": float},
        "SPEI_2y": {"mu": float, "sigma": float},
    }

    Args:
        mu_values: Array of mean predictions, shape (3,)
        sigma_values: Array of sigma predictions, shape (3,)
        target_names: List of target names (default: SPEI metrics)

    Returns:
        dict: Predictions in competition format
    """
    if target_names is None:
        target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    predictions = {}
    for i, name in enumerate(target_names):
        predictions[name] = {
            "mu": float(mu_values[i]),
            "sigma": float(sigma_values[i])
        }

    return predictions


class AverageMeter:
    """
    Computes and stores the running average of a value.

    This is useful for tracking metrics like loss during training,
    where you want to see the average over many batches.

    Example:
        meter = AverageMeter()
        for batch in dataloader:
            loss = compute_loss(batch)
            meter.update(loss.item())
        print(f"Average loss: {meter.avg:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0      # Most recent value
        self.avg = 0      # Running average
        self.sum = 0      # Total sum
        self.count = 0    # Number of updates

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val: The value to add
            n: Number of samples this value represents (for weighted averages)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric (usually validation loss) and stops training
    if it doesn't improve for a certain number of epochs.

    For beginners: Overfitting happens when a model learns the training
    data "too well" and performs poorly on new data. Early stopping
    catches this by watching the validation performance.

    Example:
        early_stop = EarlyStopping(patience=10)
        for epoch in range(100):
            val_loss = validate(model)
            if early_stop(val_loss):
                print("Early stopping triggered!")
                break
    """

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' if lower is better, 'max' if higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        """
        Check if training should stop.

        Args:
            score: Current metric value (e.g., validation loss)

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First call - initialize best score
            self.best_score = score
            return False

        # Check if score improved
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            # Score improved - reset counter and update best
            self.best_score = score
            self.counter = 0
        else:
            # No improvement - increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
