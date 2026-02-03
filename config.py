"""
Configuration settings for the Beetles Drought Prediction project.

This file centralizes all hyperparameters and settings in one place,
making it easy to experiment with different configurations.

For beginners: Think of this as a "control panel" where you can
adjust all the knobs for your machine learning experiment.
"""

import torch
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base project directory (automatically set to where this file lives)
PROJECT_DIR = Path(__file__).parent

# Where raw data is stored after downloading from HuggingFace
DATA_DIR = PROJECT_DIR / "data"

# Where pre-computed embeddings are saved (so we don't re-extract them each time)
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"

# Where model checkpoints (saved weights) are stored
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

# Where TensorBoard logs are written for visualizing training progress
LOGS_DIR = PROJECT_DIR / "logs"


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# HuggingFace dataset identifier for the beetle images
# You'll need a HuggingFace account and token to download this
DATASET_NAME = "imageomics/sentinel-beetles"

# The three drought metrics we're predicting
# SPEI = Standardized Precipitation Evapotranspiration Index
# - SPEI_30d: drought conditions over the past 30 days
# - SPEI_1y: drought conditions over the past 1 year
# - SPEI_2y: drought conditions over the past 2 years
TARGET_COLUMNS = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

# Number of target values (mu and sigma for each SPEI metric)
NUM_TARGETS = 3  # We predict 3 SPEI values


# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# Site-based validation: hold out entire sites to simulate test scenario
# These domain IDs will be used ONLY for validation, never for training
# This tests how well the model generalizes to completely unseen locations
# You can change these to experiment with different holdout sites
VALIDATION_DOMAIN_IDS = [32, 99]  # Two sites held out for validation


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Vision backbone model for extracting image features
# DINOv3 is a powerful self-supervised vision model from Meta AI
# We use the base variant (ViT-B/16) for a good balance of quality and speed
BACKBONE_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# Fallback model if DINOv3 has issues
FALLBACK_BACKBONE = "facebook/dinov2-base"

# Dimension of the embedding vector from the backbone
# DINOv3 ViT-B produces 768-dimensional embeddings
EMBEDDING_DIM = 768

# Hidden layer size in our regression head
# The regression head is a simple neural network that takes embeddings
# and outputs predictions. This controls how "wide" that network is.
HIDDEN_DIM = 128

# Dropout probability for regularization
# Dropout randomly sets neurons to 0 during training to prevent overfitting
# Higher values = more regularization (0.0 = no dropout, 0.5 = aggressive)
DROPOUT = 0.1

# Output dimension: 6 values = 3 mu + 3 sigma
# For each SPEI metric, we predict both the mean (mu) and uncertainty (sigma)
OUTPUT_DIM = 6


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Learning rate: how big of steps the optimizer takes during training
# Too high = unstable training, too low = slow training
# 1e-3 (0.001) is a reasonable starting point for Adam optimizer
LEARNING_RATE = 1e-4

# Batch size: how many samples to process at once
# Larger batches = faster training but more memory usage
# 64 is a reasonable default for most GPUs
BATCH_SIZE = 64

# Number of complete passes through the training data
# More epochs = more training, but risk overfitting
# With early stopping, this is really a maximum limit
NUM_EPOCHS = 500

# Early stopping patience: stop training if validation doesn't improve
# for this many consecutive epochs
# This prevents overfitting by stopping when the model stops getting better
EARLY_STOPPING_PATIENCE = 50

# Weight decay (L2 regularization): penalizes large weights
# Helps prevent overfitting by keeping model weights small
WEIGHT_DECAY = 1e-3

# Learning rate scheduler type
# 'cosine': CosineAnnealingLR - smoothly decreases LR over training (recommended)
# 'plateau': ReduceLROnPlateau - halves LR when validation stalls
LR_TYPE = 'cosine'

# Minimum sigma value to prevent numerical instability
# Sigma must be positive; this sets a floor to avoid division by zero
MIN_SIGMA = 1e-3


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device():
    """
    Automatically select the best available compute device.

    Returns:
        torch.device: 'cuda' if GPU available, else 'cpu'

    For beginners: GPUs are MUCH faster for deep learning, but the code
    will work fine on CPU too (just slower for feature extraction).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU detected - training will still work but be slower)")
    return device


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

# Random seed for reproducibility
# Setting this ensures you get the same results each time you run the code
# (useful for debugging and comparing experiments)
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Integer seed value (default: RANDOM_SEED from config)

    For beginners: Neural networks involve randomness (weight initialization,
    data shuffling, etc.). Setting seeds makes experiments reproducible.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to {seed}")


# =============================================================================
# PRINT CONFIGURATION SUMMARY
# =============================================================================

def print_config():
    """Print a summary of the current configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Backbone model:      {BACKBONE_MODEL}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Hidden dimension:    {HIDDEN_DIM}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"LR scheduler:        {LR_TYPE}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Max epochs:          {NUM_EPOCHS}")
    print(f"Early stopping:      {EARLY_STOPPING_PATIENCE} epochs patience")
    print(f"Validation sites:    {VALIDATION_DOMAIN_IDS}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # If you run this file directly, it will print the configuration
    print_config()
    device = get_device()
