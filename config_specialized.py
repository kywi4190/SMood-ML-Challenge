"""
Configuration for specialized target-specific training.

This file contains hyperparameters for the two specialized models:
  - Model 1 (backbone1): Predicts SPEI_30d only
  - Model 2 (backbone2): Predicts SPEI_1y and SPEI_2y

Edit these values to tune each model independently.
"""

# =============================================================================
# MODEL 1: SPEI_30d SPECIALIST (short-term drought)
# =============================================================================
# This model predicts only SPEI_30d using backbone1 (default: dinov3)

MODEL1_HIDDEN_DIM = 128      # Hidden layer size
MODEL1_BATCH_SIZE = 32       # Batch size for training
MODEL1_DROPOUT = 0.1         # Dropout probability
MODEL1_TYPE = 'deep'       # Model type: 'simple' or 'deep'

# =============================================================================
# MODEL 2: SPEI_1y + SPEI_2y SPECIALIST (long-term drought)
# =============================================================================
# This model predicts SPEI_1y and SPEI_2y using backbone2 (default: bioclip2)

MODEL2_HIDDEN_DIM = 64      # Hidden layer size (smaller - bioclip2 features are task-aligned)
MODEL2_BATCH_SIZE = 64       # Batch size for training
MODEL2_DROPOUT = 0.1         # Dropout probability
MODEL2_TYPE = 'simple'       # Model type: 'simple' or 'deep'

# =============================================================================
# SHARED TRAINING PARAMETERS
# =============================================================================
# These apply to both models

LEARNING_RATE = 1e-4         # Learning rate for both models
NUM_EPOCHS = 100             # Maximum epochs (early stopping usually triggers first)
EARLY_STOPPING_PATIENCE = 500 # Stop if no improvement for this many epochs
WEIGHT_DECAY = 1e-3          # L2 regularization strength
MIN_SIGMA = 1e-3             # Minimum sigma for numerical stability
RANDOM_SEED = 42             # Random seed for reproducibility

# Learning rate scheduler type (shared by both models)
# 'cosine': CosineAnnealingLR - smoothly decreases LR over training (recommended)
# 'plateau': ReduceLROnPlateau - halves LR when validation stalls
LR_TYPE = 'cosine'
