"""
Configuration for attention pooling training.

This file contains hyperparameters for train_attention.py, which trains
models with learned pooling (attention, gated_attention, statistics, hybrid).

Edit these values to tune the model without typing command-line arguments.

Usage:
    1. Edit values below
    2. Run: python scripts/train_attention.py
    3. View results: tensorboard --logdir logs
"""

# =============================================================================
# POOLING CONFIGURATION
# =============================================================================

# Type of learned pooling to use:
#   'attention': Attention-weighted pooling (learns which images matter)
#   'gated_attention': Gated attention (more expressive, recommended)
#   'statistics': Mean + std concatenation (captures distribution shape)
#   'hybrid': Attention-weighted mean + weighted std (best for CRPS)
POOLING_TYPE = 'attention'

# Hidden dimension for the attention network
# Larger = more expressive but more parameters
POOLING_HIDDEN_DIM = 64

# For statistics/hybrid pooling: project concatenated output back to embed_dim?
# True: pooled_dim = embed_dim (smaller regression head input)
# False: pooled_dim = embed_dim * 2 (preserves all information)
PROJECT_TO_ORIGINAL = False


# =============================================================================
# REGRESSION HEAD CONFIGURATION
# =============================================================================

# Hidden layer size in the regression head
HEAD_HIDDEN_DIM = 64

# Dropout probability for regularization (0.0 = no dropout)
DROPOUT = 0.3


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Learning rate
LEARNING_RATE = 1e-5

# Number of training epochs (early stopping usually triggers first)
NUM_EPOCHS = 100

# Early stopping: stop if no improvement for this many epochs
EARLY_STOPPING_PATIENCE = 50

# L2 regularization strength
WEIGHT_DECAY = 1e-2

# Learning rate scheduler type:
#   'cosine': CosineAnnealingLR - smoothly decreases LR (recommended)
#   'plateau': ReduceLROnPlateau - halves LR when validation stalls
LR_TYPE = 'cosine'


# =============================================================================
# BACKBONE CONFIGURATION
# =============================================================================

# Which backbone embeddings to use
# Options: 'dinov3', 'dinov2', 'bioclip', 'bioclip2', or combined like 'dinov3+bioclip2'
BACKBONE = 'dinov3'


# =============================================================================
# OTHER
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42
