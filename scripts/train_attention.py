"""
Train drought prediction models with learned pooling (attention, gated attention, etc.).

This script trains models that LEARN how to aggregate multiple beetle images per event,
rather than using fixed pooling (mean/max/sum). The pooling module is trained end-to-end
with the regression head.

Key differences from train.py:
    - Uses EventEmbeddingDataset (variable-length events) instead of AggregatedEmbeddingDataset
    - Training is batch_size=1 (one event at a time) due to variable image counts
    - Both pooling module and regression head are trained together
    - Checkpoints save both pooling and regression head weights for submission

Supported pooling types:
    - attention: Attention-weighted pooling (learns which images matter)
    - gated_attention: Gated attention (more expressive, recommended)
    - statistics: Mean + std concatenation (captures distribution shape)
    - hybrid: Attention-weighted mean + weighted std (best for CRPS)

Usage:
    # Train with attention pooling
    python scripts/train_attention.py --backbone dinov3 --pooling attention

    # Train with hybrid pooling (recommended for CRPS)
    python scripts/train_attention.py --backbone dinov3 --pooling hybrid

    # Full example with all options
    python scripts/train_attention.py \\
        --backbone bioclip2 \\
        --pooling gated_attention \\
        --epochs 100 \\
        --lr 1e-4 \\
        --hidden_dim 128 \\
        --pooling_hidden_dim 64

TensorBoard:
    tensorboard --logdir logs
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    EMBEDDINGS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    EMBEDDING_DIM, MIN_SIGMA,
    TARGET_COLUMNS, LEARNED_POOLING_TYPES,
    get_device, set_seed
)

# Import attention-specific config (edit config_attention.py to change defaults)
from config_attention import (
    POOLING_TYPE, POOLING_HIDDEN_DIM, PROJECT_TO_ORIGINAL,
    HEAD_HIDDEN_DIM, DROPOUT,
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY, LR_TYPE,
    BACKBONE, RANDOM_SEED
)
from src.data_loader import create_data_loaders
from src.pooling import create_pooling, get_pooling_output_dim
from src.model import RegressionHead
from src.loss import gaussian_nll_loss
from src.metrics import compute_competition_score
from src.utils import get_timestamp, ensure_dir, AverageMeter, EarlyStopping, format_time


# =============================================================================
# POOLED MODEL (Pooling + Regression Head as single trainable unit)
# =============================================================================

class PooledModel(nn.Module):
    """
    Combined model that pools variable-length image embeddings and predicts SPEI values.

    This model:
    1. Takes variable-length image embeddings (N_images, embed_dim)
    2. Applies learned pooling to get a single representation
    3. Passes through regression head to predict (mu, sigma) for each target

    Both pooling and regression head are trained end-to-end.
    """

    def __init__(
        self,
        pooling_type='attention',
        embed_dim=EMBEDDING_DIM,
        pooling_hidden_dim=POOLING_HIDDEN_DIM,
        head_hidden_dim=HEAD_HIDDEN_DIM,
        num_targets=3,
        dropout=DROPOUT,
        min_sigma=MIN_SIGMA,
        project_to_original=False
    ):
        """
        Args:
            pooling_type: Type of learned pooling (attention, gated_attention, statistics, hybrid)
            embed_dim: Dimension of input embeddings
            pooling_hidden_dim: Hidden dimension for pooling attention network
            head_hidden_dim: Hidden dimension for regression head
            num_targets: Number of targets to predict (default 3: SPEI_30d, 1y, 2y)
            dropout: Dropout probability for regression head
            min_sigma: Minimum sigma value for numerical stability
            project_to_original: For statistics/hybrid, project back to embed_dim
        """
        super().__init__()

        self.pooling_type = pooling_type
        self.embed_dim = embed_dim
        self.project_to_original = project_to_original

        # Create pooling module
        self.pooling = create_pooling(
            pooling_type=pooling_type,
            embed_dim=embed_dim,
            hidden_dim=pooling_hidden_dim,
            project_to_original=project_to_original
        )

        # Determine input dimension for regression head
        pooled_dim = get_pooling_output_dim(
            pooling_type=pooling_type,
            embed_dim=embed_dim,
            project_to_original=project_to_original
        )

        # Create regression head
        self.head = RegressionHead(
            embedding_dim=pooled_dim,
            hidden_dim=head_hidden_dim,
            num_targets=num_targets,
            min_sigma=min_sigma,
            dropout=dropout
        )

    def forward(self, embeddings, return_weights=False):
        """
        Forward pass.

        Args:
            embeddings: (N_images, embed_dim) tensor of image embeddings for ONE event
            return_weights: If True, also return attention weights (for visualization)

        Returns:
            mu: (1, num_targets) predicted means
            sigma: (1, num_targets) predicted uncertainties
            weights: (optional) attention weights if return_weights=True
        """
        # Apply pooling to get single representation
        if return_weights and hasattr(self.pooling, 'forward'):
            # Check if pooling supports return_weights
            import inspect
            sig = inspect.signature(self.pooling.forward)
            if 'return_weights' in sig.parameters:
                pooled, weights = self.pooling(embeddings, keepdim=True, return_weights=True)
            else:
                pooled = self.pooling(embeddings, keepdim=True)
                weights = None
        else:
            pooled = self.pooling(embeddings, keepdim=True)
            weights = None

        # Get predictions
        mu, sigma = self.head(pooled)

        if return_weights:
            return mu, sigma, weights
        return mu, sigma

    def get_config(self):
        """Return model configuration for saving."""
        return {
            'pooling_type': self.pooling_type,
            'embed_dim': self.embed_dim,
            'pooling_hidden_dim': POOLING_HIDDEN_DIM,
            'head_hidden_dim': HEAD_HIDDEN_DIM,
            'project_to_original': self.project_to_original,
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, optimizer, device, epoch):
    """
    Train for one epoch.

    Since we use batch_size=1 (one event at a time), each iteration
    processes a variable-length set of image embeddings.
    """
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for embeddings, targets in pbar:
        # embeddings: (N_images, embed_dim) for this event
        # targets: (1, num_targets)
        embeddings = embeddings.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(embeddings)

        # Compute loss
        loss = gaussian_nll_loss(mu, sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, device, epoch):
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()

    all_mu = []
    all_sigma = []
    all_targets = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    for embeddings, targets in pbar:
        embeddings = embeddings.to(device)
        targets = targets.to(device)

        mu, sigma = model(embeddings)
        loss = gaussian_nll_loss(mu, sigma, targets)

        loss_meter.update(loss.item())

        all_mu.append(mu.cpu())
        all_sigma.append(sigma.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions
    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_sigma = torch.cat(all_sigma, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Compute CRPS
    rms_crps, per_target_crps = compute_competition_score(all_mu, all_sigma, all_targets)

    return loss_meter.avg, rms_crps, per_target_crps


def save_checkpoint(model, optimizer, epoch, val_crps, path, args, extra_info=None):
    """
    Save model checkpoint with both pooling and regression head weights.

    The checkpoint format is compatible with submission/model.py.
    """
    checkpoint = {
        'epoch': epoch,
        'val_crps': val_crps,
        'model_state_dict': model.head.state_dict(),
        'pooling_state_dict': model.pooling.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Model configuration for loading (uses actual args values, not defaults)
        'pooling_type': model.pooling_type,
        'embedding_dim': model.embed_dim,
        'hidden_dim': args.hidden_dim,
        'pooling_hidden_dim': args.pooling_hidden_dim,
        'dropout': args.dropout,
        'min_sigma': MIN_SIGMA,
        'project_to_original': model.project_to_original,
    }
    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, path)


def check_embeddings_exist(backbone):
    """Check if embeddings have been extracted for the specified backbone."""
    required_files = [
        EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt',
        EMBEDDINGS_DIR / 'train_targets.pt',
        EMBEDDINGS_DIR / 'train_events.pt',
        EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt',
        EMBEDDINGS_DIR / 'val_targets.pt',
        EMBEDDINGS_DIR / 'val_events.pt'
    ]

    missing = [f for f in required_files if not f.exists()]

    if missing:
        print(f"ERROR: Embeddings not found for backbone '{backbone}'!")
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nPlease run: python scripts/extract_embeddings.py --backbone {backbone}")
        return False

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train drought prediction model with learned pooling"
    )
    parser.add_argument(
        '--backbone', type=str, default=BACKBONE,
        help=f'Which backbone embeddings to use (default: {BACKBONE})'
    )
    parser.add_argument(
        '--pooling', type=str, default=POOLING_TYPE,
        choices=LEARNED_POOLING_TYPES,
        help=f'Pooling type (default: {POOLING_TYPE}). Options: {LEARNED_POOLING_TYPES}'
    )
    parser.add_argument(
        '--epochs', type=int, default=NUM_EPOCHS,
        help=f'Number of epochs (default: {NUM_EPOCHS})'
    )
    parser.add_argument(
        '--lr', type=float, default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=HEAD_HIDDEN_DIM,
        help=f'Regression head hidden dimension (default: {HEAD_HIDDEN_DIM})'
    )
    parser.add_argument(
        '--pooling_hidden_dim', type=int, default=POOLING_HIDDEN_DIM,
        help=f'Pooling attention hidden dimension (default: {POOLING_HIDDEN_DIM})'
    )
    parser.add_argument(
        '--dropout', type=float, default=DROPOUT,
        help=f'Dropout probability (default: {DROPOUT})'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=WEIGHT_DECAY,
        help=f'Weight decay (default: {WEIGHT_DECAY})'
    )
    parser.add_argument(
        '--lr_type', type=str, default=LR_TYPE,
        choices=['cosine', 'plateau'],
        help=f'LR scheduler type (default: {LR_TYPE})'
    )
    parser.add_argument(
        '--patience', type=int, default=EARLY_STOPPING_PATIENCE,
        help=f'Early stopping patience (default: {EARLY_STOPPING_PATIENCE})'
    )
    parser.add_argument(
        '--project_to_original', action='store_true', default=PROJECT_TO_ORIGINAL,
        help=f'For statistics/hybrid pooling, project output back to embed_dim (default: {PROJECT_TO_ORIGINAL})'
    )
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Custom experiment name (default: auto-generated)'
    )
    parser.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})'
    )

    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print("BEETLES DROUGHT PREDICTION - LEARNED POOLING TRAINING")
    print("=" * 60)

    # Check embeddings
    if not check_embeddings_exist(args.backbone):
        sys.exit(1)

    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device()

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LR scheduler: {args.lr_type}")
    print(f"  Hidden dim (head): {args.hidden_dim}")
    print(f"  Hidden dim (pooling): {args.pooling_hidden_dim}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Project to original: {args.project_to_original}")

    # Load data with EventEmbeddingDataset for variable-length events
    print("\n" + "-" * 50)
    print("Loading data (variable-length events)...")
    print("-" * 50)

    train_loader, val_loader = create_data_loaders(
        train_embeddings_path=EMBEDDINGS_DIR / f'train_embeddings_{args.backbone}.pt',
        train_targets_path=EMBEDDINGS_DIR / 'train_targets.pt',
        train_events_path=EMBEDDINGS_DIR / 'train_events.pt',
        val_embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{args.backbone}.pt',
        val_targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        val_events_path=EMBEDDINGS_DIR / 'val_events.pt',
        pooling_type=args.pooling  # This triggers EventEmbeddingDataset
    )

    # Detect embedding dimension
    sample_embeddings = torch.load(
        EMBEDDINGS_DIR / f'train_embeddings_{args.backbone}.pt',
        weights_only=True
    )
    embed_dim = sample_embeddings.shape[1]
    print(f"\nEmbedding dimension: {embed_dim}")

    # Create model
    print("\n" + "-" * 50)
    print("Creating model...")
    print("-" * 50)

    model = PooledModel(
        pooling_type=args.pooling,
        embed_dim=embed_dim,
        pooling_hidden_dim=args.pooling_hidden_dim,
        head_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        project_to_original=args.project_to_original
    )
    model = model.to(device)

    # Count parameters
    pooling_params = sum(p.numel() for p in model.pooling.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    total_params = pooling_params + head_params
    print(f"  Pooling parameters: {pooling_params:,}")
    print(f"  Head parameters: {head_params:,}")
    print(f"  Total parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # LR scheduler
    if args.lr_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    # Experiment name
    if args.experiment_name is None:
        lr_str = f"{args.lr:.0e}".replace("-", "")
        args.experiment_name = (
            f"{args.backbone}_{args.pooling}_h{args.hidden_dim}_"
            f"ph{args.pooling_hidden_dim}_lr{lr_str}_{get_timestamp()}"
        )

    # TensorBoard
    log_dir = ensure_dir(LOGS_DIR / args.experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\nTensorBoard logs: {log_dir}")

    # Checkpoint directory
    checkpoint_dir = ensure_dir(CHECKPOINTS_DIR)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')

    # Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Max epochs: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print("=" * 60 + "\n")

    best_val_crps = float('inf')
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_crps, per_target_crps = validate(model, val_loader, device, epoch)

        # LR scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        if args.lr_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_crps)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('CRPS/val', val_crps, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        for target_name, crps_val in per_target_crps.items():
            writer.add_scalar(f'CRPS/{target_name}', crps_val, epoch)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"CRPS: {val_crps:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_crps < best_val_crps:
            best_val_crps = val_crps
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_crps,
                checkpoint_dir / "best_model.pt", args
            )
            print(f"  → New best! CRPS: {val_crps:.4f}")

        # Early stopping
        if early_stopping(val_crps):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Training complete
    total_time = time.time() - start_time
    writer.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {format_time(total_time)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best CRPS: {best_val_crps:.4f}")
    print(f"\nModel saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    # Save final model for submission
    submission_path = checkpoint_dir / f"model_{args.pooling}_{args.backbone}.pt"
    save_checkpoint(model, optimizer, epoch, val_crps, submission_path, args)
    print(f"\nSubmission checkpoint: {submission_path}")

    print("\nTo use this model for submission:")
    print(f"  1. Copy {submission_path} to submission/model.pth")
    print(f"  2. The submission/model.py already supports {args.pooling} pooling")
    print("=" * 60)


if __name__ == "__main__":
    main()
