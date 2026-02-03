"""
Train the drought prediction model.

This script trains the regression head on pre-computed embeddings.
Make sure to run extract_embeddings.py first!

Usage:
    python scripts/train.py

With custom parameters:
    python scripts/train.py --epochs 100 --lr 0.001 --batch_size 32

Using BioClip2 embeddings:
    python scripts/train.py --backbone bioclip2
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from config import (
    EMBEDDINGS_DIR, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, get_device, set_seed, print_config
)
from src.model import RegressionHead, create_model
from src.data_loader import create_data_loaders
from src.trainer import Trainer
from src.utils import print_model_summary, get_timestamp
from src.feature_extractor import BACKBONE_OPTIONS


def check_embeddings_exist(backbone='dinov3'):
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
        # Provide appropriate help based on backbone type
        if '+' in backbone:
            parts = backbone.split('+')
            print(f"\nFor combined backbones, run:")
            print(f"  python scripts/combine_embeddings.py --backbone1 {parts[0]} --backbone2 {parts[1]}")
        else:
            print(f"\nPlease run: python scripts/extract_embeddings.py --backbone {backbone}")
        return False

    return True


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train the drought prediction model"
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='dinov3',
        help='Which backbone embeddings to use (default: dinov3). '
             'Options: dinov3, dinov2, bioclip, bioclip2, or combined like dinov3+bioclip2'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=NUM_EPOCHS,
        help=f'Number of training epochs (default: {NUM_EPOCHS})'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=HIDDEN_DIM,
        help=f'Hidden layer dimension (default: {HIDDEN_DIM})'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DROPOUT,
        help=f'Dropout probability (default: {DROPOUT})'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['simple', 'deep'],
        default='simple',
        help='Model architecture (default: simple)'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['mean', 'max', 'sum'],
        default='mean',
        help='How to aggregate embeddings per event (default: mean)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Name for this experiment (default: auto-generated)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 60)
    print("BEETLES DROUGHT PREDICTION - TRAINING")
    print("=" * 60)

    # Check embeddings exist for this backbone
    if not check_embeddings_exist(args.backbone):
        sys.exit(1)

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Print configuration
    print(f"\nTraining Configuration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Model type: {args.model_type}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Seed: {args.seed}")

    # Create data loaders with backbone-specific embeddings
    print("\n" + "-" * 50)
    print("Loading data...")
    print("-" * 50)

    train_embeddings_path = EMBEDDINGS_DIR / f'train_embeddings_{args.backbone}.pt'
    train_loader, val_loader = create_data_loaders(
        train_embeddings_path=train_embeddings_path,
        train_targets_path=EMBEDDINGS_DIR / 'train_targets.pt',
        train_events_path=EMBEDDINGS_DIR / 'train_events.pt',
        val_embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{args.backbone}.pt',
        val_targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        val_events_path=EMBEDDINGS_DIR / 'val_events.pt',
        batch_size=args.batch_size,
        aggregated=True,
        aggregation=args.aggregation
    )

    # Detect embedding dimension from loaded embeddings
    sample_embeddings = torch.load(train_embeddings_path, weights_only=True)
    embedding_dim = sample_embeddings.shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")

    # Create model
    print("\n" + "-" * 50)
    print("Creating model...")
    print("-" * 50)

    model = create_model(
        model_type=args.model_type,
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    print_model_summary(model, "RegressionHead")

    # Generate experiment name if not provided
    # Format: {backbone}_{model_type}_{aggregation}_h{hidden_dim}_lr{learning_rate}_{timestamp}
    if args.experiment_name is None:
        lr_str = f"{args.lr:.0e}".replace("-", "")  # e.g., "1e04" for 0.0001
        args.experiment_name = f"{args.backbone}_{args.model_type}_{args.aggregation}_h{args.hidden_dim}_lr{lr_str}_{get_timestamp()}"

    # Create trainer
    print("\n" + "-" * 50)
    print("Initializing trainer...")
    print("-" * 50)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device,
        experiment_name=args.experiment_name
    )

    # Train!
    history = trainer.train(num_epochs=args.epochs)

    # Save final model
    trainer.save_final_model()

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Backbone: {args.backbone}")
    print(f"Best validation CRPS: {trainer.best_val_crps:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")
    print(f"\nModel saved to: checkpoints/best_model.pt")
    print(f"TensorBoard logs: logs/{args.experiment_name}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir logs/{args.experiment_name}")
    print("\nTo evaluate on validation set, run:")
    print(f"  python scripts/evaluate.py --backbone {args.backbone}")
    print("=" * 60)


if __name__ == "__main__":
    main()
