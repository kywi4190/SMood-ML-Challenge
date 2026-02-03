"""
Combine embeddings from two backbones into a single concatenated embedding.

This script concatenates embeddings from two different backbones (e.g., dinov3 and bioclip2)
to create a combined feature representation. The resulting embeddings can be used for
training a model that leverages features from both backbones.

Usage:
    python scripts/combine_embeddings.py --backbone1 dinov3 --backbone2 bioclip2

This will create:
    - train_embeddings_dinov3+bioclip2.pt (1536-dim if both are 768-dim)
    - val_embeddings_dinov3+bioclip2.pt
"""

import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import EMBEDDINGS_DIR
from src.feature_extractor import BACKBONE_OPTIONS


def check_embeddings_exist(backbone):
    """Check if embeddings exist for a backbone."""
    train_path = EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt'
    val_path = EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt'

    if not train_path.exists():
        print(f"ERROR: Training embeddings not found for '{backbone}'")
        print(f"  Expected: {train_path}")
        print(f"  Run: python scripts/extract_embeddings.py --backbone {backbone}")
        return False

    if not val_path.exists():
        print(f"ERROR: Validation embeddings not found for '{backbone}'")
        print(f"  Expected: {val_path}")
        print(f"  Run: python scripts/extract_embeddings.py --backbone {backbone}")
        return False

    return True


def load_embeddings(backbone):
    """Load train and val embeddings for a backbone."""
    train_path = EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt'
    val_path = EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt'

    train_emb = torch.load(train_path, weights_only=True)
    val_emb = torch.load(val_path, weights_only=True)

    return train_emb, val_emb


def combine_and_save(backbone1, backbone2, normalize=True):
    """
    Combine embeddings from two backbones and save.

    Args:
        backbone1: First backbone name (e.g., 'dinov3')
        backbone2: Second backbone name (e.g., 'bioclip2')
        normalize: Whether to L2-normalize embeddings before concatenation

    Returns:
        str: Combined backbone name (e.g., 'dinov3+bioclip2')
    """
    combined_name = f"{backbone1}+{backbone2}"

    print(f"\nLoading embeddings from '{backbone1}'...")
    train_emb1, val_emb1 = load_embeddings(backbone1)
    print(f"  Train: {train_emb1.shape}, Val: {val_emb1.shape}")

    print(f"\nLoading embeddings from '{backbone2}'...")
    train_emb2, val_emb2 = load_embeddings(backbone2)
    print(f"  Train: {train_emb2.shape}, Val: {val_emb2.shape}")

    # Verify sample counts match
    if train_emb1.shape[0] != train_emb2.shape[0]:
        raise ValueError(
            f"Training sample count mismatch: {backbone1} has {train_emb1.shape[0]}, "
            f"{backbone2} has {train_emb2.shape[0]}"
        )

    if val_emb1.shape[0] != val_emb2.shape[0]:
        raise ValueError(
            f"Validation sample count mismatch: {backbone1} has {val_emb1.shape[0]}, "
            f"{backbone2} has {val_emb2.shape[0]}"
        )

    # Optionally normalize embeddings so both backbones contribute equally
    if normalize:
        print(f"\nL2-normalizing embeddings...")
        # Print magnitude stats before normalization
        print(f"  {backbone1} train mean L2 norm: {torch.norm(train_emb1, dim=1).mean():.4f}")
        print(f"  {backbone2} train mean L2 norm: {torch.norm(train_emb2, dim=1).mean():.4f}")

        train_emb1 = torch.nn.functional.normalize(train_emb1, p=2, dim=1)
        val_emb1 = torch.nn.functional.normalize(val_emb1, p=2, dim=1)
        train_emb2 = torch.nn.functional.normalize(train_emb2, p=2, dim=1)
        val_emb2 = torch.nn.functional.normalize(val_emb2, p=2, dim=1)
        print(f"  After normalization: all embeddings have unit L2 norm")

    # Concatenate along feature dimension
    print(f"\nConcatenating embeddings...")
    train_combined = torch.cat([train_emb1, train_emb2], dim=1)
    val_combined = torch.cat([val_emb1, val_emb2], dim=1)

    combined_dim = train_combined.shape[1]
    print(f"  Combined dimension: {train_emb1.shape[1]} + {train_emb2.shape[1]} = {combined_dim}")

    # Save combined embeddings
    train_path = EMBEDDINGS_DIR / f'train_embeddings_{combined_name}.pt'
    val_path = EMBEDDINGS_DIR / f'val_embeddings_{combined_name}.pt'

    print(f"\nSaving combined embeddings...")
    torch.save(train_combined, train_path)
    torch.save(val_combined, val_path)
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    return combined_name, combined_dim


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Combine embeddings from two backbones"
    )
    parser.add_argument(
        '--backbone1',
        type=str,
        default='dinov3',
        help='First backbone (default: dinov3)'
    )
    parser.add_argument(
        '--backbone2',
        type=str,
        default='bioclip2',
        help='Second backbone (default: bioclip2)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip L2 normalization (not recommended - can cause feature dominance)'
    )

    args = parser.parse_args()

    normalize = not args.no_normalize

    print("=" * 60)
    print("COMBINE EMBEDDINGS")
    print("=" * 60)
    print(f"Backbone 1: {args.backbone1}")
    print(f"Backbone 2: {args.backbone2}")
    print(f"Normalize: {normalize}")

    # Check embeddings exist for both backbones
    if not check_embeddings_exist(args.backbone1):
        sys.exit(1)
    if not check_embeddings_exist(args.backbone2):
        sys.exit(1)

    # Combine and save
    combined_name, combined_dim = combine_and_save(args.backbone1, args.backbone2, normalize=normalize)

    print("\n" + "=" * 60)
    print("COMBINATION COMPLETE")
    print("=" * 60)
    print(f"\nCombined backbone: {combined_name}")
    print(f"Combined dimension: {combined_dim}")
    print(f"\nTo train with combined embeddings:")
    print(f"  python scripts/train.py --backbone {combined_name}")
    print(f"\nTo evaluate:")
    print(f"  python scripts/evaluate.py --backbone {combined_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
