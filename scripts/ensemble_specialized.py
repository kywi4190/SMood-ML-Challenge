"""
Combine predictions from specialized models for final evaluation.

This script loads specialized models (each trained on specific targets)
and combines their predictions into a complete 3-target prediction.

Expected setup:
  - Model 1: DINOv3 specialized for SPEI_30d
  - Model 2: BioCLIP2 specialized for SPEI_1y and SPEI_2y

Usage:
    # Using default checkpoint names
    python scripts/ensemble_specialized.py

    # Custom checkpoints
    python scripts/ensemble_specialized.py \
        --checkpoint1 checkpoints/best_model_specialized_30d.pt \
        --backbone1 dinov3 \
        --checkpoint2 checkpoints/best_model_specialized_1y_2y.pt \
        --backbone2 bioclip2
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import EMBEDDINGS_DIR, CHECKPOINTS_DIR, get_device, set_seed
from src.data_loader import AggregatedEmbeddingDataset
from src.metrics import compute_competition_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the specialized model class
from train_specialized import SpecializedRegressionHead, TARGET_CONFIG


def load_specialized_model(checkpoint_path, backbone, device):
    """
    Load a specialized model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        backbone: Backbone name (for loading embeddings to get dimension)
        device: Device to load model to

    Returns:
        tuple: (model, target_indices, target_names)
    """
    print(f"Loading specialized model: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get target info from checkpoint
    target_indices = checkpoint['target_indices']
    target_names = checkpoint['target_names']

    print(f"  Targets: {target_names} (indices: {target_indices})")

    # Get embedding dimension from embeddings
    emb_path = EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt'
    sample_emb = torch.load(emb_path, weights_only=True)
    embedding_dim = sample_emb.shape[1]

    # Infer hidden_dim from checkpoint
    # fc1.weight shape is (hidden_dim, embedding_dim)
    hidden_dim = checkpoint['model_state_dict']['fc1.weight'].shape[0]
    print(f"  Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}")

    # Create model
    model = SpecializedRegressionHead(
        embedding_dim=embedding_dim,
        target_indices=target_indices,
        hidden_dim=hidden_dim
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, target_indices, target_names


@torch.no_grad()
def get_specialized_predictions(model, backbone, device):
    """
    Get predictions from a specialized model.

    Args:
        model: Specialized model
        backbone: Backbone name (for loading embeddings)
        device: Device to run on

    Returns:
        tuple: (mu, sigma) numpy arrays
    """
    # Create dataloader for this backbone
    val_dataset = AggregatedEmbeddingDataset(
        embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt',
        targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        event_ids_path=EMBEDDINGS_DIR / 'val_events.pt',
        aggregation='mean'
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model.eval()
    all_mu = []
    all_sigma = []

    for embeddings, _ in val_loader:
        embeddings = embeddings.to(device)
        mu, sigma = model(embeddings)
        all_mu.append(mu.cpu().numpy())
        all_sigma.append(sigma.cpu().numpy())

    return np.concatenate(all_mu, axis=0), np.concatenate(all_sigma, axis=0)


def load_targets():
    """Load the full validation targets."""
    # Load from aggregated dataset to ensure correct ordering
    val_dataset = AggregatedEmbeddingDataset(
        embeddings_path=EMBEDDINGS_DIR / 'val_embeddings_dinov3.pt',  # Any backbone works
        targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        event_ids_path=EMBEDDINGS_DIR / 'val_events.pt',
        aggregation='mean'
    )

    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    _, targets = next(iter(val_loader))

    return targets.numpy()


def combine_predictions(predictions_list):
    """
    Combine predictions from multiple specialized models.

    Args:
        predictions_list: List of (mu, sigma, target_indices) tuples

    Returns:
        tuple: (combined_mu, combined_sigma) with shape (n_samples, 3)
    """
    # Get number of samples from first prediction
    n_samples = predictions_list[0][0].shape[0]

    # Initialize combined arrays
    combined_mu = np.zeros((n_samples, 3))
    combined_sigma = np.zeros((n_samples, 3))

    # Fill in predictions for each target
    for mu, sigma, target_indices in predictions_list:
        for i, target_idx in enumerate(target_indices):
            combined_mu[:, target_idx] = mu[:, i]
            combined_sigma[:, target_idx] = sigma[:, i]

    return combined_mu, combined_sigma


def print_results(mu, sigma, targets, title="RESULTS"):
    """Print detailed evaluation results."""
    target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    rms_crps, per_target_crps = compute_competition_score(mu, sigma, targets)

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"\n{'COMPETITION SCORE (RMS-CRPS)':40s}: {rms_crps:.4f}")

    print("\n" + "-" * 70)
    print(f"{'Target':<15} {'CRPS':>12} {'Mean Sigma':>12} {'Mean |Error|':>15}")
    print("-" * 70)

    for i, name in enumerate(target_names):
        crps = per_target_crps[name]
        mean_sigma = np.mean(sigma[:, i])
        mean_abs_error = np.mean(np.abs(targets[:, i] - mu[:, i]))
        print(f"{name:<15} {crps:>12.4f} {mean_sigma:>12.4f} {mean_abs_error:>15.4f}")

    print("-" * 70)
    mean_crps = np.mean(list(per_target_crps.values()))
    print(f"{'Mean':<15} {mean_crps:>12.4f}")
    print("=" * 70)

    return rms_crps, per_target_crps


def main():
    parser = argparse.ArgumentParser(
        description="Combine predictions from specialized models"
    )

    # Model 1: SPEI_30d specialist
    parser.add_argument(
        '--checkpoint1',
        type=str,
        default=str(CHECKPOINTS_DIR / 'best_model_specialized_30d.pt'),
        help='Checkpoint for SPEI_30d specialist'
    )
    parser.add_argument(
        '--backbone1',
        type=str,
        default='dinov3',
        help='Backbone for model 1 (default: dinov3)'
    )

    # Model 2: SPEI_1y/2y specialist
    parser.add_argument(
        '--checkpoint2',
        type=str,
        default=str(CHECKPOINTS_DIR / 'best_model_specialized_1y_2y.pt'),
        help='Checkpoint for SPEI_1y/2y specialist'
    )
    parser.add_argument(
        '--backbone2',
        type=str,
        default='bioclip2',
        help='Backbone for model 2 (default: bioclip2)'
    )

    # Optional: also evaluate individual models for comparison
    parser.add_argument(
        '--compare_individual',
        action='store_true',
        help='Also print results for standard (non-specialized) models if available'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SPECIALIZED MODEL ENSEMBLE EVALUATION")
    print("=" * 70)

    set_seed()
    device = get_device()

    # Check checkpoints exist
    if not Path(args.checkpoint1).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint1}")
        print("Train with: python scripts/train_specialized.py --backbone dinov3 --targets 30d")
        sys.exit(1)

    if not Path(args.checkpoint2).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint2}")
        print("Train with: python scripts/train_specialized.py --backbone bioclip2 --targets 1y 2y")
        sys.exit(1)

    # Load specialized models
    print("\nLoading specialized models...")
    print("-" * 50)

    model1, indices1, names1 = load_specialized_model(args.checkpoint1, args.backbone1, device)
    model2, indices2, names2 = load_specialized_model(args.checkpoint2, args.backbone2, device)

    # Verify targets don't overlap and cover all 3
    all_indices = set(indices1) | set(indices2)
    if len(all_indices) != 3:
        print(f"WARNING: Models don't cover all targets!")
        print(f"  Model 1 targets: {names1}")
        print(f"  Model 2 targets: {names2}")
        print(f"  Missing: {set([0,1,2]) - all_indices}")

    if set(indices1) & set(indices2):
        print(f"WARNING: Overlapping targets detected!")
        overlap = set(indices1) & set(indices2)
        print(f"  Overlap: {overlap}")
        print(f"  Model 1 predictions will be used for overlapping targets")

    # Get predictions
    print("\n" + "-" * 50)
    print("Generating predictions...")
    print("-" * 50)

    print(f"Model 1 ({args.backbone1}): {names1}")
    mu1, sigma1 = get_specialized_predictions(model1, args.backbone1, device)
    print(f"  Predictions shape: {mu1.shape}")

    print(f"Model 2 ({args.backbone2}): {names2}")
    mu2, sigma2 = get_specialized_predictions(model2, args.backbone2, device)
    print(f"  Predictions shape: {mu2.shape}")

    # Load targets
    targets = load_targets()
    print(f"Targets shape: {targets.shape}")

    # Combine predictions
    print("\n" + "-" * 50)
    print("Combining predictions...")
    print("-" * 50)

    predictions_list = [
        (mu1, sigma1, indices1),
        (mu2, sigma2, indices2)
    ]

    combined_mu, combined_sigma = combine_predictions(predictions_list)

    print(f"Combined predictions shape: {combined_mu.shape}")
    print(f"Target assignment:")
    for i, name in enumerate(["SPEI_30d", "SPEI_1y", "SPEI_2y"]):
        if i in indices1:
            source = f"{args.backbone1} (Model 1)"
        else:
            source = f"{args.backbone2} (Model 2)"
        print(f"  {name}: {source}")

    # Print ensemble results
    rms_crps, per_target = print_results(
        combined_mu, combined_sigma, targets,
        "SPECIALIZED ENSEMBLE RESULTS"
    )

    # Save predictions
    output_path = CHECKPOINTS_DIR / 'specialized_ensemble_predictions.npz'
    np.savez(
        output_path,
        mu=combined_mu,
        sigma=combined_sigma,
        targets=targets,
        model1_backbone=args.backbone1,
        model1_targets=names1,
        model2_backbone=args.backbone2,
        model2_targets=names2
    )
    print(f"\nPredictions saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ensemble RMS-CRPS: {rms_crps:.4f}")
    print(f"\nThis score combines:")
    print(f"  - {args.backbone1} for {', '.join(names1)}")
    print(f"  - {args.backbone2} for {', '.join(names2)}")
    print("\nCompare this to your individual model scores to verify improvement!")
    print("=" * 70)


if __name__ == "__main__":
    main()
