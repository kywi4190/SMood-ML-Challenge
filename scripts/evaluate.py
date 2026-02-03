"""
Evaluate the trained model on the validation set.

This script loads the best trained model and computes detailed metrics
including CRPS, MAE, calibration, and per-target breakdowns.

Usage:
    python scripts/evaluate.py

With a specific checkpoint:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from config import (
    EMBEDDINGS_DIR, CHECKPOINTS_DIR, EMBEDDING_DIM, HIDDEN_DIM,
    get_device, set_seed
)
from src.model import create_model
from src.data_loader import AggregatedEmbeddingDataset
from src.metrics import (
    compute_competition_score, compute_crps_per_target,
    compute_mae, compute_rmse, compute_calibration,
    print_metrics_summary
)
from src.feature_extractor import BACKBONE_OPTIONS
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_model(checkpoint_path, device, model_type='simple', hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model to
        model_type: 'simple' or 'deep'
        hidden_dim: Hidden dimension size
        embedding_dim: Input embedding dimension

    Returns:
        nn.Module: The loaded model
    """
    print(f"Loading model from {checkpoint_path}")
    print(f"  Model type: {model_type}, hidden_dim: {hidden_dim}, embedding_dim: {embedding_dim}")

    # Create model architecture (must match what was trained)
    model = create_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    # Load weights (weights_only=False needed for checkpoints with numpy/extra metadata)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_crps' in checkpoint:
            print(f"  Saved validation CRPS: {checkpoint['val_crps']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def run_evaluation(model, dataloader, device):
    """
    Run evaluation on a dataset.

    Args:
        model: The trained model
        dataloader: DataLoader for evaluation data
        device: Device to run on

    Returns:
        tuple: (all_mu, all_sigma, all_targets) as numpy arrays
    """
    model.eval()

    all_mu = []
    all_sigma = []
    all_targets = []

    for embeddings, targets in tqdm(dataloader, desc="Evaluating"):
        embeddings = embeddings.to(device)

        mu, sigma = model(embeddings)

        all_mu.append(mu.cpu())
        all_sigma.append(sigma.cpu())
        all_targets.append(targets)

    # Concatenate and convert to numpy
    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_sigma = torch.cat(all_sigma, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    return all_mu, all_sigma, all_targets


def print_detailed_results(mu, sigma, targets, target_names=None):
    """
    Print detailed evaluation results.

    Args:
        mu: Predicted means
        sigma: Predicted sigmas
        targets: True values
        target_names: Names for each target
    """
    if target_names is None:
        target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    print("\n" + "=" * 70)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 70)

    # Overall competition score
    rms_crps, per_target_crps = compute_competition_score(mu, sigma, targets)
    print(f"\n{'COMPETITION SCORE (RMS-CRPS)':40s}: {rms_crps:.4f}")

    # Per-target breakdown
    print("\n" + "-" * 70)
    print(f"{'Metric':<15} {'SPEI_30d':>12} {'SPEI_1y':>12} {'SPEI_2y':>12} {'Mean':>12}")
    print("-" * 70)

    # CRPS per target
    crps_row = ["CRPS"]
    crps_values = []
    for name in target_names:
        crps_values.append(per_target_crps[name])
        crps_row.append(f"{per_target_crps[name]:.4f}")
    crps_row.append(f"{np.mean(crps_values):.4f}")
    print(f"{crps_row[0]:<15} {crps_row[1]:>12} {crps_row[2]:>12} {crps_row[3]:>12} {crps_row[4]:>12}")

    # MAE per target
    mae_row = ["MAE"]
    mae_values = []
    for i, name in enumerate(target_names):
        mae = compute_mae(mu[:, i], targets[:, i])
        mae_values.append(mae)
        mae_row.append(f"{mae:.4f}")
    mae_row.append(f"{np.mean(mae_values):.4f}")
    print(f"{mae_row[0]:<15} {mae_row[1]:>12} {mae_row[2]:>12} {mae_row[3]:>12} {mae_row[4]:>12}")

    # RMSE per target
    rmse_row = ["RMSE"]
    rmse_values = []
    for i, name in enumerate(target_names):
        rmse = compute_rmse(mu[:, i], targets[:, i])
        rmse_values.append(rmse)
        rmse_row.append(f"{rmse:.4f}")
    rmse_row.append(f"{np.mean(rmse_values):.4f}")
    print(f"{rmse_row[0]:<15} {rmse_row[1]:>12} {rmse_row[2]:>12} {rmse_row[3]:>12} {rmse_row[4]:>12}")

    # Mean sigma (predicted uncertainty)
    sigma_row = ["Mean Sigma"]
    sigma_means = []
    for i in range(3):
        sigma_mean = np.mean(sigma[:, i])
        sigma_means.append(sigma_mean)
        sigma_row.append(f"{sigma_mean:.4f}")
    sigma_row.append(f"{np.mean(sigma_means):.4f}")
    print(f"{sigma_row[0]:<15} {sigma_row[1]:>12} {sigma_row[2]:>12} {sigma_row[3]:>12} {sigma_row[4]:>12}")

    # Calibration check
    print("\n" + "-" * 70)
    print("UNCERTAINTY CALIBRATION")
    print("-" * 70)
    print(f"{'Target':<15} {'Coverage @ 1σ':>18} {'Coverage @ 2σ':>18}")
    print(f"{'(Expected)':^15} {'(68.27%)':>18} {'(95.45%)':>18}")
    print("-" * 70)

    errors = targets - mu
    for i, name in enumerate(target_names):
        cal = compute_calibration(sigma[:, i], errors[:, i])
        actual_1s = cal['1_sigma']['actual'] * 100
        actual_2s = cal['2_sigma']['actual'] * 100
        print(f"{name:<15} {actual_1s:>17.1f}% {actual_2s:>17.1f}%")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    # Check if uncertainty is calibrated
    avg_1s_coverage = np.mean([
        compute_calibration(sigma[:, i], errors[:, i])['1_sigma']['actual']
        for i in range(3)
    ])

    if avg_1s_coverage < 0.60:
        print("  - Model is OVERCONFIDENT (coverage < expected)")
        print("    Consider: increasing min_sigma or using regularization")
    elif avg_1s_coverage > 0.75:
        print("  - Model is UNDERCONFIDENT (coverage > expected)")
        print("    Consider: the uncertainty estimates are conservative")
    else:
        print("  - Uncertainty estimates appear well-calibrated")

    # Which target is hardest
    hardest_target = max(per_target_crps, key=per_target_crps.get)
    easiest_target = min(per_target_crps, key=per_target_crps.get)
    print(f"  - Easiest to predict: {easiest_target} (CRPS: {per_target_crps[easiest_target]:.4f})")
    print(f"  - Hardest to predict: {hardest_target} (CRPS: {per_target_crps[hardest_target]:.4f})")

    print("=" * 70)


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained drought prediction model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(CHECKPOINTS_DIR / 'best_model.pt'),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='dinov3',
        help='Which backbone embeddings to use (default: dinov3). '
             'Options: dinov3, dinov2, bioclip, bioclip2, or combined like dinov3+bioclip2'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['mean', 'max', 'sum'],
        default='mean',
        help='How to aggregate embeddings per event'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['simple', 'deep'],
        default='simple',
        help='Model architecture (must match trained model)'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=HIDDEN_DIM,
        help='Hidden dimension (must match trained model)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BEETLES DROUGHT PREDICTION - EVALUATION")
    print("=" * 60)

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Please train a model first: python scripts/train.py")
        sys.exit(1)

    # Check embeddings exist for this backbone
    val_embeddings_path = EMBEDDINGS_DIR / f'val_embeddings_{args.backbone}.pt'
    if not val_embeddings_path.exists():
        print(f"ERROR: Validation embeddings not found for backbone '{args.backbone}'!")
        print(f"Expected: {val_embeddings_path}")
        if '+' in args.backbone:
            parts = args.backbone.split('+')
            print(f"\nFor combined backbones, run:")
            print(f"  python scripts/combine_embeddings.py --backbone1 {parts[0]} --backbone2 {parts[1]}")
        else:
            print(f"Please run: python scripts/extract_embeddings.py --backbone {args.backbone}")
        sys.exit(1)

    # Set seed
    set_seed()

    # Get device
    device = get_device()

    # Detect embedding dimension from validation embeddings
    sample_embeddings = torch.load(val_embeddings_path, weights_only=True)
    embedding_dim = sample_embeddings.shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")

    # Load model
    model = load_model(args.checkpoint, device, args.model_type, args.hidden_dim, embedding_dim)

    # Create validation dataset and loader
    print(f"\nLoading validation data (backbone: {args.backbone})...")
    val_dataset = AggregatedEmbeddingDataset(
        embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{args.backbone}.pt',
        targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        event_ids_path=EMBEDDINGS_DIR / 'val_events.pt',
        aggregation=args.aggregation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Validation set: {len(val_dataset)} events")

    # Run evaluation
    print("\nRunning evaluation...")
    mu, sigma, targets = run_evaluation(model, val_loader, device)

    # Print detailed results
    print_detailed_results(mu, sigma, targets)

    # Save predictions for analysis
    output_path = CHECKPOINTS_DIR / 'val_predictions.npz'
    np.savez(
        output_path,
        mu=mu,
        sigma=sigma,
        targets=targets
    )
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()
