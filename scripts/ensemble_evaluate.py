"""
Ensemble evaluation using target-specific model selection.

This script combines predictions from multiple models, using the best
model for each target based on validation performance.

Based on observed results:
- DINOv3 excels at SPEI_30d (short-term)
- BioCLIP2 excels at SPEI_1y and SPEI_2y (long-term)

Usage:
    # Target-specific selection (use best model per target)
    python scripts/ensemble_evaluate.py --mode select

    # Weighted average (tune weights per target)
    python scripts/ensemble_evaluate.py --mode weighted --weights 0.7 0.3 0.3

    # Simple average
    python scripts/ensemble_evaluate.py --mode average
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import EMBEDDINGS_DIR, CHECKPOINTS_DIR, HIDDEN_DIM, get_device, set_seed
from src.model import create_model
from src.data_loader import AggregatedEmbeddingDataset
from src.metrics import compute_competition_score, compute_crps_per_target
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_model_and_embeddings(checkpoint_path, backbone, device, model_type='simple', hidden_dim=HIDDEN_DIM):
    """Load a model and its corresponding validation embeddings."""
    # Load embeddings to get dimension
    val_emb_path = EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt'
    if not val_emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {val_emb_path}")

    sample_emb = torch.load(val_emb_path, weights_only=True)
    embedding_dim = sample_emb.shape[1]

    # Create and load model
    model = create_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Create dataloader
    val_dataset = AggregatedEmbeddingDataset(
        embeddings_path=val_emb_path,
        targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        event_ids_path=EMBEDDINGS_DIR / 'val_events.pt',
        aggregation='mean'
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return model, val_loader


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get predictions from a model."""
    model.eval()
    all_mu, all_sigma, all_targets = [], [], []

    for embeddings, targets in dataloader:
        embeddings = embeddings.to(device)
        mu, sigma = model(embeddings)
        all_mu.append(mu.cpu())
        all_sigma.append(sigma.cpu())
        all_targets.append(targets)

    return (
        torch.cat(all_mu, dim=0).numpy(),
        torch.cat(all_sigma, dim=0).numpy(),
        torch.cat(all_targets, dim=0).numpy()
    )


def ensemble_select(predictions_dict, target_assignment):
    """
    Select predictions from different models per target.

    Args:
        predictions_dict: {model_name: (mu, sigma, targets)}
        target_assignment: {target_idx: model_name} e.g., {0: 'dinov3', 1: 'bioclip2', 2: 'bioclip2'}

    Returns:
        Combined (mu, sigma, targets)
    """
    first_model = list(predictions_dict.keys())[0]
    mu_ref, sigma_ref, targets = predictions_dict[first_model]
    n_samples, n_targets = mu_ref.shape

    mu_combined = np.zeros_like(mu_ref)
    sigma_combined = np.zeros_like(sigma_ref)

    for target_idx, model_name in target_assignment.items():
        mu, sigma, _ = predictions_dict[model_name]
        mu_combined[:, target_idx] = mu[:, target_idx]
        sigma_combined[:, target_idx] = sigma[:, target_idx]

    return mu_combined, sigma_combined, targets


def ensemble_weighted(predictions_list, weights):
    """
    Weighted average of predictions.

    Args:
        predictions_list: [(mu1, sigma1, targets), (mu2, sigma2, targets), ...]
        weights: [w1, w2, ...] weights for each model (should sum to 1)

    Returns:
        Combined (mu, sigma, targets)
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    mu_combined = sum(w * pred[0] for w, pred in zip(weights, predictions_list))
    # For sigma, combine variances: sigma_combined = sqrt(sum(w^2 * sigma^2))
    sigma_combined = np.sqrt(sum((w ** 2) * (pred[1] ** 2) for w, pred in zip(weights, predictions_list)))
    targets = predictions_list[0][2]

    return mu_combined, sigma_combined, targets


def ensemble_average(predictions_list):
    """Simple average of predictions."""
    n_models = len(predictions_list)
    weights = [1.0 / n_models] * n_models
    return ensemble_weighted(predictions_list, weights)


def print_results(mu, sigma, targets, title="ENSEMBLE RESULTS"):
    """Print evaluation results."""
    target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    rms_crps, per_target_crps = compute_competition_score(mu, sigma, targets)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"\n{'COMPETITION SCORE (RMS-CRPS)':40s}: {rms_crps:.4f}")

    print("\n" + "-" * 60)
    print(f"{'Target':<15} {'CRPS':>12} {'Mean Sigma':>12}")
    print("-" * 60)

    for i, name in enumerate(target_names):
        crps = per_target_crps[name]
        mean_sigma = np.mean(sigma[:, i])
        print(f"{name:<15} {crps:>12.4f} {mean_sigma:>12.4f}")

    print("-" * 60)
    print(f"{'Mean':<15} {np.mean(list(per_target_crps.values())):>12.4f}")
    print("=" * 60)

    return rms_crps, per_target_crps


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation with multiple models")

    parser.add_argument(
        '--mode',
        type=str,
        choices=['select', 'weighted', 'average'],
        default='select',
        help='Ensemble mode: select (best per target), weighted, or average'
    )
    parser.add_argument(
        '--checkpoint1',
        type=str,
        default=str(CHECKPOINTS_DIR / 'best_model_dinov3.pt'),
        help='Checkpoint for model 1 (default: best_model_dinov3.pt)'
    )
    parser.add_argument(
        '--backbone1',
        type=str,
        default='dinov3',
        help='Backbone for model 1'
    )
    parser.add_argument(
        '--hidden_dim1',
        type=int,
        default=128,
        help='Hidden dim for model 1'
    )
    parser.add_argument(
        '--checkpoint2',
        type=str,
        default=str(CHECKPOINTS_DIR / 'best_model_bioclip2.pt'),
        help='Checkpoint for model 2 (default: best_model_bioclip2.pt)'
    )
    parser.add_argument(
        '--backbone2',
        type=str,
        default='bioclip2',
        help='Backbone for model 2'
    )
    parser.add_argument(
        '--hidden_dim2',
        type=int,
        default=64,
        help='Hidden dim for model 2'
    )
    parser.add_argument(
        '--weights',
        type=float,
        nargs=3,
        default=[0.7, 0.3, 0.3],
        help='Weights for model 1 per target [SPEI_30d, SPEI_1y, SPEI_2y] (only for weighted mode)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='simple',
        help='Model architecture type'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ENSEMBLE EVALUATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")

    set_seed()
    device = get_device()

    # Load both models
    print(f"\nLoading model 1: {args.backbone1} (hidden_dim={args.hidden_dim1})")
    model1, loader1 = load_model_and_embeddings(
        args.checkpoint1, args.backbone1, device,
        model_type=args.model_type, hidden_dim=args.hidden_dim1
    )

    print(f"Loading model 2: {args.backbone2} (hidden_dim={args.hidden_dim2})")
    model2, loader2 = load_model_and_embeddings(
        args.checkpoint2, args.backbone2, device,
        model_type=args.model_type, hidden_dim=args.hidden_dim2
    )

    # Get predictions
    print("\nGenerating predictions...")
    pred1 = get_predictions(model1, loader1, device)
    pred2 = get_predictions(model2, loader2, device)

    # Print individual model results for reference
    print_results(pred1[0], pred1[1], pred1[2], f"MODEL 1: {args.backbone1}")
    print_results(pred2[0], pred2[1], pred2[2], f"MODEL 2: {args.backbone2}")

    # Ensemble based on mode
    if args.mode == 'select':
        # Use model1 (dinov3) for SPEI_30d, model2 (bioclip2) for SPEI_1y and SPEI_2y
        predictions_dict = {
            args.backbone1: pred1,
            args.backbone2: pred2
        }
        target_assignment = {
            0: args.backbone1,  # SPEI_30d -> dinov3
            1: args.backbone2,  # SPEI_1y -> bioclip2
            2: args.backbone2   # SPEI_2y -> bioclip2
        }
        mu, sigma, targets = ensemble_select(predictions_dict, target_assignment)
        print(f"\nTarget assignment: SPEI_30d->{args.backbone1}, SPEI_1y->{args.backbone2}, SPEI_2y->{args.backbone2}")

    elif args.mode == 'weighted':
        # Per-target weighted average
        # weights are for model1, model2 gets (1-weight)
        mu = np.zeros_like(pred1[0])
        sigma = np.zeros_like(pred1[1])
        targets = pred1[2]

        for i, w1 in enumerate(args.weights):
            w2 = 1.0 - w1
            mu[:, i] = w1 * pred1[0][:, i] + w2 * pred2[0][:, i]
            sigma[:, i] = np.sqrt((w1 ** 2) * (pred1[1][:, i] ** 2) + (w2 ** 2) * (pred2[1][:, i] ** 2))

        print(f"\nWeights for {args.backbone1}: {args.weights}")

    else:  # average
        mu, sigma, targets = ensemble_average([pred1, pred2])
        print("\nSimple average of both models")

    # Print ensemble results
    rms_crps, per_target = print_results(mu, sigma, targets, "ENSEMBLE COMBINED")

    # Save ensemble predictions
    output_path = CHECKPOINTS_DIR / 'ensemble_predictions.npz'
    np.savez(output_path, mu=mu, sigma=sigma, targets=targets)
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()
