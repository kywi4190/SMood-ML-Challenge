"""
Ensemble evaluation using Optuna-searched specialized models.

This script loads specialized models from Optuna search checkpoints
(which use FlexibleModel / FlexiblePooledModel) and combines their
predictions into a complete 3-target prediction.

Expected setup:
  - Model 1: Optuna-searched SPEI_30d specialist (e.g., DINOv3 short_term)
  - Model 2: Optuna-searched SPEI_1y/2y specialist (e.g., BioCLIP2 long_term)

Usage:
    # Using default Optuna checkpoint names
    python scripts/ensemble_optuna.py

    # Custom checkpoints and backbones
    python scripts/ensemble_optuna.py \
        --checkpoint1 checkpoints/best_model_optuna_dinov3_short_term.pt \
        --backbone1 dinov3 \
        --checkpoint2 checkpoints/best_model_optuna_bioclip2_long_term.pt \
        --backbone2 bioclip2

    # Compare against a single all-targets model
    python scripts/ensemble_optuna.py --compare_all_targets
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    EMBEDDINGS_DIR, CHECKPOINTS_DIR, TARGET_COLUMNS,
    TARGET_INDICES_SHORT_TERM, TARGET_INDICES_LONG_TERM,
    get_device, set_seed
)
from src.pooling import create_pooling, get_pooling_output_dim
from src.metrics import compute_competition_score

# Import the Optuna model classes
from optuna_search import FlexibleModel, FlexiblePooledModel


def load_optuna_model(checkpoint_path, device):
    """
    Load an Optuna-searched model from its checkpoint.

    Reconstructs the correct model architecture (FlexibleModel or
    FlexiblePooledModel) from the saved model_config.

    Args:
        checkpoint_path: Path to the Optuna checkpoint file
        device: Device to load model to

    Returns:
        tuple: (model, model_config) where model_config contains all
               reconstruction info including training_mode, backbone, etc.
    """
    print(f"Loading Optuna model: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['model_config']
    training_mode = config['training_mode']

    print(f"  Training mode: {training_mode}")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Targets: {config['target_names']}")
    print(f"  Architecture: {config['hidden_dims']}")
    print(f"  CRPS: {checkpoint['val_crps']:.6f} (trial {checkpoint['trial_number']})")

    if training_mode == 'fixed':
        model = FlexibleModel(
            embedding_dim=config['embed_dim'],
            hidden_dims=config['hidden_dims'],
            num_targets=config['num_targets'],
            min_sigma=config['min_sigma'],
            dropout=config['dropout'],
            use_species=config.get('use_species', False),
            num_species=config.get('num_species', 0),
            species_embed_dim=config.get('species_embed_dim', 32),
        )
        print(f"  Aggregation: {config.get('aggregation', 'mean')}")
        if config.get('use_species'):
            print(f"  Species: enabled (embed_dim={config.get('species_embed_dim')})")
    else:
        pooling_type = config['pooling_type']
        pooling_module = create_pooling(
            pooling_type=pooling_type,
            embed_dim=config['embed_dim'],
            hidden_dim=config['pooling_hidden_dim'],
            project_to_original=config.get('project_to_original', False),
        )
        model = FlexiblePooledModel(
            pooling_module=pooling_module,
            pooled_dim=config['pooled_dim'],
            hidden_dims=config['hidden_dims'],
            num_targets=config['num_targets'],
            min_sigma=config['min_sigma'],
            dropout=config['dropout'],
        )
        print(f"  Pooling: {pooling_type} (hidden={config['pooling_hidden_dim']})")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def _aggregate_embeddings(embeddings, events, targets, aggregation, species=None):
    """Aggregate per-image embeddings into per-event embeddings."""
    event_to_indices = defaultdict(list)
    for idx, eid in enumerate(events):
        event_to_indices[eid.item()].append(idx)

    agg_embs = []
    agg_targets = []
    agg_species = []

    for eid, indices in event_to_indices.items():
        event_embs = embeddings[indices]
        if aggregation == 'mean':
            agg = event_embs.mean(dim=0)
        elif aggregation == 'max':
            agg = event_embs.max(dim=0)[0]
        else:
            agg = event_embs.sum(dim=0)

        agg_embs.append(agg)
        agg_targets.append(targets[indices[0]])

        if species is not None:
            mode_sp = torch.mode(species[indices]).values.item()
            agg_species.append(mode_sp)

    result = {
        'embeddings': torch.stack(agg_embs),
        'targets': torch.stack(agg_targets),
    }
    if species is not None:
        result['species'] = torch.tensor(agg_species, dtype=torch.long)
    return result


def _group_events(embeddings, events, targets):
    """Group embeddings by event for learned pooling."""
    event_to_indices = defaultdict(list)
    for idx, eid in enumerate(events):
        event_to_indices[eid.item()].append(idx)

    event_embs = []
    event_targets = []
    for eid, indices in event_to_indices.items():
        event_embs.append(embeddings[indices])
        event_targets.append(targets[indices[0]])

    return event_embs, torch.stack(event_targets)


@torch.no_grad()
def get_predictions(model, model_config, device):
    """
    Get predictions from an Optuna-searched model on validation data.

    Automatically sets up the correct data pipeline (fixed aggregation
    or learned pooling) based on model_config.

    Args:
        model: The loaded model (FlexibleModel or FlexiblePooledModel)
        model_config: Config dict from the Optuna checkpoint
        device: Device to run on

    Returns:
        tuple: (mu, sigma, targets) numpy arrays
    """
    backbone = model_config['backbone']
    training_mode = model_config['training_mode']

    # Load raw data
    val_embeddings = torch.load(
        EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt', weights_only=True)
    val_targets = torch.load(
        EMBEDDINGS_DIR / 'val_targets.pt', weights_only=True)
    val_events = torch.load(
        EMBEDDINGS_DIR / 'val_events.pt', weights_only=True)

    model.eval()

    if training_mode == 'fixed':
        aggregation = model_config.get('aggregation', 'mean')
        use_species = model_config.get('use_species', False)

        val_species = None
        if use_species:
            sp_path = EMBEDDINGS_DIR / 'val_species.pt'
            if sp_path.exists():
                val_species = torch.load(sp_path, weights_only=True)

        data = _aggregate_embeddings(
            val_embeddings, val_events, val_targets, aggregation, val_species)

        embs = data['embeddings'].to(device)

        if use_species and 'species' in data:
            species = data['species'].to(device)
            mu, sigma = model(embs, species)
        else:
            mu, sigma = model(embs)

        targets = data['targets']
        return mu.cpu().numpy(), sigma.cpu().numpy(), targets.numpy()

    else:
        # Learned pooling: process one event at a time
        event_embs, event_targets = _group_events(
            val_embeddings, val_events, val_targets)

        all_mu = []
        all_sigma = []

        for embs in event_embs:
            embs = embs.to(device)
            mu, sigma = model(embs)
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())

        mu = torch.cat(all_mu, dim=0).numpy()
        sigma = torch.cat(all_sigma, dim=0).numpy()
        return mu, sigma, event_targets.numpy()


def combine_specialized_predictions(mu1, sigma1, target_indices1,
                                     mu2, sigma2, target_indices2,
                                     n_targets=3):
    """
    Combine predictions from two specialized models into full 3-target predictions.

    Args:
        mu1, sigma1: Predictions from model 1, shape (n_samples, len(target_indices1))
        target_indices1: Which target columns model 1 predicts (e.g., [0] for SPEI_30d)
        mu2, sigma2: Predictions from model 2, shape (n_samples, len(target_indices2))
        target_indices2: Which target columns model 2 predicts (e.g., [1, 2])
        n_targets: Total number of targets (3)

    Returns:
        tuple: (combined_mu, combined_sigma) with shape (n_samples, 3)
    """
    n_samples = mu1.shape[0]
    combined_mu = np.zeros((n_samples, n_targets))
    combined_sigma = np.ones((n_samples, n_targets))  # Default sigma=1

    for i, idx in enumerate(target_indices1):
        combined_mu[:, idx] = mu1[:, i]
        combined_sigma[:, idx] = sigma1[:, i]

    for i, idx in enumerate(target_indices2):
        combined_mu[:, idx] = mu2[:, i]
        combined_sigma[:, idx] = sigma2[:, i]

    return combined_mu, combined_sigma


def get_target_indices_for_mode(mode):
    """Get the original target column indices for a specialized mode."""
    if mode == 'short_term':
        return TARGET_INDICES_SHORT_TERM
    elif mode == 'long_term':
        return TARGET_INDICES_LONG_TERM
    else:
        return [0, 1, 2]


def infer_mode_from_config(model_config):
    """Infer the target mode from a model_config's target_names."""
    names = model_config['target_names']
    if names == ['SPEI_30d']:
        return 'short_term'
    elif set(names) == {'SPEI_1y', 'SPEI_2y'}:
        return 'long_term'
    else:
        return 'all_targets'


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
        description="Ensemble evaluation using Optuna-searched specialized models"
    )

    # Model 1: short-term specialist
    parser.add_argument(
        '--checkpoint1',
        type=str,
        default=str(CHECKPOINTS_DIR / 'best_model_optuna_dinov3_short_term.pt'),
        help='Optuna checkpoint for short-term specialist'
    )
    parser.add_argument(
        '--backbone1',
        type=str,
        default=None,
        help='Override backbone for model 1 (default: read from checkpoint)'
    )

    # Model 2: long-term specialist
    parser.add_argument(
        '--checkpoint2',
        type=str,
        default=str(CHECKPOINTS_DIR / 'best_model_optuna_bioclip2_long_term.pt'),
        help='Optuna checkpoint for long-term specialist'
    )
    parser.add_argument(
        '--backbone2',
        type=str,
        default=None,
        help='Override backbone for model 2 (default: read from checkpoint)'
    )

    # Optional: compare against all-targets model
    parser.add_argument(
        '--compare_all_targets',
        action='store_true',
        help='Also evaluate an all-targets Optuna model for comparison'
    )
    parser.add_argument(
        '--checkpoint_all',
        type=str,
        default=None,
        help='Optuna checkpoint for all-targets model (auto-detected if not specified)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("OPTUNA SPECIALIZED MODEL ENSEMBLE")
    print("=" * 70)

    set_seed()
    device = get_device()

    # Check checkpoints exist
    cp1 = Path(args.checkpoint1)
    cp2 = Path(args.checkpoint2)

    if not cp1.exists():
        print(f"ERROR: Checkpoint not found: {cp1}")
        print("Train with: python scripts/optuna_search.py --backbone dinov3 --mode short_term")
        sys.exit(1)

    if not cp2.exists():
        print(f"ERROR: Checkpoint not found: {cp2}")
        print("Train with: python scripts/optuna_search.py --backbone bioclip2 --mode long_term")
        sys.exit(1)

    # Load models
    print("\nLoading specialized models...")
    print("-" * 50)

    model1, config1 = load_optuna_model(cp1, device)
    mode1 = infer_mode_from_config(config1)
    indices1 = get_target_indices_for_mode(mode1)

    # Override backbone if specified
    if args.backbone1:
        config1['backbone'] = args.backbone1

    print()
    model2, config2 = load_optuna_model(cp2, device)
    mode2 = infer_mode_from_config(config2)
    indices2 = get_target_indices_for_mode(mode2)

    if args.backbone2:
        config2['backbone'] = args.backbone2

    # Verify target coverage
    all_indices = set(indices1) | set(indices2)
    if len(all_indices) != 3:
        print(f"\nWARNING: Models don't cover all 3 targets!")
        print(f"  Model 1 targets: {config1['target_names']} (indices {indices1})")
        print(f"  Model 2 targets: {config2['target_names']} (indices {indices2})")
        missing = set([0, 1, 2]) - all_indices
        print(f"  Missing indices: {missing}")

    overlap = set(indices1) & set(indices2)
    if overlap:
        print(f"\nWARNING: Overlapping targets: {overlap}")
        print(f"  Model 1 predictions will be used for overlapping targets")

    # Get predictions
    print("\n" + "-" * 50)
    print("Generating predictions...")
    print("-" * 50)

    print(f"\nModel 1 ({config1['backbone']}, {mode1}): {config1['target_names']}")
    mu1, sigma1, targets1 = get_predictions(model1, config1, device)
    print(f"  Predictions: mu={mu1.shape}, sigma={sigma1.shape}")

    print(f"\nModel 2 ({config2['backbone']}, {mode2}): {config2['target_names']}")
    mu2, sigma2, targets2 = get_predictions(model2, config2, device)
    print(f"  Predictions: mu={mu2.shape}, sigma={sigma2.shape}")

    # Load full targets for evaluation
    # Use targets from all_targets mode (all 3 columns)
    full_targets = targets1 if targets1.shape[1] == 3 else targets2
    if full_targets.shape[1] != 3:
        # Neither model has all targets; load separately
        val_targets_raw = torch.load(
            EMBEDDINGS_DIR / 'val_targets.pt', weights_only=True)
        val_events = torch.load(
            EMBEDDINGS_DIR / 'val_events.pt', weights_only=True)
        event_to_indices = defaultdict(list)
        for idx, eid in enumerate(val_events):
            event_to_indices[eid.item()].append(idx)
        agg_targets = []
        for eid in sorted(event_to_indices.keys()):
            agg_targets.append(val_targets_raw[event_to_indices[eid][0]])
        full_targets = torch.stack(agg_targets).numpy()

    # Combine predictions
    print("\n" + "-" * 50)
    print("Combining predictions...")
    print("-" * 50)

    combined_mu, combined_sigma = combine_specialized_predictions(
        mu1, sigma1, indices1,
        mu2, sigma2, indices2
    )

    print(f"Combined predictions shape: {combined_mu.shape}")
    print(f"Target assignment:")
    for i, name in enumerate(TARGET_COLUMNS):
        if i in indices1:
            source = f"{config1['backbone']} {mode1} (Model 1)"
        elif i in indices2:
            source = f"{config2['backbone']} {mode2} (Model 2)"
        else:
            source = "NOT COVERED"
        print(f"  {name}: {source}")

    # Evaluate ensemble
    rms_crps, per_target = print_results(
        combined_mu, combined_sigma, full_targets,
        "OPTUNA SPECIALIZED ENSEMBLE"
    )

    # Optionally compare against all-targets model
    if args.compare_all_targets:
        cp_all = args.checkpoint_all
        if cp_all is None:
            # Try to auto-detect
            candidates = [
                CHECKPOINTS_DIR / f'best_model_optuna_{config1["backbone"]}_all_targets.pt',
                CHECKPOINTS_DIR / f'best_model_optuna_{config2["backbone"]}_all_targets.pt',
            ]
            for c in candidates:
                if c.exists():
                    cp_all = str(c)
                    break

        if cp_all and Path(cp_all).exists():
            print("\n\nLoading all-targets model for comparison...")
            model_all, config_all = load_optuna_model(Path(cp_all), device)
            mu_all, sigma_all, targets_all = get_predictions(
                model_all, config_all, device)
            print_results(
                mu_all, sigma_all, full_targets,
                f"ALL-TARGETS MODEL ({config_all['backbone']})"
            )
        else:
            print("\nNo all-targets checkpoint found for comparison.")

    # Save predictions
    output_path = CHECKPOINTS_DIR / 'optuna_ensemble_predictions.npz'
    np.savez(
        output_path,
        mu=combined_mu,
        sigma=combined_sigma,
        targets=full_targets,
        model1_backbone=config1['backbone'],
        model1_mode=mode1,
        model1_targets=config1['target_names'],
        model1_crps=str(checkpoint_crps(Path(args.checkpoint1))),
        model2_backbone=config2['backbone'],
        model2_mode=mode2,
        model2_targets=config2['target_names'],
        model2_crps=str(checkpoint_crps(Path(args.checkpoint2))),
    )
    print(f"\nPredictions saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ensemble RMS-CRPS: {rms_crps:.4f}")
    print(f"\nThis score combines:")
    print(f"  - {config1['backbone']} ({mode1}) for {', '.join(config1['target_names'])}")
    print(f"  - {config2['backbone']} ({mode2}) for {', '.join(config2['target_names'])}")
    print("\nCompare this to your individual model scores to verify improvement!")
    print("=" * 70)


def checkpoint_crps(path):
    """Quick read of CRPS from a checkpoint without loading full model."""
    try:
        cp = torch.load(path, map_location='cpu', weights_only=False)
        return cp.get('val_crps', 'unknown')
    except Exception:
        return 'unknown'


if __name__ == "__main__":
    main()
