"""
Hyperparameter Grid Search for Drought Prediction Models.

This script performs exhaustive grid search over specified hyperparameter combinations
to find the best model configuration for each backbone type.

Supported modes:
    - Standard backbones: dinov3, bioclip2, dinov3+bioclip2
    - Specialized: Separate grid search for SPEI_30d (dinov3) and SPEI_1y/2y (bioclip2)

Usage:
    # Standard backbone grid search
    python scripts/grid_search.py --backbone dinov3

    # Combined backbone grid search
    python scripts/grid_search.py --backbone dinov3+bioclip2

    # Specialized model grid search (runs both dinov3 and bioclip2)
    python scripts/grid_search.py --backbone specialized

The script logs:
    - Best model training curves to TensorBoard (CRPS/SPEI_30d, etc.)
    - Parameter analysis graphs showing each parameter's effect on performance
    - Top 3 model configurations printed to terminal
"""

import sys
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import EMBEDDINGS_DIR, CHECKPOINTS_DIR, LOGS_DIR, get_device, set_seed
from src.data_loader import create_data_loaders
from src.model import create_model
from src.metrics import compute_competition_score
from src.utils import get_timestamp

# =============================================================================
# GRID SEARCH PARAMETER CONFIGURATION
# =============================================================================
# Edit these arrays to specify which values to test for each hyperparameter.
# The grid search will test ALL combinations (Cartesian product).

PARAM_GRID = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'lr_type': ['cosine', 'plateau'],
    'hidden_dim': [64, 128, 256],
    'dropout': [0.0, 0.1, 0.2],
    'model_type': ['simple', 'deep'],
    'num_epochs': [100, 150],
    'batch_size': [32, 64],
    'eta_min': [1e-6, 1e-7],      # Cosine annealing minimum LR
    'size_factor': [0.5, 0.25],   # Deep model layer decay factor
    'weight_decay': [1e-3, 1e-4],
}

# For specialized model grid search, you can use different grids
PARAM_GRID_SPECIALIZED_30D = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'lr_type': ['cosine', 'plateau'],
    'hidden_dim': [64, 128, 256],
    'dropout': [0.0, 0.1, 0.2],
    'model_type': ['simple', 'deep'],
    'num_epochs': [100, 150],
    'batch_size': [32, 64],
    'eta_min': [1e-6, 1e-7],
    'size_factor': [0.5, 0.25],
    'weight_decay': [1e-3, 1e-4],
}

PARAM_GRID_SPECIALIZED_1Y2Y = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'lr_type': ['cosine', 'plateau'],
    'hidden_dim': [64, 128, 256],
    'dropout': [0.0, 0.1, 0.2],
    'model_type': ['simple', 'deep'],
    'num_epochs': [100, 150],
    'batch_size': [32, 64],
    'eta_min': [1e-6, 1e-7],
    'size_factor': [0.5, 0.25],
    'weight_decay': [1e-3, 1e-4],
}

# Minimum sigma for numerical stability
MIN_SIGMA = 1e-3


# =============================================================================
# SPECIALIZED MODEL CLASSES (for specialized grid search)
# =============================================================================

class SpecializedRegressionHead(nn.Module):
    """Simple regression head for specialized targets."""

    def __init__(self, embedding_dim, num_targets, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_targets = num_targets
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_targets * 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + MIN_SIGMA
        return mu, sigma


class DeeperSpecializedRegressionHead(nn.Module):
    """Deeper regression head for specialized targets."""

    def __init__(self, embedding_dim, num_targets, hidden_dim, dropout=0.1, size_factor=0.5):
        super().__init__()
        self.num_targets = num_targets
        h1 = hidden_dim
        h2 = max(1, int(hidden_dim * size_factor))
        h3 = max(1, int(hidden_dim * size_factor * size_factor))
        self.fc1 = nn.Linear(embedding_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_out = nn.Linear(h3, num_targets * 2)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        output = self.fc_out(x)
        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + MIN_SIGMA
        return mu, sigma


def create_specialized_model(embedding_dim, num_targets, hidden_dim, dropout, model_type, size_factor):
    """Factory for specialized models."""
    if model_type == 'simple':
        return SpecializedRegressionHead(embedding_dim, num_targets, hidden_dim, dropout)
    else:
        return DeeperSpecializedRegressionHead(embedding_dim, num_targets, hidden_dim, dropout, size_factor)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def gaussian_nll_loss(mu, sigma, targets):
    """Compute Gaussian negative log-likelihood loss."""
    nll = 0.5 * np.log(2 * np.pi) + torch.log(sigma) + 0.5 * ((targets - mu) / sigma) ** 2
    return nll.mean()


def train_model_with_params(
    model, train_loader, val_loader, params, device,
    target_indices=None, early_stopping_patience=50
):
    """
    Train a model with given hyperparameters.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        params: Dictionary of hyperparameters
        device: Device to train on
        target_indices: If specified, only evaluate on these target indices
        early_stopping_patience: Epochs without improvement before stopping

    Returns:
        dict: Training results including best CRPS and training history
    """
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )

    # Set up scheduler
    if params['lr_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params['num_epochs'],
            eta_min=params['eta_min']
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    best_crps = float('inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        'train_loss': [],
        'val_crps': [],
        'learning_rate': []
    }

    for epoch in range(1, params['num_epochs'] + 1):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0

        for embeddings, all_targets in train_loader:
            embeddings = embeddings.to(device)
            if target_indices is not None:
                targets = all_targets[:, target_indices].to(device)
            else:
                targets = all_targets.to(device)

            optimizer.zero_grad()
            mu, sigma = model(embeddings)
            loss = gaussian_nll_loss(mu, sigma, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches

        # Validation
        model.eval()
        all_mu, all_sigma, all_targets_np = [], [], []

        with torch.no_grad():
            for embeddings, all_targets in val_loader:
                embeddings = embeddings.to(device)
                if target_indices is not None:
                    targets = all_targets[:, target_indices]
                else:
                    targets = all_targets

                mu, sigma = model(embeddings)
                all_mu.append(mu.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())
                all_targets_np.append(targets.numpy())

        all_mu = np.concatenate(all_mu, axis=0)
        all_sigma = np.concatenate(all_sigma, axis=0)
        all_targets_np = np.concatenate(all_targets_np, axis=0)

        # Compute CRPS
        val_crps, _ = compute_competition_score(all_mu, all_sigma, all_targets_np)

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if params['lr_type'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_crps)

        history['train_loss'].append(train_loss)
        history['val_crps'].append(val_crps)
        history['learning_rate'].append(current_lr)

        # Check for improvement
        if val_crps < best_crps:
            best_crps = val_crps
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'best_crps': best_crps,
        'best_epoch': best_epoch,
        'final_epoch': epoch,
        'history': history,
        'model_state': best_state
    }


def get_embedding_dim(backbone):
    """Get embedding dimension from saved embeddings."""
    emb_path = EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt'
    embeddings = torch.load(emb_path, weights_only=True)
    return embeddings.shape[1]


def generate_param_combinations(param_grid):
    """Generate all combinations of parameters from a grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def count_combinations(param_grid):
    """Count total number of parameter combinations."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


# =============================================================================
# GRID SEARCH FUNCTIONS
# =============================================================================

def run_standard_grid_search(backbone, param_grid, device, writer):
    """
    Run grid search for a standard (non-specialized) backbone.

    Args:
        backbone: Backbone name (e.g., 'dinov3', 'bioclip2', 'dinov3+bioclip2')
        param_grid: Dictionary of parameter arrays
        device: Device to train on
        writer: TensorBoard SummaryWriter

    Returns:
        list: Sorted list of (params, crps) tuples (best first)
    """
    print(f"\n{'='*70}")
    print(f"GRID SEARCH: {backbone}")
    print(f"{'='*70}")

    # Check embeddings exist
    emb_path = EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt'
    if not emb_path.exists():
        print(f"ERROR: Embeddings not found for '{backbone}'")
        print(f"Run: python scripts/extract_embeddings.py --backbone {backbone}")
        return []

    embedding_dim = get_embedding_dim(backbone)
    print(f"Embedding dimension: {embedding_dim}")

    total_combinations = count_combinations(param_grid)
    print(f"Total parameter combinations: {total_combinations}")

    results = []
    best_result = None
    best_params = None

    # Progress bar for all combinations
    pbar = tqdm(
        enumerate(generate_param_combinations(param_grid)),
        total=total_combinations,
        desc="Grid Search"
    )

    for run_idx, params in pbar:
        # Create data loaders with current batch size
        train_loader, val_loader = create_data_loaders(
            train_embeddings_path=EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt',
            train_targets_path=EMBEDDINGS_DIR / 'train_targets.pt',
            train_events_path=EMBEDDINGS_DIR / 'train_events.pt',
            val_embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt',
            val_targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
            val_events_path=EMBEDDINGS_DIR / 'val_events.pt',
            batch_size=params['batch_size'],
            aggregated=True,
            aggregation='mean'
        )

        # Create model
        model = create_model(
            model_type=params['model_type'],
            embedding_dim=embedding_dim,
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
            size_factor=params['size_factor']
        )

        # Train
        result = train_model_with_params(model, train_loader, val_loader, params, device)

        # Store results
        results.append({
            'params': params.copy(),
            'crps': result['best_crps'],
            'best_epoch': result['best_epoch'],
            'history': result['history'],
            'model_state': result['model_state']
        })

        # Track best
        if best_result is None or result['best_crps'] < best_result['best_crps']:
            best_result = result
            best_params = params.copy()

        pbar.set_postfix({
            'crps': f"{result['best_crps']:.4f}",
            'best': f"{best_result['best_crps']:.4f}"
        })

    # Sort results by CRPS (best first)
    results.sort(key=lambda x: x['crps'])

    return results, best_params, best_result


def run_specialized_grid_search(param_grid_30d, param_grid_1y2y, device, writer):
    """
    Run grid search for specialized models.

    Runs separate searches for:
        - dinov3 → SPEI_30d
        - bioclip2 → SPEI_1y, SPEI_2y

    Args:
        param_grid_30d: Parameter grid for SPEI_30d model
        param_grid_1y2y: Parameter grid for SPEI_1y/2y model
        device: Device to train on
        writer: TensorBoard SummaryWriter

    Returns:
        tuple: (results_30d, results_1y2y) sorted lists
    """
    print(f"\n{'='*70}")
    print("SPECIALIZED GRID SEARCH")
    print(f"{'='*70}")

    # =========================================================================
    # Grid search for SPEI_30d (dinov3)
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Part 1: dinov3 → SPEI_30d")
    print(f"{'-'*70}")

    backbone1 = 'dinov3'
    emb_path1 = EMBEDDINGS_DIR / f'train_embeddings_{backbone1}.pt'
    if not emb_path1.exists():
        print(f"ERROR: Embeddings not found for '{backbone1}'")
        return [], []

    embedding_dim1 = get_embedding_dim(backbone1)
    total_30d = count_combinations(param_grid_30d)
    print(f"Combinations to test: {total_30d}")

    results_30d = []
    best_crps_30d = float('inf')

    pbar = tqdm(
        enumerate(generate_param_combinations(param_grid_30d)),
        total=total_30d,
        desc="SPEI_30d Search"
    )

    for run_idx, params in pbar:
        train_loader, val_loader = create_data_loaders(
            train_embeddings_path=EMBEDDINGS_DIR / f'train_embeddings_{backbone1}.pt',
            train_targets_path=EMBEDDINGS_DIR / 'train_targets.pt',
            train_events_path=EMBEDDINGS_DIR / 'train_events.pt',
            val_embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{backbone1}.pt',
            val_targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
            val_events_path=EMBEDDINGS_DIR / 'val_events.pt',
            batch_size=params['batch_size'],
            aggregated=True,
            aggregation='mean'
        )

        model = create_specialized_model(
            embedding_dim=embedding_dim1,
            num_targets=1,
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
            model_type=params['model_type'],
            size_factor=params['size_factor']
        )

        result = train_model_with_params(
            model, train_loader, val_loader, params, device,
            target_indices=[0]  # SPEI_30d only
        )

        results_30d.append({
            'params': params.copy(),
            'crps': result['best_crps'],
            'best_epoch': result['best_epoch'],
            'history': result['history'],
            'model_state': result['model_state']
        })

        if result['best_crps'] < best_crps_30d:
            best_crps_30d = result['best_crps']

        pbar.set_postfix({
            'crps': f"{result['best_crps']:.4f}",
            'best': f"{best_crps_30d:.4f}"
        })

    results_30d.sort(key=lambda x: x['crps'])

    # =========================================================================
    # Grid search for SPEI_1y/2y (bioclip2)
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Part 2: bioclip2 → SPEI_1y, SPEI_2y")
    print(f"{'-'*70}")

    backbone2 = 'bioclip2'
    emb_path2 = EMBEDDINGS_DIR / f'train_embeddings_{backbone2}.pt'
    if not emb_path2.exists():
        print(f"ERROR: Embeddings not found for '{backbone2}'")
        return results_30d, []

    embedding_dim2 = get_embedding_dim(backbone2)
    total_1y2y = count_combinations(param_grid_1y2y)
    print(f"Combinations to test: {total_1y2y}")

    results_1y2y = []
    best_crps_1y2y = float('inf')

    pbar = tqdm(
        enumerate(generate_param_combinations(param_grid_1y2y)),
        total=total_1y2y,
        desc="SPEI_1y/2y Search"
    )

    for run_idx, params in pbar:
        train_loader, val_loader = create_data_loaders(
            train_embeddings_path=EMBEDDINGS_DIR / f'train_embeddings_{backbone2}.pt',
            train_targets_path=EMBEDDINGS_DIR / 'train_targets.pt',
            train_events_path=EMBEDDINGS_DIR / 'train_events.pt',
            val_embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{backbone2}.pt',
            val_targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
            val_events_path=EMBEDDINGS_DIR / 'val_events.pt',
            batch_size=params['batch_size'],
            aggregated=True,
            aggregation='mean'
        )

        model = create_specialized_model(
            embedding_dim=embedding_dim2,
            num_targets=2,
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
            model_type=params['model_type'],
            size_factor=params['size_factor']
        )

        result = train_model_with_params(
            model, train_loader, val_loader, params, device,
            target_indices=[1, 2]  # SPEI_1y, SPEI_2y only
        )

        results_1y2y.append({
            'params': params.copy(),
            'crps': result['best_crps'],
            'best_epoch': result['best_epoch'],
            'history': result['history'],
            'model_state': result['model_state']
        })

        if result['best_crps'] < best_crps_1y2y:
            best_crps_1y2y = result['best_crps']

        pbar.set_postfix({
            'crps': f"{result['best_crps']:.4f}",
            'best': f"{best_crps_1y2y:.4f}"
        })

    results_1y2y.sort(key=lambda x: x['crps'])

    return results_30d, results_1y2y


# =============================================================================
# TENSORBOARD LOGGING
# =============================================================================

def log_best_model_training(writer, best_result, prefix='Best'):
    """Log the best model's training history to TensorBoard."""
    history = best_result['history']

    for epoch, (loss, crps, lr) in enumerate(zip(
        history['train_loss'],
        history['val_crps'],
        history['learning_rate']
    ), 1):
        writer.add_scalar(f'{prefix}/Loss/train', loss, epoch)
        writer.add_scalar(f'{prefix}/CRPS/val', crps, epoch)
        writer.add_scalar(f'{prefix}/Learning_Rate', lr, epoch)


def log_parameter_analysis(writer, results, param_grid):
    """
    Log parameter analysis graphs to TensorBoard.

    For each parameter, creates a graph showing the parameter value (x-axis)
    vs performance, sorted from worst (left) to best (right).

    Args:
        writer: TensorBoard SummaryWriter
        results: List of result dictionaries sorted by CRPS (best first)
        param_grid: Parameter grid dictionary
    """
    # Reverse to get worst first (for graphing left to right)
    results_worst_first = list(reversed(results))

    for param_name in param_grid.keys():
        # Extract parameter values and CRPS scores in worst-to-best order
        for idx, result in enumerate(results_worst_first):
            param_value = result['params'][param_name]
            crps = result['crps']

            # For numeric parameters, log the value
            if isinstance(param_value, (int, float)):
                writer.add_scalar(f'ParamAnalysis/{param_name}_value', param_value, idx)
            else:
                # For string parameters (like lr_type, model_type), encode as index
                unique_values = list(param_grid[param_name])
                encoded = unique_values.index(param_value)
                writer.add_scalar(f'ParamAnalysis/{param_name}_encoded', encoded, idx)

            # Always log the CRPS for this run
            writer.add_scalar(f'ParamAnalysis/{param_name}_crps', crps, idx)

    # Log ordered results list as text
    results_text = "# Ranked Results (Best to Worst)\n\n"
    for rank, result in enumerate(results, 1):
        results_text += f"## Rank {rank}: CRPS = {result['crps']:.4f}\n"
        for k, v in result['params'].items():
            results_text += f"- {k}: {v}\n"
        results_text += "\n"

    writer.add_text('Results/RankedList', results_text)


def log_specialized_analysis(writer, results_30d, results_1y2y, param_grid_30d, param_grid_1y2y):
    """Log parameter analysis for specialized models."""
    # Log SPEI_30d analysis
    for param_name in param_grid_30d.keys():
        results_worst_first = list(reversed(results_30d))
        for idx, result in enumerate(results_worst_first):
            param_value = result['params'][param_name]
            crps = result['crps']

            if isinstance(param_value, (int, float)):
                writer.add_scalar(f'ParamAnalysis_30d/{param_name}_value', param_value, idx)
            else:
                unique_values = list(param_grid_30d[param_name])
                encoded = unique_values.index(param_value)
                writer.add_scalar(f'ParamAnalysis_30d/{param_name}_encoded', encoded, idx)
            writer.add_scalar(f'ParamAnalysis_30d/{param_name}_crps', crps, idx)

    # Log SPEI_1y2y analysis
    for param_name in param_grid_1y2y.keys():
        results_worst_first = list(reversed(results_1y2y))
        for idx, result in enumerate(results_worst_first):
            param_value = result['params'][param_name]
            crps = result['crps']

            if isinstance(param_value, (int, float)):
                writer.add_scalar(f'ParamAnalysis_1y2y/{param_name}_value', param_value, idx)
            else:
                unique_values = list(param_grid_1y2y[param_name])
                encoded = unique_values.index(param_value)
                writer.add_scalar(f'ParamAnalysis_1y2y/{param_name}_encoded', encoded, idx)
            writer.add_scalar(f'ParamAnalysis_1y2y/{param_name}_crps', crps, idx)


# =============================================================================
# RESULT PRINTING
# =============================================================================

def print_top_results(results, n=3, title="TOP MODELS"):
    """Print the top N model configurations."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    for rank, result in enumerate(results[:n], 1):
        print(f"\n--- Rank {rank}: CRPS = {result['crps']:.4f} (epoch {result['best_epoch']}) ---")
        for key, value in result['params'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6g}")
            else:
                print(f"  {key}: {value}")


def save_best_model(result, backbone, checkpoint_dir):
    """Save the best model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f'best_model_gridsearch_{backbone}.pt'

    torch.save({
        'model_state_dict': result['model_state'],
        'params': result['params'],
        'crps': result['crps'],
        'best_epoch': result['best_epoch']
    }, path)

    print(f"\nBest model saved to: {path}")
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search for drought prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--backbone',
        type=str,
        required=True,
        choices=['dinov3', 'bioclip2', 'dinov3+bioclip2', 'specialized'],
        help='Backbone to search (or "specialized" for separate 30d/1y2y search)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Custom experiment name (default: auto-generated)'
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = f"gridsearch_{args.backbone}_{get_timestamp()}"

    # Setup TensorBoard
    log_dir = LOGS_DIR / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"\nTensorBoard logs: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}")

    if args.backbone == 'specialized':
        # Run specialized grid search
        results_30d, results_1y2y = run_specialized_grid_search(
            PARAM_GRID_SPECIALIZED_30D,
            PARAM_GRID_SPECIALIZED_1Y2Y,
            device,
            writer
        )

        # Log parameter analysis
        if results_30d and results_1y2y:
            log_specialized_analysis(
                writer, results_30d, results_1y2y,
                PARAM_GRID_SPECIALIZED_30D, PARAM_GRID_SPECIALIZED_1Y2Y
            )

            # Log best models' training curves
            log_best_model_training(writer, results_30d[0], prefix='Best_30d')
            log_best_model_training(writer, results_1y2y[0], prefix='Best_1y2y')

            # Print results
            print_top_results(results_30d, n=3, title="TOP 3 MODELS: dinov3 → SPEI_30d")
            print_top_results(results_1y2y, n=3, title="TOP 3 MODELS: bioclip2 → SPEI_1y/2y")

            # Save best models
            save_best_model(results_30d[0], 'specialized_30d', CHECKPOINTS_DIR)
            save_best_model(results_1y2y[0], 'specialized_1y2y', CHECKPOINTS_DIR)

            # Log ranked lists as text
            text_30d = "# SPEI_30d Ranked Results\n\n"
            for rank, result in enumerate(results_30d, 1):
                text_30d += f"## Rank {rank}: CRPS = {result['crps']:.4f}\n"
                for k, v in result['params'].items():
                    text_30d += f"- {k}: {v}\n"
                text_30d += "\n"
            writer.add_text('Results/SPEI_30d_Ranked', text_30d)

            text_1y2y = "# SPEI_1y2y Ranked Results\n\n"
            for rank, result in enumerate(results_1y2y, 1):
                text_1y2y += f"## Rank {rank}: CRPS = {result['crps']:.4f}\n"
                for k, v in result['params'].items():
                    text_1y2y += f"- {k}: {v}\n"
                text_1y2y += "\n"
            writer.add_text('Results/SPEI_1y2y_Ranked', text_1y2y)

    else:
        # Run standard grid search
        results, best_params, best_result = run_standard_grid_search(
            args.backbone, PARAM_GRID, device, writer
        )

        if results:
            # Log parameter analysis
            log_parameter_analysis(writer, results, PARAM_GRID)

            # Log best model's training curves
            log_best_model_training(writer, results[0])

            # Print top 3
            print_top_results(results, n=3, title=f"TOP 3 MODELS: {args.backbone}")

            # Save best model
            save_best_model(results[0], args.backbone, CHECKPOINTS_DIR)

    writer.close()

    print(f"\n{'='*70}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"\nView results: tensorboard --logdir {log_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
