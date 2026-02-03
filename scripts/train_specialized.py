"""
Train specialized models for drought prediction.

This script trains TWO specialized models:
  - Model 1 (backbone1): Predicts SPEI_30d only (short-term drought)
  - Model 2 (backbone2): Predicts SPEI_1y and SPEI_2y (long-term drought)

The combined predictions are logged to TensorBoard for easy comparison
with standard single-backbone models.

Usage:
    # Train with defaults (dinov3 for 30d, bioclip2 for 1y+2y)
    python scripts/train_specialized.py

    # Custom backbones
    python scripts/train_specialized.py --backbone1 dinov3 --backbone2 bioclip2

Hyperparameters are configured in config_specialized.py for easy editing.

Workflow:
    1. Edit config_specialized.py to set hyperparameters
    2. Run: python scripts/train_specialized.py
    3. View results: tensorboard --logdir logs
    4. Repeat with different hyperparameters
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from main config (paths, device, seed)
from config import (
    EMBEDDINGS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    get_device, set_seed
)

# Import specialized hyperparameters
from config_specialized import (
    MODEL1_HIDDEN_DIM, MODEL1_BATCH_SIZE, MODEL1_DROPOUT, MODEL1_TYPE,
    MODEL2_HIDDEN_DIM, MODEL2_BATCH_SIZE, MODEL2_DROPOUT, MODEL2_TYPE,
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY, MIN_SIGMA, RANDOM_SEED, LR_TYPE
)

from src.data_loader import create_data_loaders
from src.metrics import compute_competition_score
from src.utils import get_timestamp


# =============================================================================
# SPECIALIZED MODELS
# =============================================================================

class SpecializedRegressionHead(nn.Module):
    """
    Simple regression head that predicts only specific SPEI targets.

    Architecture: embedding → hidden → output

    This model outputs predictions for a subset of targets, eliminating
    gradient interference from unrelated targets.

    Args:
        embedding_dim: Size of input embeddings
        num_targets: Number of targets to predict (1 for 30d, 2 for 1y+2y)
        hidden_dim: Size of hidden layer
        dropout: Dropout probability
    """

    def __init__(self, embedding_dim, num_targets, hidden_dim, dropout=0.1):
        super().__init__()

        self.num_targets = num_targets

        # Network layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_targets * 2)  # mu + sigma per target

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better gradient flow."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Forward pass.

        Returns:
            tuple: (mu, sigma) each of shape (batch_size, num_targets)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)

        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + MIN_SIGMA

        return mu, sigma


class DeeperSpecializedRegressionHead(nn.Module):
    """
    Deeper regression head with multiple hidden layers.

    Architecture: embedding → hidden1 → hidden2 → hidden3 → output

    Layer sizes decrease based on size_factor:
    [hidden_dim, hidden_dim*factor, hidden_dim*factor^2]

    Args:
        embedding_dim: Size of input embeddings
        num_targets: Number of targets to predict
        hidden_dim: Size of first hidden layer (subsequent layers are smaller)
        dropout: Dropout probability
        size_factor: Layer size decay factor (default 0.5)
    """

    def __init__(self, embedding_dim, num_targets, hidden_dim, dropout=0.1, size_factor=0.5):
        super().__init__()

        self.num_targets = num_targets

        # Decreasing layer sizes based on size_factor
        h1 = hidden_dim
        h2 = max(1, int(hidden_dim * size_factor))
        h3 = max(1, int(hidden_dim * size_factor * size_factor))

        # Network layers
        self.fc1 = nn.Linear(embedding_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc_out = nn.Linear(h3, num_targets * 2)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through multiple layers."""
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


def create_specialized_model(embedding_dim, num_targets, hidden_dim, dropout, model_type='simple', size_factor=0.5):
    """
    Factory function to create specialized models.

    Args:
        embedding_dim: Size of input embeddings
        num_targets: Number of targets to predict
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
        model_type: 'simple' or 'deep'
        size_factor: For 'deep' models, controls layer size decay (default 0.5)

    Returns:
        nn.Module: The created model
    """
    if model_type == 'simple':
        return SpecializedRegressionHead(embedding_dim, num_targets, hidden_dim, dropout)
    elif model_type == 'deep':
        return DeeperSpecializedRegressionHead(embedding_dim, num_targets, hidden_dim, dropout, size_factor)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'simple' or 'deep'")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def gaussian_nll_loss(mu, sigma, targets):
    """Compute Gaussian negative log-likelihood loss."""
    nll = 0.5 * np.log(2 * np.pi) + torch.log(sigma) + 0.5 * ((targets - mu) / sigma) ** 2
    return nll.mean()


def compute_crps_numpy(mu, sigma, targets):
    """Compute CRPS for Gaussian predictions (numpy arrays)."""
    from scipy import stats
    z = (targets - mu) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return crps.mean()


def train_single_model(
    model,
    train_loader,
    val_loader,
    target_indices,
    target_names,
    device,
    writer,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=EARLY_STOPPING_PATIENCE,
    lr_type=LR_TYPE
):
    """
    Train a single specialized model.

    Args:
        model: SpecializedRegressionHead instance
        train_loader: Training data loader
        val_loader: Validation data loader
        target_indices: Which target indices this model predicts [0], [1,2], etc.
        target_names: Names of targets for logging ['SPEI_30d'] or ['SPEI_1y', 'SPEI_2y']
        device: Device to train on
        writer: TensorBoard SummaryWriter for logging
        num_epochs: Maximum epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        lr_type: Learning rate scheduler type ('cosine' or 'plateau')

    Returns:
        dict: Training results including best model state, metrics, and per-epoch predictions
    """
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    # Set up learning rate scheduler
    if lr_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
    else:  # 'plateau'
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
        'val_crps_per_target': {name: [] for name in target_names},
        'val_predictions': []  # Store predictions at each epoch for combined logging
    }

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0

        for embeddings, all_targets in train_loader:
            embeddings = embeddings.to(device)
            targets = all_targets[:, target_indices].to(device)

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
                targets = all_targets[:, target_indices]

                mu, sigma = model(embeddings)
                all_mu.append(mu.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())
                all_targets_np.append(targets.numpy())

        all_mu = np.concatenate(all_mu, axis=0)
        all_sigma = np.concatenate(all_sigma, axis=0)
        all_targets_np = np.concatenate(all_targets_np, axis=0)

        # Compute CRPS for this model's targets
        crps_values = []
        for i, name in enumerate(target_names):
            crps = compute_crps_numpy(all_mu[:, i], all_sigma[:, i], all_targets_np[:, i])
            crps_values.append(crps)
            history['val_crps_per_target'][name].append(crps)
            # Log per-target CRPS to TensorBoard (same format as standard training)
            writer.add_scalar(f'CRPS/{name}', crps, epoch)

        val_crps = np.mean(crps_values)

        # Update learning rate scheduler
        if lr_type == 'cosine':
            scheduler.step()  # Cosine scheduler steps by epoch
        else:
            scheduler.step(val_crps)  # Plateau scheduler needs metric

        history['train_loss'].append(train_loss)
        history['val_crps'].append(val_crps)

        # Store predictions for this epoch (for combined evaluation later)
        history['val_predictions'].append({
            'mu': all_mu.copy(),
            'sigma': all_sigma.copy(),
            'targets': all_targets_np.copy()
        })

        # Log training loss and learning rate
        writer.add_scalar('Loss/train', train_loss, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Check for improvement
        if val_crps < best_crps:
            best_crps = val_crps
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        crps_str = " | ".join([f"{name}: {crps:.4f}" for name, crps in zip(target_names, crps_values)])
        improved = "✓" if epochs_without_improvement == 0 else ""
        print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | {crps_str} {improved}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best model
    model.load_state_dict(best_state)

    return {
        'model': model,
        'best_crps': best_crps,
        'best_epoch': best_epoch,
        'final_epoch': epoch,
        'history': history,
        'target_indices': target_indices,
        'target_names': target_names
    }


def get_combined_predictions(model1, model2, val_loader1, val_loader2, device):
    """
    Get combined predictions from both specialized models.

    Args:
        model1: Model for SPEI_30d
        model2: Model for SPEI_1y and SPEI_2y
        val_loader1: Validation loader for backbone1
        val_loader2: Validation loader for backbone2
        device: Device to run on

    Returns:
        tuple: (combined_mu, combined_sigma, targets) as numpy arrays
    """
    model1.eval()
    model2.eval()

    # Get predictions from model1 (SPEI_30d)
    mu1_list, sigma1_list = [], []
    targets_list = []

    with torch.no_grad():
        for embeddings, all_targets in val_loader1:
            embeddings = embeddings.to(device)
            mu, sigma = model1(embeddings)
            mu1_list.append(mu.cpu().numpy())
            sigma1_list.append(sigma.cpu().numpy())
            targets_list.append(all_targets.numpy())

    mu1 = np.concatenate(mu1_list, axis=0)  # Shape: (N, 1)
    sigma1 = np.concatenate(sigma1_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)  # Shape: (N, 3)

    # Get predictions from model2 (SPEI_1y, SPEI_2y)
    mu2_list, sigma2_list = [], []

    with torch.no_grad():
        for embeddings, _ in val_loader2:
            embeddings = embeddings.to(device)
            mu, sigma = model2(embeddings)
            mu2_list.append(mu.cpu().numpy())
            sigma2_list.append(sigma.cpu().numpy())

    mu2 = np.concatenate(mu2_list, axis=0)  # Shape: (N, 2)
    sigma2 = np.concatenate(sigma2_list, axis=0)

    # Combine: [30d from model1, 1y and 2y from model2]
    combined_mu = np.concatenate([mu1, mu2], axis=1)  # Shape: (N, 3)
    combined_sigma = np.concatenate([sigma1, sigma2], axis=1)

    return combined_mu, combined_sigma, targets


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_embeddings_exist(backbone):
    """Check if embeddings exist for the specified backbone."""
    train_path = EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt'
    val_path = EMBEDDINGS_DIR / f'val_embeddings_{backbone}.pt'

    if not train_path.exists() or not val_path.exists():
        return False
    return True


def get_embedding_dim(backbone):
    """Get embedding dimension from saved embeddings."""
    emb_path = EMBEDDINGS_DIR / f'train_embeddings_{backbone}.pt'
    embeddings = torch.load(emb_path, weights_only=True)
    return embeddings.shape[1]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train specialized models for SPEI prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hyperparameters are set in config_specialized.py

Example:
    python scripts/train_specialized.py
    python scripts/train_specialized.py --backbone1 dinov3 --backbone2 bioclip2
        """
    )

    parser.add_argument(
        '--backbone1',
        type=str,
        default='dinov3',
        help='Backbone for SPEI_30d prediction (default: dinov3)'
    )
    parser.add_argument(
        '--backbone2',
        type=str,
        default='bioclip2',
        help='Backbone for SPEI_1y/2y prediction (default: bioclip2)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Custom experiment name (default: auto-generated)'
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("SPECIALIZED MODEL TRAINING")
    print("=" * 70)
    print(f"\nModel 1: {args.backbone1} → SPEI_30d")
    print(f"Model 2: {args.backbone2} → SPEI_1y, SPEI_2y")

    # Check embeddings exist
    for backbone in [args.backbone1, args.backbone2]:
        if not check_embeddings_exist(backbone):
            print(f"\nERROR: Embeddings not found for '{backbone}'")
            if '+' in backbone:
                parts = backbone.split('+')
                print(f"Run: python scripts/combine_embeddings.py --backbone1 {parts[0]} --backbone2 {parts[1]}")
            else:
                print(f"Run: python scripts/extract_embeddings.py --backbone {backbone}")
            sys.exit(1)

    # Set seed
    set_seed(RANDOM_SEED)
    device = get_device()

    # Print configuration
    print("\n" + "-" * 70)
    print("CONFIGURATION (from config_specialized.py)")
    print("-" * 70)
    print(f"\nModel 1 ({args.backbone1} → SPEI_30d):")
    print(f"  model_type: {MODEL1_TYPE}")
    print(f"  hidden_dim: {MODEL1_HIDDEN_DIM}")
    print(f"  batch_size: {MODEL1_BATCH_SIZE}")
    print(f"  dropout: {MODEL1_DROPOUT}")

    print(f"\nModel 2 ({args.backbone2} → SPEI_1y, SPEI_2y):")
    print(f"  model_type: {MODEL2_TYPE}")
    print(f"  hidden_dim: {MODEL2_HIDDEN_DIM}")
    print(f"  batch_size: {MODEL2_BATCH_SIZE}")
    print(f"  dropout: {MODEL2_DROPOUT}")

    print(f"\nShared:")
    print(f"  learning_rate: {LEARNING_RATE}")
    print(f"  lr_scheduler: {LR_TYPE}")
    print(f"  max_epochs: {NUM_EPOCHS}")
    print(f"  early_stopping_patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  weight_decay: {WEIGHT_DECAY}")

    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = f"specialized_{args.backbone1}+{args.backbone2}_{get_timestamp()}"

    # Setup TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
    log_dir = LOGS_DIR / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"\nTensorBoard logs: {log_dir}")

    # ==========================================================================
    # TRAIN MODEL 1: backbone1 → SPEI_30d
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"TRAINING MODEL 1: {args.backbone1} → SPEI_30d")
    print("=" * 70)

    # Load data for backbone1
    train_loader1, val_loader1 = create_data_loaders(
        train_embeddings_path=EMBEDDINGS_DIR / f'train_embeddings_{args.backbone1}.pt',
        train_targets_path=EMBEDDINGS_DIR / 'train_targets.pt',
        train_events_path=EMBEDDINGS_DIR / 'train_events.pt',
        val_embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{args.backbone1}.pt',
        val_targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        val_events_path=EMBEDDINGS_DIR / 'val_events.pt',
        batch_size=MODEL1_BATCH_SIZE,
        aggregated=True,
        aggregation='mean'
    )

    embedding_dim1 = get_embedding_dim(args.backbone1)
    print(f"Embedding dimension: {embedding_dim1}")

    # Create and train model 1
    model1 = create_specialized_model(
        embedding_dim=embedding_dim1,
        num_targets=1,  # Only SPEI_30d
        hidden_dim=MODEL1_HIDDEN_DIM,
        dropout=MODEL1_DROPOUT,
        model_type=MODEL1_TYPE
    )
    print(f"Model 1 architecture: {MODEL1_TYPE}")

    print(f"Training Model 1...")
    result1 = train_single_model(
        model=model1,
        train_loader=train_loader1,
        val_loader=val_loader1,
        target_indices=[0],  # SPEI_30d
        target_names=['SPEI_30d'],
        device=device,
        writer=writer
    )

    print(f"\nModel 1 Results:")
    print(f"  Best CRPS (SPEI_30d): {result1['best_crps']:.4f}")
    print(f"  Best epoch: {result1['best_epoch']}")
    print(f"  Final epoch: {result1['final_epoch']}")

    # ==========================================================================
    # TRAIN MODEL 2: backbone2 → SPEI_1y, SPEI_2y
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"TRAINING MODEL 2: {args.backbone2} → SPEI_1y, SPEI_2y")
    print("=" * 70)

    # Load data for backbone2
    train_loader2, val_loader2 = create_data_loaders(
        train_embeddings_path=EMBEDDINGS_DIR / f'train_embeddings_{args.backbone2}.pt',
        train_targets_path=EMBEDDINGS_DIR / 'train_targets.pt',
        train_events_path=EMBEDDINGS_DIR / 'train_events.pt',
        val_embeddings_path=EMBEDDINGS_DIR / f'val_embeddings_{args.backbone2}.pt',
        val_targets_path=EMBEDDINGS_DIR / 'val_targets.pt',
        val_events_path=EMBEDDINGS_DIR / 'val_events.pt',
        batch_size=MODEL2_BATCH_SIZE,
        aggregated=True,
        aggregation='mean'
    )

    embedding_dim2 = get_embedding_dim(args.backbone2)
    print(f"Embedding dimension: {embedding_dim2}")

    # Create and train model 2
    model2 = create_specialized_model(
        embedding_dim=embedding_dim2,
        num_targets=2,  # SPEI_1y and SPEI_2y
        hidden_dim=MODEL2_HIDDEN_DIM,
        dropout=MODEL2_DROPOUT,
        model_type=MODEL2_TYPE
    )
    print(f"Model 2 architecture: {MODEL2_TYPE}")

    print(f"Training Model 2...")
    result2 = train_single_model(
        model=model2,
        train_loader=train_loader2,
        val_loader=val_loader2,
        target_indices=[1, 2],  # SPEI_1y, SPEI_2y
        target_names=['SPEI_1y', 'SPEI_2y'],
        device=device,
        writer=writer
    )

    print(f"\nModel 2 Results:")
    print(f"  Best CRPS (SPEI_1y, SPEI_2y): {result2['best_crps']:.4f}")
    print(f"  Best epoch: {result2['best_epoch']}")
    print(f"  Final epoch: {result2['final_epoch']}")

    # ==========================================================================
    # COMPUTE AND LOG COMBINED RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMBINED ENSEMBLE EVALUATION")
    print("=" * 70)

    # Get combined predictions from best models
    combined_mu, combined_sigma, targets = get_combined_predictions(
        result1['model'], result2['model'],
        val_loader1, val_loader2,
        device
    )

    # Compute competition score
    rms_crps, per_target_crps = compute_competition_score(combined_mu, combined_sigma, targets)

    # Print results
    print(f"\n{'Target':<15} {'CRPS':>12} {'Source':>20}")
    print("-" * 50)
    print(f"{'SPEI_30d':<15} {per_target_crps['SPEI_30d']:>12.4f} {args.backbone1:>20}")
    print(f"{'SPEI_1y':<15} {per_target_crps['SPEI_1y']:>12.4f} {args.backbone2:>20}")
    print(f"{'SPEI_2y':<15} {per_target_crps['SPEI_2y']:>12.4f} {args.backbone2:>20}")
    print("-" * 50)
    print(f"{'RMS-CRPS':<15} {rms_crps:>12.4f}")

    # ==========================================================================
    # LOG PER-EPOCH COMBINED RMS-CRPS TO TENSORBOARD
    # ==========================================================================
    # Compute combined RMS-CRPS for each epoch using stored predictions
    # This allows the specialized model to appear on the same CRPS/val graph
    # as standard models trained with train.py (Trainer class uses 'CRPS/val')

    print("\nLogging combined RMS-CRPS per epoch to TensorBoard...")

    # Only log combined CRPS for epochs where BOTH models were actively training
    # Beyond min_epochs, the combined score would mix active training with stale predictions
    min_epochs = min(result1['final_epoch'], result2['final_epoch'])

    for epoch in range(1, min_epochs + 1):
        # Get predictions for this epoch from both models
        pred1 = result1['history']['val_predictions'][epoch - 1]
        pred2 = result2['history']['val_predictions'][epoch - 1]

        # Combine predictions: [30d from model1, 1y and 2y from model2]
        epoch_mu = np.concatenate([pred1['mu'], pred2['mu']], axis=1)
        epoch_sigma = np.concatenate([pred1['sigma'], pred2['sigma']], axis=1)

        # Reconstruct full targets from stored partial targets
        # pred1['targets'] shape: (N, 1) for SPEI_30d
        # pred2['targets'] shape: (N, 2) for SPEI_1y and SPEI_2y
        full_targets = np.zeros((epoch_mu.shape[0], 3))
        full_targets[:, 0] = pred1['targets'].squeeze()  # SPEI_30d
        full_targets[:, 1:] = pred2['targets']  # SPEI_1y, SPEI_2y

        # Compute combined RMS-CRPS
        epoch_rms_crps, _ = compute_competition_score(epoch_mu, epoch_sigma, full_targets)

        # Log to TensorBoard with same name as standard training uses
        writer.add_scalar('CRPS/val', epoch_rms_crps, epoch)

    # Log individual model training curves (for detailed analysis)
    for epoch, (loss1, crps1) in enumerate(zip(result1['history']['train_loss'], result1['history']['val_crps']), 1):
        writer.add_scalar('Model1_30d/train_loss', loss1, epoch)
        writer.add_scalar('Model1_30d/val_crps', crps1, epoch)

    for epoch, (loss2, crps2) in enumerate(zip(result2['history']['train_loss'], result2['history']['val_crps']), 1):
        writer.add_scalar('Model2_1y2y/train_loss', loss2, epoch)
        writer.add_scalar('Model2_1y2y/val_crps', crps2, epoch)

    # Also log a summary text
    writer.add_text('Summary', f"""
    **Specialized Ensemble Results**

    - Backbone1: {args.backbone1} (SPEI_30d)
    - Backbone2: {args.backbone2} (SPEI_1y, SPEI_2y)

    **Per-Target CRPS:**
    - SPEI_30d: {per_target_crps['SPEI_30d']:.4f}
    - SPEI_1y: {per_target_crps['SPEI_1y']:.4f}
    - SPEI_2y: {per_target_crps['SPEI_2y']:.4f}

    **Combined RMS-CRPS: {rms_crps:.4f}**

    Model 1 config: type={MODEL1_TYPE}, hidden_dim={MODEL1_HIDDEN_DIM}, batch_size={MODEL1_BATCH_SIZE}, dropout={MODEL1_DROPOUT}
    Model 2 config: type={MODEL2_TYPE}, hidden_dim={MODEL2_HIDDEN_DIM}, batch_size={MODEL2_BATCH_SIZE}, dropout={MODEL2_DROPOUT}
    """, min_epochs)

    writer.close()

    # ==========================================================================
    # SAVE MODELS
    # ==========================================================================
    print("\n" + "-" * 70)
    print("SAVING MODELS")
    print("-" * 70)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model 1
    checkpoint1 = {
        'model_state_dict': result1['model'].state_dict(),
        'backbone': args.backbone1,
        'targets': ['SPEI_30d'],
        'target_indices': [0],
        'model_type': MODEL1_TYPE,
        'hidden_dim': MODEL1_HIDDEN_DIM,
        'dropout': MODEL1_DROPOUT,
        'embedding_dim': embedding_dim1,
        'best_crps': result1['best_crps'],
        'best_epoch': result1['best_epoch']
    }
    path1 = CHECKPOINTS_DIR / 'best_model_specialized_30d.pt'
    torch.save(checkpoint1, path1)
    print(f"Model 1 saved: {path1}")

    # Save model 2
    checkpoint2 = {
        'model_state_dict': result2['model'].state_dict(),
        'backbone': args.backbone2,
        'targets': ['SPEI_1y', 'SPEI_2y'],
        'target_indices': [1, 2],
        'model_type': MODEL2_TYPE,
        'hidden_dim': MODEL2_HIDDEN_DIM,
        'dropout': MODEL2_DROPOUT,
        'embedding_dim': embedding_dim2,
        'best_crps': result2['best_crps'],
        'best_epoch': result2['best_epoch']
    }
    path2 = CHECKPOINTS_DIR / 'best_model_specialized_1y_2y.pt'
    torch.save(checkpoint2, path2)
    print(f"Model 2 saved: {path2}")

    # Save combined predictions
    predictions_path = CHECKPOINTS_DIR / 'specialized_ensemble_predictions.npz'
    np.savez(
        predictions_path,
        mu=combined_mu,
        sigma=combined_sigma,
        targets=targets,
        backbone1=args.backbone1,
        backbone2=args.backbone2
    )
    print(f"Predictions saved: {predictions_path}")

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Combined RMS-CRPS: {rms_crps:.4f}")
    print(f"\n  SPEI_30d:  {per_target_crps['SPEI_30d']:.4f} ({args.backbone1})")
    print(f"  SPEI_1y:   {per_target_crps['SPEI_1y']:.4f} ({args.backbone2})")
    print(f"  SPEI_2y:   {per_target_crps['SPEI_2y']:.4f} ({args.backbone2})")
    print(f"\nTo tune hyperparameters:")
    print(f"  1. Edit config_specialized.py")
    print(f"  2. Run: python scripts/train_specialized.py")
    print(f"  3. View: tensorboard --logdir logs")
    print("=" * 70)


if __name__ == "__main__":
    main()
