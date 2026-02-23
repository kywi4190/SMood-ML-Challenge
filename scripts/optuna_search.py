"""
Optuna Hyperparameter Optimization for Beetle Drought Prediction Models.

Uses Bayesian optimization (TPE) to efficiently search over:
- Training mode: fixed pooling (mean/max/sum) vs learned pooling (attention/gated/statistics/hybrid)
- Architecture: number of hidden layers (1-5), per-layer neuron count (16-1024)
- Hyperparameters: learning rate, dropout, weight decay, batch size, LR scheduler
- Optional features: species embeddings
- Loss parameters: minimum sigma

The search uses MedianPruner to kill bad trials early and persists results
to a SQLite database so studies can be resumed across sessions.

Usage:
    # Search for best dinov3 model (200 trials)
    python scripts/optuna_search.py --backbone dinov3 --n_trials 200

    # Search for best bioclip2 model
    python scripts/optuna_search.py --backbone bioclip2 --n_trials 200

    # Combined backbone
    python scripts/optuna_search.py --backbone dinov3+bioclip2 --n_trials 200

    # Only search fixed pooling (faster)
    python scripts/optuna_search.py --backbone dinov3 --n_trials 200 --fixed_only

    # Only search learned pooling
    python scripts/optuna_search.py --backbone dinov3 --n_trials 200 --learned_only

    # Specialized: optimize only for SPEI_30d
    python scripts/optuna_search.py --backbone dinov3 --n_trials 200 --mode short_term

    # Resume a previous study
    python scripts/optuna_search.py --backbone dinov3 --n_trials 100 --study_name my_previous_study
"""

import sys
import argparse
import time
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config import (
    EMBEDDINGS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    TARGET_COLUMNS, TARGET_INDICES_SHORT_TERM, TARGET_INDICES_LONG_TERM,
    MIN_SIGMA, get_device, set_seed
)
from src.pooling import create_pooling, get_pooling_output_dim
from src.loss import gaussian_nll_loss
from src.metrics import compute_competition_score
from src.utils import get_timestamp, ensure_dir, format_time


# =============================================================================
# FLEXIBLE MODEL CLASSES
# =============================================================================

class FlexibleModel(nn.Module):
    """
    Regression head with arbitrary layer dimensions and optional species embeddings.

    Unlike the fixed RegressionHead (1 hidden layer) or DeeperRegressionHead
    (3 layers with geometric decay), this model accepts any list of layer sizes,
    allowing Optuna to discover the optimal architecture.

    Args:
        embedding_dim: Input embedding dimension
        hidden_dims: List of hidden layer sizes (e.g., [256, 128, 32])
        num_targets: Number of SPEI targets to predict
        min_sigma: Floor for predicted sigma
        dropout: Dropout probability applied after each hidden layer
        use_species: Whether to include learned species embeddings
        num_species: Number of unique species (excluding unknown at index 0)
        species_embed_dim: Dimension of species embedding vectors
    """

    def __init__(self, embedding_dim, hidden_dims, num_targets, min_sigma, dropout,
                 use_species=False, num_species=0, species_embed_dim=32):
        super().__init__()
        self.use_species = use_species
        self.num_targets = num_targets
        self.min_sigma = min_sigma

        input_dim = embedding_dim
        if use_species:
            self.species_embedding = nn.Embedding(
                num_species + 1, species_embed_dim, padding_idx=0
            )
            input_dim += species_embed_dim

        # Build hidden layers dynamically
        layers = []
        in_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hdim
        self.hidden = nn.Sequential(*layers)

        # Output: mu + sigma per target
        self.output = nn.Linear(in_dim, num_targets * 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, species_idx=None):
        if self.use_species and species_idx is not None:
            species_emb = self.species_embedding(species_idx)
            x = torch.cat([x, species_emb], dim=-1)
        x = self.hidden(x)
        out = self.output(x)
        mu = out[:, :self.num_targets]
        sigma_raw = out[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + self.min_sigma
        return mu, sigma


class FlexiblePooledModel(nn.Module):
    """
    Learned pooling + flexible regression head.

    Pools variable-length image embeddings (N_images, embed_dim) into a
    single representation, then passes through a flexible MLP head.

    Args:
        pooling_module: A pooling nn.Module (attention, gated_attention, etc.)
        pooled_dim: Output dimension of the pooling module
        hidden_dims: List of hidden layer sizes for the regression head
        num_targets: Number of SPEI targets to predict
        min_sigma: Floor for predicted sigma
        dropout: Dropout probability
    """

    def __init__(self, pooling_module, pooled_dim, hidden_dims, num_targets,
                 min_sigma, dropout):
        super().__init__()
        self.pooling = pooling_module
        self.num_targets = num_targets
        self.min_sigma = min_sigma

        layers = []
        in_dim = pooled_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hdim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, num_targets * 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, embeddings):
        pooled = self.pooling(embeddings, keepdim=True)  # (1, pooled_dim)
        x = self.hidden(pooled)
        out = self.output(x)
        mu = out[:, :self.num_targets]
        sigma_raw = out[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + self.min_sigma
        return mu, sigma


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class DataManager:
    """
    Pre-loads all data once and creates data loaders on demand.

    Avoids redundant disk I/O across trials by caching:
    - Raw embeddings, targets, events, species tensors
    - Pre-aggregated datasets for each fixed pooling method (mean/max/sum)
    - Event-grouped data for learned pooling
    """

    def __init__(self, backbone, embeddings_dir, mode='all_targets'):
        self.backbone = backbone
        self.embeddings_dir = Path(embeddings_dir)
        self.mode = mode

        print(f"Loading data for backbone: {backbone}")

        # Load raw tensors
        self.train_embeddings = torch.load(
            self.embeddings_dir / f'train_embeddings_{backbone}.pt', weights_only=True)
        self.train_targets = torch.load(
            self.embeddings_dir / 'train_targets.pt', weights_only=True)
        self.train_events = torch.load(
            self.embeddings_dir / 'train_events.pt', weights_only=True)
        self.val_embeddings = torch.load(
            self.embeddings_dir / f'val_embeddings_{backbone}.pt', weights_only=True)
        self.val_targets = torch.load(
            self.embeddings_dir / 'val_targets.pt', weights_only=True)
        self.val_events = torch.load(
            self.embeddings_dir / 'val_events.pt', weights_only=True)

        self.embed_dim = self.train_embeddings.shape[1]

        # Species data (optional)
        self.has_species = False
        self.num_species = 0
        self.train_species = None
        self.val_species = None
        sp_train = self.embeddings_dir / 'train_species.pt'
        sp_val = self.embeddings_dir / 'val_species.pt'
        vocab_path = self.embeddings_dir / 'species_vocab.json'
        if sp_train.exists() and sp_val.exists() and vocab_path.exists():
            self.train_species = torch.load(sp_train, weights_only=True)
            self.val_species = torch.load(sp_val, weights_only=True)
            with open(vocab_path) as f:
                self.num_species = len(json.load(f)) - 1  # exclude <unknown>
            self.has_species = True
            print(f"  Species data loaded: {self.num_species} species + unknown")

        # Target configuration based on mode
        if mode == 'short_term':
            self.target_indices = TARGET_INDICES_SHORT_TERM
            self.target_names = [TARGET_COLUMNS[i] for i in self.target_indices]
        elif mode == 'long_term':
            self.target_indices = TARGET_INDICES_LONG_TERM
            self.target_names = [TARGET_COLUMNS[i] for i in self.target_indices]
        else:
            self.target_indices = None
            self.target_names = list(TARGET_COLUMNS)
        self.num_targets = len(self.target_names)

        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Targets: {self.target_names}")
        print(f"  Train images: {len(self.train_embeddings)}")
        print(f"  Val images: {len(self.val_embeddings)}")

        # Pre-compute cached datasets
        print("  Pre-computing aggregated datasets...")
        self._precompute_aggregated()
        print("  Pre-computing event datasets...")
        self._precompute_events()
        print("  Data ready.\n")

    def _slice_targets(self, targets_tensor):
        """Slice targets to only include relevant columns for the mode."""
        if self.target_indices is not None:
            return targets_tensor[:, self.target_indices]
        return targets_tensor

    def _slice_target_single(self, target_tensor):
        """Slice a single target vector."""
        if self.target_indices is not None:
            return target_tensor[self.target_indices]
        return target_tensor

    def _precompute_aggregated(self):
        """Pre-compute aggregated (per-event) embeddings for fixed pooling."""
        self.aggregated_cache = {}

        for agg_method in ['mean', 'max', 'sum']:
            for split in ['train', 'val']:
                embeddings = self.train_embeddings if split == 'train' else self.val_embeddings
                targets = self.train_targets if split == 'train' else self.val_targets
                events = self.train_events if split == 'train' else self.val_events
                species = self.train_species if split == 'train' else self.val_species

                event_to_indices = defaultdict(list)
                for idx, eid in enumerate(events):
                    event_to_indices[eid.item()].append(idx)

                agg_embs = []
                agg_targets = []
                agg_species = []

                for eid, indices in event_to_indices.items():
                    event_embs = embeddings[indices]
                    if agg_method == 'mean':
                        agg = event_embs.mean(dim=0)
                    elif agg_method == 'max':
                        agg = event_embs.max(dim=0)[0]
                    else:
                        agg = event_embs.sum(dim=0)

                    t = self._slice_target_single(targets[indices[0]])
                    agg_embs.append(agg)
                    agg_targets.append(t)

                    if species is not None:
                        mode_sp = torch.mode(species[indices]).values.item()
                        agg_species.append(mode_sp)

                self.aggregated_cache[(split, agg_method)] = {
                    'embeddings': torch.stack(agg_embs),
                    'targets': torch.stack(agg_targets),
                    'species': torch.tensor(agg_species, dtype=torch.long) if species is not None else None,
                }

        n_train = len(self.aggregated_cache[('train', 'mean')]['embeddings'])
        n_val = len(self.aggregated_cache[('val', 'mean')]['embeddings'])
        print(f"    Aggregated: {n_train} train events, {n_val} val events")

    def _precompute_events(self):
        """Group embeddings by event for learned pooling."""
        self.train_event_embeddings = []
        self.train_event_targets = []
        self.val_event_embeddings = []
        self.val_event_targets = []

        for split in ['train', 'val']:
            embeddings = self.train_embeddings if split == 'train' else self.val_embeddings
            targets = self.train_targets if split == 'train' else self.val_targets
            events = self.train_events if split == 'train' else self.val_events

            event_to_indices = defaultdict(list)
            for idx, eid in enumerate(events):
                event_to_indices[eid.item()].append(idx)

            event_embs_list = []
            event_targets_list = []

            for eid, indices in event_to_indices.items():
                event_embs_list.append(embeddings[indices])
                t = self._slice_target_single(targets[indices[0]])
                event_targets_list.append(t)

            if split == 'train':
                self.train_event_embeddings = event_embs_list
                self.train_event_targets = torch.stack(event_targets_list)
            else:
                self.val_event_embeddings = event_embs_list
                self.val_event_targets = torch.stack(event_targets_list)

        print(f"    Events: {len(self.train_event_embeddings)} train, "
              f"{len(self.val_event_embeddings)} val")

    def get_fixed_loaders(self, aggregation, batch_size, use_species=False):
        """Create DataLoaders for fixed pooling training."""
        from torch.utils.data import TensorDataset, DataLoader

        train_data = self.aggregated_cache[('train', aggregation)]
        val_data = self.aggregated_cache[('val', aggregation)]

        if use_species and train_data['species'] is not None:
            train_ds = TensorDataset(
                train_data['embeddings'], train_data['targets'], train_data['species'])
            val_ds = TensorDataset(
                val_data['embeddings'], val_data['targets'], val_data['species'])
        else:
            train_ds = TensorDataset(train_data['embeddings'], train_data['targets'])
            val_ds = TensorDataset(val_data['embeddings'], val_data['targets'])

        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
        return train_loader, val_loader

    def get_event_loaders(self):
        """Create DataLoaders for learned pooling training (batch_size=1)."""
        from torch.utils.data import Dataset, DataLoader

        class _EventDataset(Dataset):
            def __init__(self, event_embeddings, event_targets):
                self.event_embeddings = event_embeddings
                self.event_targets = event_targets

            def __len__(self):
                return len(self.event_embeddings)

            def __getitem__(self, idx):
                return self.event_embeddings[idx], self.event_targets[idx]

        def _collate(batch):
            embs = [b[0] for b in batch]
            tgts = torch.stack([b[1] for b in batch])
            if len(batch) == 1:
                return embs[0], tgts
            return embs, tgts

        train_ds = _EventDataset(self.train_event_embeddings, self.train_event_targets)
        val_ds = _EventDataset(self.val_event_embeddings, self.val_event_targets)

        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_collate)
        return train_loader, val_loader


# =============================================================================
# BEST MODEL TRACKING
# =============================================================================

class BestModelTracker:
    """
    Holds onto the single best model found during the search.

    Previous best states are dereferenced and garbage-collected when a
    new best is found, so memory usage stays constant.
    """

    def __init__(self):
        self.best_crps = float('inf')
        self.best_params = None
        self.best_state = None
        self.best_config = None
        self.best_trial_number = None

    def update(self, crps, params, state_dict, config, trial_number):
        if crps < self.best_crps:
            self.best_crps = crps
            self.best_params = params.copy()
            self.best_state = {k: v.cpu().clone() for k, v in state_dict.items()}
            self.best_config = config.copy()
            self.best_trial_number = trial_number
            return True
        return False


# =============================================================================
# TRAINING
# =============================================================================

def train_and_evaluate(model, train_loader, val_loader, config, device, trial,
                       num_targets, target_names, is_learned_pooling=False):
    """
    Train a model and return (best_crps, best_state_dict).

    Reports validation CRPS to the Optuna trial each epoch for pruning.
    Uses early stopping internally to avoid wasting epochs.
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    if config['lr_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['max_epochs'], eta_min=config['eta_min']
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    best_crps = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, config['max_epochs'] + 1):
        # ---- TRAIN ----
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if is_learned_pooling:
                embeddings, targets = batch
                embeddings = embeddings.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                mu, sigma = model(embeddings)
            else:
                if len(batch) == 3:
                    embeddings, targets, species = batch
                    embeddings = embeddings.to(device)
                    targets = targets.to(device)
                    species = species.to(device)
                    optimizer.zero_grad()
                    mu, sigma = model(embeddings, species)
                else:
                    embeddings, targets = batch
                    embeddings = embeddings.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    mu, sigma = model(embeddings)

            loss = gaussian_nll_loss(mu, sigma, targets)
            loss.backward()

            if config['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config['grad_clip'])

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # ---- VALIDATE ----
        model.eval()
        all_mu, all_sigma, all_targets = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                if is_learned_pooling:
                    embeddings, targets = batch
                    embeddings = embeddings.to(device)
                    targets = targets.to(device)
                    mu, sigma = model(embeddings)
                else:
                    if len(batch) == 3:
                        embeddings, targets, species = batch
                        embeddings = embeddings.to(device)
                        targets = targets.to(device)
                        species = species.to(device)
                        mu, sigma = model(embeddings, species)
                    else:
                        embeddings, targets = batch
                        embeddings = embeddings.to(device)
                        targets = targets.to(device)
                        mu, sigma = model(embeddings)

                all_mu.append(mu.cpu())
                all_sigma.append(sigma.cpu())
                all_targets.append(targets.cpu())

        all_mu = torch.cat(all_mu).numpy()
        all_sigma = torch.cat(all_sigma).numpy()
        all_targets_np = torch.cat(all_targets).numpy()

        val_crps, _ = compute_competition_score(
            all_mu, all_sigma, all_targets_np, target_names)

        # Scheduler step
        if config['lr_type'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_crps)

        # Track best
        if val_crps < best_crps:
            best_crps = val_crps
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Report to Optuna for pruning
        trial.report(val_crps, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if epochs_no_improve >= config['patience']:
            break

    return best_crps, best_state


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def create_objective(data_manager, device, search_config, tracker):
    """
    Create the Optuna objective function as a closure.

    The closure captures data_manager, device, search_config, and tracker
    so the objective signature matches what Optuna expects: f(trial) -> float.
    """
    dm = data_manager

    def objective(trial):
        # ----- TRAINING MODE -----
        if search_config['fixed_only']:
            training_mode = 'fixed'
        elif search_config['learned_only']:
            training_mode = 'learned'
        else:
            training_mode = trial.suggest_categorical(
                'training_mode', ['fixed', 'learned'])

        # ----- COMMON HYPERPARAMETERS -----
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        lr_type = trial.suggest_categorical('lr_type', ['cosine', 'plateau'])
        min_sigma = trial.suggest_float('min_sigma', 1e-4, 1e-2, log=True)

        eta_min = 1e-7
        if lr_type == 'cosine':
            eta_min = trial.suggest_float('eta_min', 1e-8, 1e-4, log=True)

        use_grad_clip = trial.suggest_categorical('use_grad_clip', [True, False])
        grad_clip = None
        if use_grad_clip:
            grad_clip = trial.suggest_float('grad_clip_norm', 0.1, 10.0, log=True)

        # ----- ARCHITECTURE (shared across modes) -----
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f'layer_{i}_dim', 16, 1024, log=True)
            hidden_dims.append(dim)

        # ----- TRAINING CONFIG -----
        config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'lr_type': lr_type,
            'eta_min': eta_min,
            'max_epochs': search_config['max_epochs'],
            'patience': search_config['patience'],
            'grad_clip': grad_clip,
        }

        # ----- BUILD MODEL & LOADERS -----
        model_config = {
            'training_mode': training_mode,
            'hidden_dims': hidden_dims,
            'num_targets': dm.num_targets,
            'target_names': dm.target_names,
            'embed_dim': dm.embed_dim,
            'min_sigma': min_sigma,
            'dropout': dropout,
            'backbone': dm.backbone,
        }

        if training_mode == 'fixed':
            aggregation = trial.suggest_categorical(
                'aggregation', ['mean', 'max', 'sum'])
            batch_size = trial.suggest_categorical(
                'batch_size', [16, 32, 64, 128])

            # Species (optional)
            use_species = False
            species_embed_dim = 0
            if dm.has_species and not search_config['no_species']:
                use_species = trial.suggest_categorical(
                    'use_species', [True, False])
                if use_species:
                    species_embed_dim = trial.suggest_int(
                        'species_embed_dim', 8, 64, log=True)

            model_config.update({
                'aggregation': aggregation,
                'batch_size': batch_size,
                'use_species': use_species,
                'species_embed_dim': species_embed_dim,
                'num_species': dm.num_species,
            })

            model = FlexibleModel(
                embedding_dim=dm.embed_dim,
                hidden_dims=hidden_dims,
                num_targets=dm.num_targets,
                min_sigma=min_sigma,
                dropout=dropout,
                use_species=use_species,
                num_species=dm.num_species,
                species_embed_dim=species_embed_dim,
            )

            train_loader, val_loader = dm.get_fixed_loaders(
                aggregation, batch_size, use_species)
            is_learned = False

        else:  # learned pooling
            pooling_type = trial.suggest_categorical(
                'pooling_type',
                ['attention', 'gated_attention', 'statistics', 'hybrid'])
            pooling_hidden_dim = trial.suggest_int(
                'pooling_hidden_dim', 16, 256, log=True)

            project_to_original = False
            if pooling_type in ['statistics', 'hybrid']:
                project_to_original = trial.suggest_categorical(
                    'project_to_original', [True, False])

            pooling_module = create_pooling(
                pooling_type=pooling_type,
                embed_dim=dm.embed_dim,
                hidden_dim=pooling_hidden_dim,
                project_to_original=project_to_original,
            )
            pooled_dim = get_pooling_output_dim(
                pooling_type, dm.embed_dim, project_to_original)

            model_config.update({
                'pooling_type': pooling_type,
                'pooling_hidden_dim': pooling_hidden_dim,
                'project_to_original': project_to_original,
                'pooled_dim': pooled_dim,
            })

            model = FlexiblePooledModel(
                pooling_module=pooling_module,
                pooled_dim=pooled_dim,
                hidden_dims=hidden_dims,
                num_targets=dm.num_targets,
                min_sigma=min_sigma,
                dropout=dropout,
            )

            train_loader, val_loader = dm.get_event_loaders()
            is_learned = True

        # ----- TRAIN -----
        best_crps, best_state = train_and_evaluate(
            model, train_loader, val_loader, config, device, trial,
            dm.num_targets, dm.target_names, is_learned_pooling=is_learned)

        # ----- TRACK BEST -----
        full_config = {**config, **model_config}
        if best_state is not None:
            tracker.update(
                best_crps, trial.params, best_state, full_config, trial.number)

        return best_crps

    return objective


# =============================================================================
# RESULT REPORTING & SAVING
# =============================================================================

def _infer_training_mode(params):
    """Infer training mode from trial params (handles forced modes)."""
    if 'training_mode' in params:
        return params['training_mode']
    # When mode is forced, it's not in params. Infer from mode-specific keys.
    if 'pooling_type' in params:
        return 'learned'
    return 'fixed'


def format_trial_summary(trial):
    """Format a single trial's parameters for display."""
    p = trial.params
    mode = _infer_training_mode(p)
    dims = [p.get(f'layer_{i}_dim', '?') for i in range(p.get('n_layers', 0))]
    dims_str = ', '.join(str(d) for d in dims)

    parts = [f"CRPS: {trial.value:.4f}"]

    if mode == 'fixed':
        agg = p.get('aggregation', '?')
        bs = p.get('batch_size', '?')
        species = p.get('use_species', False)
        parts.append(f"fixed/{agg} bs={bs}")
        if species:
            parts.append(f"species(d={p.get('species_embed_dim', '?')})")
    else:
        pool = p.get('pooling_type', '?')
        ph = p.get('pooling_hidden_dim', '?')
        parts.append(f"learned/{pool} ph={ph}")

    parts.append(f"arch=[{dims_str}]")
    parts.append(f"lr={p.get('learning_rate', 0):.1e}")
    parts.append(f"drop={p.get('dropout', 0):.2f}")
    parts.append(f"wd={p.get('weight_decay', 0):.1e}")
    parts.append(f"{p.get('lr_type', '?')}")

    return ' | '.join(parts)


def print_study_results(study, tracker, n_top=10):
    """Print comprehensive results after the study completes."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials
              if t.state == optuna.trial.TrialState.FAIL]

    print(f"\n{'='*80}")
    print("OPTUNA SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"  Total trials:     {len(study.trials)}")
    print(f"  Completed:        {len(completed)}")
    print(f"  Pruned:           {len(pruned)}")
    print(f"  Failed:           {len(failed)}")

    if not completed:
        print("\nNo completed trials. Cannot report results.")
        return

    # Top N trials
    sorted_trials = sorted(completed, key=lambda t: t.value)
    n_show = min(n_top, len(sorted_trials))

    print(f"\n{'-'*80}")
    print(f"TOP {n_show} CONFIGURATIONS")
    print(f"{'-'*80}")

    for rank, trial in enumerate(sorted_trials[:n_show], 1):
        print(f"\n  #{rank:2d} (Trial {trial.number:4d}) {format_trial_summary(trial)}")

    # Best configuration details
    best = study.best_trial
    print(f"\n{'='*80}")
    print(f"BEST MODEL (Trial {best.number})")
    print(f"{'='*80}")
    print(f"  Validation CRPS: {best.value:.6f}")
    print()

    p = best.params
    mode = _infer_training_mode(p)

    print(f"  Training Mode: {mode}")
    if mode == 'fixed':
        print(f"    Aggregation:  {p.get('aggregation', 'N/A')}")
        print(f"    Batch Size:   {p.get('batch_size', 'N/A')}")
        use_sp = p.get('use_species', False)
        print(f"    Use Species:  {use_sp}", end='')
        if use_sp:
            print(f" (embed_dim={p.get('species_embed_dim', 'N/A')})")
        else:
            print()
    else:
        print(f"    Pooling Type:        {p.get('pooling_type', 'N/A')}")
        print(f"    Pooling Hidden Dim:  {p.get('pooling_hidden_dim', 'N/A')}")
        proj = p.get('project_to_original', False)
        if p.get('pooling_type') in ['statistics', 'hybrid']:
            print(f"    Project to Original: {proj}")

    n_layers = p.get('n_layers', 0)
    print(f"\n  Architecture: {n_layers} hidden layer(s)")
    for i in range(n_layers):
        dim = p.get(f'layer_{i}_dim', '?')
        print(f"    Layer {i}: {dim} neurons")

    print(f"\n  Optimizer:")
    print(f"    Learning Rate: {p.get('learning_rate', 0):.6e}")
    print(f"    Weight Decay:  {p.get('weight_decay', 0):.6e}")
    print(f"    LR Scheduler:  {p.get('lr_type', 'N/A')}")
    if p.get('lr_type') == 'cosine':
        print(f"    Eta Min:       {p.get('eta_min', 0):.6e}")
    print(f"    Dropout:       {p.get('dropout', 0):.4f}")
    use_gc = p.get('use_grad_clip', False)
    print(f"    Grad Clip:     ", end='')
    if use_gc:
        print(f"{p.get('grad_clip_norm', 0):.4f}")
    else:
        print("disabled")
    print(f"    Min Sigma:     {p.get('min_sigma', 0):.6e}")

    print(f"{'='*80}")


def save_best_checkpoint(tracker, save_path, study_name):
    """Save the best model checkpoint with full reconstruction info."""
    if tracker.best_state is None:
        print("No model to save (no completed trials).")
        return

    checkpoint = {
        'model_state_dict': tracker.best_state,
        'val_crps': tracker.best_crps,
        'trial_number': tracker.best_trial_number,
        'trial_params': tracker.best_params,
        'model_config': tracker.best_config,
        'study_name': study_name,
    }

    ensure_dir(save_path.parent)
    torch.save(checkpoint, save_path)
    print(f"\nBest model saved to: {save_path}")
    print(f"  CRPS: {tracker.best_crps:.6f}")
    print(f"  Trial: {tracker.best_trial_number}")


def save_results_json(study, save_path):
    """Save all trial results to a JSON file for analysis."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]

    results = []
    for trial in sorted(completed, key=lambda t: t.value):
        results.append({
            'trial_number': trial.number,
            'crps': trial.value,
            'params': trial.params,
            'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
        })

    ensure_dir(save_path.parent)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {save_path}")


# =============================================================================
# TRIAL PROGRESS CALLBACK
# =============================================================================

def make_progress_callback(n_trials, start_time):
    """Create a callback that prints formatted progress after each trial."""
    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            status = "PRUNED"
            value_str = "N/A"
        elif trial.state == optuna.trial.TrialState.COMPLETE:
            status = "OK"
            value_str = f"{trial.value:.4f}"
        else:
            status = str(trial.state.name)
            value_str = "N/A"

        completed = len([t for t in study.trials
                         if t.state in (optuna.trial.TrialState.COMPLETE,
                                        optuna.trial.TrialState.PRUNED)])
        best_val = study.best_value if study.best_trial else float('inf')

        elapsed = time.time() - start_time
        avg_per_trial = elapsed / max(completed, 1)
        remaining = avg_per_trial * max(n_trials - completed, 0)

        # Build summary
        p = trial.params
        mode = _infer_training_mode(p)
        dims = [p.get(f'layer_{i}_dim', '?') for i in range(p.get('n_layers', 0))]
        dims_str = ','.join(str(d) for d in dims)

        if mode == 'fixed':
            mode_detail = f"fixed/{p.get('aggregation', '?')}"
        else:
            mode_detail = f"learned/{p.get('pooling_type', '?')}"

        is_best = (trial.state == optuna.trial.TrialState.COMPLETE
                   and trial.value <= best_val)
        star = " *" if is_best else ""

        print(f"[{completed:4d}/{n_trials}] "
              f"CRPS: {value_str:>8s} | "
              f"Best: {best_val:.4f}{star} | "
              f"{mode_detail:<22s} | "
              f"[{dims_str}] | "
              f"{status:<7s} | "
              f"ETA: {format_time(remaining)}")

    return callback


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for drought prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optuna_search.py --backbone dinov3 --n_trials 200
  python scripts/optuna_search.py --backbone bioclip2 --n_trials 300 --fixed_only
  python scripts/optuna_search.py --backbone dinov3 --n_trials 100 --study_name prev_study
        """
    )
    parser.add_argument(
        '--backbone', type=str, required=True,
        help='Backbone embeddings to use (dinov3, bioclip2, dinov3+bioclip2, etc.)')
    parser.add_argument(
        '--n_trials', type=int, default=200,
        help='Number of Optuna trials to run (default: 200)')
    parser.add_argument(
        '--max_epochs', type=int, default=200,
        help='Maximum training epochs per trial (default: 200)')
    parser.add_argument(
        '--patience', type=int, default=30,
        help='Early stopping patience per trial (default: 30)')
    parser.add_argument(
        '--mode', type=str, default='all_targets',
        choices=['all_targets', 'short_term', 'long_term'],
        help='Which targets to predict (default: all_targets)')
    parser.add_argument(
        '--fixed_only', action='store_true',
        help='Only search fixed pooling (skip learned pooling)')
    parser.add_argument(
        '--learned_only', action='store_true',
        help='Only search learned pooling (skip fixed pooling)')
    parser.add_argument(
        '--no_species', action='store_true',
        help='Exclude species embeddings from search')
    parser.add_argument(
        '--study_name', type=str, default=None,
        help='Study name for persistence/resumption (default: auto-generated)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)')
    parser.add_argument(
        '--n_startup_trials', type=int, default=15,
        help='Trials before pruner activates (default: 15)')
    parser.add_argument(
        '--n_warmup_steps', type=int, default=30,
        help='Epochs before pruning within a trial (default: 30)')

    args = parser.parse_args()

    if args.fixed_only and args.learned_only:
        parser.error("Cannot use both --fixed_only and --learned_only")

    # ---- SETUP ----
    print("=" * 80)
    print("OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 80)

    set_seed(args.seed)
    device = get_device()

    # Check embeddings exist
    train_emb_path = EMBEDDINGS_DIR / f'train_embeddings_{args.backbone}.pt'
    val_emb_path = EMBEDDINGS_DIR / f'val_embeddings_{args.backbone}.pt'
    if not train_emb_path.exists() or not val_emb_path.exists():
        print(f"\nERROR: Embeddings not found for backbone '{args.backbone}'")
        print(f"Expected: {train_emb_path}")
        print(f"Expected: {val_emb_path}")
        if '+' in args.backbone:
            parts = args.backbone.split('+')
            print(f"\nRun: python scripts/combine_embeddings.py "
                  f"--backbone1 {parts[0]} --backbone2 {parts[1]}")
        else:
            print(f"\nRun: python scripts/extract_embeddings.py "
                  f"--backbone {args.backbone}")
        sys.exit(1)

    # Load data
    dm = DataManager(args.backbone, EMBEDDINGS_DIR, mode=args.mode)

    # Study name
    if args.study_name is None:
        args.study_name = f"optuna_{args.backbone}_{args.mode}_{get_timestamp()}"

    # Search config
    search_config = {
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'fixed_only': args.fixed_only,
        'learned_only': args.learned_only,
        'no_species': args.no_species,
    }

    print(f"\nSearch Configuration:")
    print(f"  Study name:       {args.study_name}")
    print(f"  Backbone:         {args.backbone}")
    print(f"  Mode:             {args.mode} ({dm.target_names})")
    print(f"  Trials:           {args.n_trials}")
    print(f"  Max epochs/trial: {args.max_epochs}")
    print(f"  Patience:         {args.patience}")
    print(f"  Pruner startup:   {args.n_startup_trials} trials")
    print(f"  Pruner warmup:    {args.n_warmup_steps} epochs")
    modes_searched = []
    if not args.learned_only:
        modes_searched.append("fixed pooling")
    if not args.fixed_only:
        modes_searched.append("learned pooling")
    print(f"  Modes:            {', '.join(modes_searched)}")
    print(f"  Species:          {'excluded' if args.no_species else 'included' if dm.has_species else 'not available'}")

    # ---- CREATE STUDY ----
    db_path = ensure_dir(LOGS_DIR) / 'optuna_studies.db'
    storage = f"sqlite:///{db_path}"

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='minimize',
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=5,
        ),
        load_if_exists=True,
    )

    existing = len(study.trials)
    has_completed = any(
        t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if existing > 0:
        print(f"\n  Resuming study with {existing} existing trials")
        if has_completed:
            print(f"  Previous best CRPS: {study.best_value:.6f}")

    # ---- RUN ----
    tracker = BestModelTracker()

    # If resuming, initialize tracker with previous best
    if has_completed:
        tracker.best_crps = study.best_value
        tracker.best_params = study.best_trial.params
        tracker.best_trial_number = study.best_trial.number

    objective_fn = create_objective(dm, device, search_config, tracker)
    start_time = time.time()
    progress_cb = make_progress_callback(args.n_trials + existing, start_time)

    print(f"\n{'-'*80}")
    print("STARTING OPTIMIZATION")
    print(f"{'-'*80}")
    print(f"{'[Trial]':>8s}  {'CRPS':>8s}   {'Best':>12s}   "
          f"{'Mode':<22s}   {'Architecture':<12s}   {'Status':<7s}   {'ETA'}")
    print(f"{'-'*80}")

    try:
        study.optimize(
            objective_fn,
            n_trials=args.n_trials,
            callbacks=[progress_cb],
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print(f"\n\nSearch interrupted by user after {len(study.trials) - existing} "
              f"new trials ({len(study.trials)} total)")

    total_time = time.time() - start_time

    # ---- RESULTS ----
    print(f"\nTotal search time: {format_time(total_time)}")

    print_study_results(study, tracker)

    # Save best model
    checkpoint_path = CHECKPOINTS_DIR / f'best_model_optuna_{args.backbone}_{args.mode}.pt'
    save_best_checkpoint(tracker, checkpoint_path, args.study_name)

    # Save results JSON
    results_path = LOGS_DIR / f'optuna_results_{args.study_name}.json'
    save_results_json(study, results_path)

    # Resumption instructions
    print(f"\nTo resume this study, run:")
    print(f"  python scripts/optuna_search.py --backbone {args.backbone} "
          f"--n_trials N --study_name {args.study_name}")
    print(f"\nStudy database: {db_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
