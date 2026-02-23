"""
Data loading utilities for the Beetles Drought Prediction project.

This module provides PyTorch Dataset classes for:
1. Loading raw images (for embedding extraction)
2. Loading pre-computed embeddings (for fast training)

For beginners: A Dataset tells PyTorch how to load individual samples.
A DataLoader then batches these samples together and handles shuffling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TARGET_COLUMNS, VALIDATION_DOMAIN_IDS,
    BATCH_SIZE, RANDOM_SEED, POOLING_TYPE, LEARNED_POOLING_TYPES
)


class EmbeddingDataset(Dataset):
    """
    Dataset for loading pre-computed embeddings.

    This is the FAST dataset used during training. Instead of loading images
    and extracting features each time (slow), we load pre-computed embeddings
    from disk (fast).

    For beginners: Think of this as loading pre-prepared ingredients instead
    of preparing them from scratch every time you want to cook.

    Example:
        dataset = EmbeddingDataset(
            embeddings_path="embeddings/train.pt",
            targets_path="embeddings/train_targets.pt",
            event_ids_path="embeddings/train_events.pt"
        )
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(self, embeddings_path, targets_path, event_ids_path=None):
        """
        Initialize the dataset.

        Args:
            embeddings_path: Path to .pt file with embeddings tensor
            targets_path: Path to .pt file with target values tensor
            event_ids_path: Optional path to event IDs (for aggregation)
        """
        # Load pre-computed embeddings
        # Shape: (num_samples, embedding_dim)
        self.embeddings = torch.load(embeddings_path, weights_only=True)

        # Load target values (SPEI metrics)
        # Shape: (num_samples, 3)
        self.targets = torch.load(targets_path, weights_only=True)

        # Optionally load event IDs for grouping
        self.event_ids = None
        if event_ids_path is not None and Path(event_ids_path).exists():
            self.event_ids = torch.load(event_ids_path, weights_only=True)

        # Validate shapes match
        assert len(self.embeddings) == len(self.targets), \
            f"Mismatch: {len(self.embeddings)} embeddings vs {len(self.targets)} targets"

        print(f"Loaded dataset with {len(self)} samples")
        print(f"  Embedding shape: {self.embeddings.shape}")
        print(f"  Targets shape: {self.targets.shape}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            tuple: (embedding, target) tensors
        """
        embedding = self.embeddings[idx]
        target = self.targets[idx]

        return embedding, target


class AggregatedEmbeddingDataset(Dataset):
    """
    Dataset that aggregates multiple beetle images per event.

    In this competition, each "sampling event" (location + date) has multiple
    beetle images. This dataset aggregates (e.g., averages) all embeddings
    for each event into a single embedding.

    For beginners: This is like taking the average of all beetle photos
    from one location to make a single prediction for that location.
    """

    def __init__(
        self,
        embeddings_path,
        targets_path,
        event_ids_path,
        aggregation='mean',
        species_path=None
    ):
        """
        Initialize the dataset.

        Args:
            embeddings_path: Path to .pt file with embeddings
            targets_path: Path to .pt file with targets
            event_ids_path: Path to .pt file with event IDs
            aggregation: How to combine embeddings ('mean', 'max', 'sum')
            species_path: Optional path to .pt file with species indices
        """
        # Load raw data
        all_embeddings = torch.load(embeddings_path, weights_only=True)
        all_targets = torch.load(targets_path, weights_only=True)
        all_event_ids = torch.load(event_ids_path, weights_only=True)

        # Optionally load species indices
        all_species = None
        if species_path is not None and Path(species_path).exists():
            all_species = torch.load(species_path, weights_only=True)
            assert len(all_species) == len(all_embeddings), \
                f"Species length {len(all_species)} != embeddings length {len(all_embeddings)}"

        self.aggregation = aggregation
        self.has_species = all_species is not None

        # Group by event ID
        event_to_indices = defaultdict(list)
        for idx, event_id in enumerate(all_event_ids):
            event_to_indices[event_id.item()].append(idx)

        # Aggregate embeddings for each event
        self.embeddings = []
        self.targets = []
        self.event_ids = []
        self.species_indices = []

        for event_id, indices in event_to_indices.items():
            # Get all embeddings for this event
            event_embeddings = all_embeddings[indices]

            # Aggregate based on strategy
            if aggregation == 'mean':
                aggregated = event_embeddings.mean(dim=0)
            elif aggregation == 'max':
                aggregated = event_embeddings.max(dim=0)[0]
            elif aggregation == 'sum':
                aggregated = event_embeddings.sum(dim=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            # Targets should be the same for all images in an event
            # (they all have the same SPEI values)
            target = all_targets[indices[0]]

            self.embeddings.append(aggregated)
            self.targets.append(target)
            self.event_ids.append(event_id)

            # Pick mode (most common) species for this event
            if self.has_species:
                event_species = all_species[indices]
                # Mode: most common species index
                mode_species = torch.mode(event_species).values.item()
                self.species_indices.append(mode_species)

        # Stack into tensors
        self.embeddings = torch.stack(self.embeddings)
        self.targets = torch.stack(self.targets)
        self.event_ids = torch.tensor(self.event_ids)
        if self.has_species:
            self.species_indices = torch.tensor(self.species_indices, dtype=torch.long)

        print(f"Aggregated {len(all_embeddings)} images into {len(self)} events")
        print(f"  Using {aggregation} aggregation")
        print(f"  Embedding shape: {self.embeddings.shape}")
        if self.has_species:
            print(f"  Species indices: {len(self.species_indices)} (unique: {len(self.species_indices.unique())})")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.has_species:
            return self.embeddings[idx], self.targets[idx], self.species_indices[idx]
        return self.embeddings[idx], self.targets[idx]


class EventEmbeddingDataset(Dataset):
    """
    Dataset that returns ALL embeddings for each event (variable length).

    Unlike AggregatedEmbeddingDataset which pre-aggregates embeddings,
    this dataset returns all individual embeddings for an event, allowing
    LEARNED pooling (attention, gated attention, etc.) to be applied
    during training as part of the model's forward pass.

    For beginners: Instead of averaging beetle images beforehand, this
    keeps all images and lets the model learn which ones are important.

    Note: This dataset returns variable-length samples, so it requires
    a custom collate function (event_collate_fn) for batching.

    Example:
        dataset = EventEmbeddingDataset(
            embeddings_path="embeddings/train.pt",
            targets_path="embeddings/train_targets.pt",
            event_ids_path="embeddings/train_events.pt"
        )
        loader = DataLoader(dataset, batch_size=1, collate_fn=event_collate_fn)
    """

    def __init__(self, embeddings_path, targets_path, event_ids_path, species_path=None):
        """
        Initialize the dataset.

        Args:
            embeddings_path: Path to .pt file with embeddings
            targets_path: Path to .pt file with targets
            event_ids_path: Path to .pt file with event IDs
            species_path: Optional path to .pt file with species indices
        """
        # Load raw data
        all_embeddings = torch.load(embeddings_path, weights_only=True)
        all_targets = torch.load(targets_path, weights_only=True)
        all_event_ids = torch.load(event_ids_path, weights_only=True)

        # Optionally load species indices
        all_species = None
        if species_path is not None and Path(species_path).exists():
            all_species = torch.load(species_path, weights_only=True)
            assert len(all_species) == len(all_embeddings), \
                f"Species length {len(all_species)} != embeddings length {len(all_embeddings)}"

        self.has_species = all_species is not None

        # Group by event ID
        event_to_indices = defaultdict(list)
        for idx, event_id in enumerate(all_event_ids):
            event_to_indices[event_id.item()].append(idx)

        # Store event data (all embeddings per event, not aggregated)
        self.event_embeddings = []  # List of (N_i, embed_dim) tensors
        self.targets = []           # List of (num_targets,) tensors
        self.event_ids = []         # List of event IDs
        self.event_species = []     # List of (N_i,) species index tensors
        self.num_images_per_event = []  # For analysis

        for event_id, indices in event_to_indices.items():
            event_embs = all_embeddings[indices]  # (N_i, embed_dim)
            target = all_targets[indices[0]]      # Same for all images in event

            self.event_embeddings.append(event_embs)
            self.targets.append(target)
            self.event_ids.append(event_id)
            self.num_images_per_event.append(len(indices))

            if self.has_species:
                self.event_species.append(all_species[indices])

        # Stack targets for easy access
        self.targets = torch.stack(self.targets)
        self.event_ids = torch.tensor(self.event_ids)

        print(f"Created EventEmbeddingDataset with {len(self)} events")
        print(f"  Total images: {sum(self.num_images_per_event)}")
        print(f"  Images per event: min={min(self.num_images_per_event)}, "
              f"max={max(self.num_images_per_event)}, "
              f"mean={np.mean(self.num_images_per_event):.1f}")
        if self.has_species:
            print(f"  Species data loaded for each image")

    def __len__(self):
        return len(self.event_embeddings)

    def __getitem__(self, idx):
        """
        Get all embeddings for a single event.

        Returns:
            tuple: (event_embeddings, target) or (event_embeddings, target, event_species)
                - event_embeddings: (N_i, embed_dim) tensor
                - target: (num_targets,) tensor
                - event_species: (N_i,) tensor of species indices (when species loaded)
        """
        if self.has_species:
            return self.event_embeddings[idx], self.targets[idx], self.event_species[idx]
        return self.event_embeddings[idx], self.targets[idx]


def event_collate_fn(batch):
    """
    Custom collate function for variable-length event batches.

    Since events have different numbers of images, we can't simply stack them.
    This function returns a list of embeddings and a stacked target tensor.

    Handles both 2-element tuples (embeddings, target) and 3-element tuples
    (embeddings, target, species) for species-aware training.

    For batch_size > 1, this returns:
        - embeddings: List of (N_i, embed_dim) tensors
        - targets: (batch_size, num_targets) tensor
        - species (if present): List of (N_i,) tensors

    For batch_size = 1 (recommended for attention pooling training):
        - embeddings: Single (N, embed_dim) tensor
        - targets: (1, num_targets) tensor
        - species (if present): Single (N,) tensor

    Usage:
        loader = DataLoader(dataset, batch_size=1, collate_fn=event_collate_fn)
    """
    has_species = len(batch[0]) == 3

    embeddings_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    if has_species:
        species_list = [item[2] for item in batch]

        # For batch_size=1, return the single tensor directly
        if len(batch) == 1:
            return embeddings_list[0], targets, species_list[0]
        return embeddings_list, targets, species_list

    # For batch_size=1, return the single tensor directly
    if len(batch) == 1:
        return embeddings_list[0], targets

    return embeddings_list, targets


def event_collate_fn_padded(batch, pad_value=0.0):
    """
    Collate function that pads variable-length sequences.

    Pads all event embeddings to the same length (max in batch).
    Returns a mask indicating valid positions.

    Returns:
        tuple: (padded_embeddings, targets, mask)
            - padded_embeddings: (batch_size, max_len, embed_dim)
            - targets: (batch_size, num_targets)
            - mask: (batch_size, max_len) boolean mask (True = valid)
    """
    embeddings_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    # Find max sequence length
    max_len = max(emb.size(0) for emb in embeddings_list)
    embed_dim = embeddings_list[0].size(1)
    batch_size = len(batch)

    # Pad embeddings
    padded = torch.full((batch_size, max_len, embed_dim), pad_value)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, emb in enumerate(embeddings_list):
        length = emb.size(0)
        padded[i, :length] = emb
        mask[i, :length] = True

    return padded, targets, mask


def create_data_loaders(
    train_embeddings_path,
    train_targets_path,
    train_events_path,
    val_embeddings_path,
    val_targets_path,
    val_events_path,
    batch_size=BATCH_SIZE,
    aggregated=True,
    aggregation='mean',
    pooling_type=None,
    use_species=False,
    train_species_path=None,
    val_species_path=None
):
    """
    Create training and validation data loaders.

    Args:
        train_embeddings_path: Path to training embeddings
        train_targets_path: Path to training targets
        train_events_path: Path to training event IDs
        val_embeddings_path: Path to validation embeddings
        val_targets_path: Path to validation targets
        val_events_path: Path to validation event IDs
        batch_size: Batch size for data loaders
        aggregated: Whether to use aggregated dataset (one sample per event)
        aggregation: Aggregation method if using aggregated dataset ('mean', 'max', 'sum')
        pooling_type: If specified, overrides aggregation. For learned pooling types
                      (attention, gated_attention, statistics, hybrid), uses
                      EventEmbeddingDataset with batch_size=1.
        use_species: Whether to include species indices in the dataset
        train_species_path: Path to training species indices (required if use_species=True)
        val_species_path: Path to validation species indices (required if use_species=True)

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Resolve species paths
    sp_train = train_species_path if use_species else None
    sp_val = val_species_path if use_species else None

    # Determine if we need event-level data (for learned pooling)
    use_event_dataset = (
        pooling_type is not None and
        pooling_type.lower() in LEARNED_POOLING_TYPES
    )

    if use_event_dataset:
        # For learned pooling, return all embeddings per event
        # and use batch_size=1 with custom collate
        print(f"\nUsing EventEmbeddingDataset for learned pooling ({pooling_type})")

        print("\nLoading training data...")
        train_dataset = EventEmbeddingDataset(
            embeddings_path=train_embeddings_path,
            targets_path=train_targets_path,
            event_ids_path=train_events_path,
            species_path=sp_train
        )

        print("\nLoading validation data...")
        val_dataset = EventEmbeddingDataset(
            embeddings_path=val_embeddings_path,
            targets_path=val_targets_path,
            event_ids_path=val_events_path,
            species_path=sp_val
        )

        # For learned pooling, use batch_size=1 (one event at a time)
        # This allows variable-length input to the pooling module
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=event_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=event_collate_fn
        )

    elif aggregated:
        # For fixed pooling (mean, max, sum), pre-aggregate embeddings
        DatasetClass = AggregatedEmbeddingDataset
        agg_method = pooling_type if pooling_type else aggregation

        print(f"\nUsing AggregatedEmbeddingDataset with {agg_method} pooling")

        print("\nLoading training data...")
        train_dataset = DatasetClass(
            embeddings_path=train_embeddings_path,
            targets_path=train_targets_path,
            event_ids_path=train_events_path,
            aggregation=agg_method,
            species_path=sp_train
        )

        print("\nLoading validation data...")
        val_dataset = DatasetClass(
            embeddings_path=val_embeddings_path,
            targets_path=val_targets_path,
            event_ids_path=val_events_path,
            aggregation=agg_method,
            species_path=sp_val
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    else:
        # Return individual image embeddings (no event-level grouping)
        print("\nUsing EmbeddingDataset (individual images)")

        print("\nLoading training data...")
        train_dataset = EmbeddingDataset(
            embeddings_path=train_embeddings_path,
            targets_path=train_targets_path,
            event_ids_path=train_events_path
        )

        print("\nLoading validation data...")
        val_dataset = EmbeddingDataset(
            embeddings_path=val_embeddings_path,
            targets_path=val_targets_path,
            event_ids_path=val_events_path
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    print(f"\nDataLoaders created:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def split_by_domain(embeddings, targets, event_ids, domain_ids, val_domains):
    """
    Split data into train/val by domain ID.

    This implements site-based holdout validation, where entire sites
    are held out for validation to test generalization.

    Args:
        embeddings: All embeddings tensor
        targets: All targets tensor
        event_ids: All event IDs tensor
        domain_ids: Domain ID for each sample
        val_domains: List of domain IDs to use for validation

    Returns:
        dict: Contains train and val splits for embeddings, targets, events
    """
    # Convert val_domains to a set for fast lookup
    val_domains_set = set(val_domains)

    # Create masks for train and val
    val_mask = torch.tensor([d.item() in val_domains_set for d in domain_ids])
    train_mask = ~val_mask

    # Split the data
    splits = {
        'train_embeddings': embeddings[train_mask],
        'train_targets': targets[train_mask],
        'train_events': event_ids[train_mask],
        'val_embeddings': embeddings[val_mask],
        'val_targets': targets[val_mask],
        'val_events': event_ids[val_mask]
    }

    print(f"\nSplit data by domain:")
    print(f"  Training samples: {train_mask.sum().item()}")
    print(f"  Validation samples: {val_mask.sum().item()}")
    print(f"  Validation domains: {val_domains}")

    return splits


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing data loaders with dummy data...")

    # Create dummy data
    num_samples = 100
    embedding_dim = 768
    num_targets = 3

    dummy_embeddings = torch.randn(num_samples, embedding_dim)
    dummy_targets = torch.randn(num_samples, num_targets)
    dummy_events = torch.randint(0, 20, (num_samples,))  # 20 unique events

    # Save to temp files
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save dummy data
        torch.save(dummy_embeddings, tmpdir / "emb.pt")
        torch.save(dummy_targets, tmpdir / "targets.pt")
        torch.save(dummy_events, tmpdir / "events.pt")

        # Test EmbeddingDataset
        print("\n--- Testing EmbeddingDataset ---")
        dataset = EmbeddingDataset(
            embeddings_path=tmpdir / "emb.pt",
            targets_path=tmpdir / "targets.pt",
            event_ids_path=tmpdir / "events.pt"
        )
        print(f"Dataset length: {len(dataset)}")
        emb, target = dataset[0]
        print(f"Sample shapes: embedding={emb.shape}, target={target.shape}")

        # Test AggregatedEmbeddingDataset
        print("\n--- Testing AggregatedEmbeddingDataset ---")
        agg_dataset = AggregatedEmbeddingDataset(
            embeddings_path=tmpdir / "emb.pt",
            targets_path=tmpdir / "targets.pt",
            event_ids_path=tmpdir / "events.pt",
            aggregation='mean'
        )
        print(f"Aggregated dataset length: {len(agg_dataset)}")

        # Test DataLoader with aggregated data
        print("\n--- Testing DataLoader (aggregated) ---")
        loader = DataLoader(agg_dataset, batch_size=8, shuffle=True)
        for batch_emb, batch_target in loader:
            print(f"Batch shapes: embeddings={batch_emb.shape}, targets={batch_target.shape}")
            break

        # Test EventEmbeddingDataset (for learned pooling)
        print("\n--- Testing EventEmbeddingDataset ---")
        event_dataset = EventEmbeddingDataset(
            embeddings_path=tmpdir / "emb.pt",
            targets_path=tmpdir / "targets.pt",
            event_ids_path=tmpdir / "events.pt"
        )
        print(f"Event dataset length: {len(event_dataset)}")
        event_emb, event_target = event_dataset[0]
        print(f"Event 0: {event_emb.shape[0]} images, embedding shape={event_emb.shape}")

        # Test DataLoader with event collate function
        print("\n--- Testing DataLoader (event-based) ---")
        event_loader = DataLoader(
            event_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=event_collate_fn
        )
        for batch_emb, batch_target in event_loader:
            print(f"Event batch: embeddings={batch_emb.shape}, targets={batch_target.shape}")
            break

        # Test padded collate function
        print("\n--- Testing Padded Collate ---")
        event_loader_padded = DataLoader(
            event_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=event_collate_fn_padded
        )
        for padded_emb, batch_target, mask in event_loader_padded:
            print(f"Padded batch: embeddings={padded_emb.shape}, targets={batch_target.shape}, mask={mask.shape}")
            print(f"  Valid images per event: {mask.sum(dim=1).tolist()}")
            break

    print("\nAll tests passed!")
