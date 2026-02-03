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
    BATCH_SIZE, RANDOM_SEED
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
        aggregation='mean'
    ):
        """
        Initialize the dataset.

        Args:
            embeddings_path: Path to .pt file with embeddings
            targets_path: Path to .pt file with targets
            event_ids_path: Path to .pt file with event IDs
            aggregation: How to combine embeddings ('mean', 'max', 'sum')
        """
        # Load raw data
        all_embeddings = torch.load(embeddings_path, weights_only=True)
        all_targets = torch.load(targets_path, weights_only=True)
        all_event_ids = torch.load(event_ids_path, weights_only=True)

        self.aggregation = aggregation

        # Group by event ID
        event_to_indices = defaultdict(list)
        for idx, event_id in enumerate(all_event_ids):
            event_to_indices[event_id.item()].append(idx)

        # Aggregate embeddings for each event
        self.embeddings = []
        self.targets = []
        self.event_ids = []

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

        # Stack into tensors
        self.embeddings = torch.stack(self.embeddings)
        self.targets = torch.stack(self.targets)
        self.event_ids = torch.tensor(self.event_ids)

        print(f"Aggregated {len(all_embeddings)} images into {len(self)} events")
        print(f"  Using {aggregation} aggregation")
        print(f"  Embedding shape: {self.embeddings.shape}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


def create_data_loaders(
    train_embeddings_path,
    train_targets_path,
    train_events_path,
    val_embeddings_path,
    val_targets_path,
    val_events_path,
    batch_size=BATCH_SIZE,
    aggregated=True,
    aggregation='mean'
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
        aggregation: Aggregation method if using aggregated dataset

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Choose dataset class
    if aggregated:
        DatasetClass = AggregatedEmbeddingDataset
        extra_args = {'aggregation': aggregation}
    else:
        DatasetClass = EmbeddingDataset
        extra_args = {}

    # Create datasets
    print("\nLoading training data...")
    train_dataset = DatasetClass(
        embeddings_path=train_embeddings_path,
        targets_path=train_targets_path,
        event_ids_path=train_events_path,
        **extra_args
    ) if aggregated else EmbeddingDataset(
        embeddings_path=train_embeddings_path,
        targets_path=train_targets_path,
        event_ids_path=train_events_path
    )

    print("\nLoading validation data...")
    val_dataset = DatasetClass(
        embeddings_path=val_embeddings_path,
        targets_path=val_targets_path,
        event_ids_path=val_events_path,
        **extra_args
    ) if aggregated else EmbeddingDataset(
        embeddings_path=val_embeddings_path,
        targets_path=val_targets_path,
        event_ids_path=val_events_path
    )

    # Create data loaders
    # - shuffle=True for training (randomize order each epoch)
    # - shuffle=False for validation (consistent evaluation)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
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

        # Test DataLoader
        print("\n--- Testing DataLoader ---")
        loader = DataLoader(agg_dataset, batch_size=8, shuffle=True)
        for batch_emb, batch_target in loader:
            print(f"Batch shapes: embeddings={batch_emb.shape}, targets={batch_target.shape}")
            break

    print("\nAll tests passed!")
