"""
Extract embeddings from beetle images using DINOv2 or BioClip2.

This script:
1. Loads the beetle dataset from HuggingFace
2. Extracts image embeddings using the selected backbone
3. Saves embeddings to disk for fast training later

Running this once per backbone saves significant time during training.

Usage:
    # Extract with DINOv2 (default)
    python scripts/extract_embeddings.py

    # Extract with BioClip2
    python scripts/extract_embeddings.py --backbone bioclip2

You can have both dinov2 and bioclip2 embeddings saved simultaneously.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    DATASET_NAME, EMBEDDINGS_DIR, TARGET_COLUMNS,
    VALIDATION_DOMAIN_IDS, get_device, set_seed
)
from src.feature_extractor import FeatureExtractor, BACKBONE_OPTIONS, list_available_backbones


def get_hf_token(token_arg=None):
    """Get HuggingFace token from argument or environment variable."""
    if token_arg:
        return token_arg
    for env_var in ['HF_TOKEN', 'HUGGING_FACE_TOKEN', 'HUGGINGFACE_TOKEN']:
        token = os.environ.get(env_var)
        if token:
            return token
    raise ValueError(
        "No HuggingFace token found! Set HF_TOKEN or pass --hf_token"
    )


def load_dataset(hf_token):
    """Load the beetle dataset from HuggingFace."""
    from datasets import load_dataset

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, token=hf_token)
    return dataset


def extract_split_embeddings(
    split_data,
    extractor,
    batch_size=16,
    device=None
):
    """
    Extract embeddings for all images in a dataset split.

    Args:
        split_data: HuggingFace dataset split
        extractor: FeatureExtractor instance
        batch_size: Batch size for extraction
        device: Device to use

    Returns:
        tuple: (embeddings, targets, event_ids, domain_ids)
    """
    device = device or get_device()

    all_embeddings = []
    all_targets = []
    all_event_ids = []
    all_domain_ids = []

    # Process in batches
    num_samples = len(split_data)
    print(f"Extracting embeddings from {num_samples} images...")

    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Extracting"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = split_data[start_idx:end_idx]

        # Get images (they come as PIL Images from HuggingFace)
        images = batch_data['file_path']

        # Extract embeddings for this batch
        batch_embeddings = extractor.extract_batch(
            images,
            batch_size=batch_size,
            show_progress=False
        )

        # Get targets
        batch_targets = torch.tensor([
            [batch_data['SPEI_30d'][i], batch_data['SPEI_1y'][i], batch_data['SPEI_2y'][i]]
            for i in range(len(images))
        ], dtype=torch.float32)

        # Get event IDs
        batch_events = torch.tensor(batch_data['eventID'], dtype=torch.long)

        # Get domain IDs
        batch_domains = torch.tensor(batch_data['domainID'], dtype=torch.long)

        all_embeddings.append(batch_embeddings)
        all_targets.append(batch_targets)
        all_event_ids.append(batch_events)
        all_domain_ids.append(batch_domains)

    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    targets = torch.cat(all_targets, dim=0)
    event_ids = torch.cat(all_event_ids, dim=0)
    domain_ids = torch.cat(all_domain_ids, dim=0)

    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"Targets shape: {targets.shape}")

    return embeddings, targets, event_ids, domain_ids


def split_by_domain(embeddings, targets, event_ids, domain_ids, val_domains):
    """
    Split data into train/val based on domain IDs.

    Args:
        embeddings: All embeddings
        targets: All targets
        event_ids: All event IDs
        domain_ids: All domain IDs
        val_domains: List of domain IDs to use for validation

    Returns:
        dict: Split data
    """
    val_domains_set = set(val_domains)

    # Create masks
    val_mask = torch.tensor([d.item() in val_domains_set for d in domain_ids])
    train_mask = ~val_mask

    return {
        'train': {
            'embeddings': embeddings[train_mask],
            'targets': targets[train_mask],
            'event_ids': event_ids[train_mask],
            'domain_ids': domain_ids[train_mask]
        },
        'val': {
            'embeddings': embeddings[val_mask],
            'targets': targets[val_mask],
            'event_ids': event_ids[val_mask],
            'domain_ids': domain_ids[val_mask]
        }
    }


def save_embeddings(data_dict, output_dir, prefix, backbone):
    """
    Save embeddings and metadata to disk with backbone suffix.

    Args:
        data_dict: Dictionary with embeddings, targets, event_ids, domain_ids
        output_dir: Output directory
        prefix: Prefix for filenames (e.g., 'train' or 'val')
        backbone: Backbone name for suffix (e.g., 'dinov2' or 'bioclip2')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings with backbone suffix
    torch.save(data_dict['embeddings'], output_dir / f'{prefix}_embeddings_{backbone}.pt')

    # Save targets and events WITHOUT backbone suffix (they're the same regardless of backbone)
    # Only save if they don't exist yet
    targets_path = output_dir / f'{prefix}_targets.pt'
    events_path = output_dir / f'{prefix}_events.pt'
    domains_path = output_dir / f'{prefix}_domains.pt'

    if not targets_path.exists():
        torch.save(data_dict['targets'], targets_path)
    if not events_path.exists():
        torch.save(data_dict['event_ids'], events_path)
    if not domains_path.exists():
        torch.save(data_dict['domain_ids'], domains_path)

    print(f"Saved {prefix} data ({backbone}):")
    print(f"  Embeddings: {data_dict['embeddings'].shape} -> {prefix}_embeddings_{backbone}.pt")
    print(f"  Targets: {data_dict['targets'].shape}")
    print(f"  Events: {data_dict['event_ids'].shape}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from beetle images"
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        choices=list(BACKBONE_OPTIONS.keys()),
        default='dinov3',
        help='Which backbone to use for feature extraction (default: dinov3)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for embedding extraction'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(EMBEDDINGS_DIR),
        help='Directory to save embeddings'
    )
    parser.add_argument(
        '--val_domains',
        type=int,
        nargs='+',
        default=VALIDATION_DOMAIN_IDS,
        help='Domain IDs to use for validation'
    )
    parser.add_argument(
        '--list_backbones',
        action='store_true',
        help='List available backbones and exit'
    )

    args = parser.parse_args()

    # Just list backbones if requested
    if args.list_backbones:
        list_available_backbones()
        sys.exit(0)

    print("=" * 60)
    print("EMBEDDING EXTRACTION")
    print("=" * 60)
    print(f"Backbone: {args.backbone}")

    # Set random seed for reproducibility
    set_seed()

    # Get device
    device = get_device()

    # Get token
    try:
        hf_token = get_hf_token(args.hf_token)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(hf_token)

    # Initialize feature extractor with selected backbone
    print("\nInitializing feature extractor...")
    extractor = FeatureExtractor(backbone=args.backbone, device=device)
    extractor.load()

    # Extract embeddings from training split
    print("\n" + "-" * 50)
    print("Processing training data...")
    print("-" * 50)

    train_data = dataset['train']
    embeddings, targets, event_ids, domain_ids = extract_split_embeddings(
        train_data,
        extractor,
        batch_size=args.batch_size,
        device=device
    )

    # Split by domain for validation
    print(f"\nSplitting by domain (validation domains: {args.val_domains})")
    splits = split_by_domain(
        embeddings, targets, event_ids, domain_ids,
        val_domains=args.val_domains
    )

    # Report split sizes
    print(f"\nSplit results:")
    print(f"  Training samples: {len(splits['train']['embeddings'])}")
    print(f"  Validation samples: {len(splits['val']['embeddings'])}")

    # Count unique events in each split
    train_events = len(set(splits['train']['event_ids'].tolist()))
    val_events = len(set(splits['val']['event_ids'].tolist()))
    print(f"  Training events: {train_events}")
    print(f"  Validation events: {val_events}")

    # Save embeddings
    print("\n" + "-" * 50)
    print("Saving embeddings...")
    print("-" * 50)

    save_embeddings(splits['train'], args.output_dir, 'train', args.backbone)
    save_embeddings(splits['val'], args.output_dir, 'val', args.backbone)

    # Save/update metadata
    import json
    metadata_path = Path(args.output_dir) / 'metadata.json'

    # Load existing metadata if it exists
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Update with this backbone's info
    metadata['val_domains'] = args.val_domains
    metadata['train_samples'] = len(splits['train']['embeddings'])
    metadata['val_samples'] = len(splits['val']['embeddings'])
    metadata['train_events'] = train_events
    metadata['val_events'] = val_events

    # Track which backbones have been extracted
    if 'backbones' not in metadata:
        metadata['backbones'] = []
    if args.backbone not in metadata['backbones']:
        metadata['backbones'].append(args.backbone)

    metadata[f'embedding_dim_{args.backbone}'] = splits['train']['embeddings'].shape[1]

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")

    print("\n" + "=" * 60)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nBackbone: {args.backbone}")
    print(f"Embeddings saved to: {args.output_dir}")
    print(f"  - train_embeddings_{args.backbone}.pt")
    print(f"  - val_embeddings_{args.backbone}.pt")
    print(f"\nNext step: Run training with this backbone:")
    print(f"  python scripts/train.py --backbone {args.backbone}")


if __name__ == "__main__":
    main()
