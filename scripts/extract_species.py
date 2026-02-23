"""
Extract species names from beetle dataset as a feature for training.

This script:
1. Loads the HuggingFace dataset (metadata only, no image processing)
2. Iterates in the same sequential order as extract_embeddings.py
3. Extracts scientificName for each sample
4. Builds a vocabulary mapping species names to integer indices
5. Splits by domain (same logic as extract_embeddings.py)
6. Saves species index tensors and vocabulary

Usage:
    python scripts/extract_species.py
    python scripts/extract_species.py --hf_token YOUR_TOKEN
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    DATASET_NAME, EMBEDDINGS_DIR, VALIDATION_DOMAIN_IDS, set_seed
)


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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract species names from beetle dataset"
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(EMBEDDINGS_DIR),
        help='Directory to save species data'
    )
    parser.add_argument(
        '--val_domains',
        type=int,
        nargs='+',
        default=VALIDATION_DOMAIN_IDS,
        help='Domain IDs to use for validation'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SPECIES NAME EXTRACTION")
    print("=" * 60)

    # Set random seed for reproducibility
    set_seed()

    # Get token
    try:
        hf_token = get_hf_token(args.hf_token)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Load dataset
    from datasets import load_dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, token=hf_token)

    train_data = dataset['train']
    num_samples = len(train_data)
    print(f"Total samples: {num_samples}")

    # Extract species names and domain IDs using column-level access
    # (much faster than per-sample access, avoids image decoding)
    print("\nExtracting species names...")
    all_species_names = train_data['scientificName']
    all_species_names = [name if name is not None else '' for name in all_species_names]
    all_domain_ids = train_data['domainID']

    # Build vocabulary from all species (before splitting)
    # Index 0 is reserved for unknown
    species_counter = Counter(all_species_names)
    print(f"\nFound {len(species_counter)} unique species names")
    print("Top 10 most common species:")
    for name, count in species_counter.most_common(10):
        display_name = name if name else '<empty>'
        print(f"  {display_name}: {count}")

    # Build vocab: index 0 = unknown, then sorted by frequency (most common first)
    vocab = {"<unknown>": 0}
    for idx, (species_name, _) in enumerate(species_counter.most_common(), start=1):
        if species_name:  # Skip empty strings
            vocab[species_name] = idx

    # Map empty/missing species to unknown (index 0)
    print(f"\nVocabulary size: {len(vocab)} (including <unknown>)")

    # Convert species names to indices
    all_species_indices = []
    for name in all_species_names:
        if name and name in vocab:
            all_species_indices.append(vocab[name])
        else:
            all_species_indices.append(0)  # unknown

    all_species_indices = torch.tensor(all_species_indices, dtype=torch.long)
    all_domain_ids = torch.tensor(all_domain_ids, dtype=torch.long)

    # Split by domain (same logic as extract_embeddings.py)
    val_domains_set = set(args.val_domains)
    val_mask = torch.tensor([d.item() in val_domains_set for d in all_domain_ids])
    train_mask = ~val_mask

    train_species = all_species_indices[train_mask]
    val_species = all_species_indices[val_mask]

    print(f"\nSplit results:")
    print(f"  Training samples: {len(train_species)}")
    print(f"  Validation samples: {len(val_species)}")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_species_path = output_dir / 'train_species.pt'
    val_species_path = output_dir / 'val_species.pt'
    vocab_path = output_dir / 'species_vocab.json'

    torch.save(train_species, train_species_path)
    torch.save(val_species, val_species_path)

    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved:")
    print(f"  {train_species_path} - shape: {train_species.shape}")
    print(f"  {val_species_path} - shape: {val_species.shape}")
    print(f"  {vocab_path} - {len(vocab)} entries")

    # Print some stats
    train_unique = len(train_species.unique())
    val_unique = len(val_species.unique())
    print(f"\nUnique species in train: {train_unique}")
    print(f"Unique species in val: {val_unique}")

    # Check how many val species are unseen in train
    train_set = set(train_species.tolist())
    val_set = set(val_species.tolist())
    unseen = val_set - train_set
    if unseen:
        print(f"Species indices in val but not in train: {unseen}")
        # Reverse lookup
        idx_to_name = {v: k for k, v in vocab.items()}
        for idx in unseen:
            print(f"  Index {idx}: {idx_to_name.get(idx, '???')}")

    print("\n" + "=" * 60)
    print("SPECIES EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
