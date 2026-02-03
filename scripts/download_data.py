"""
Download the beetle dataset from HuggingFace.

This script downloads the training and validation data from the
HuggingFace Hub. You'll need a HuggingFace account and API token.

Usage:
    python scripts/download_data.py --hf_token YOUR_TOKEN

Or set the HF_TOKEN environment variable:
    set HF_TOKEN=YOUR_TOKEN  (Windows)
    export HF_TOKEN=YOUR_TOKEN  (Linux/Mac)
    python scripts/download_data.py
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DATASET_NAME, DATA_DIR


def get_hf_token(token_arg=None):
    """
    Get HuggingFace token from argument or environment variable.

    Args:
        token_arg: Token passed as command line argument

    Returns:
        str: The HuggingFace token

    Raises:
        ValueError: If no token is found
    """
    # Try command line argument first
    if token_arg:
        return token_arg

    # Try environment variables
    for env_var in ['HF_TOKEN', 'HUGGING_FACE_TOKEN', 'HUGGINGFACE_TOKEN']:
        token = os.environ.get(env_var)
        if token:
            print(f"Using token from {env_var} environment variable")
            return token

    raise ValueError(
        "No HuggingFace token found!\n"
        "Please either:\n"
        "  1. Pass --hf_token YOUR_TOKEN\n"
        "  2. Set HF_TOKEN environment variable\n"
        "\n"
        "Get your token at: https://huggingface.co/settings/tokens"
    )


def download_dataset(hf_token):
    """
    Download the beetle dataset from HuggingFace.

    Args:
        hf_token: HuggingFace API token

    Returns:
        Dataset: The loaded dataset
    """
    print(f"\nDownloading dataset: {DATASET_NAME}")
    print("This may take a while depending on your internet connection...")
    print("-" * 50)

    from datasets import load_dataset

    # Load the dataset from HuggingFace
    # This will download and cache the data locally
    dataset = load_dataset(
        DATASET_NAME,
        token=hf_token,
    )

    return dataset


def explore_dataset(dataset):
    """
    Print information about the downloaded dataset.

    Args:
        dataset: The loaded dataset
    """
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    # Print available splits
    print(f"\nAvailable splits: {list(dataset.keys())}")

    # Info about each split
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} split:")
        print(f"  Number of samples: {len(split_data)}")

        # Show column names
        if hasattr(split_data, 'column_names'):
            print(f"  Columns: {split_data.column_names}")

        # Show a sample
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"  Sample keys: {list(sample.keys())}")

            # Check for expected columns
            for col in ['SPEI_30d', 'SPEI_1y', 'SPEI_2y', 'eventID', 'domainID']:
                if col in sample:
                    print(f"    {col}: {sample[col]}")

    # Count unique events and domains
    if 'train' in dataset:
        train_data = dataset['train']

        if 'eventID' in train_data.column_names:
            unique_events = len(set(train_data['eventID']))
            print(f"\nUnique events in training: {unique_events}")

        if 'domainID' in train_data.column_names:
            unique_domains = sorted(set(train_data['domainID']))
            print(f"Unique domains in training: {unique_domains}")
            print(f"Number of domains: {len(unique_domains)}")

    print("=" * 60)


def save_dataset_info(dataset, output_dir):
    """
    Save dataset metadata to disk.

    Args:
        dataset: The loaded dataset
        output_dir: Directory to save info
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save basic info
    info = {
        'dataset_name': DATASET_NAME,
        'splits': {}
    }

    for split_name, split_data in dataset.items():
        info['splits'][split_name] = {
            'num_samples': len(split_data),
            'columns': split_data.column_names if hasattr(split_data, 'column_names') else []
        }

        # Save unique domain IDs if available
        if 'domainID' in split_data.column_names:
            info['splits'][split_name]['domain_ids'] = sorted(set(split_data['domainID']))

    # Save to JSON
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset info saved to {info_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download the beetle dataset from HuggingFace"
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token (or set HF_TOKEN env variable)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DATA_DIR),
        help='Directory to save dataset info'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BEETLE DATASET DOWNLOADER")
    print("=" * 60)

    # Get token
    try:
        hf_token = get_hf_token(args.hf_token)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Download dataset
    try:
        dataset = download_dataset(hf_token)
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. Verify your HuggingFace token is valid")
        print("  3. Make sure you have access to the dataset")
        print(f"     (visit: https://huggingface.co/datasets/{DATASET_NAME})")
        sys.exit(1)

    # Explore and save info
    explore_dataset(dataset)
    save_dataset_info(dataset, args.output_dir)

    print("\nDataset download complete!")
    print(f"The dataset is cached by HuggingFace and will load quickly next time.")
    print("\nNext step: Run scripts/extract_embeddings.py to compute image features")


if __name__ == "__main__":
    main()
