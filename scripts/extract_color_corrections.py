"""
Extract color correction matrices from colorpicker reference images.

This script pre-computes color correction matrices (CCMs) for all unique
colorpicker images in the beetle dataset. The CCMs are cached for fast
lookup during embedding extraction.

Usage:
    python scripts/extract_color_corrections.py
    python scripts/extract_color_corrections.py --hf_token YOUR_TOKEN --visualize

Output:
    embeddings/color_corrections.pkl    - Cached CCMs (973 matrices)
    embeddings/colorpicker_mapping.json - Sample index -> colorpicker path
    embeddings/failed_colorpickers.txt  - Failed detections (if any)
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DATASET_NAME, EMBEDDINGS_DIR
from src.color_calibration import (
    ColorCheckerDetector,
    ColorCorrectionMatrix,
    ColorCalibrationCache,
    get_reference_colors,
    COLORCHECKER_24_SRGB,
    COLORCHECKER_30_SRGB
)


def get_hf_token(token_arg=None):
    """Get HuggingFace token from argument or environment variable."""
    if token_arg:
        return token_arg
    for env_var in ['HF_TOKEN', 'HUGGING_FACE_TOKEN', 'HUGGINGFACE_TOKEN']:
        token = os.environ.get(env_var)
        if token:
            return token
    # Try reading from hugtoken.txt
    token_file = project_root / 'hugtoken.txt'
    if token_file.exists():
        with open(token_file, 'r') as f:
            token = f.read().strip()
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


def visualize_detection(image, patches, reference, ccm, output_path):
    """
    Create visualization of color checker detection.

    Args:
        image: Original colorpicker image
        patches: Detected patch colors
        reference: Reference colors used
        ccm: Computed color correction matrix
        output_path: Path to save visualization
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Colorpicker')
    axes[0, 0].axis('off')

    # Detected patches (as color swatches)
    n_patches = len(patches)
    cols = 6
    rows = (n_patches + cols - 1) // cols
    patch_img = np.zeros((rows * 50, cols * 50, 3), dtype=np.uint8)
    for i, color in enumerate(patches):
        r, c = i // cols, i % cols
        patch_img[r*50:(r+1)*50, c*50:(c+1)*50] = color.astype(np.uint8)
    axes[0, 1].imshow(patch_img)
    axes[0, 1].set_title(f'Detected Patches ({n_patches})')
    axes[0, 1].axis('off')

    # Reference patches
    ref_img = np.zeros((rows * 50, cols * 50, 3), dtype=np.uint8)
    for i, color in enumerate(reference[:n_patches]):
        r, c = i // cols, i % cols
        ref_img[r*50:(r+1)*50, c*50:(c+1)*50] = color.astype(np.uint8)
    axes[0, 2].imshow(ref_img)
    axes[0, 2].set_title(f'Reference Colors')
    axes[0, 2].axis('off')

    # Corrected patches
    corrected_patches = ccm.apply(patches.reshape(1, -1, 3)[0:1]).reshape(-1, 3)
    corr_img = np.zeros((rows * 50, cols * 50, 3), dtype=np.uint8)
    for i, color in enumerate(corrected_patches):
        r, c = i // cols, i % cols
        corr_img[r*50:(r+1)*50, c*50:(c+1)*50] = np.clip(color, 0, 255).astype(np.uint8)
    axes[1, 0].imshow(corr_img)
    axes[1, 0].set_title(f'Corrected Patches (RMSE: {ccm.rmse:.1f})')
    axes[1, 0].axis('off')

    # Color error comparison
    ax = axes[1, 1]
    errors_before = np.sqrt(np.sum((patches - reference[:n_patches]) ** 2, axis=1))
    errors_after = np.sqrt(np.sum((corrected_patches - reference[:n_patches]) ** 2, axis=1))
    x = np.arange(n_patches)
    width = 0.35
    ax.bar(x - width/2, errors_before, width, label='Before', alpha=0.8)
    ax.bar(x + width/2, errors_after, width, label='After', alpha=0.8)
    ax.set_xlabel('Patch Index')
    ax.set_ylabel('Color Error (Euclidean)')
    ax.set_title('Per-Patch Error')
    ax.legend()

    # CCM visualization
    ax = axes[1, 2]
    im = ax.imshow(ccm.matrix, cmap='RdBu', vmin=-0.5, vmax=1.5)
    ax.set_title('Correction Matrix')
    ax.set_xlabel('Input RGB')
    ax.set_ylabel('Output RGB')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['R', 'G', 'B'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['R', 'G', 'B'])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{ccm.matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract color correction matrices from colorpicker images"
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
        help='Directory to save color corrections'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization images for each colorpicker'
    )
    parser.add_argument(
        '--max_visualize',
        type=int,
        default=50,
        help='Maximum number of visualizations to generate'
    )
    parser.add_argument(
        '--rmse_threshold',
        type=float,
        default=15.0,
        help='RMSE threshold for quality warnings'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("COLOR CORRECTION EXTRACTION")
    print("=" * 60)

    # Get token
    try:
        hf_token = get_hf_token(args.hf_token)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(hf_token)
    train_data = dataset['train']

    # Find unique colorpicker images
    print("\nAnalyzing colorpicker images...")

    # Check if colorpicker columns exist
    columns = train_data.column_names
    print(f"Available columns: {columns}")

    # Try different possible column names
    colorpicker_col = None
    colorpicker_full_col = None
    for col in ['colorpicker_path', 'colorpicker', 'color_picker_path']:
        if col in columns:
            colorpicker_col = col
            break
    for col in ['colorpicker_full_path', 'colorpicker_full', 'colorpicker_image']:
        if col in columns:
            colorpicker_full_col = col
            break

    if colorpicker_col is None:
        print("\nWARNING: No colorpicker path column found!")
        print("Checking for colorpicker image column directly...")
        # Check if colorpicker images are stored directly
        if 'colorpicker' in columns:
            colorpicker_full_col = 'colorpicker'
        else:
            print("No colorpicker column found. Listing available columns:")
            for col in columns:
                print(f"  - {col}")
            sys.exit(1)

    # Build mapping of colorpicker paths to sample indices
    print(f"\nUsing colorpicker column: {colorpicker_full_col or colorpicker_col}")

    colorpicker_to_samples = defaultdict(list)
    sample_to_colorpicker = {}

    num_samples = len(train_data)
    print(f"Total samples: {num_samples}")

    # Get colorpicker paths for all samples
    if colorpicker_col:
        colorpicker_paths = train_data[colorpicker_col]
    else:
        # If no path column, use indices
        colorpicker_paths = [str(i) for i in range(num_samples)]

    for idx, cp_path in enumerate(tqdm(colorpicker_paths, desc="Mapping colorpickers")):
        # Use string representation as key
        cp_key = str(cp_path) if cp_path is not None else f"missing_{idx}"
        colorpicker_to_samples[cp_key].append(idx)
        sample_to_colorpicker[idx] = cp_key

    unique_colorpickers = list(colorpicker_to_samples.keys())
    print(f"Unique colorpickers: {len(unique_colorpickers)}")

    # Initialize detector and cache
    detector = ColorCheckerDetector(use_opencv_mcc=True)
    cache = ColorCalibrationCache()

    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        viz_dir = output_dir / 'color_viz'
        viz_dir.mkdir(exist_ok=True)

    # Process each unique colorpicker
    failed_colorpickers = []
    high_rmse_count = 0
    visualize_count = 0

    print(f"\nProcessing {len(unique_colorpickers)} unique colorpickers...")

    for cp_key in tqdm(unique_colorpickers, desc="Extracting CCMs"):
        try:
            # Get a sample index that uses this colorpicker
            sample_idx = colorpicker_to_samples[cp_key][0]

            # Load the colorpicker image
            sample = train_data[sample_idx]

            # Try to get the colorpicker image
            if colorpicker_full_col and colorpicker_full_col in sample:
                cp_image = sample[colorpicker_full_col]
            elif 'colorpicker' in sample:
                cp_image = sample['colorpicker']
            else:
                # No colorpicker image available
                cache.mark_failure(cp_key)
                failed_colorpickers.append((cp_key, "No colorpicker image found"))
                continue

            if cp_image is None:
                cache.mark_failure(cp_key)
                failed_colorpickers.append((cp_key, "Colorpicker image is None"))
                continue

            # Convert PIL Image to numpy array
            import numpy as np
            from PIL import Image

            if isinstance(cp_image, Image.Image):
                img_array = np.array(cp_image.convert('RGB'))
            else:
                # Assume it's already an array
                img_array = np.array(cp_image)
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)

            # Detect patches
            patches = detector.detect_patches(img_array)

            if patches is None:
                cache.mark_failure(cp_key)
                failed_colorpickers.append((cp_key, "Patch detection failed"))
                continue

            # Get reference colors based on number of detected patches
            num_patches = len(patches)
            reference = get_reference_colors(num_patches)

            # Compute color correction matrix
            ccm = ColorCorrectionMatrix.from_patches(patches, reference)

            # Check quality
            if ccm.rmse > args.rmse_threshold:
                high_rmse_count += 1

            # Add to cache
            cache.add(cp_key, ccm)

            # Visualize if requested
            if args.visualize and visualize_count < args.max_visualize:
                viz_path = viz_dir / f"colorpicker_{visualize_count:04d}.png"
                visualize_detection(img_array, patches, reference, ccm, viz_path)
                visualize_count += 1

        except Exception as e:
            cache.mark_failure(cp_key)
            failed_colorpickers.append((cp_key, str(e)))

    # Save cache
    cache_path = output_dir / 'color_corrections.pkl'
    cache.save(cache_path)

    # Save sample-to-colorpicker mapping
    mapping_path = output_dir / 'colorpicker_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(sample_to_colorpicker, f)
    print(f"Saved colorpicker mapping to {mapping_path}")

    # Save failed colorpickers
    if failed_colorpickers:
        failed_path = output_dir / 'failed_colorpickers.txt'
        with open(failed_path, 'w') as f:
            for cp_key, reason in failed_colorpickers:
                f.write(f"{cp_key}: {reason}\n")
        print(f"Saved {len(failed_colorpickers)} failed colorpickers to {failed_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    stats = cache.get_stats()
    print(f"Total colorpickers: {len(unique_colorpickers)}")
    print(f"Successfully processed: {stats['total_cached'] - stats['identity_count']}")
    print(f"Failed (using identity): {stats['identity_count']}")
    print(f"High RMSE warnings (>{args.rmse_threshold}): {high_rmse_count}")
    print(f"\nOutput files:")
    print(f"  - {cache_path}")
    print(f"  - {mapping_path}")
    if args.visualize:
        print(f"  - {viz_dir}/ ({visualize_count} visualizations)")
    print(f"\nNext step: Extract embeddings with color normalization:")
    print(f"  python scripts/extract_embeddings.py --backbone bioclip2 --colornorm")


if __name__ == "__main__":
    main()
