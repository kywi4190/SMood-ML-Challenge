"""
Feature extraction using DINOv2 or BioClip2.

This module handles loading pretrained vision backbones and extracting
image embeddings. The embeddings are the "features" that capture what's
in each beetle image.

Supported backbones:
- DINOv2: General-purpose self-supervised vision model from Meta
- BioClip2: Biology-specific model trained on organism images (recommended for this task)

For beginners: A pretrained model has already learned to "see" from millions
of images. We use its learned representations as features for our task,
rather than training a vision model from scratch (which would need way more data).
"""

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import BACKBONE_MODEL, FALLBACK_BACKBONE, EMBEDDING_DIM, get_device


# Available backbone options
BACKBONE_OPTIONS = {
    'dinov3': {
        'model_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'type': 'transformers',
        'embedding_dim': 768,
        'description': 'DINOv3 ViT-B/16 - latest self-supervised vision model from Meta (DEFAULT)'
    },
    'dinov2': {
        'model_name': 'facebook/dinov2-base',
        'type': 'transformers',
        'embedding_dim': 768,
        'description': 'DINOv2 base model - general purpose vision features (fallback)'
    },
    'bioclip': {
        'model_name': 'hf-hub:imageomics/bioclip',
        'type': 'open_clip',
        'embedding_dim': 512,
        'description': 'BioCLIP - trained on TreeOfLife-10M biological images (ViT-B/16)'
    },
    'bioclip2': {
        'model_name': 'hf-hub:imageomics/bioclip-2',
        'type': 'open_clip',
        'embedding_dim': 768,
        'description': 'BioCLIP-2 - trained on TreeOfLife-200M, 18% better than BioCLIP (ViT-L/14)'
    }
}


class FeatureExtractor:
    """
    Extract image features using DINOv2 or BioClip2.

    This class wraps pretrained vision models and provides a simple
    interface to convert images into feature vectors (embeddings).

    Example:
        # Using DINOv2 (default)
        extractor = FeatureExtractor(backbone='dinov2')
        extractor.load()
        embedding = extractor.extract_single(image)

        # Using BioClip2
        extractor = FeatureExtractor(backbone='bioclip2')
        extractor.load()
        embedding = extractor.extract_single(image)
    """

    def __init__(self, backbone='dinov2', device=None):
        """
        Initialize the feature extractor.

        Args:
            backbone: Which backbone to use ('dinov2' or 'bioclip2')
            device: Device to run on (default: auto-detect)
        """
        if backbone not in BACKBONE_OPTIONS:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from: {list(BACKBONE_OPTIONS.keys())}")

        self.backbone = backbone
        self.backbone_config = BACKBONE_OPTIONS[backbone]
        self.device = device or get_device()
        self.model = None
        self.processor = None
        self.tokenizer = None  # For BioClip2
        self.embedding_dim = self.backbone_config['embedding_dim']

    def load(self):
        """
        Load the pretrained model and image processor.

        The model is set to evaluation mode and frozen (no gradient computation).
        """
        print(f"Loading feature extractor: {self.backbone}")
        print(f"  Model: {self.backbone_config['model_name']}")
        print(f"  Description: {self.backbone_config['description']}")

        if self.backbone_config['type'] == 'transformers':
            self._load_transformers_model()
        elif self.backbone_config['type'] == 'open_clip':
            self._load_bioclip_model()

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

        # Move model to the appropriate device (GPU if available)
        self.model.to(self.device)

        # Freeze all parameters (we don't want to train the backbone)
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Model loaded on {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def _load_transformers_model(self):
        """Load a model using HuggingFace transformers."""
        from transformers import AutoModel, AutoImageProcessor

        model_name = self.backbone_config['model_name']

        # Load the image processor (handles resizing, normalization, etc.)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Load the model
        self.model = AutoModel.from_pretrained(model_name)

        # Verify embedding dimension
        if hasattr(self.model.config, 'hidden_size'):
            self.embedding_dim = self.model.config.hidden_size

    def _load_bioclip_model(self):
        """Load BioCLIP using open_clip from HuggingFace Hub."""
        import open_clip

        model_name = self.backbone_config['model_name']

        # Load BioCLIP model and preprocessing from HuggingFace Hub
        # Format: hf-hub:imageomics/bioclip
        self.model, _, self.processor = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # BioCLIP uses ViT-B/16 architecture with 512-dim features
        self.embedding_dim = self.backbone_config['embedding_dim']

    def preprocess_image(self, image):
        """
        Preprocess a single image for the model.

        Args:
            image: PIL Image object

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Ensure image is RGB (some images might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.backbone_config['type'] == 'transformers':
            # HuggingFace transformers preprocessing
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values']
        else:
            # open_clip preprocessing (returns tensor directly)
            return self.processor(image).unsqueeze(0)

    def preprocess_batch(self, images):
        """
        Preprocess a batch of images.

        Args:
            images: List of PIL Image objects

        Returns:
            torch.Tensor: Batch of preprocessed image tensors
        """
        # Convert any non-RGB images
        rgb_images = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            rgb_images.append(img)

        if self.backbone_config['type'] == 'transformers':
            # HuggingFace transformers preprocessing
            inputs = self.processor(images=rgb_images, return_tensors="pt")
            return inputs['pixel_values']
        else:
            # open_clip preprocessing
            tensors = [self.processor(img) for img in rgb_images]
            return torch.stack(tensors)

    @torch.no_grad()
    def extract_single(self, image):
        """
        Extract features from a single image.

        Args:
            image: PIL Image object

        Returns:
            torch.Tensor: Feature vector of shape (embedding_dim,)
        """
        # Preprocess the image
        pixel_values = self.preprocess_image(image).to(self.device)

        if self.backbone_config['type'] == 'transformers':
            # Run through the model
            outputs = self.model(pixel_values)

            # Get the [CLS] token embedding
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output[0]
            else:
                embedding = outputs.last_hidden_state[0, 0]
        else:
            # BioClip2: use encode_image method
            embedding = self.model.encode_image(pixel_values)[0]

        return embedding.cpu()

    @torch.no_grad()
    def extract_batch(self, images, batch_size=16, show_progress=True):
        """
        Extract features from a batch of images.

        Args:
            images: List of PIL Image objects
            batch_size: Number of images to process at once
            show_progress: Whether to show a progress bar

        Returns:
            torch.Tensor: Feature matrix of shape (num_images, embedding_dim)
        """
        all_embeddings = []

        # Process images in batches
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features", unit="batch")

        for i in iterator:
            batch_images = images[i:i + batch_size]

            # Preprocess batch
            pixel_values = self.preprocess_batch(batch_images).to(self.device)

            if self.backbone_config['type'] == 'transformers':
                # Run through model
                outputs = self.model(pixel_values)

                # Get embeddings
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    batch_embeddings = outputs.last_hidden_state[:, 0]
            else:
                # BioClip2: use encode_image method
                batch_embeddings = self.model.encode_image(pixel_values)

            all_embeddings.append(batch_embeddings.cpu())

        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)

    def get_transforms(self):
        """
        Get the image transforms/processor for use in data pipelines.

        Returns:
            The image processor object
        """
        return self.processor


def get_embedding_filename(base_name, backbone):
    """
    Generate embedding filename with backbone suffix.

    Args:
        base_name: Base filename (e.g., 'train_embeddings')
        backbone: Backbone name ('dinov2' or 'bioclip2')

    Returns:
        str: Filename with backbone suffix (e.g., 'train_embeddings_dinov2.pt')
    """
    return f"{base_name}_{backbone}.pt"


def extract_and_save_embeddings(
    images,
    output_path,
    backbone='dinov2',
    batch_size=16,
    device=None
):
    """
    Extract embeddings from images and save to disk.

    This is a convenience function for the embedding extraction script.

    Args:
        images: List of PIL Image objects
        output_path: Where to save the embeddings (as .pt file)
        backbone: Which backbone to use ('dinov2' or 'bioclip2')
        batch_size: Batch size for extraction
        device: Device to use (default: auto-detect)

    Returns:
        torch.Tensor: The extracted embeddings
    """
    # Initialize and load the feature extractor
    extractor = FeatureExtractor(backbone=backbone, device=device)
    extractor.load()

    # Extract features
    print(f"Extracting embeddings from {len(images)} images...")
    embeddings = extractor.extract_batch(images, batch_size=batch_size)

    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Embeddings saved to {output_path}")
    print(f"Shape: {embeddings.shape}")

    return embeddings


def load_embeddings(path):
    """
    Load pre-computed embeddings from disk.

    Args:
        path: Path to the .pt file

    Returns:
        torch.Tensor: The loaded embeddings
    """
    embeddings = torch.load(path, weights_only=True)
    print(f"Loaded embeddings from {path}")
    print(f"Shape: {embeddings.shape}")
    return embeddings


def list_available_backbones():
    """Print information about available backbone options."""
    print("\nAvailable backbones:")
    print("-" * 60)
    for name, config in BACKBONE_OPTIONS.items():
        print(f"  {name}:")
        print(f"    Model: {config['model_name']}")
        print(f"    Embedding dim: {config['embedding_dim']}")
        print(f"    Description: {config['description']}")
    print("-" * 60)


if __name__ == "__main__":
    # Quick test of the feature extractor
    print("Testing FeatureExtractor...")

    # List available backbones
    list_available_backbones()

    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')

    # Test with DINOv2
    print("\n--- Testing DINOv2 ---")
    extractor = FeatureExtractor(backbone='dinov2')
    extractor.load()
    embedding = extractor.extract_single(dummy_image)
    print(f"Single image embedding shape: {embedding.shape}")

    print("\nTest complete!")
