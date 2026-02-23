"""
Competition submission model for the Beetles Drought Prediction Challenge.

This file follows the required Model class interface for the competition.
It uses BioCLIP2 for feature extraction and a trained regression head for predictions.

Supports both fixed pooling (mean) and learned pooling (attention, hybrid, etc.)
depending on the checkpoint configuration.

To submit:
1. Copy this file to your submission folder
2. Include model.pth (trained regression head weights + optional pooling weights)
3. Include requirements.txt
4. Create tarball and submit to Codabench
"""

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MODEL ARCHITECTURES (self-contained for submission)
# =============================================================================

class RegressionHead(nn.Module):
    """
    Regression head that predicts SPEI values with uncertainty.

    This is a copy of the model architecture from src/model.py.
    We include it here to make the submission self-contained.
    """

    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=128,
        num_targets=3,
        min_sigma=1e-3,
        dropout=0.1
    ):
        super().__init__()
        self.num_targets = num_targets
        self.min_sigma = min_sigma

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
        sigma = F.softplus(sigma_raw) + self.min_sigma

        return mu, sigma


class SpeciesAwareRegressionHead(nn.Module):
    """
    Regression head that incorporates species name as a learned embedding.
    Concatenates species embedding with image embedding before the regression head.
    """

    def __init__(
        self,
        num_species,
        species_embed_dim=32,
        image_embed_dim=768,
        hidden_dim=128,
        num_targets=3,
        min_sigma=1e-3,
        dropout=0.1
    ):
        super().__init__()
        self.num_species = num_species
        self.species_embed_dim = species_embed_dim

        self.species_embedding = nn.Embedding(
            num_species + 1, species_embed_dim, padding_idx=0
        )

        combined_dim = image_embed_dim + species_embed_dim
        self.head = RegressionHead(
            embedding_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_targets=num_targets,
            min_sigma=min_sigma,
            dropout=dropout
        )

    def forward(self, image_embedding, species_idx):
        species_emb = self.species_embedding(species_idx)
        combined = torch.cat([image_embedding, species_emb], dim=-1)
        return self.head(combined)


# =============================================================================
# POOLING MODULES (self-contained for submission)
# =============================================================================

class MeanPooling(nn.Module):
    """Simple mean pooling across images."""

    def forward(self, embeddings, keepdim=False):
        pooled = embeddings.mean(dim=0)
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        return 1


class AttentionPooling(nn.Module):
    """Attention-weighted pooling."""

    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings, keepdim=False):
        scores = self.attention(embeddings)
        weights = F.softmax(scores, dim=0)
        pooled = (weights * embeddings).sum(dim=0)
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        return 1


class GatedAttentionPooling(nn.Module):
    """Gated attention pooling."""

    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings, keepdim=False):
        A_V = self.attention_V(embeddings)
        A_U = self.attention_U(embeddings)
        scores = self.attention_w(A_V * A_U)
        weights = F.softmax(scores, dim=0)
        pooled = (weights * embeddings).sum(dim=0)
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        return 1


class StatisticsPooling(nn.Module):
    """Statistics pooling: concatenates mean and std."""

    def __init__(self, embed_dim=768, project_to_original=False):
        super().__init__()
        self.project_to_original = project_to_original
        if project_to_original:
            self.projection = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, embeddings, keepdim=False):
        mean = embeddings.mean(dim=0)
        std = embeddings.std(dim=0) if embeddings.size(0) > 1 else torch.zeros_like(mean)
        pooled = torch.cat([mean, std], dim=-1)
        if self.project_to_original:
            pooled = self.projection(pooled)
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        return 1 if self.project_to_original else 2


class HybridAttentionPooling(nn.Module):
    """Hybrid attention pooling: attention-weighted mean + weighted std."""

    def __init__(self, embed_dim=768, hidden_dim=128, project_to_original=False):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.project_to_original = project_to_original
        if project_to_original:
            self.projection = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, embeddings, keepdim=False):
        scores = self.attention(embeddings)
        weights = F.softmax(scores, dim=0)
        weighted_mean = (weights * embeddings).sum(dim=0)
        weighted_var = (weights * (embeddings - weighted_mean).pow(2)).sum(dim=0)
        weighted_std = torch.sqrt(weighted_var + 1e-6)
        pooled = torch.cat([weighted_mean, weighted_std], dim=-1)
        if self.project_to_original:
            pooled = self.projection(pooled)
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        return 1 if self.project_to_original else 2


def create_pooling(pooling_type, embed_dim=768, hidden_dim=128, project_to_original=False):
    """Factory function to create pooling modules."""
    pooling_type = pooling_type.lower()
    if pooling_type == 'mean':
        return MeanPooling()
    elif pooling_type == 'attention':
        return AttentionPooling(embed_dim=embed_dim, hidden_dim=hidden_dim)
    elif pooling_type == 'gated_attention':
        return GatedAttentionPooling(embed_dim=embed_dim, hidden_dim=hidden_dim)
    elif pooling_type == 'statistics':
        return StatisticsPooling(embed_dim=embed_dim, project_to_original=project_to_original)
    elif pooling_type == 'hybrid':
        return HybridAttentionPooling(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                       project_to_original=project_to_original)
    else:
        # Default to mean pooling for unknown types
        print(f"Unknown pooling type '{pooling_type}', using mean pooling")
        return MeanPooling()


# =============================================================================
# MAIN MODEL CLASS
# =============================================================================

class Model:
    """
    Main model class for the competition.

    This class implements the required interface:
    - load(): Load model weights and preprocessing
    - predict(datapoints): Make predictions on a batch of images

    The model uses:
    1. BioCLIP2 backbone for feature extraction (frozen)
    2. Optional learned pooling module
    3. Trained regression head for SPEI prediction
    """

    def __init__(self):
        """Initialize model attributes."""
        self.model = None
        self.backbone = None
        self.preprocess = None
        self.pooling = None
        self.device = None
        self.pooling_type = 'mean'  # Default
        self.use_species = False
        self.species_vocab = None  # {species_name: int_index}

    def load(self):
        """
        Load model weights and preprocessing transforms.

        This method is called once at startup to initialize everything.
        """
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load BioCLIP2 backbone
        print("Loading BioCLIP2 backbone...")
        self._load_backbone()

        # Load trained regression head (and pooling if present)
        print("Loading regression head...")
        self._load_regressor()

        print("Model loaded successfully!")

    def _load_backbone(self):
        """Load the BioCLIP2 feature extraction backbone."""
        from open_clip import create_model_and_transforms

        # Load BioCLIP2 model
        self.backbone, _, self.preprocess = create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2",
            output_dict=True,
            require_pretrained=True
        )

        # Move to device and set to eval mode
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        # Freeze backbone (no gradient computation needed)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _load_regressor(self):
        """Load the trained regression head and pooling module."""
        weights_path = Path(__file__).parent / "model.pth"

        # Default configuration
        embedding_dim = 768
        hidden_dim = 64  # BioCLIP2 models often use smaller hidden dim
        num_targets = 3
        min_sigma = 1e-3
        dropout = 0.1
        pooling_type = 'mean'
        pooling_hidden_dim = 128
        project_to_original = False
        species_config = None

        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Extract configuration from checkpoint if available
            if isinstance(checkpoint, dict):
                # Get model config
                if 'hidden_dim' in checkpoint:
                    hidden_dim = checkpoint['hidden_dim']
                if 'embedding_dim' in checkpoint:
                    embedding_dim = checkpoint['embedding_dim']
                if 'dropout' in checkpoint:
                    dropout = checkpoint['dropout']
                if 'min_sigma' in checkpoint:
                    min_sigma = checkpoint['min_sigma']

                # Get pooling config
                if 'pooling_type' in checkpoint:
                    pooling_type = checkpoint['pooling_type']
                if 'pooling_hidden_dim' in checkpoint:
                    pooling_hidden_dim = checkpoint['pooling_hidden_dim']
                if 'project_to_original' in checkpoint:
                    project_to_original = checkpoint['project_to_original']

                # Get species config
                if 'species_config' in checkpoint:
                    species_config = checkpoint['species_config']

            print(f"Pooling type: {pooling_type}")

        # Create pooling module
        self.pooling_type = pooling_type
        self.pooling = create_pooling(
            pooling_type=pooling_type,
            embed_dim=embedding_dim,
            hidden_dim=pooling_hidden_dim,
            project_to_original=project_to_original
        )

        # Determine input dimension for regression head
        # (depends on pooling output dimension)
        pooled_dim = embedding_dim * self.pooling.output_dim_multiplier

        # Handle species-aware model
        if species_config is not None:
            self.use_species = True
            num_species = species_config['num_species']
            species_embed_dim = species_config['species_embed_dim']

            # Load species vocabulary
            vocab_path = Path(__file__).parent / "species_vocab.json"
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    self.species_vocab = json.load(f)
                print(f"Loaded species vocabulary: {len(self.species_vocab)} entries")
            else:
                print(f"WARNING: species_vocab.json not found at {vocab_path}")
                self.species_vocab = {"<unknown>": 0}

            # Create species-aware regression head
            self.model = SpeciesAwareRegressionHead(
                num_species=num_species,
                species_embed_dim=species_embed_dim,
                image_embed_dim=pooled_dim,
                hidden_dim=hidden_dim,
                num_targets=num_targets,
                min_sigma=min_sigma,
                dropout=dropout
            )
            print(f"Species-aware model: {num_species} species, embed_dim={species_embed_dim}")
        else:
            # Create the standard regression head architecture
            self.model = RegressionHead(
                embedding_dim=pooled_dim,
                hidden_dim=hidden_dim,
                num_targets=num_targets,
                min_sigma=min_sigma,
                dropout=dropout
            )

        # Load trained weights
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                # Load model state
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])

                # Load pooling state if present
                if 'pooling_state_dict' in checkpoint and self.pooling_type != 'mean':
                    self.pooling.load_state_dict(checkpoint['pooling_state_dict'])
                    print(f"Loaded pooling weights for {pooling_type}")
            else:
                # Old format: just model state dict
                self.model.load_state_dict(checkpoint)

            print(f"Loaded weights from {weights_path}")
        else:
            print(f"WARNING: No weights found at {weights_path}")
            print("Model will use random weights!")

        self.model.eval()
        self.model.to(self.device)
        self.pooling.to(self.device)

    @torch.no_grad()
    def predict(self, datapoints):
        """
        Make predictions on a batch of images.

        This method is called by the competition framework for each
        sampling event. All images in datapoints belong to the same event.

        Args:
            datapoints: List of dicts, each containing:
                - "relative_img": PIL Image object
                - "domainID": Integer domain identifier
                - Other metadata fields

        Returns:
            dict: Predictions in the required format:
            {
                "SPEI_30d": {"mu": float, "sigma": float},
                "SPEI_1y": {"mu": float, "sigma": float},
                "SPEI_2y": {"mu": float, "sigma": float},
            }
        """
        # Extract images from datapoints
        images = [entry["relative_img"] for entry in datapoints]

        # Convert images to RGB if needed and preprocess
        processed_images = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_images.append(self.preprocess(img))

        # Stack into batch tensor
        pixel_values = torch.stack(processed_images).to(self.device)

        # Extract features using BioCLIP2 backbone
        # Process in smaller batches if needed
        batch_size = 8
        all_embeddings = []

        for i in range(0, len(pixel_values), batch_size):
            batch = pixel_values[i:i + batch_size]
            outputs = self.backbone(batch)

            # Get image features from BioCLIP2
            embeddings = outputs["image_features"]
            all_embeddings.append(embeddings)

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Apply pooling (learned or fixed)
        aggregated_embedding = self.pooling(all_embeddings, keepdim=True)

        # Get predictions from regression head
        if self.use_species and self.species_vocab is not None:
            # Extract species names from datapoints and find mode (most common)
            species_names = []
            for entry in datapoints:
                name = entry.get("scientificName", "")
                if name is None:
                    name = ""
                species_names.append(name)

            # Map to indices
            species_indices = []
            for name in species_names:
                idx = self.species_vocab.get(name, 0)  # 0 = unknown
                species_indices.append(idx)

            # Pick mode (most common) species for this event
            if species_indices:
                mode_species = Counter(species_indices).most_common(1)[0][0]
            else:
                mode_species = 0

            species_idx = torch.tensor([mode_species], dtype=torch.long, device=self.device)
            mu, sigma = self.model(aggregated_embedding, species_idx)
        else:
            mu, sigma = self.model(aggregated_embedding)

        # Convert to numpy for output
        mu = mu.cpu().numpy()[0]
        sigma = sigma.cpu().numpy()[0]

        # Format output
        predictions = {
            "SPEI_30d": {"mu": float(mu[0]), "sigma": float(sigma[0])},
            "SPEI_1y": {"mu": float(mu[1]), "sigma": float(sigma[1])},
            "SPEI_2y": {"mu": float(mu[2]), "sigma": float(sigma[2])},
        }

        return predictions


# For testing
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    print("Testing Model class...")

    # Create model
    model = Model()
    model.load()

    # Create dummy datapoints (simulating competition input)
    dummy_images = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    datapoints = [{"relative_img": img, "domainID": 1} for img in dummy_images]

    # Make prediction
    predictions = model.predict(datapoints)

    print("\nPredictions:")
    for name, values in predictions.items():
        print(f"  {name}: mu={values['mu']:.4f}, sigma={values['sigma']:.4f}")

    # Verify format
    assert "SPEI_30d" in predictions
    assert "SPEI_1y" in predictions
    assert "SPEI_2y" in predictions
    for v in predictions.values():
        assert "mu" in v
        assert "sigma" in v
        assert v["sigma"] > 0

    print("\nAll tests passed!")
