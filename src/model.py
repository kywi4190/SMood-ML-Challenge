"""
Neural network model for the Beetles Drought Prediction project.

This module defines the regression head that takes beetle image embeddings
and predicts SPEI drought metrics with uncertainty estimates.

Architecture:
    [Embedding (768-dim)] → [Linear] → [ReLU] → [Linear] → [6 outputs]
                                                            ↓
                                                    [3 mu, 3 sigma]

For beginners: This is a Multi-Layer Perceptron (MLP), the simplest type
of neural network. It takes the image embedding and learns to predict
the drought values through a series of transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_TARGETS, MIN_SIGMA, DROPOUT,
    TARGET_COLUMNS, TARGET_INDICES_SHORT_TERM, TARGET_INDICES_LONG_TERM
)


class RegressionHead(nn.Module):
    """
    Simple MLP that predicts SPEI values with uncertainty from embeddings.

    This model outputs 6 values:
    - 3 mu values: predicted means for SPEI_30d, SPEI_1y, SPEI_2y
    - 3 sigma values: predicted uncertainties (standard deviations)

    The sigma values use softplus activation to ensure they're always positive.

    Example:
        model = RegressionHead(embedding_dim=768)
        embedding = torch.randn(32, 768)  # Batch of 32 embeddings
        mu, sigma = model(embedding)
        # mu: shape (32, 3), sigma: shape (32, 3)
    """

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_targets=NUM_TARGETS,
        min_sigma=MIN_SIGMA,
        dropout=DROPOUT
    ):
        """
        Initialize the regression head.

        Args:
            embedding_dim: Size of input embeddings (768 for DINOv3-base)
            hidden_dim: Size of the hidden layer
            num_targets: Number of SPEI metrics to predict (3)
            min_sigma: Minimum allowed sigma value
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.num_targets = num_targets
        self.min_sigma = min_sigma

        # First linear layer: compress embedding to hidden dimension
        # This learns what aspects of the embedding are relevant for our task
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)

        # Dropout for regularization (prevents overfitting)
        # During training, randomly sets some values to 0
        self.dropout = nn.Dropout(dropout)

        # Second linear layer: hidden to outputs
        # Outputs 6 values: 3 for mu, 3 for sigma
        self.fc2 = nn.Linear(hidden_dim, num_targets * 2)

        # Initialize weights (helps training converge faster)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize network weights.

        Using Xavier initialization for better gradient flow.
        This helps the model train more smoothly from the start.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input embeddings, shape (batch_size, embedding_dim)

        Returns:
            tuple: (mu, sigma) each of shape (batch_size, num_targets)
        """
        # First layer with ReLU activation
        # ReLU (Rectified Linear Unit) introduces non-linearity
        # It simply turns negative values to 0
        x = F.relu(self.fc1(x))

        # Apply dropout (only active during training)
        x = self.dropout(x)

        # Output layer - get 6 raw values
        output = self.fc2(x)

        # Split into mu (first 3) and sigma (last 3)
        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]

        # Apply softplus to sigma to ensure it's positive
        # softplus(x) = log(1 + exp(x)), which is always > 0
        # Then add min_sigma to prevent values too close to 0
        sigma = F.softplus(sigma_raw) + self.min_sigma

        return mu, sigma

    def predict(self, x):
        """
        Make predictions (same as forward, but clearer name for inference).

        Args:
            x: Input embeddings

        Returns:
            tuple: (mu, sigma)
        """
        return self.forward(x)

    def get_prediction_dict(self, x, target_names=None):
        """
        Get predictions in the competition output format.

        Args:
            x: Input embedding for a SINGLE sample, shape (embedding_dim,)
            target_names: Names for each target (default: config.TARGET_COLUMNS)

        Returns:
            dict: Competition format predictions
        """
        if target_names is None:
            target_names = TARGET_COLUMNS

        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Get predictions
        mu, sigma = self.forward(x)

        # Convert to competition format
        predictions = {}
        for i, name in enumerate(target_names):
            predictions[name] = {
                "mu": mu[0, i].item(),
                "sigma": sigma[0, i].item()
            }

        return predictions


class DeeperRegressionHead(nn.Module):
    """
    A deeper version of the regression head with more layers.

    This might learn more complex patterns but also risks overfitting.
    Use this if the simple model underfits (validation performance is poor).

    Architecture:
        [768] → [512] → [256] → [128] → [6]
    """

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=[512, 256, 128],
        num_targets=NUM_TARGETS,
        min_sigma=MIN_SIGMA,
        dropout=DROPOUT
    ):
        """
        Initialize the deeper regression head.

        Args:
            embedding_dim: Size of input embeddings
            hidden_dims: List of hidden layer sizes
            num_targets: Number of SPEI metrics to predict
            min_sigma: Minimum allowed sigma value
            dropout: Dropout probability
        """
        super().__init__()

        self.num_targets = num_targets
        self.min_sigma = min_sigma

        # Build the layers dynamically
        layers = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Create sequential module from layers
        self.hidden_layers = nn.Sequential(*layers)

        # Final output layer
        self.output_layer = nn.Linear(in_dim, num_targets * 2)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input embeddings, shape (batch_size, embedding_dim)

        Returns:
            tuple: (mu, sigma)
        """
        # Pass through hidden layers
        x = self.hidden_layers(x)

        # Get outputs
        output = self.output_layer(x)

        # Split and process mu/sigma
        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + self.min_sigma

        return mu, sigma


class SpeciesAwareRegressionHead(nn.Module):
    """
    Regression head that incorporates species name as a learned embedding.

    Concatenates a learned species embedding with the image embedding before
    passing through the regression head. This allows the model to leverage
    species identity as supplementary information.

    The species embedding is small (default 32-dim) relative to the image
    embedding (768-dim), treating species as auxiliary info rather than the
    primary signal. Index 0 is reserved for unknown species.
    """

    def __init__(
        self,
        num_species,
        species_embed_dim=32,
        image_embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_targets=NUM_TARGETS,
        min_sigma=MIN_SIGMA,
        dropout=DROPOUT,
        model_type='simple',
        size_factor=0.5
    ):
        """
        Initialize the species-aware regression head.

        Args:
            num_species: Number of unique species in vocabulary (excluding unknown)
            species_embed_dim: Dimension of species embedding
            image_embed_dim: Dimension of image embeddings
            hidden_dim: Hidden layer dimension for the regression head
            num_targets: Number of SPEI metrics to predict
            min_sigma: Minimum sigma value
            dropout: Dropout probability
            model_type: 'simple' or 'deep' for the inner regression head
            size_factor: Layer size decay factor for deep models
        """
        super().__init__()

        self.num_species = num_species
        self.species_embed_dim = species_embed_dim

        # Species embedding layer (padding_idx=0 for unknown species)
        self.species_embedding = nn.Embedding(
            num_species + 1, species_embed_dim, padding_idx=0
        )

        # Combined dimension: image embedding + species embedding
        combined_dim = image_embed_dim + species_embed_dim

        # Inner regression head operates on the combined features
        self.head = create_model(
            model_type=model_type,
            size_factor=size_factor,
            embedding_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_targets=num_targets,
            min_sigma=min_sigma,
            dropout=dropout
        )

    def forward(self, image_embedding, species_idx):
        """
        Forward pass.

        Args:
            image_embedding: Image embeddings, shape (batch_size, image_embed_dim)
            species_idx: Species indices, shape (batch_size,)

        Returns:
            tuple: (mu, sigma) each of shape (batch_size, num_targets)
        """
        species_emb = self.species_embedding(species_idx)  # (batch, species_embed_dim)
        combined = torch.cat([image_embedding, species_emb], dim=-1)
        return self.head(combined)


def create_model(model_type='simple', size_factor=0.5, **kwargs):
    """
    Factory function to create models.

    Args:
        model_type: 'simple' or 'deep'
        size_factor: For 'deep' models, controls layer size decay (default 0.5).
                     Layer sizes are: [base_dim, base_dim*factor, base_dim*factor^2]
                     E.g., size_factor=0.5 with hidden_dim=128 gives [128, 64, 32]
                     E.g., size_factor=0.25 with hidden_dim=128 gives [128, 32, 8]
        **kwargs: Additional arguments for the model

    Returns:
        nn.Module: The created model
    """
    if model_type == 'simple':
        # Remove size_factor if present (not used for simple model)
        kwargs.pop('size_factor', None)
        return RegressionHead(**kwargs)
    elif model_type == 'deep':
        # DeeperRegressionHead uses hidden_dims (list) not hidden_dim (int)
        # Convert hidden_dim to a list of decreasing sizes if provided
        if 'hidden_dim' in kwargs:
            base_dim = kwargs.pop('hidden_dim')
            # Create decreasing layer sizes using size_factor
            layer1 = base_dim
            layer2 = max(1, int(base_dim * size_factor))
            layer3 = max(1, int(base_dim * size_factor * size_factor))
            kwargs['hidden_dims'] = [layer1, layer2, layer3]
        return DeeperRegressionHead(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# SPECIALIZED MODELS (for target-specific training)
# =============================================================================

class SpecializedRegressionHead(nn.Module):
    """
    Simple regression head that predicts only specific SPEI targets.

    Unlike RegressionHead which predicts all 3 targets, this model predicts
    a subset of targets (e.g., only SPEI_30d or only SPEI_1y + SPEI_2y).
    This eliminates gradient interference from unrelated targets and allows
    different backbones/hyperparameters for different prediction horizons.

    Architecture: embedding → hidden → output

    Example:
        # Model for short-term (30d) prediction
        model_30d = SpecializedRegressionHead(
            embedding_dim=768,
            num_targets=1,  # Only SPEI_30d
            hidden_dim=128
        )

        # Model for long-term (1y, 2y) prediction
        model_1y2y = SpecializedRegressionHead(
            embedding_dim=768,
            num_targets=2,  # SPEI_1y and SPEI_2y
            hidden_dim=64
        )
    """

    def __init__(
        self,
        embedding_dim,
        num_targets,
        hidden_dim,
        dropout=DROPOUT,
        min_sigma=MIN_SIGMA
    ):
        """
        Initialize the specialized regression head.

        Args:
            embedding_dim: Size of input embeddings
            num_targets: Number of targets to predict (1 for 30d, 2 for 1y+2y)
            hidden_dim: Size of hidden layer
            dropout: Dropout probability
            min_sigma: Minimum sigma value for numerical stability
        """
        super().__init__()

        self.num_targets = num_targets
        self.min_sigma = min_sigma

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

        Args:
            x: Input embeddings, shape (batch_size, embedding_dim)

        Returns:
            tuple: (mu, sigma) each of shape (batch_size, num_targets)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)

        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + self.min_sigma

        return mu, sigma


class DeeperSpecializedRegressionHead(nn.Module):
    """
    Deeper regression head with multiple hidden layers for specialized targets.

    Architecture: embedding → hidden1 → hidden2 → hidden3 → output

    Layer sizes decrease based on size_factor:
    [hidden_dim, hidden_dim*factor, hidden_dim*factor^2]

    Example:
        model = DeeperSpecializedRegressionHead(
            embedding_dim=768,
            num_targets=1,
            hidden_dim=128,
            size_factor=0.5  # Layers: 128 → 64 → 32
        )
    """

    def __init__(
        self,
        embedding_dim,
        num_targets,
        hidden_dim,
        dropout=DROPOUT,
        size_factor=0.5,
        min_sigma=MIN_SIGMA
    ):
        """
        Initialize the deeper specialized regression head.

        Args:
            embedding_dim: Size of input embeddings
            num_targets: Number of targets to predict
            hidden_dim: Size of first hidden layer (subsequent layers are smaller)
            dropout: Dropout probability
            size_factor: Layer size decay factor (default 0.5)
            min_sigma: Minimum sigma value for numerical stability
        """
        super().__init__()

        self.num_targets = num_targets
        self.min_sigma = min_sigma

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
        """
        Forward pass through multiple layers.

        Args:
            x: Input embeddings, shape (batch_size, embedding_dim)

        Returns:
            tuple: (mu, sigma) each of shape (batch_size, num_targets)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        output = self.fc_out(x)

        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + self.min_sigma

        return mu, sigma


def create_specialized_model(
    embedding_dim,
    num_targets,
    hidden_dim,
    dropout=DROPOUT,
    model_type='simple',
    size_factor=0.5,
    min_sigma=MIN_SIGMA
):
    """
    Factory function to create specialized models.

    Specialized models predict a subset of targets (e.g., only SPEI_30d or
    only SPEI_1y + SPEI_2y), allowing different backbones and hyperparameters
    for different prediction horizons.

    Args:
        embedding_dim: Size of input embeddings
        num_targets: Number of targets to predict (1 for 30d, 2 for 1y+2y)
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
        model_type: 'simple' or 'deep'
        size_factor: For 'deep' models, controls layer size decay (default 0.5)
        min_sigma: Minimum sigma value for numerical stability

    Returns:
        nn.Module: The created specialized model
    """
    if model_type == 'simple':
        return SpecializedRegressionHead(
            embedding_dim=embedding_dim,
            num_targets=num_targets,
            hidden_dim=hidden_dim,
            dropout=dropout,
            min_sigma=min_sigma
        )
    elif model_type == 'deep':
        return DeeperSpecializedRegressionHead(
            embedding_dim=embedding_dim,
            num_targets=num_targets,
            hidden_dim=hidden_dim,
            dropout=dropout,
            size_factor=size_factor,
            min_sigma=min_sigma
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'simple' or 'deep'")


if __name__ == "__main__":
    # Quick test of the models
    print("Testing RegressionHead...")

    # Create model
    model = RegressionHead(embedding_dim=768, hidden_dim=256)
    print(f"Model architecture:\n{model}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    embedding = torch.randn(batch_size, 768)
    mu, sigma = model(embedding)

    print(f"\nInput shape: {embedding.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Mu values: {mu[0]}")
    print(f"Sigma values: {sigma[0]} (should be positive)")

    # Verify sigma is positive
    assert (sigma > 0).all(), "Sigma should be positive!"
    print("\n✓ Sigma values are all positive")

    # Test prediction dict format
    single_embedding = embedding[0]
    pred_dict = model.get_prediction_dict(single_embedding)
    print(f"\nPrediction dict format:")
    for name, values in pred_dict.items():
        print(f"  {name}: mu={values['mu']:.4f}, sigma={values['sigma']:.4f}")

    # Test deeper model
    print("\n" + "="*50)
    print("Testing DeeperRegressionHead...")
    deep_model = DeeperRegressionHead(embedding_dim=768)
    num_params_deep = sum(p.numel() for p in deep_model.parameters())
    print(f"Deep model parameters: {num_params_deep:,}")

    mu_deep, sigma_deep = deep_model(embedding)
    print(f"Deep model output shapes: mu={mu_deep.shape}, sigma={sigma_deep.shape}")

    # Test specialized models
    print("\n" + "="*50)
    print("Testing SpecializedRegressionHead (1 target)...")
    spec_model_1 = SpecializedRegressionHead(
        embedding_dim=768,
        num_targets=1,
        hidden_dim=128
    )
    mu_spec1, sigma_spec1 = spec_model_1(embedding)
    print(f"Output shapes: mu={mu_spec1.shape}, sigma={sigma_spec1.shape}")
    assert mu_spec1.shape == (batch_size, 1), "Expected 1 target output"

    print("\n" + "="*50)
    print("Testing SpecializedRegressionHead (2 targets)...")
    spec_model_2 = SpecializedRegressionHead(
        embedding_dim=768,
        num_targets=2,
        hidden_dim=64
    )
    mu_spec2, sigma_spec2 = spec_model_2(embedding)
    print(f"Output shapes: mu={mu_spec2.shape}, sigma={sigma_spec2.shape}")
    assert mu_spec2.shape == (batch_size, 2), "Expected 2 target outputs"

    print("\n" + "="*50)
    print("Testing create_specialized_model factory...")
    for model_type in ['simple', 'deep']:
        model = create_specialized_model(
            embedding_dim=768,
            num_targets=1,
            hidden_dim=128,
            model_type=model_type
        )
        mu, sigma = model(embedding)
        print(f"  {model_type}: mu={mu.shape}, sigma={sigma.shape}")

    print("\nAll tests passed!")
