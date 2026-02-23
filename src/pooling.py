"""
Pooling mechanisms for aggregating multiple beetle images per sampling event.

This module provides both fixed and learnable pooling strategies:

Fixed Pooling (no learnable parameters):
    - MeanPooling: Average embeddings (default, good baseline)
    - MaxPooling: Element-wise maximum (captures strongest features)
    - SumPooling: Sum embeddings (sensitive to number of images)

Learned Pooling (trainable parameters):
    - AttentionPooling: Attention-weighted pooling (learns which images matter)
    - GatedAttentionPooling: Gated attention (more expressive)
    - StatisticsPooling: Concatenates mean + std (captures distribution shape)
    - HybridAttentionPooling: Attention-weighted mean + weighted std (recommended)

For beginners: When predicting drought conditions from multiple beetle images,
we need to combine their features into a single representation. Fixed pooling
uses simple statistics (mean, max), while learned pooling can learn which
images are most informative for the prediction task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import POOLING_HIDDEN_DIM, EMBEDDING_DIM


# =============================================================================
# FIXED POOLING METHODS (no learnable parameters)
# =============================================================================

class MeanPooling(nn.Module):
    """
    Simple mean pooling across images.

    Takes the average embedding across all images in an event.
    This is the default pooling method and a strong baseline.

    Input: (N, embed_dim) where N = number of images in event
    Output: (embed_dim,) or (1, embed_dim) if keepdim=True
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, keepdim=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of image embeddings
            keepdim: If True, output shape is (1, embed_dim)

        Returns:
            Pooled embedding
        """
        pooled = embeddings.mean(dim=0)
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        """Output dimension is same as input dimension."""
        return 1


class MaxPooling(nn.Module):
    """
    Element-wise maximum pooling across images.

    Takes the maximum value for each embedding dimension across all images.
    Good for capturing the strongest features present in any image.

    Input: (N, embed_dim) where N = number of images in event
    Output: (embed_dim,) or (1, embed_dim) if keepdim=True
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, keepdim=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of image embeddings
            keepdim: If True, output shape is (1, embed_dim)

        Returns:
            Pooled embedding
        """
        pooled = embeddings.max(dim=0)[0]
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        """Output dimension is same as input dimension."""
        return 1


class SumPooling(nn.Module):
    """
    Sum pooling across images.

    Sums all embeddings together. This makes the output sensitive to
    the number of images (more images = larger values).

    Input: (N, embed_dim) where N = number of images in event
    Output: (embed_dim,) or (1, embed_dim) if keepdim=True
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, keepdim=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of image embeddings
            keepdim: If True, output shape is (1, embed_dim)

        Returns:
            Pooled embedding
        """
        pooled = embeddings.sum(dim=0)
        if keepdim:
            pooled = pooled.unsqueeze(0)
        return pooled

    @property
    def output_dim_multiplier(self):
        """Output dimension is same as input dimension."""
        return 1


# =============================================================================
# LEARNED POOLING METHODS (trainable parameters)
# =============================================================================

class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling.

    Learns to assign different importance weights to different images.
    Images that are more informative for drought prediction get higher weights.

    Architecture:
        embedding → Linear → Tanh → Linear → scalar score
        scores → softmax → weights
        weighted sum of embeddings

    Input: (N, embed_dim) where N = number of images in event
    Output: (embed_dim,) or (1, embed_dim) if keepdim=True
    """

    def __init__(self, embed_dim=EMBEDDING_DIM, hidden_dim=POOLING_HIDDEN_DIM):
        """
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden layer size for attention network
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Scalar attention score per image
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize attention network weights."""
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, embeddings, keepdim=False, return_weights=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of image embeddings
            keepdim: If True, output shape is (1, embed_dim)
            return_weights: If True, also return attention weights

        Returns:
            Pooled embedding, and optionally attention weights
        """
        # Compute attention scores
        scores = self.attention(embeddings)  # (N, 1)
        weights = F.softmax(scores, dim=0)   # Normalize across images

        # Weighted sum
        pooled = (weights * embeddings).sum(dim=0)  # (embed_dim,)

        if keepdim:
            pooled = pooled.unsqueeze(0)

        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled

    @property
    def output_dim_multiplier(self):
        """Output dimension is same as input dimension."""
        return 1


class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling (more expressive than simple attention).

    Uses a gating mechanism that combines two pathways:
    - Tanh pathway: captures what features to attend to
    - Sigmoid pathway: gates which features to pass through

    This is more expressive than simple attention and has been shown
    to work well in multiple-instance learning problems.

    Architecture:
        V = Tanh(Linear(embedding))
        U = Sigmoid(Linear(embedding))
        score = Linear(V * U)  # element-wise product
        weights = softmax(scores)

    Input: (N, embed_dim) where N = number of images in event
    Output: (embed_dim,) or (1, embed_dim) if keepdim=True
    """

    def __init__(self, embed_dim=EMBEDDING_DIM, hidden_dim=POOLING_HIDDEN_DIM):
        """
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden layer size for attention network
        """
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()  # Gate
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize attention network weights."""
        for module in [self.attention_V, self.attention_U, self.attention_w]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    nn.init.zeros_(submodule.bias)

    def forward(self, embeddings, keepdim=False, return_weights=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of image embeddings
            keepdim: If True, output shape is (1, embed_dim)
            return_weights: If True, also return attention weights

        Returns:
            Pooled embedding, and optionally attention weights
        """
        # Gated attention: tanh(Vx) * sigmoid(Ux)
        A_V = self.attention_V(embeddings)  # (N, hidden_dim)
        A_U = self.attention_U(embeddings)  # (N, hidden_dim)
        scores = self.attention_w(A_V * A_U)  # (N, 1)

        weights = F.softmax(scores, dim=0)
        pooled = (weights * embeddings).sum(dim=0)

        if keepdim:
            pooled = pooled.unsqueeze(0)

        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled

    @property
    def output_dim_multiplier(self):
        """Output dimension is same as input dimension."""
        return 1


class StatisticsPooling(nn.Module):
    """
    Statistics pooling: concatenates mean and standard deviation.

    Captures both the central tendency (mean) and spread (std) of the
    embedding distribution. This can help capture image diversity within
    an event, which may correlate with ecological conditions.

    Input: (N, embed_dim) where N = number of images in event
    Output: (embed_dim * 2,) or (1, embed_dim * 2) if keepdim=True

    Note: This doubles the embedding dimension, so the regression head
    input dimension must be adjusted accordingly.
    """

    def __init__(self, embed_dim=EMBEDDING_DIM, project_to_original=False):
        """
        Args:
            embed_dim: Dimension of input embeddings
            project_to_original: If True, project back to embed_dim
        """
        super().__init__()

        self.project_to_original = project_to_original
        if project_to_original:
            self.projection = nn.Linear(embed_dim * 2, embed_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, embeddings, keepdim=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of image embeddings
            keepdim: If True, output shape is (1, output_dim)

        Returns:
            Pooled embedding (concatenation of mean and std)
        """
        mean = embeddings.mean(dim=0)
        # Add small epsilon for numerical stability with single image
        std = embeddings.std(dim=0) if embeddings.size(0) > 1 else torch.zeros_like(mean)

        pooled = torch.cat([mean, std], dim=-1)

        if self.project_to_original:
            pooled = self.projection(pooled)

        if keepdim:
            pooled = pooled.unsqueeze(0)

        return pooled

    @property
    def output_dim_multiplier(self):
        """Output dimension is 2x input if not projecting, else 1x."""
        return 1 if self.project_to_original else 2


class HybridAttentionPooling(nn.Module):
    """
    Hybrid attention pooling: attention-weighted mean + weighted variance.

    This is the recommended pooling method for CRPS-based evaluation because:
    1. Attention learns which images are most informative
    2. Weighted variance captures uncertainty about the aggregated representation
    3. This uncertainty signal can help calibrate the predicted sigma

    Architecture:
        attention scores → softmax → weights
        weighted_mean = sum(weights * embeddings)
        weighted_var = sum(weights * (embeddings - weighted_mean)^2)
        output = concat(weighted_mean, sqrt(weighted_var))

    Input: (N, embed_dim) where N = number of images in event
    Output: (embed_dim * 2,) or (1, embed_dim * 2) if keepdim=True

    Note: This doubles the embedding dimension.
    """

    def __init__(self, embed_dim=EMBEDDING_DIM, hidden_dim=POOLING_HIDDEN_DIM,
                 project_to_original=False):
        """
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden layer size for attention network
            project_to_original: If True, project back to embed_dim
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.project_to_original = project_to_original
        if project_to_original:
            self.projection = nn.Linear(embed_dim * 2, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.project_to_original:
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, embeddings, keepdim=False, return_weights=False):
        """
        Args:
            embeddings: (N, embed_dim) tensor of image embeddings
            keepdim: If True, output shape is (1, output_dim)
            return_weights: If True, also return attention weights

        Returns:
            Pooled embedding, and optionally attention weights
        """
        # Compute attention weights
        scores = self.attention(embeddings)  # (N, 1)
        weights = F.softmax(scores, dim=0)

        # Weighted mean
        weighted_mean = (weights * embeddings).sum(dim=0)

        # Weighted standard deviation (as uncertainty signal)
        # Add epsilon for numerical stability
        weighted_var = (weights * (embeddings - weighted_mean).pow(2)).sum(dim=0)
        weighted_std = torch.sqrt(weighted_var + 1e-6)

        # Concatenate mean and std
        pooled = torch.cat([weighted_mean, weighted_std], dim=-1)

        if self.project_to_original:
            pooled = self.projection(pooled)

        if keepdim:
            pooled = pooled.unsqueeze(0)

        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled

    @property
    def output_dim_multiplier(self):
        """Output dimension is 2x input if not projecting, else 1x."""
        return 1 if self.project_to_original else 2


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pooling(pooling_type, embed_dim=EMBEDDING_DIM, hidden_dim=POOLING_HIDDEN_DIM,
                   project_to_original=False):
    """
    Factory function to create pooling modules.

    Args:
        pooling_type: Type of pooling to use:
            - 'mean': Simple mean pooling (fixed)
            - 'max': Element-wise max pooling (fixed)
            - 'sum': Sum pooling (fixed)
            - 'attention': Attention-weighted pooling (learned)
            - 'gated_attention': Gated attention pooling (learned)
            - 'statistics': Mean + std concatenation (learned projection optional)
            - 'hybrid': Attention-weighted mean + std (learned, recommended)
        embed_dim: Dimension of input embeddings
        hidden_dim: Hidden dimension for attention networks
        project_to_original: For statistics/hybrid, whether to project back to embed_dim

    Returns:
        nn.Module: The pooling module
    """
    pooling_type = pooling_type.lower()

    if pooling_type == 'mean':
        return MeanPooling()
    elif pooling_type == 'max':
        return MaxPooling()
    elif pooling_type == 'sum':
        return SumPooling()
    elif pooling_type == 'attention':
        return AttentionPooling(embed_dim=embed_dim, hidden_dim=hidden_dim)
    elif pooling_type == 'gated_attention':
        return GatedAttentionPooling(embed_dim=embed_dim, hidden_dim=hidden_dim)
    elif pooling_type == 'statistics':
        return StatisticsPooling(embed_dim=embed_dim, project_to_original=project_to_original)
    elif pooling_type == 'hybrid':
        return HybridAttentionPooling(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            project_to_original=project_to_original
        )
    else:
        raise ValueError(
            f"Unknown pooling type: {pooling_type}. "
            f"Choose from: mean, max, sum, attention, gated_attention, statistics, hybrid"
        )


def get_pooling_output_dim(pooling_type, embed_dim, project_to_original=False):
    """
    Get the output dimension of a pooling module.

    Args:
        pooling_type: Type of pooling
        embed_dim: Input embedding dimension
        project_to_original: For statistics/hybrid, whether projecting back

    Returns:
        int: Output dimension after pooling
    """
    pooling_type = pooling_type.lower()

    if pooling_type in ['mean', 'max', 'sum', 'attention', 'gated_attention']:
        return embed_dim
    elif pooling_type in ['statistics', 'hybrid']:
        return embed_dim if project_to_original else embed_dim * 2
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


# =============================================================================
# HELPER FUNCTION FOR APPLYING POOLING TO BATCHED DATA
# =============================================================================

def apply_pooling_to_event_batch(pooling_module, embeddings_list):
    """
    Apply pooling to a batch of events with variable numbers of images.

    Args:
        pooling_module: A pooling module instance
        embeddings_list: List of (N_i, embed_dim) tensors, one per event

    Returns:
        (batch_size, pooled_dim) tensor of pooled embeddings
    """
    pooled = []
    for event_embeddings in embeddings_list:
        pooled_emb = pooling_module(event_embeddings, keepdim=True)
        pooled.append(pooled_emb)
    return torch.cat(pooled, dim=0)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Pooling Modules...")
    print("=" * 60)

    # Test parameters
    num_images = 10
    embed_dim = 768
    batch_embeddings = torch.randn(num_images, embed_dim)

    # Test all pooling types
    pooling_types = ['mean', 'max', 'sum', 'attention', 'gated_attention',
                     'statistics', 'hybrid']

    for pooling_type in pooling_types:
        print(f"\nTesting {pooling_type} pooling...")

        pooling = create_pooling(pooling_type, embed_dim=embed_dim)
        output_dim = get_pooling_output_dim(pooling_type, embed_dim)

        # Test forward pass
        pooled = pooling(batch_embeddings)
        print(f"  Input shape: {batch_embeddings.shape}")
        print(f"  Output shape: {pooled.shape}")
        print(f"  Expected output dim: {output_dim}")
        assert pooled.shape[-1] == output_dim, f"Dimension mismatch!"

        # Test with keepdim
        pooled_keepdim = pooling(batch_embeddings, keepdim=True)
        print(f"  Output shape (keepdim): {pooled_keepdim.shape}")
        assert pooled_keepdim.shape == (1, output_dim)

        # Count parameters
        num_params = sum(p.numel() for p in pooling.parameters())
        print(f"  Trainable parameters: {num_params:,}")

    # Test attention weights return
    print("\n" + "=" * 60)
    print("Testing attention weight extraction...")
    attention_pooling = create_pooling('attention', embed_dim=embed_dim)
    pooled, weights = attention_pooling(batch_embeddings, return_weights=True)
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum: {weights.sum().item():.4f} (should be ~1.0)")
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Weights should sum to 1"

    # Test with project_to_original
    print("\n" + "=" * 60)
    print("Testing statistics pooling with projection...")
    stats_pooling = create_pooling('statistics', embed_dim=embed_dim, project_to_original=True)
    pooled_proj = stats_pooling(batch_embeddings)
    print(f"  Output shape: {pooled_proj.shape}")
    assert pooled_proj.shape[-1] == embed_dim, "Projected output should match embed_dim"

    # Test batched application
    print("\n" + "=" * 60)
    print("Testing batched application...")
    events = [
        torch.randn(5, embed_dim),   # Event with 5 images
        torch.randn(10, embed_dim),  # Event with 10 images
        torch.randn(3, embed_dim),   # Event with 3 images
    ]
    pooling = create_pooling('attention', embed_dim=embed_dim)
    batched_output = apply_pooling_to_event_batch(pooling, events)
    print(f"  Number of events: {len(events)}")
    print(f"  Batched output shape: {batched_output.shape}")
    assert batched_output.shape == (3, embed_dim)

    print("\n" + "=" * 60)
    print("All pooling tests passed!")
