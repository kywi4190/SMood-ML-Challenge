"""
Loss functions for the Beetles Drought Prediction project.

This module implements the Gaussian Negative Log-Likelihood (NLL) loss,
which is ideal for training models that predict both a value AND its uncertainty.

For beginners: A loss function measures how "wrong" the model's predictions are.
During training, we try to minimize this loss, making predictions better.
"""

import torch
import torch.nn as nn
import math


def gaussian_nll_loss(mu, sigma, target, reduction='mean'):
    """
    Compute Gaussian Negative Log-Likelihood loss.

    This loss is special because it trains the model to predict TWO things:
    1. mu (μ): The expected value - what we think the answer is
    2. sigma (σ): The uncertainty - how confident we are

    The math behind it:
    -----------------
    We assume our predictions follow a Gaussian (normal) distribution.
    The likelihood of observing the true value 'target' given our prediction
    (mu, sigma) is:

        p(target | mu, sigma) = (1 / sqrt(2π σ²)) * exp(-(target - mu)² / (2σ²))

    The negative log-likelihood is:

        NLL = 0.5 * log(σ²) + 0.5 * (target - mu)² / σ²

    Why this works:
    ---------------
    - The first term (log σ²) penalizes overconfidence (small sigma)
    - The second term penalizes prediction errors
    - If sigma is small (confident), errors are heavily penalized
    - If sigma is large (uncertain), errors are less penalized
    - This encourages the model to be confident when it can be, and
      uncertain when predictions are difficult

    Args:
        mu: Predicted means, shape (batch_size, num_targets)
        sigma: Predicted standard deviations, shape (batch_size, num_targets)
        target: True values, shape (batch_size, num_targets)
        reduction: How to combine losses - 'mean', 'sum', or 'none'

    Returns:
        Loss value (scalar if reduction='mean' or 'sum', tensor if 'none')
    """
    # Ensure sigma is positive by clamping to a minimum value
    # This prevents numerical issues like log(0) or division by 0
    sigma = torch.clamp(sigma, min=1e-6)

    # Compute variance (sigma squared)
    variance = sigma ** 2

    # Compute the two terms of the NLL loss
    # Term 1: log(variance) - penalizes small sigma (overconfidence)
    log_variance_term = 0.5 * torch.log(variance)

    # Term 2: squared error normalized by variance
    # Large errors with small variance = big penalty (model was wrongly confident)
    # Large errors with large variance = smaller penalty (model admitted uncertainty)
    squared_error_term = 0.5 * ((target - mu) ** 2) / variance

    # Combine the two terms
    loss = log_variance_term + squared_error_term

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss as a PyTorch module.

    This is a wrapper around the gaussian_nll_loss function that can be
    used like any other PyTorch loss function (e.g., nn.MSELoss).

    Example:
        criterion = GaussianNLLLoss()
        loss = criterion(predicted_mu, predicted_sigma, targets)
        loss.backward()  # Backpropagate gradients
    """

    def __init__(self, reduction='mean', min_sigma=1e-6):
        """
        Initialize the loss function.

        Args:
            reduction: How to combine losses ('mean', 'sum', or 'none')
            min_sigma: Minimum allowed sigma value to prevent numerical issues
        """
        super().__init__()
        self.reduction = reduction
        self.min_sigma = min_sigma

    def forward(self, mu, sigma, target):
        """
        Compute the loss.

        Args:
            mu: Predicted means, shape (batch_size, num_targets)
            sigma: Predicted standard deviations, shape (batch_size, num_targets)
            target: True values, shape (batch_size, num_targets)

        Returns:
            Loss value
        """
        # Clamp sigma to minimum value
        sigma = torch.clamp(sigma, min=self.min_sigma)
        return gaussian_nll_loss(mu, sigma, target, reduction=self.reduction)


def compute_per_target_loss(mu, sigma, target, target_names=None):
    """
    Compute loss for each target separately (useful for analysis).

    This helps you understand which SPEI metrics are easier/harder to predict.

    Args:
        mu: Predicted means, shape (batch_size, 3)
        sigma: Predicted standard deviations, shape (batch_size, 3)
        target: True values, shape (batch_size, 3)
        target_names: Names for each target (default: SPEI metrics)

    Returns:
        dict: Loss value for each target
    """
    if target_names is None:
        target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    losses = {}
    for i, name in enumerate(target_names):
        # Extract single target column
        mu_i = mu[:, i]
        sigma_i = sigma[:, i]
        target_i = target[:, i]

        # Compute loss for this target
        loss_i = gaussian_nll_loss(
            mu_i.unsqueeze(1),
            sigma_i.unsqueeze(1),
            target_i.unsqueeze(1),
            reduction='mean'
        )
        losses[name] = loss_i.item()

    return losses


# =============================================================================
# ALTERNATIVE LOSS FUNCTIONS (for experimentation)
# =============================================================================

def heteroscedastic_mse_loss(mu, sigma, target, reduction='mean'):
    """
    Heteroscedastic MSE loss - a simpler alternative to Gaussian NLL.

    This loss downweights the contribution of samples with high uncertainty,
    encouraging the model to focus on samples it can predict well.

    The formula is: (target - mu)² / (2 * sigma²)

    Note: This ignores the log(variance) term, so sigma can grow unbounded.
    Use with caution - the model might learn to predict huge sigma for
    everything to minimize loss.

    Args:
        mu: Predicted means
        sigma: Predicted standard deviations
        target: True values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    sigma = torch.clamp(sigma, min=1e-6)
    loss = 0.5 * ((target - mu) ** 2) / (sigma ** 2)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


if __name__ == "__main__":
    # Quick test of the loss functions
    print("Testing Gaussian NLL Loss...")

    # Create some dummy data
    batch_size = 4
    num_targets = 3

    mu = torch.randn(batch_size, num_targets)
    sigma = torch.abs(torch.randn(batch_size, num_targets)) + 0.1
    target = torch.randn(batch_size, num_targets)

    # Test functional version
    loss = gaussian_nll_loss(mu, sigma, target)
    print(f"Functional loss: {loss.item():.4f}")

    # Test module version
    criterion = GaussianNLLLoss()
    loss_module = criterion(mu, sigma, target)
    print(f"Module loss: {loss_module.item():.4f}")

    # Test per-target loss
    per_target = compute_per_target_loss(mu, sigma, target)
    print(f"Per-target losses: {per_target}")

    print("\nAll tests passed!")
