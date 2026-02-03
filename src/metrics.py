"""
Evaluation metrics for the Beetles Drought Prediction project.

This module implements the Continuous Ranked Probability Score (CRPS),
which is the official evaluation metric for the competition.

For beginners: CRPS measures both prediction accuracy AND uncertainty quality.
A model that predicts well AND has well-calibrated uncertainty will score best.
"""

import torch
import numpy as np
from scipy import stats
import math


def crps_gaussian(mu, sigma, observation):
    """
    Compute the Continuous Ranked Probability Score (CRPS) for Gaussian predictions.

    CRPS measures how good a probabilistic prediction is. Unlike simple metrics
    like MAE or MSE, CRPS rewards predictions that:
    1. Are close to the true value (accuracy)
    2. Have well-calibrated uncertainty (reliability)

    The formula for Gaussian CRPS is:
        CRPS = σ * [z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π]

    where:
        z = (observation - μ) / σ  (standardized error)
        Φ(z) = CDF of standard normal (probability z is less than value)
        φ(z) = PDF of standard normal (density at z)

    Lower CRPS is better (like loss).

    Args:
        mu: Predicted mean (float, array, or tensor)
        sigma: Predicted standard deviation (float, array, or tensor)
        observation: True value (float, array, or tensor)

    Returns:
        CRPS value(s) - same shape as inputs
    """
    # Convert to numpy if tensors
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(observation, torch.Tensor):
        observation = observation.detach().cpu().numpy()

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)

    # Standardize the observation
    z = (observation - mu) / sigma

    # Get standard normal PDF and CDF values
    phi_z = stats.norm.pdf(z)  # PDF: density at z
    Phi_z = stats.norm.cdf(z)  # CDF: probability <= z

    # Compute CRPS using the closed-form formula
    # CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))

    return crps


def compute_crps_batch(mu_batch, sigma_batch, target_batch):
    """
    Compute CRPS for a batch of predictions.

    Args:
        mu_batch: Predicted means, shape (batch_size, num_targets)
        sigma_batch: Predicted sigmas, shape (batch_size, num_targets)
        target_batch: True values, shape (batch_size, num_targets)

    Returns:
        Mean CRPS across batch and targets (scalar)
    """
    crps_values = crps_gaussian(mu_batch, sigma_batch, target_batch)
    return np.mean(crps_values)


def compute_crps_per_target(mu_batch, sigma_batch, target_batch, target_names=None):
    """
    Compute CRPS separately for each target.

    This helps identify which SPEI metrics are easier/harder to predict.

    Args:
        mu_batch: Predicted means, shape (batch_size, 3)
        sigma_batch: Predicted sigmas, shape (batch_size, 3)
        target_batch: True values, shape (batch_size, 3)
        target_names: Names for each target

    Returns:
        dict: CRPS value for each target
    """
    if target_names is None:
        target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    # Convert to numpy if needed
    if isinstance(mu_batch, torch.Tensor):
        mu_batch = mu_batch.detach().cpu().numpy()
    if isinstance(sigma_batch, torch.Tensor):
        sigma_batch = sigma_batch.detach().cpu().numpy()
    if isinstance(target_batch, torch.Tensor):
        target_batch = target_batch.detach().cpu().numpy()

    crps_dict = {}
    for i, name in enumerate(target_names):
        crps_values = crps_gaussian(
            mu_batch[:, i],
            sigma_batch[:, i],
            target_batch[:, i]
        )
        crps_dict[name] = np.mean(crps_values)

    return crps_dict


def compute_rms_crps(crps_values):
    """
    Compute Root Mean Square of CRPS values.

    This is the final competition metric:
    RMS_CRPS = sqrt(mean(CRPS^2))

    Args:
        crps_values: Array or list of CRPS values

    Returns:
        RMS-CRPS (scalar)
    """
    crps_array = np.array(crps_values)
    return np.sqrt(np.mean(crps_array ** 2))


def compute_competition_score(all_mu, all_sigma, all_targets, target_names=None):
    """
    Compute the final competition score.

    The competition uses RMS of CRPS across all predictions and SPEI types.

    Args:
        all_mu: All predicted means, shape (num_samples, 3)
        all_sigma: All predicted sigmas, shape (num_samples, 3)
        all_targets: All true values, shape (num_samples, 3)
        target_names: Names for each target

    Returns:
        tuple: (rms_crps, per_target_crps_dict)
    """
    if target_names is None:
        target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    # Convert to numpy
    if isinstance(all_mu, torch.Tensor):
        all_mu = all_mu.detach().cpu().numpy()
    if isinstance(all_sigma, torch.Tensor):
        all_sigma = all_sigma.detach().cpu().numpy()
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.detach().cpu().numpy()

    # Compute CRPS for all predictions
    all_crps = crps_gaussian(all_mu, all_sigma, all_targets)

    # Compute RMS across all CRPS values (flattened)
    rms_crps = compute_rms_crps(all_crps.flatten())

    # Also compute per-target CRPS for analysis
    per_target_crps = {}
    for i, name in enumerate(target_names):
        per_target_crps[name] = np.mean(all_crps[:, i])

    return rms_crps, per_target_crps


# =============================================================================
# ADDITIONAL METRICS (for analysis)
# =============================================================================

def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        MAE (scalar)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return np.mean(np.abs(predictions - targets))


def compute_rmse(predictions, targets):
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        RMSE (scalar)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_calibration(sigma, errors):
    """
    Check if the predicted uncertainties are well-calibrated.

    For a well-calibrated model:
    - ~68% of observations should fall within ±1 sigma of the prediction
    - ~95% should fall within ±2 sigma

    Args:
        sigma: Predicted standard deviations
        errors: Prediction errors (target - mu)

    Returns:
        dict: Percentage of observations within 1, 2, and 3 sigma
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(errors, torch.Tensor):
        errors = errors.detach().cpu().numpy()

    # Standardize errors
    z_scores = np.abs(errors) / np.maximum(sigma, 1e-6)

    # Expected vs actual coverage
    calibration = {
        '1_sigma': {
            'expected': 0.6827,  # ~68.27% for normal distribution
            'actual': np.mean(z_scores <= 1)
        },
        '2_sigma': {
            'expected': 0.9545,  # ~95.45%
            'actual': np.mean(z_scores <= 2)
        },
        '3_sigma': {
            'expected': 0.9973,  # ~99.73%
            'actual': np.mean(z_scores <= 3)
        }
    }

    return calibration


def print_metrics_summary(mu, sigma, targets, target_names=None):
    """
    Print a comprehensive summary of all metrics.

    Args:
        mu: Predicted means
        sigma: Predicted sigmas
        targets: True values
        target_names: Names for each target
    """
    if target_names is None:
        target_names = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]

    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)

    # Competition score
    rms_crps, per_target_crps = compute_competition_score(mu, sigma, targets, target_names)
    print(f"\nCompetition Score (RMS-CRPS): {rms_crps:.4f}")

    # Per-target metrics
    print("\nPer-Target CRPS:")
    for name, crps in per_target_crps.items():
        print(f"  {name}: {crps:.4f}")

    # Convert for additional metrics
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # MAE and RMSE per target
    print("\nMAE per target:")
    for i, name in enumerate(target_names):
        mae = compute_mae(mu[:, i], targets[:, i])
        print(f"  {name}: {mae:.4f}")

    # Calibration check
    errors = targets - mu
    print("\nCalibration check (% of predictions within n sigma):")
    for i, name in enumerate(target_names):
        cal = compute_calibration(sigma[:, i], errors[:, i])
        print(f"  {name}:")
        for k, v in cal.items():
            print(f"    {k}: Expected {v['expected']*100:.1f}%, Actual {v['actual']*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    # Quick test of metrics
    print("Testing CRPS calculation...")

    # Test case: perfect prediction should have low CRPS
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.5, 0.5, 0.5])
    obs = np.array([1.0, 2.0, 3.0])  # Perfect predictions

    crps = crps_gaussian(mu, sigma, obs)
    print(f"CRPS for perfect predictions: {crps}")
    print(f"Mean CRPS: {np.mean(crps):.4f}")

    # Test case: bad predictions should have high CRPS
    obs_bad = np.array([5.0, 6.0, 7.0])  # Very wrong
    crps_bad = crps_gaussian(mu, sigma, obs_bad)
    print(f"\nCRPS for bad predictions: {crps_bad}")
    print(f"Mean CRPS: {np.mean(crps_bad):.4f}")

    # Test batch computation
    print("\nTesting batch computation...")
    mu_batch = np.random.randn(100, 3)
    sigma_batch = np.abs(np.random.randn(100, 3)) + 0.1
    target_batch = np.random.randn(100, 3)

    rms_crps, per_target = compute_competition_score(mu_batch, sigma_batch, target_batch)
    print(f"RMS-CRPS: {rms_crps:.4f}")
    print(f"Per-target CRPS: {per_target}")

    print("\nAll tests passed!")
