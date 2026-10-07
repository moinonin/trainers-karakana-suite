"""Module II: Pure Probability SGO Matrix (Discretised).

Provides a scale-invariant, discretised view of outcome geometry using
Direction (Loss/Neutral/Win) and Magnitude (Small/Medium/Large) bins.
"""

from typing import Any

import numpy as np

_EPS = 1e-12


def discretize_magnitudes(magnitudes: np.ndarray, bins: int = 3) -> np.ndarray:
    """
    Map absolute magnitudes into quantile-based bins.

    Returns:
        np.ndarray: Integer bin indices [0, bins-1]
    """
    abs_mags = np.abs(magnitudes)
    if abs_mags.size == 0:
        return np.array([], dtype=int)

    if abs_mags.max() < _EPS:
        return np.zeros_like(abs_mags, dtype=int)

    # Use only active (non-zero) magnitudes to define quantiles
    active_mask = abs_mags > _EPS
    active_mags = abs_mags[active_mask]

    if active_mags.size < bins:
        # Fallback to simple linear scaling if not enough data for quantiles
        indices = np.zeros_like(abs_mags, dtype=int)
        if active_mags.size > 0:
            max_val = active_mags.max()
            indices[active_mask] = np.minimum((active_mags / (max_val + _EPS) * bins).astype(int), bins - 1)
        return indices

    # Calculate quantile thresholds
    thresholds = np.quantile(active_mags, np.linspace(0, 1, bins + 1)[1:-1])

    # Map all magnitudes (including zeros, though they'll usually be 'Small')
    bin_indices = np.digitize(abs_mags, thresholds)
    return np.clip(bin_indices, 0, bins - 1)


def build_probability_sgo_matrix(outcomes: np.ndarray, bins: int = 3) -> np.ndarray:
    """
    Construct the 3x3 (or 3xBins) Probability SGO Matrix.

    Rows: 0=Loss, 1=Neutral, 2=Win
    Cols: 0=Small, 1=Medium, 2=Large (or up to Bins-1)

    Returns:
        np.ndarray: Normalized joint probability matrix.
    """
    if outcomes.size == 0:
        return np.zeros((3, bins))

    # 1. Determine Direction
    directions = np.ones_like(outcomes, dtype=int)  # Default Neutral (1)
    directions[outcomes > _EPS] = 2  # Win
    directions[outcomes < -_EPS] = 0  # Loss

    # 2. Determine Magnitude Bins
    magnitude_bins = discretize_magnitudes(outcomes, bins=bins)

    # 3. Populate Matrix
    matrix = np.zeros((3, bins))
    for d, m in zip(directions, magnitude_bins):
        matrix[d, m] += 1

    # 4. Normalize
    return matrix / (outcomes.size + _EPS)


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate observed-to-target KL-Divergence between two probability matrices.

    D(P || Q) = sum(P * log(P / Q))

    This requires a smoothed target when Q contains zero-mass cells. It is
    useful for offline diagnostics, but it is not the same objective as the
    PyTorch ``probability_kl`` training loss.
    """
    p = p + _EPS  # Smoothing
    q = q + _EPS
    return float(np.sum(p * np.log(p / q)))


def calculate_target_likelihood_kl(
    p_observed: np.ndarray,
    target: np.ndarray | None = None,
    reduction: str = "batchmean",
) -> float:
    """
    Calculate the target-to-observed KL used by PyTorch ``probability_kl``.

    With the default one-hot target this is primarily a large-win target-cell
    likelihood penalty: low observed mass in the target cell produces high
    loss. ``reduction="batchmean"`` matches ``torch.nn.functional.kl_div`` for
    a 2-D probability matrix by dividing the summed divergence by row count.
    """
    observed = p_observed + _EPS
    if target is None:
        target = get_ideal_probability_matrix(bins=p_observed.shape[1])
    target = target + _EPS

    loss = float(np.sum(target * np.log(target / observed)))
    if reduction == "batchmean":
        return loss / float(p_observed.shape[0])
    if reduction == "sum":
        return loss
    raise ValueError("reduction must be 'batchmean' or 'sum'")


def get_ideal_probability_matrix(bins: int = 3) -> np.ndarray:
    """
    Returns the target 'Safe' matrix: All mass in the (Win, Large) cell.
    """
    ideal = np.zeros((3, bins))
    ideal[2, -1] = 1.0
    return ideal


def evaluate_probability_geometry(outcomes: np.ndarray, bins: int = 3) -> dict[str, Any]:
    """
    Full evaluation using Module II logic.
    """
    p_obs = build_probability_sgo_matrix(outcomes, bins=bins)
    p_ideal = get_ideal_probability_matrix(bins=bins)

    observed_to_target_kl = calculate_kl_divergence(p_obs, p_ideal)
    target_likelihood_kl = calculate_target_likelihood_kl(p_obs, p_ideal)
    fro_dist = float(np.linalg.norm(p_obs - p_ideal))

    return {
        "probability_matrix": p_obs.tolist(),
        "kl_divergence": observed_to_target_kl,
        "kl_divergence_direction": "observed_to_target",
        "observed_to_target_kl": observed_to_target_kl,
        "target_likelihood_kl": target_likelihood_kl,
        "target_likelihood_kl_reduction": "batchmean",
        "frobenius_distance": fro_dist,
        "bins": bins,
        "entropy": float(-np.sum(p_obs * np.log2(p_obs + _EPS))),
    }
