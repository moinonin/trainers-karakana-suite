"""Core metrics for structural evaluation of binary-outcome systems."""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_EPS = 1e-12
_DEFAULT_COF_NORM_DIFF_TOLERANCE = 0.1

# Ranking Profiles
RANKING_PROFILES = {
    "balanced": {
        "structure_weight": 0.50,
        "alpha_s_sil_weight": 0.15,
        "entropy_weight": 0.13,
        "positive_alpha_weight": 0.10,
        "expectancy_weight": 0.07,
        "smse_weight": 0.05,
        "neutral_penalty_weight": 0.03,
    },
    "conservative": {
        "structure_weight": 0.60,
        "alpha_s_sil_weight": 0.14,
        "entropy_weight": 0.10,
        "positive_alpha_weight": 0.07,
        "expectancy_weight": 0.05,
        "smse_weight": 0.04,
        "neutral_penalty_weight": 0.02,
    },
    "aggressive": {
        "structure_weight": 0.35,
        "expectancy_weight": 0.25,
        "entropy_weight": 0.15,
        "positive_alpha_weight": 0.13,
        "alpha_s_sil_weight": 0.10,
        "smse_weight": 0.02,
        "neutral_penalty_weight": 0.05,
    },
}


def apply_utility_transformation(magnitudes: np.ndarray, power: float = 1.1) -> np.ndarray:
    """
    Apply a non-linear utility transformation to magnitudes.

    Penalizes larger values (especially losses) more than linearly to capture
    tail risk and ruin potential.
    """
    signs = np.sign(magnitudes)
    abs_mags = np.abs(magnitudes)
    return signs * (abs_mags**power)


def calculate_cofactor_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute the cofactor matrix for a square numeric matrix."""
    cofactors = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            cofactors[i, j] = ((-1) ** (i + j)) * float(np.linalg.det(minor))
    return cofactors


def _as_outcome_units(outcome_units: Iterable[Sequence[float] | np.ndarray]) -> list[np.ndarray]:
    units = [np.asarray(unit, dtype=float).reshape(-1) for unit in outcome_units]
    if not units:
        raise ValueError("Cannot compute moment SGO matrix with 0 outcome units.")
    return units


def compute_sgo_matrix(
    outcome_units: Iterable[Sequence[float] | np.ndarray],
    utility_power: float = 1.1,
) -> np.ndarray:
    """
    Compute the theoretical Moment SGO matrix.

    Each unit is an episode, chunk, trade window, or other coherent experience
    block. For unit ``u``:

    ``P_u = sum(max(g(r_t), 0))`` and ``N_u = -sum(min(g(r_t), 0))``,
    where ``g(r) = sign(r) * abs(r) ** utility_power``. Pass
    ``utility_power=1.0`` for the raw theoretical moment matrix.

    The returned matrix is ``E[[1, P_u, N_u] [1, P_u, N_u]^T]``.
    This is the matrix defined in ``docs/THEORY.md``. It is intentionally
    separate from the legacy count/rate ``n_matrix`` used by
    ``evaluate_structural_geometry``.
    """
    units = _as_outcome_units(outcome_units)
    moment_sum = np.zeros((3, 3), dtype=float)

    for unit in units:
        transformed = apply_utility_transformation(unit, power=utility_power)
        p_u = float(np.maximum(transformed, 0.0).sum())
        n_u = float(-np.minimum(transformed, 0.0).sum())
        v_u = np.array([1.0, p_u, n_u], dtype=float)
        moment_sum += np.outer(v_u, v_u)

    return moment_sum / float(len(units))


def calculate_moment_sgo_metrics(
    outcome_units: Iterable[Sequence[float] | np.ndarray],
    utility_power: float = 1.1,
) -> dict[str, Any]:
    """
    Evaluate the theoretical Moment SGO matrix and its decoupled ideal.

    The ideal preserves the observed first and second marginal moments but sets
    the cross-moment ``E[PN]`` to zero. This isolates win/loss co-occurrence
    without changing the marginal size of wins or losses.
    """
    matrix = compute_sgo_matrix(outcome_units, utility_power=utility_power)
    ideal = matrix.copy()
    ideal[1, 2] = 0.0
    ideal[2, 1] = 0.0

    e_p = float(matrix[0, 1])
    e_n = float(matrix[0, 2])
    e_p2 = float(matrix[1, 1])
    e_n2 = float(matrix[2, 2])
    e_pn = float(matrix[1, 2])
    rho_pn = float(e_pn / np.sqrt(max(e_p2 * e_n2, _EPS)))
    rho_pn_sq = float(rho_pn * rho_pn)
    fro_error = float(np.linalg.norm(matrix - ideal))
    normalized_fro_error = float(fro_error / max(np.linalg.norm(ideal), _EPS))

    cofactors = calculate_cofactor_matrix(matrix)
    ideal_cofactors = calculate_cofactor_matrix(ideal)
    cofactor_c23 = float(cofactors[1, 2])
    expected_decoupled_c23 = float(e_p * e_n)

    return {
        "moment_matrix": matrix.tolist(),
        "moment_ideal_matrix": ideal.tolist(),
        "moment_cofactors": cofactors.tolist(),
        "moment_ideal_cofactors": ideal_cofactors.tolist(),
        "e_p": e_p,
        "e_n": e_n,
        "e_p2": e_p2,
        "e_n2": e_n2,
        "e_pn": e_pn,
        "rho_pn": rho_pn,
        "rho_pn_sq": rho_pn_sq,
        "moment_fro_error": fro_error,
        "moment_normalized_fro_error": normalized_fro_error,
        "cofactor_c23": cofactor_c23,
        "expected_decoupled_c23": expected_decoupled_c23,
        "cofactor_consistency_gap": float(cofactor_c23 - expected_decoupled_c23),
    }


def extract_matrix_alpha(cofactors: np.ndarray) -> float:
    """
    Extract the primary structural metric (alpha) from the cofactor matrix.

    C20 (Row 3, Col 1 of cofactors) captures the structural tension between
    failure rate and success magnitude.
    """
    return float(cofactors[1, 2])


def evaluate_balance_diagnostics(
    success_rate: float,
    failure_rate: float,
    avg_success_magnitude: float,
    avg_failure_magnitude: float,
) -> dict[str, Any]:
    """
    Evaluate a neutral rate-vs-magnitude balance diagnostic.

    The core quantity compares failure-rate weighted success magnitude against
    success-rate weighted failure magnitude. Values near zero indicate close
    cancellation between those two cross-weighted terms.
    """
    success_rate = float(success_rate)
    failure_rate = float(failure_rate)
    avg_success_magnitude = float(avg_success_magnitude)
    avg_failure_magnitude = float(avg_failure_magnitude)
    magnitude_spread = float(avg_success_magnitude - avg_failure_magnitude)
    rate_weighted_success_mass = float(failure_rate * avg_success_magnitude)
    rate_weighted_failure_mass = float(success_rate * avg_failure_magnitude)
    rate_weighted_balance_gap = float(rate_weighted_success_mass - rate_weighted_failure_mass)

    return {
        "magnitude_spread": magnitude_spread,
        "rate_weighted_success_mass": rate_weighted_success_mass,
        "rate_weighted_failure_mass": rate_weighted_failure_mass,
        "rate_weighted_balance_gap": rate_weighted_balance_gap,
        "rate_weighted_balance_gap_abs": abs(rate_weighted_balance_gap),
    }


def evaluate_cofactors_norm_diff_diagnostics(
    cofactors_norm_diff: float,
    tolerance: float = _DEFAULT_COF_NORM_DIFF_TOLERANCE,
) -> dict[str, Any]:
    """
    Compatibility helper for simple near-zero checks on the raw norm
    difference.

    This is useful for downstream monitoring, but it is not one of the core
    structural metrics that `karakana` is built around.
    """
    cofactors_norm_diff = float(cofactors_norm_diff)
    tolerance = float(max(0.0, tolerance))
    return {
        "cofactors_norm_diff_tolerance": tolerance,
        "cofactors_norm_diff_abs": abs(cofactors_norm_diff),
        "cofactors_norm_diff_near_zero": abs(cofactors_norm_diff) <= tolerance,
    }


def calculate_entropy(returns: np.ndarray) -> float:
    """
    Calculate normalized Shannon entropy for a distribution of returns.

    This measure quantifies the randomness of trade outcomes, normalized to [0, 1].
    """
    if returns.size <= 1:
        return 0.0

    unique_values = np.unique(returns)
    # Use dynamic binning based on variety of outcomes, capped at 10
    bin_count = int(min(10, max(2, unique_values.size)))
    counts, _ = np.histogram(returns, bins=bin_count)
    counts = counts[counts > 0]

    if counts.size <= 1:
        return 0.0

    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(probabilities.size)

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def calculate_ranking_score(
    normalized_structure_term: float,
    alpha_s_sil: float,
    entropy: float,
    expectancy: float,
    normalized_expectancy: float = 0.0,
    smse: float = 0.0,
    profile: str = "balanced",
    structural_alpha: float = 0.0,
    positive_alpha_weight: float | None = None,
    positive_alpha_scale: float = 1.0,
    neutral_ratio: float = 0.0,
    **custom_weights: float,
) -> float:
    """
    Generalized ranking score for binary-outcome processes.

    Higher structural accuracy and performance result in a lower (better) score.
    Weights are pulled from a profile or provided as custom overrides.

    Now uses normalized_expectancy with a penalty for negative values.
    """
    weights = RANKING_PROFILES.get(profile, RANKING_PROFILES["balanced"]).copy()
    weights.update(custom_weights)

    if positive_alpha_weight is None:
        positive_alpha_weight = weights.get("positive_alpha_weight", 0.0)

    positive_alpha_penalty = calculate_positive_alpha_penalty(
        structural_alpha,
        scale=positive_alpha_scale,
    )

    # Penalize negative normalized expectancy
    # If norm_exp < 0, it adds to the score (worsens it)
    # We use the existing expectancy_weight for this term.
    norm_exp_term = normalized_expectancy
    if norm_exp_term < 0:
        # Extra penalty for negative normalized expectancy (asymmetry trap)
        norm_exp_term *= 2.0

    neutral_penalty_weight = weights.get("neutral_penalty_weight", 0.03)
    return float(
        (normalized_structure_term * weights["structure_weight"])
        + (alpha_s_sil * weights["alpha_s_sil_weight"])
        + (positive_alpha_penalty * positive_alpha_weight)
        + (entropy * weights["entropy_weight"])
        + (smse * weights["smse_weight"])
        + (neutral_ratio * neutral_penalty_weight)
        - (norm_exp_term * weights["expectancy_weight"])
    )


def calculate_positive_alpha_penalty(
    structural_alpha: float,
    scale: float = 1.0,
) -> float:
    """Return a normalized penalty for alpha values above the zero boundary."""
    scale = max(abs(float(scale)), _EPS)
    return float(max(0.0, float(structural_alpha)) / scale)


def calculate_normalized_structure_metrics(
    cofactors: np.ndarray,
    ideal_cofactors: np.ndarray,
    total_outcomes: int = 1,
    credibility_half_life: int = 50,
) -> dict[str, Any]:
    """
    Calculate normalized structural metrics comparing observed and ideal cofactors.

    Includes an energy penalty for passive models and a credibility factor
    based on sample size.
    """
    cofactors = np.asarray(cofactors, dtype=float)
    ideal_cofactors = np.asarray(ideal_cofactors, dtype=float)

    ideal_cofactors_norm = float(np.linalg.norm(ideal_cofactors))
    cofactors_norm = float(np.linalg.norm(cofactors))
    cofactors_norm_diff = float(cofactors_norm - ideal_cofactors_norm)

    cofactor_error = cofactors - ideal_cofactors
    mse_cofactors = float(np.mean(np.square(cofactor_error)))

    ideal_cofactors_mse = float(np.mean(np.square(ideal_cofactors)))
    observed_cofactors_mse = float(np.mean(np.square(cofactors)))

    normalization_scale = float(max(cofactors_norm, ideal_cofactors_norm, _EPS))
    normalization_mse_scale = float(max(observed_cofactors_mse, ideal_cofactors_mse, _EPS))

    relative_cofactors_fro_error = float(np.linalg.norm(cofactor_error) / normalization_scale)
    normalized_mse_cofactors = float(mse_cofactors / normalization_mse_scale)
    normalized_cofactors_norm_diff = float(cofactors_norm_diff / normalization_scale)
    normalized_cofactors_norm_diff_abs = abs(normalized_cofactors_norm_diff)

    # Bayesian Credibility: How much do we trust this sample size?
    credibility = float(total_outcomes / (total_outcomes + credibility_half_life))

    # Energy Ratio: Observed energy vs Ideal energy (scale invariant)
    energy_ratio = float(cofactors_norm / (ideal_cofactors_norm + _EPS))

    # Energy Penalty: Discourage passive "do-nothing" models
    # We use a non-linear absolute energy penalty.
    # Low absolute cofactor energy indicates a passive or degenerate system.
    # We use log1p to handle scale while remaining sensitive to near-zero energy.
    absolute_energy_signal = float(np.log1p(cofactors_norm))
    energy_penalty = 1.0 / (1.0 + absolute_energy_signal)

    # Combine metrics into a single structural term, adjusted by credibility and energy
    # We invert credibility for the penalty (low N -> higher "error" term)
    normalized_structure_term = (
        (normalized_cofactors_norm_diff_abs + normalized_mse_cofactors + relative_cofactors_fro_error)
        / max(credibility, 0.01)
    ) + energy_penalty

    return {
        "cofactors_norm": cofactors_norm,
        "ideal_cofactors_norm": ideal_cofactors_norm,
        "energy_ratio": energy_ratio,
        "energy_penalty": energy_penalty,
        "credibility": credibility,
        "normalization_scale": normalization_scale,
        "normalization_mse_scale": normalization_mse_scale,
        "cofactors_norm_diff": cofactors_norm_diff,
        "normalized_cofactors_norm_diff": normalized_cofactors_norm_diff,
        "normalized_cofactors_norm_diff_abs": normalized_cofactors_norm_diff_abs,
        "mse_cofactors": mse_cofactors,
        "rmse_cofactors": float(np.sqrt(mse_cofactors)),
        "ideal_cofactors_mse": ideal_cofactors_mse,
        "observed_cofactors_mse": observed_cofactors_mse,
        "normalized_mse_cofactors": normalized_mse_cofactors,
        "relative_cofactors_fro_error": relative_cofactors_fro_error,
        "normalized_structure_term": normalized_structure_term,
        "is_structurally_sound": bool(normalized_mse_cofactors < 1e-6 and credibility > 0.5),
        "normalization_degenerate": bool(min(cofactors_norm, ideal_cofactors_norm) < _EPS),
    }


def calculate_regime_aware_ideal(s: float, N: int, entropy: float, min_failure_rate: float = 0.01) -> np.ndarray:
    """
    Compute a regime-aware Ideal Matrix.

    In high-entropy regimes, a 'zero failure' ideal is unattainable.
    This anchor scales the ideal failure rate based on the observed entropy.
    """
    # Scale ideal failure rate by entropy (more noise -> more expected failures)
    # entropy is normalized [0, 1]
    adjusted_failure_rate = max(min_failure_rate, entropy * 0.2)
    adjusted_success_rate = 1.0 - adjusted_failure_rate

    n_l_ideal = adjusted_failure_rate * N
    n_p_ideal = adjusted_success_rate * N

    return np.array(
        [[s, 0.0, s], [n_l_ideal, n_p_ideal, N], [adjusted_failure_rate, adjusted_success_rate, 1.0]], dtype=float
    )


from .config import get_config


def evaluate_structural_geometry(
    positive_outcomes: np.ndarray,
    negative_outcomes: np.ndarray,
    utility_power: float = 1.1,
    ranking_profile: str | None = None,
    credibility_half_life: int = 50,
    regime_aware: bool = False,
    positive_alpha_weight: float | None = None,
    neutral_count: int = 0,
) -> dict[str, Any]:
    """
    Evaluate the legacy count/rate cofactor geometry of a binary-outcome process.

    This function builds the historical ``n_matrix`` from average magnitudes,
    counts, and rates. It is retained for existing reports and calibration
    compatibility. Use ``compute_sgo_matrix`` or
    ``calculate_moment_sgo_metrics`` when per-unit moment coupling
    ``E[P_u N_u]`` is required.

    Parameters:
    - `positive_outcomes`: magnitudes of success cases
    - `negative_outcomes`: magnitudes of failure cases
    - `utility_power`: non-linear scaling power (default 1.1)
    - `ranking_profile`: "balanced", "conservative", or "aggressive"
    - `credibility_half_life`: sample size for 50% credibility
    - `regime_aware`: if True, use an entropy-adjusted ideal matrix
    - `positive_alpha_weight`: optional ranking penalty weight for structural
      alpha above zero. Use this when best results should satisfy alpha <= 0.
      If None, it defaults to the weight defined in the ranking profile.

    Returns:
    Dictionary containing structural metrics, descriptive statistics, and the
    observed/ideal matrices.
    """
    if ranking_profile is None:
        ranking_profile = get_config.get("ranking_profile", "conservative")
    positive_outcomes = np.asarray(positive_outcomes, dtype=float)
    negative_outcomes = np.asarray(negative_outcomes, dtype=float)

    # Calculate entropy on raw outcomes first for regime awareness
    all_outcomes_raw = np.concatenate([positive_outcomes, negative_outcomes])
    entropy = calculate_entropy(all_outcomes_raw)

    # Apply Utility Transformation
    transformed_pos = apply_utility_transformation(positive_outcomes, power=utility_power)
    transformed_neg = apply_utility_transformation(negative_outcomes, power=utility_power)

    pos = int(transformed_pos.size)
    negs = int(transformed_neg.size)
    N = pos + negs

    if N == 0:
        raise ValueError("Cannot evaluate geometry with 0 outcomes.")

    win_rate = float(pos / N)
    loss_rate = float(negs / N)

    avg_win = float(np.mean(transformed_pos)) if pos else 0.0
    avg_loss = float(abs(np.mean(transformed_neg))) if negs else 0.0
    s = avg_win - avg_loss
    s_ideal_value = avg_win

    n_mat = np.array(
        [
            [avg_win, avg_loss, s],
            [negs, pos, N],
            [loss_rate, win_rate, 1.0],
        ],
        dtype=float,
    )

    if regime_aware and win_rate > 0:
        ideal_mat = calculate_regime_aware_ideal(s_ideal_value, N, entropy)
    else:
        ideal_mat = np.array([[s_ideal_value, 0.0, s_ideal_value], [0.0, N, N], [0.0, 1.0, 1.0]], dtype=float)

    s_observed = float(n_mat[0, 2])
    s_ideal = float(ideal_mat[0, 2])
    smse = float((s_observed - s_ideal) ** 2)

    cofactors = calculate_cofactor_matrix(n_mat)
    ideal_cofactors = calculate_cofactor_matrix(ideal_mat)
    structural_alpha = extract_matrix_alpha(cofactors)
    # Calculate ideal alpha (C12): second row, third column
    ideal_alpha = float(ideal_cofactors[1, 2])
    # Scale-invariant opposite-sign alignment loss between ideal alpha and s.
    # Perfect legacy alignment has -ideal_alpha == s_observed, with ideal
    # alpha negative and s_observed positive, so this term is 0.
    # Raw mathematical bounds are [0, 2].
    a, b = ideal_alpha, s_observed

    alpha_s_sil = float((a + b) ** 2 / (a**2 + b**2 + 1e-12))

    # Use the modular calculation for structural metrics
    structural_metrics = calculate_normalized_structure_metrics(
        cofactors, ideal_cofactors, total_outcomes=N, credibility_half_life=credibility_half_life
    )
    structural_metrics["smse"] = smse
    structural_metrics["alpha_s_sil"] = alpha_s_sil
    # Compatibility alias for early SGO integrations and docs that used the
    # older name before the metric became scale-invariant.
    structural_metrics["alpha_s_mse"] = alpha_s_sil
    alpha_penalty_scale = max(
        abs(structural_alpha),
        abs(ideal_alpha),
        abs(s_observed),
        abs(s_ideal),
        1.0,
    )
    positive_alpha_penalty = calculate_positive_alpha_penalty(
        structural_alpha,
        scale=alpha_penalty_scale,
    )

    # Sprint 5: Higher-order moments
    variance = float(np.var(all_outcomes_raw))
    skewness = 0.0
    if variance > _EPS:
        skewness = float(np.mean((all_outcomes_raw - np.mean(all_outcomes_raw)) ** 3) / (variance**1.5))

    # Expectancy and Normalized Expectancy
    # normalized_expectancy = (be_win_rate * avg_win) - (win_rate * avg_loss)
    be_win_rate = avg_loss / (avg_win + avg_loss + 1e-8)
    normalized_expectancy = (be_win_rate * avg_win) - (win_rate * avg_loss)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    balance_diagnostics = evaluate_balance_diagnostics(
        success_rate=win_rate,
        failure_rate=loss_rate,
        avg_success_magnitude=avg_win,
        avg_failure_magnitude=avg_loss,
    )
    norm_diff_diagnostics = evaluate_cofactors_norm_diff_diagnostics(structural_metrics["cofactors_norm_diff"])
    diagnostics = {
        **norm_diff_diagnostics,
        **balance_diagnostics,
        "variance": variance,
        "skewness": skewness,
        "normalized_expectancy": normalized_expectancy,
        "be_win_rate": be_win_rate,
    }

    # Calculate ranking score
    if positive_alpha_weight is None:
        positive_alpha_weight = RANKING_PROFILES.get(ranking_profile, RANKING_PROFILES["balanced"]).get(
            "positive_alpha_weight", 0.0
        )

    # Neutral ratio: systems with high idle/inactive rates are penalised
    total_with_neutrals = N + neutral_count
    neutral_ratio = float(neutral_count) / max(1, total_with_neutrals)

    ranking_score = calculate_ranking_score(
        normalized_structure_term=structural_metrics["normalized_structure_term"],
        alpha_s_sil=alpha_s_sil,
        entropy=entropy,
        expectancy=expectancy,
        normalized_expectancy=normalized_expectancy,
        smse=smse,
        profile=ranking_profile,
        structural_alpha=structural_alpha,
        positive_alpha_weight=positive_alpha_weight,
        positive_alpha_scale=alpha_penalty_scale,
        neutral_ratio=neutral_ratio,
    )

    return {
        "n_matrix": n_mat.tolist(),
        "ideal_matrix": ideal_mat.tolist(),
        "cofactors": cofactors.tolist(),
        "ideal_cofactors": ideal_cofactors.tolist(),
        "entropy": entropy,
        "ranking_score": ranking_score,
        "utility_power": utility_power,
        "ranking_profile": ranking_profile,
        "positive_alpha_weight": float(positive_alpha_weight),
        "positive_alpha_penalty": positive_alpha_penalty,
        "positive_alpha_scale": alpha_penalty_scale,
        "alpha_s_sil": structural_metrics["alpha_s_sil"],
        "alpha_s_mse": structural_metrics["alpha_s_mse"],
        **structural_metrics,
        "expectancy": expectancy,
        "normalized_expectancy": normalized_expectancy,
        "be_win_rate": be_win_rate,
        "structural_alpha": structural_alpha,
        "total_outcomes": N,
        "success_rate": win_rate,
        "avg_success_magnitude": avg_win,
        "avg_failure_magnitude": avg_loss,
        "diagnostics": diagnostics,
    }


def calculate_probability_sgo_matrix(all_outcomes: np.ndarray, bins: int = 3) -> np.ndarray:
    """
    Calculate the joint probability matrix (Direction x Magnitude).

    Directions: Loss (0), Neutral (1), Win (2)
    Magnitudes: Discretised into 'bins' quantiles (default tertiles).
    """
    if all_outcomes.size == 0:
        return np.zeros((3, bins))

    # 1. Determine Direction
    directions = np.zeros_like(all_outcomes, dtype=int)
    directions[all_outcomes > _EPS] = 2  # Win
    directions[all_outcomes < -_EPS] = 0  # Loss
    directions[(all_outcomes >= -_EPS) & (all_outcomes <= _EPS)] = 1  # Neutral

    # 2. Determine Magnitude Level
    abs_magnitudes = np.abs(all_outcomes)
    if abs_magnitudes.max() < _EPS:
        magnitudes = np.zeros_like(all_outcomes, dtype=int)
    else:
        # Use quantiles for discretisation
        try:
            # We use non-zero magnitudes to define quantiles
            active_mags = abs_magnitudes[abs_magnitudes > _EPS]
            if active_mags.size >= bins:
                thresholds = np.quantile(active_mags, np.linspace(0, 1, bins + 1)[1:-1])
                magnitudes = np.digitize(abs_magnitudes, thresholds)
            else:
                magnitudes = np.zeros_like(all_outcomes, dtype=int)
        except Exception:
            magnitudes = np.zeros_like(all_outcomes, dtype=int)

    # 3. Construct Joint Matrix
    matrix = np.zeros((3, bins))
    for d, m in zip(directions, magnitudes):
        if m < bins:  # digitize can return 'bins' for max values
            matrix[d, m] += 1
        else:
            matrix[d, bins - 1] += 1

    # Normalise to probability mass
    return matrix / (all_outcomes.size + _EPS)


def calculate_sgo_divergence(p_matrix: np.ndarray, target_matrix: np.ndarray | None = None) -> float:
    """
    Calculate the Frobenius distance from an ideal structural target.
    (Note: For differentiable KL divergence, use `torch_calculate_sgo_divergence`).
    """
    if target_matrix is None:
        # Default target: all mass in the 'Large Win' cell (2, 2)
        target_matrix = np.zeros_like(p_matrix)
        target_matrix[2, -1] = 1.0

    # Use Frobenius distance for stability in small samples
    return float(np.linalg.norm(p_matrix - target_matrix))


def evaluate_distributional_geometry(
    positive_outcomes: np.ndarray, negative_outcomes: np.ndarray, bins: int = 5, neutral_count: int = 0, **kwargs
) -> dict[str, Any]:
    """
    Evaluate structural geometry across different outcome magnitude segments.

    Uses soft/overlapping binning (sliding windows) to prevent hard cutoffs
    from creating discrete jumps in the evaluation landscape.
    """
    positive_outcomes = np.asarray(positive_outcomes, dtype=float)
    negative_outcomes = np.asarray(negative_outcomes, dtype=float)
    all_outcomes = np.concatenate([positive_outcomes, negative_outcomes])

    if all_outcomes.size < bins * 2:
        return evaluate_structural_geometry(positive_outcomes, negative_outcomes, **kwargs)

    sorted_outcomes = np.sort(all_outcomes)

    # Soft overlapping binning (Sliding window)
    chunk_size = max(4, len(sorted_outcomes) // bins + (len(sorted_outcomes) // bins) // 2)
    step = max(1, (len(sorted_outcomes) - chunk_size) // (bins - 1)) if bins > 1 else len(sorted_outcomes)

    chunks = []
    for i in range(bins):
        start_idx = i * step
        end_idx = min(start_idx + chunk_size, len(sorted_outcomes))
        if end_idx - start_idx >= 4:
            chunks.append(sorted_outcomes[start_idx:end_idx])

    if not chunks:
        chunks = [sorted_outcomes]

    bin_results = []
    for chunk in chunks:
        pos = chunk[chunk > 0]
        neg = chunk[chunk <= 0]
        if pos.size > 0 or neg.size > 0:
            try:
                res = evaluate_structural_geometry(pos, neg, **kwargs)
                bin_results.append(res)
            except Exception:
                continue

    global_res = evaluate_structural_geometry(
        positive_outcomes, negative_outcomes, neutral_count=neutral_count, **kwargs
    )

    if not bin_results:
        return global_res

    avg_structural_term = float(np.mean([r["normalized_structure_term"] for r in bin_results]))
    avg_alpha_s_sil = float(np.mean([r["alpha_s_sil"] for r in bin_results]))

    return {
        **global_res,
        "distributional_structural_term": avg_structural_term,
        "distributional_alpha_s_sil": avg_alpha_s_sil,
        "bin_count": len(bin_results),
        "is_distributionally_stable": bool(avg_structural_term < global_res["normalized_structure_term"] * 1.5),
    }


def compute_distributional_concentration(
    outcomes: np.ndarray,
    deciles: int = 10,
    concentration_threshold: float = 0.40,
) -> dict[str, Any]:
    """Compute P&L concentration across magnitude deciles.

    Flags when a single decile drives more than ``concentration_threshold``
    of total P&L — a tail-risk signal that the strategy's performance is
    concentrated in one outcome magnitude bucket.

    Sprint 17: Distributional Stability.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(outcomes)
    if n < deciles * 2:
        return {
            "decile_fractions": [],
            "max_decile_fraction": 0.0,
            "max_decile_index": -1,
            "is_concentrated": False,
            "concentration_threshold": concentration_threshold,
            "error": f"need at least {deciles * 2} outcomes, got {n}",
        }

    total_pnl = float(np.sum(np.abs(outcomes)))
    if total_pnl < 1e-12:
        return {
            "decile_fractions": [0.0] * deciles,
            "max_decile_fraction": 0.0,
            "max_decile_index": -1,
            "is_concentrated": False,
            "concentration_threshold": concentration_threshold,
        }

    sorted_idx = np.argsort(np.abs(outcomes))
    decile_size = max(1, n // deciles)
    decile_fractions = []

    for d in range(deciles):
        start = d * decile_size
        end = start + decile_size if d < deciles - 1 else n
        bucket = outcomes[sorted_idx[start:end]]
        bucket_pnl = float(np.sum(np.abs(bucket)))
        decile_fractions.append(round(bucket_pnl / total_pnl, 4))

    max_frac = max(decile_fractions)
    max_idx = decile_fractions.index(max_frac)

    return {
        "decile_fractions": decile_fractions,
        "max_decile_fraction": max_frac,
        "max_decile_index": max_idx,
        "is_concentrated": bool(max_frac > concentration_threshold),
        "concentration_threshold": concentration_threshold,
    }


class StructuralTracker:
    """
    Tracks structural error over time to compute velocity and momentum.

    .. deprecated:: Sprint 17
        Use :class:`AlphaMomentumTracker` instead.  ``StructuralTracker``
        uses raw pairwise differences with no EMA smoothing, no α-floor
        gate, and no horizon estimation.  It is retained for backward
        compatibility but will be removed in a future release.

    Used to dynamically adjust learning parameters or detect regime shifts.
    """

    def __init__(self, window_size: int = 10):
        import warnings

        warnings.warn(
            "StructuralTracker is deprecated; use AlphaMomentumTracker instead. "
            "See docs/SPEC_SGO_LOSS_TRAJECTORY.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.history: list[float] = []
        self.window_size = window_size

    def push(self, structural_error: float):
        """Add a new structural error observation."""
        self.history.append(float(structural_error))
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_velocity(self) -> float:
        """First derivative (rate of change) of structural error."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1] - self.history[-2]

    def get_momentum(self) -> float:
        """Second derivative (acceleration) of structural error."""
        if len(self.history) < 3:
            return 0.0
        v1 = self.history[-1] - self.history[-2]
        v2 = self.history[-2] - self.history[-3]
        return v1 - v2

    def get_stats(self) -> dict[str, float]:
        """Return a summary of structural trends."""
        return {
            "last_error": self.history[-1] if self.history else 0.0,
            "velocity": self.get_velocity(),
            "momentum": self.get_momentum(),
            "window_mean": float(np.mean(self.history)) if self.history else 0.0,
        }


class AlphaMomentumTracker:
    """
    Advanced Geometric Reasoning Governor (GRG) Tracker.
    Uses Exponential Moving Averages (EMA) and Structural Momentum to detect
    non-linear 'Tipping Points' in logical integrity.
    """

    def __init__(self, window_size: int = 20, alpha_ema: float = 0.3):
        self.history: list[float] = []
        self.window_size = window_size
        self.alpha_ema = alpha_ema  # Weight for EMA

        self.ema_alpha: float = 0.0
        self.ema_velocity: float = 0.0
        self.ema_momentum: float = 0.0

    def push(self, alpha: float):
        alpha = float(alpha)
        self.history.append(alpha)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # 1. Update EMA Alpha
        if len(self.history) == 1:
            self.ema_alpha = alpha
        else:
            prev_ema = self.ema_alpha
            self.ema_alpha = (self.alpha_ema * alpha) + (1 - self.alpha_ema) * prev_ema

            # 2. Update EMA Velocity (Vα) - First Derivative
            current_velocity = self.ema_alpha - prev_ema
            prev_velocity = self.ema_velocity
            self.ema_velocity = (self.alpha_ema * current_velocity) + (1 - self.alpha_ema) * prev_velocity

            # 3. Update EMA Momentum (Mα) - Second Derivative
            current_momentum = self.ema_velocity - prev_velocity
            self.ema_momentum = (self.alpha_ema * current_momentum) + (1 - self.alpha_ema) * self.ema_momentum

    def get_structural_horizon(self, floor: float = 0.75) -> float:
        """Predict remaining 'safe' tokens/turns before logical collapse."""
        dist_to_floor = self.ema_alpha - floor
        if dist_to_floor <= 0:
            return 0.0

        if abs(self.ema_velocity) < 1e-6:
            return 999.0  # Infinite horizon (stable)

        # Linear approximation of remaining distance
        # For accelerating collapse (Mα < 0), this will be conservative
        # If velocity is positive, horizon is infinite
        if self.ema_velocity >= 0:
            return 999.0

        return float(dist_to_floor / abs(self.ema_velocity))

    def get_stats(self, v_panic: float | None = None, floor: float | None = None) -> dict[str, Any]:
        from .config import get_config

        v_panic = get_config.get("v_alpha_panic_threshold", -0.02) if v_panic is None else float(v_panic)
        floor = get_config.get("alpha_safety_floor", 0.10) if floor is None else float(floor)

        horizon = self.get_structural_horizon(floor=floor)

        return {
            "alpha": self.ema_alpha,
            "v_alpha": self.ema_velocity,
            "m_alpha": self.ema_momentum,
            "horizon": horizon,
            "is_collapsing": bool(self.ema_velocity < v_panic or self.ema_momentum < v_panic / 2.0),
            "is_stable": bool(abs(self.ema_velocity) < abs(v_panic) / 4.0 and self.ema_alpha > floor),
        }

    def should_stop(
        self, safety_floor: float = 0.10, panic_threshold: float = -0.02, e_stop_horizon: float = 5.0
    ) -> bool:
        """Return True if structural collapse conditions are met."""
        stats = self.get_stats(v_panic=panic_threshold, floor=safety_floor)
        horizon = stats["horizon"]
        is_collapsing = stats["is_collapsing"]
        alpha_below_floor = self.ema_alpha < safety_floor
        accelerating_collapse = is_collapsing and stats["m_alpha"] < panic_threshold / 2.0
        horizon_exhausted = horizon <= e_stop_horizon
        return bool(accelerating_collapse or horizon_exhausted or alpha_below_floor)


# Backward-compatible exports. EVX is isolated from the canonical SGO math above.
from karakana.evx import calculate_evx, calculate_evx_coefficient  # noqa: E402,F401
