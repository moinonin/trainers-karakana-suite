"""Differentiable structural geometry metrics for PyTorch."""

import torch


def torch_apply_utility_transformation(magnitudes: torch.Tensor, power: float = 1.1) -> torch.Tensor:
    """Differentiable utility transformation."""
    signs = torch.sign(magnitudes)
    abs_mags = torch.abs(magnitudes)
    return signs * (abs_mags**power)


def torch_calculate_cofactor_matrix_3x3(M: torch.Tensor) -> torch.Tensor:
    """
    Explicit differentiable cofactor matrix for 3x3.
    Safe for singular matrices where inverse() would fail.
    """
    # M shape: (3, 3) or (Batch, 3, 3)
    # Using explicit expansion for 3x3
    if M.dim() == 2:
        m = M
    else:
        # We'll handle batching if needed, but for SGO n_matrix is usually single
        m = M[0] if M.size(0) == 1 else M

    # Manual cofactors for 3x3
    c = torch.zeros_like(m)

    # Row 0
    c[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    c[0, 1] = -(m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
    c[0, 2] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]

    # Row 1
    c[1, 0] = -(m[0, 1] * m[2, 2] - m[0, 2] * m[2, 1])
    c[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
    c[1, 2] = -(m[0, 0] * m[2, 1] - m[0, 1] * m[2, 0])

    # Row 2
    c[2, 0] = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
    c[2, 1] = -(m[0, 0] * m[1, 2] - m[0, 2] * m[1, 0])
    c[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]

    return c


def torch_calculate_ranking_score(
    normalized_mse_cofactors: torch.Tensor,
    alpha_s_sil: torch.Tensor,
    normalized_expectancy: torch.Tensor,
    expectancy: torch.Tensor,
    profile: str = "balanced",
) -> torch.Tensor:
    """Differentiable ranking score."""
    # Weights based on profile
    if profile == "aggressive":
        w_structure = 0.30
        w_alpha = 0.30
        w_exp = 0.20
    elif profile == "conservative":
        w_structure = 0.70
        w_alpha = 0.10
        w_exp = 0.05
    else:  # balanced
        w_structure = 0.50
        w_alpha = 0.15
        w_exp = 0.07

    # Penalize negative normalized expectancy
    norm_exp_term = normalized_expectancy
    if norm_exp_term < 0:
        norm_exp_term = norm_exp_term * 2.0

    return (normalized_mse_cofactors * w_structure) + (alpha_s_sil * w_alpha) - (norm_exp_term * w_exp)


def torch_calculate_probability_sgo_matrix(outcomes: torch.Tensor, bins: int = 3, sigma: float = 0.1) -> torch.Tensor:
    """
    Differentiable Probability SGO Matrix using soft-assignment.

    Assigns each outcome to a (direction, magnitude) cell using a
    differentiable approximation of discretisation.

    Uses the softmax construction from the whitepaper (eq. 6.7):
        (w_-, w_0, w_+) = softmax([-τ·x, -τ·|x|, τ·x])

    This replaces the sigmoid construction where the neutral channel
    was identically zero (sigmoid(x) + sigmoid(-x) = 1 for all x).
    The softmax formulation gives non-trivial probability mass to all
    three directions.
    """
    # Flatten to [Batch] for robust processing
    outcomes = outcomes.flatten()
    batch_size = outcomes.size(0)
    if batch_size == 0:
        return torch.zeros(3, bins, device=outcomes.device)

    # 1. Soft-Direction (Loss, Neutral, Win) using softmax (eq. 6.7)
    tau = 1.0 / sigma
    x = outcomes
    # Columns: [loss, neutral, win]
    #   loss:   -τ·x      (large when x is negative)
    #   neutral: -τ·|x|   (large when x is near zero)
    #   win:    τ·x       (large when x is positive)
    logits = torch.stack(
        [
            -tau * x,
            -tau * torch.abs(x),
            tau * x,
        ],
        dim=1,
    )
    soft_directions = torch.softmax(logits, dim=1)  # (N, 3)

    # 2. Soft-Magnitude
    abs_mags = torch.abs(outcomes)
    max_mag = abs_mags.max().detach() + 1e-8
    norm_mags = abs_mags / max_mag

    # We use fixed centers [0, 1] for scale-invariant probability mass
    centers = torch.linspace(0, 1, bins, device=outcomes.device)
    dist = (norm_mags.unsqueeze(1) - centers) ** 2
    soft_magnitudes = torch.softmax(-dist / sigma, dim=1)  # (N, bins)

    # 3. Join (Outer Product per row)
    joint_probs = torch.bmm(soft_directions.unsqueeze(2), soft_magnitudes.unsqueeze(1))
    return joint_probs.sum(dim=0) / (batch_size + 1e-12)


def torch_calculate_sgo_divergence(
    p_matrix: torch.Tensor,
    target_matrix: torch.Tensor | None = None,
    method: str = "kl",  # "kl" or "fro"
) -> torch.Tensor:
    """Differentiable distance/divergence from target."""
    if target_matrix is None:
        target_matrix = torch.zeros_like(p_matrix)
        target_matrix[2, -1] = 1.0  # Target: Large Wins

    if method == "kl":
        # PyTorch kl_div(log_observed, target) computes D(Target || Observed).
        # With a one-hot target this is a target-cell likelihood penalty, not
        # observed-to-target KL.
        return torch.nn.functional.kl_div((p_matrix + 1e-12).log(), target_matrix + 1e-12, reduction="batchmean")
    else:
        # Frobenius distance
        return torch.sqrt(torch.sum((p_matrix - target_matrix) ** 2) + 1e-8)


def torch_evaluate_structural_geometry(
    positives: torch.Tensor,
    negatives: torch.Tensor,
    utility_power: float = 1.1,
    ranking_profile: str = "balanced",
    regime_aware: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Differentiable version of evaluate_structural_geometry for PyTorch.

    Returns a dictionary of tensors that track gradients back to outcomes.
    """
    # Outcomes are tensors, possibly with gradients
    transformed_pos = torch_apply_utility_transformation(positives, power=utility_power)
    transformed_neg = torch_apply_utility_transformation(negatives, power=utility_power)

    pos_count = transformed_pos.shape[0]
    neg_count = transformed_neg.shape[0]
    N = float(pos_count + neg_count)

    win_rate = pos_count / N
    loss_rate = neg_count / N

    avg_win = torch.mean(transformed_pos) if pos_count > 0 else torch.tensor(0.0, device=positives.device)
    avg_loss = torch.abs(torch.mean(transformed_neg)) if neg_count > 0 else torch.tensor(0.0, device=negatives.device)

    s = avg_win - avg_loss

    # Expectancy and Normalized Expectancy
    be_win_rate = avg_loss / (avg_win + avg_loss + 1e-8)
    normalized_expectancy = (be_win_rate * avg_win) - (win_rate * avg_loss)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    # Construct N-Matrix (Differentiably)
    row0 = torch.stack([avg_win, avg_loss, s])
    row1 = torch.tensor([neg_count, pos_count, N], device=positives.device, dtype=torch.float32)
    row2 = torch.tensor([loss_rate, win_rate, 1.0], device=positives.device, dtype=torch.float32)
    n_mat = torch.stack([row0, row1, row2])

    # Ideal Matrix
    s_ideal_value = avg_win
    i_row0 = torch.stack([s_ideal_value, torch.tensor(0.0, device=positives.device), s_ideal_value])
    i_row1 = torch.tensor([0.0, N, N], device=positives.device, dtype=torch.float32)
    i_row2 = torch.tensor([0.0, 1.0, 1.0], device=positives.device, dtype=torch.float32)
    ideal_mat = torch.stack([i_row0, i_row1, i_row2])

    # Cofactors
    cofactors = torch_calculate_cofactor_matrix_3x3(n_mat)
    ideal_cofactors = torch_calculate_cofactor_matrix_3x3(ideal_mat)

    # Metrics
    ideal_alpha = ideal_cofactors[1, 2]
    s_observed = n_mat[0, 2]
    # Raw mathematical bounds are [0, 2]
    alpha_s_sil = (ideal_alpha + s_observed) ** 2 / (ideal_alpha**2 + s_observed**2 + 1e-8)

    # NMSE
    mse_cofactors = torch.mean((cofactors - ideal_cofactors) ** 2)
    observed_mse = torch.mean(cofactors**2)
    ideal_mse = torch.mean(ideal_cofactors**2)
    norm_mse_scale = torch.max(observed_mse, ideal_mse)
    normalized_mse_cofactors = mse_cofactors / (norm_mse_scale + 1e-8)

    # Ranking Score
    ranking_score = torch_calculate_ranking_score(
        normalized_mse_cofactors, alpha_s_sil, normalized_expectancy, expectancy, profile=ranking_profile
    )

    return {
        "alpha_s_sil": alpha_s_sil,
        "normalized_mse_cofactors": normalized_mse_cofactors,
        "normalized_expectancy": normalized_expectancy,
        "ranking_score": ranking_score,
        "s_observed": s_observed,
        "expectancy": expectancy,
    }
