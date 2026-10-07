"""Differentiable PyTorch implementation of Structural Geometry Optimization."""

import torch
import torch.nn as nn

from .torch_loss import TorchSGOLoss  # noqa: F401
from .torch_metrics import (
    torch_calculate_probability_sgo_matrix as _canonical_probability_sgo_matrix,
)
from .torch_metrics import (
    torch_calculate_sgo_divergence as _canonical_sgo_divergence,
)

_EPS = 1e-12


def apply_utility_transformation_pt(magnitudes: torch.Tensor, power: float = 1.1) -> torch.Tensor:
    """Apply non-linear utility transformation differentiably."""
    signs = torch.sign(magnitudes)
    abs_mags = torch.abs(magnitudes)
    # Adding a small eps for numerical stability during backprop
    return signs * (abs_mags + _EPS) ** power


def calculate_cofactor_matrix_pt(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the cofactor matrix for a 3x3 numeric matrix differentiably.
    For a 3x3 matrix, the cofactor matrix is the transpose of the adjugate.
    det(A) * A^-1 = adj(A), cofactor(A) = adj(A)^T
    """
    # For a general batched or single 3x3 matrix:
    # cofactor(A)_{ij} = (-1)^{i+j} det(M_{ij})
    # Luckily, for 3x3, we can just use torch's inverse and determinant if det != 0
    # Or explicitly compute the 2x2 determinants for stability.

    cofactors = torch.empty_like(matrix)
    for i in range(3):
        for j in range(3):
            # rows and cols to keep
            rows = [r for r in range(3) if r != i]
            cols = [c for c in range(3) if c != j]
            # 2x2 minor
            minor = matrix[torch.tensor(rows).unsqueeze(1), torch.tensor(cols)]
            det = minor[0, 0] * minor[1, 1] - minor[0, 1] * minor[1, 0]
            cofactors[i, j] = ((-1) ** (i + j)) * det

    return cofactors


def evaluate_structural_geometry_pt(
    positive_outcomes: torch.Tensor,
    negative_outcomes: torch.Tensor,
    utility_power: float = 1.1,
    credibility_half_life: float = 50.0,
) -> dict[str, torch.Tensor]:
    """
    Differentiable evaluation of structural geometry.
    This can be used natively in PyTorch loss graphs.
    """
    if positive_outcomes.numel() == 0 or negative_outcomes.numel() == 0:
        # Prevent graph breaks by returning high structural penalty if empty
        dummy_penalty = torch.tensor(1.0, device=positive_outcomes.device, requires_grad=True)
        return {"alpha_s_sil": dummy_penalty, "smse": dummy_penalty, "normalized_structure_term": dummy_penalty}

    pos_trans = apply_utility_transformation_pt(positive_outcomes, power=utility_power)
    neg_trans = apply_utility_transformation_pt(negative_outcomes, power=utility_power)

    pos_count = pos_trans.numel()
    neg_count = neg_trans.numel()
    N = pos_count + neg_count

    win_rate = pos_count / N
    loss_rate = neg_count / N

    avg_win = torch.mean(pos_trans)
    avg_loss = torch.abs(torch.mean(neg_trans))

    s_observed = avg_win - avg_loss
    s_ideal = avg_win

    n_mat = torch.stack(
        [
            torch.stack([avg_win, avg_loss, s_observed]),
            torch.stack(
                [
                    torch.tensor(neg_count, dtype=torch.float32, device=pos_trans.device),
                    torch.tensor(pos_count, dtype=torch.float32, device=pos_trans.device),
                    torch.tensor(N, dtype=torch.float32, device=pos_trans.device),
                ]
            ),
            torch.stack(
                [
                    torch.tensor(loss_rate, dtype=torch.float32, device=pos_trans.device),
                    torch.tensor(win_rate, dtype=torch.float32, device=pos_trans.device),
                    torch.tensor(1.0, dtype=torch.float32, device=pos_trans.device),
                ]
            ),
        ]
    )

    ideal_mat = torch.stack(
        [
            torch.stack([s_ideal, torch.tensor(0.0, device=pos_trans.device), s_ideal]),
            torch.stack(
                [
                    torch.tensor(0.0, device=pos_trans.device),
                    torch.tensor(N, dtype=torch.float32, device=pos_trans.device),
                    torch.tensor(N, dtype=torch.float32, device=pos_trans.device),
                ]
            ),
            torch.stack(
                [
                    torch.tensor(0.0, device=pos_trans.device),
                    torch.tensor(1.0, dtype=torch.float32, device=pos_trans.device),
                    torch.tensor(1.0, dtype=torch.float32, device=pos_trans.device),
                ]
            ),
        ]
    )

    smse = (s_observed - s_ideal) ** 2

    cofactors = calculate_cofactor_matrix_pt(n_mat)
    ideal_cofactors = calculate_cofactor_matrix_pt(ideal_mat)

    structural_alpha = cofactors[1, 2]
    ideal_alpha = ideal_cofactors[1, 2]

    alpha_s_sil = ((ideal_alpha + s_observed) ** 2) / (ideal_alpha**2 + s_observed**2 + 1e-12)

    cofactors_norm = torch.norm(cofactors)
    ideal_cofactors_norm = torch.norm(ideal_cofactors)
    cofactor_error = cofactors - ideal_cofactors
    mse_cofactors = torch.mean(torch.square(cofactor_error))

    normalization_scale = torch.max(
        torch.stack([cofactors_norm, ideal_cofactors_norm, torch.tensor(_EPS, device=pos_trans.device)])
    )
    relative_cofactors_fro_error = torch.norm(cofactor_error) / normalization_scale

    absolute_energy_signal = torch.log1p(cofactors_norm)
    energy_penalty = 1.0 / (1.0 + absolute_energy_signal)

    credibility = torch.tensor(N / (N + credibility_half_life), device=pos_trans.device)

    normalized_structure_term = (
        (mse_cofactors + relative_cofactors_fro_error) / torch.clamp(credibility, min=0.01)
    ) + energy_penalty

    return {
        "alpha_s_sil": alpha_s_sil,
        "smse": smse,
        "normalized_structure_term": normalized_structure_term,
        "structural_alpha": structural_alpha,
    }


def calculate_rho_pn_sq(outcomes: torch.Tensor, utility_power: float = 1.1) -> torch.Tensor:
    """
    Calculate the scale-invariant Squared Correlation Penalty (rho_PN^2).

    This function computes the squared normalized correlation between
    positive and negative outcomes within a unit of experience (episode).

    Args:
        outcomes: Tensor of shape (batch_size, sequence_length).
                  If 1D, it's treated as (batch_size, 1).
    """
    if outcomes.dim() == 1:
        outcomes = outcomes.unsqueeze(1)

    transformed = apply_utility_transformation_pt(outcomes, power=utility_power)
    P_u = torch.sum(torch.relu(transformed), dim=-1)
    N_u = torch.sum(torch.relu(-transformed), dim=-1)

    E_P2 = torch.mean(P_u**2)
    E_N2 = torch.mean(N_u**2)
    E_PN = torch.mean(P_u * N_u)

    # Scale-invariant squared correlation
    rho_pn_sq = (E_PN**2) / (E_P2 * E_N2 + _EPS)
    return rho_pn_sq


def torch_calculate_probability_sgo_matrix(
    outcomes: torch.Tensor,
    bins: int = 3,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Canonical differentiable Probability SGO Matrix using soft-assignment."""
    return _canonical_probability_sgo_matrix(outcomes, bins=bins, sigma=temperature)


def torch_calculate_sgo_divergence(
    p_matrix: torch.Tensor,
    target_matrix: torch.Tensor | None = None,
    method: str = "kl",  # "kl" or "fro"
) -> torch.Tensor:
    """Canonical differentiable distance/divergence from target."""
    return _canonical_sgo_divergence(p_matrix, target_matrix=target_matrix, method=method)


class SGOLossModule(nn.Module):
    """
    PyTorch native SGO Loss for Deep RL models.
    """

    def __init__(
        self,
        lambda_sgo: float = 0.1,
        metric_key: str = "rho_pn_sq",
        utility_power: float = 1.1,
        loss_module: str = "probability_kl",
        ranking_profile: str = "balanced",
    ):
        super().__init__()
        self.lambda_sgo = lambda_sgo
        self.metric_key = metric_key
        self.utility_power = utility_power
        self.loss_module = loss_module
        self.ranking_profile = ranking_profile

    def forward(self, outcomes: torch.Tensor, context_entropy: float = 0.0) -> torch.Tensor:
        """
        Compute structural loss with an optional uncertainty penalty.
        """
        if "probability" in self.loss_module:
            p_matrix = torch_calculate_probability_sgo_matrix(outcomes)
            method = "kl" if "kl" in self.loss_module else "fro"
            # Divergence from ideal (Large Wins)
            divergence = torch_calculate_sgo_divergence(p_matrix, method=method)
            return self.lambda_sgo * divergence

        if self.metric_key == "rho_pn_sq":
            return self.lambda_sgo * calculate_rho_pn_sq(
                outcomes,
                utility_power=self.utility_power,
            )

        # Fallback to older metrics...
        positives = outcomes[outcomes > 0]
        negatives = outcomes[outcomes <= 0]

        if positives.numel() == 0 or negatives.numel() == 0:
            return torch.tensor(self.lambda_sgo * 1.0, device=outcomes.device, requires_grad=True)

        metrics = evaluate_structural_geometry_pt(positives, negatives, utility_power=self.utility_power)
        return self.lambda_sgo * metrics.get(self.metric_key, torch.tensor(1.0, device=outcomes.device))
