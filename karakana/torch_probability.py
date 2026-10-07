"""Differentiable Module II: Probability SGO in PyTorch.

Uses soft-assignment binning to allow backpropagation through discretised
probability matrices and KL-Divergence loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchProbabilitySGO(nn.Module):
    """
    Differentiable Probability SGO Matrix (3xBins).

    Soft-assigns outcomes to bins based on their distance to bin centers.
    """

    def __init__(self, bins: int = 3, sigma: float = 0.1):
        super().__init__()
        self.bins = bins
        self.sigma = sigma

        # Bin centers [0, 1]
        self.register_buffer("centers", torch.linspace(0, 1, bins))

    def forward(self, outcomes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outcomes: [Batch] Tensor of rewards. Can be 1D or 2D.

        Returns:
            [3, Bins] Probability Matrix.

        Uses the softmax construction from the whitepaper (eq. 6.7):
            (w_-, w_0, w_+) = softmax([-τ·x, -τ·|x|, τ·x])

        This replaces the sigmoid construction where neutral was identically
        zero (since sigmoid(x) + sigmoid(-x) = 1 for all x). The softmax
        formulation gives a non-trivial neutral channel.
        """
        # Flatten to [Batch] for robust processing
        outcomes = outcomes.flatten()
        batch_size = outcomes.size(0)
        if batch_size == 0:
            return torch.zeros(3, self.bins, device=outcomes.device)

        # 1. Soft Direction Assignment [Batch, 3]
        # Loss (<0), Neutral (~0), Win (>0)
        # Using softmax construction (eq. 6.7) instead of sigmoid:
        #   sigmoid(x) + sigmoid(-x) = 1  → neutral is identically zero
        #   softmax([-τ·x, -τ·|x|, τ·x]) → all three channels are non-trivial
        tau = 1.0 / self.sigma
        x = outcomes
        # Columns: [loss_prob, neut_prob, win_prob]
        #   loss:   -τ·x      (large when x is negative)
        #   neutral: -τ·|x|   (large when x is near zero)
        #   win:    τ·x       (large when x is positive)
        direction_logits = torch.stack(
            [
                -tau * x,
                -tau * torch.abs(x),
                tau * x,
            ],
            dim=1,
        )  # [Batch, 3]
        directions = F.softmax(direction_logits, dim=1)  # [Batch, 3]

        # 2. Soft Magnitude Assignment [Batch, Bins]
        # Normalize magnitudes to [0, 1] for binning
        abs_mags = torch.abs(outcomes)
        max_mag = abs_mags.max().detach() + 1e-8
        norm_mags = abs_mags / max_mag

        # Calculate squared distance to each bin center
        # [Batch, 1] - [Bins] -> [Batch, Bins]
        dists = (norm_mags.unsqueeze(1) - self.centers) ** 2
        # Softmax over negative distances to assign to nearest bin
        mag_assignment = F.softmax(-dists / self.sigma, dim=1)

        # 3. Outer Product and Sum [3, Bins]
        # directions: [Batch, 3, 1], mag_assignment: [Batch, 1, Bins]
        joint = torch.bmm(directions.unsqueeze(2), mag_assignment.unsqueeze(1))

        # Sum and Normalize
        matrix = joint.sum(dim=0) / (batch_size + 1e-12)
        return matrix


class ProbabilitySGOLoss(nn.Module):
    """
    KL-Divergence Loss for Module II.
    """

    def __init__(self, bins: int = 3, lambda_prob: float = 0.1):
        super().__init__()
        self.sgo_module = TorchProbabilitySGO(bins=bins)
        self.lambda_prob = lambda_prob

        # Ideal: All mass in [Win (2), Large (Bins-1)]
        target = torch.zeros(3, bins)
        target[2, -1] = 1.0
        self.register_buffer("target", target)

    def forward(self, outcomes: torch.Tensor, base_loss: torch.Tensor) -> torch.Tensor:
        p_obs = self.sgo_module(outcomes)

        # PyTorch computes D(Target || Observed) for this argument order.
        # With a one-hot target this is a target-cell likelihood penalty.
        kl_div = F.kl_div((p_obs + 1e-12).log(), self.target + 1e-12, reduction="batchmean", log_target=False)

        return base_loss + self.lambda_prob * kl_div
