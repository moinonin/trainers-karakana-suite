"""Differentiable SGO Loss for PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .torch_metrics import (
    torch_evaluate_structural_geometry,
)
from .torch_probability import TorchProbabilitySGO


class TorchSGOLoss(nn.Module):
    """
    PyTorch module for active Structural Geometry Optimization.
    """

    def __init__(
        self,
        lambda_sgo: float = 0.1,
        metric_key: str = "alpha_s_sil",
        utility_power: float = 1.1,
        loss_module: str = "probability_kl",  # "moment", "probability_fro", or "probability_kl"
        bins: int = 3,
        ranking_profile: str = "balanced",
        use_coupling_trajectory: bool = False,
        lambda_coupling: float = 0.0,
        coupling_k: int = 10,
        coupling_gradient: bool = False,
    ):
        super().__init__()
        self.lambda_sgo = lambda_sgo
        self.metric_key = metric_key
        self.utility_power = utility_power
        self.loss_module = loss_module
        self.ranking_profile = ranking_profile
        self.use_coupling_trajectory = use_coupling_trajectory
        self.lambda_coupling = lambda_coupling
        self.coupling_k = coupling_k
        self.coupling_gradient = coupling_gradient
        self.last_rho_sq: float | None = None

        if "probability" in loss_module:
            self.prob_sgo = TorchProbabilitySGO(bins=bins)
            # Ideal Target: [Win, Large]
            target = torch.zeros(3, bins)
            target[2, -1] = 1.0
            self.register_buffer("target", target)

    def forward(self, outcomes: torch.Tensor, base_loss: torch.Tensor | None = None) -> torch.Tensor:
        """
        Calculate the combined loss with structural regularization.
        """
        self.last_rho_sq = None
        if self.loss_module == "moment":
            positives = outcomes[outcomes > 0]
            negatives = outcomes[outcomes <= 0]

            if positives.shape[0] == 0 or negatives.shape[0] == 0:
                structural_penalty = torch.tensor(1.0, device=outcomes.device, requires_grad=True)
            else:
                metrics = torch_evaluate_structural_geometry(
                    positives, negatives, utility_power=self.utility_power, ranking_profile=self.ranking_profile
                )
                structural_penalty = metrics.get(self.metric_key, torch.tensor(1.0, device=outcomes.device)) / 2.0

        elif self.loss_module == "probability_kl":
            p_obs = self.prob_sgo(outcomes)
            # PyTorch computes D(target || observed) for this argument order.
            # Keep behavior stable; see docs/MATH_CONTRACT.md for naming.
            structural_penalty = functional.kl_div(
                (p_obs + 1e-12).log(), self.target + 1e-12, reduction="batchmean", log_target=False
            )

        elif self.loss_module == "probability_fro":
            p_obs = self.prob_sgo(outcomes)
            structural_penalty = torch.sqrt(torch.sum((p_obs - self.target) ** 2) + 1e-8)

        else:
            raise ValueError(f"Unknown SGO loss module: {self.loss_module}")

        # ── K-configurable coupling trajectory penalty (SGPO extension) ──
        coupling_penalty = torch.tensor(0.0, device=outcomes.device, requires_grad=True)
        if (
            self.use_coupling_trajectory
            and self.lambda_coupling > 0
            and self.coupling_k > 0
            and outcomes.numel() >= self.coupling_k
        ):
            K = int(self.coupling_k)
            # Chunk sequence into non-overlapping units of size K from the back
            # to ensure terminal rewards (which have highest variance) are included.
            num_chunks = outcomes.numel() // K
            if num_chunks > 0:
                chunks = outcomes[-num_chunks * K :].view(num_chunks, K)
                # Apply utility transform using self.utility_power
                p = self.utility_power
                signs = torch.sign(chunks)
                abs_chunks = torch.abs(chunks)
                transformed = signs * (abs_chunks**p)
                P_u = torch.sum(torch.clamp(transformed, min=0.0), dim=1)
                N_u = torch.sum(torch.clamp(-transformed, min=0.0), dim=1)
                E_PN = torch.mean(P_u * N_u)
                E_P2 = torch.mean(P_u**2)
                E_N2 = torch.mean(N_u**2)
                denom = E_P2 * E_N2 + 1e-12
                rho_sq = (E_PN**2) / denom
                # Direct coupling penalty (scale-invariant, bounded [0,1])
                coupling_penalty = self.lambda_coupling * rho_sq
                self.last_rho_sq = float(rho_sq.detach().cpu().item())
                # Gradient-velocity decay: if enabled, penalize positive change
                # over the last chunk difference. Simplified here as negative
                # bonus when rho_sq is non-zero but below 1.0.
                if self.coupling_gradient:
                    # In full training loop, compare rho_sq(t) vs rho_sq(t-1)
                    # Here we approximate with a small decay incentive when
                    # rho_sq is non-zero but below 1.0.
                    coupling_penalty += self.lambda_coupling * 0.1 * torch.clamp(1.0 - rho_sq, min=0.0)

        reg_term = self.lambda_sgo * structural_penalty + coupling_penalty

        if base_loss is not None:
            return base_loss + reg_term
        return reg_term
