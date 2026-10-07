"""Shared Geometric Reasoning Governor (GRG) controller for training loops.

Drop-in structural steering for any PyTorch training loop.  Monitors
alpha velocity (V_α), triggers LR slashing on collapse, restores LR
when stability returns, and fires e-stop when the structural horizon
is exhausted.

Usage::

    from karakana.trainers.grg import GRGController

    grg = GRGController(optimizer, base_lr=1e-4, profile="finance_rag")
    alpha_tracker = grg.tracker  # same AlphaMomentumTracker instance

    for epoch in range(epochs):
        for batch in loader:
            ...
            alpha = compute_alpha(outputs)
            grg.step(alpha)

            if grg.should_stop():
                print("GRG e-stop triggered")
                break
"""

from __future__ import annotations

from typing import Any

import torch

from karakana.config import get_config
from karakana.metrics import AlphaMomentumTracker


class GRGController:
    """Geometric Reasoning Governor for training-loop structural steering.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer whose learning rate will be modulated.
    base_lr: float
        Baseline learning rate to restore when stability returns.
    profile: str or None
        Config profile name for domain-specific thresholds.
        Falls back to ``global.grg`` defaults.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        profile: str | None = None,
        ranking_profile: str = "balanced",
        grg_window_size: int | None = None,
    ):
        grg_cfg = get_config.get_profile_grg(profile)

        self.optimizer = optimizer
        self.base_lr = base_lr
        self.ranking_profile = ranking_profile
        self.warmup_steps = int(grg_cfg.get("warmup_steps", 50))
        self.panic_threshold = float(grg_cfg.get("v_alpha_panic_threshold", -0.01))
        self.safety_floor = float(grg_cfg.get("alpha_safety_floor", 0.40))
        self.lr_slash_factor = float(grg_cfg.get("lr_slash_factor", 0.1))
        self.lr_restore_factor = float(grg_cfg.get("lr_restore_factor", 1.0))
        self.e_stop_horizon = int(grg_cfg.get("e_stop_horizon", 5))
        self.stability_threshold = float(grg_cfg.get("stability_v_alpha_threshold", 0.005))

        window_size = int(grg_cfg.get("window_size", 20)) if grg_window_size is None else grg_window_size

        self.tracker = AlphaMomentumTracker(
            window_size=window_size,
            alpha_ema=float(grg_cfg.get("ema_weight", 0.3)),
        )

        self._slashed: bool = False
        self._e_stop: bool = False
        self._step_count: int = 0
        self._panic_count: int = 0
        self._recovery_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, alpha: float) -> dict[str, Any]:
        """Push a new alpha observation and apply steering rules.

        Returns the tracker stats dict for logging.
        """
        self.tracker.push(float(alpha))
        self._step_count += 1
        stats = self.tracker.get_stats(
            v_panic=self.panic_threshold,
            floor=self.safety_floor,
        )

        v_alpha = float(stats["v_alpha"])
        m_alpha = float(stats["m_alpha"])
        horizon = float(stats["horizon"])
        is_collapsing = bool(stats["is_collapsing"])
        is_stable = bool(stats["is_stable"])

        # 1. Structural LR scheduling
        if is_collapsing and not self._slashed:
            self._slash_lr(v_alpha)
            self._panic_count += 1
        elif is_stable and self._slashed:
            self._restore_lr(v_alpha)
            self._recovery_count += 1

        # 2. E-stop: linear horizon exhaustion OR non-linear acceleration collapse
        # Use the controller's safety_floor for horizon, not the tracker's global default
        horizon = self.tracker.get_structural_horizon(floor=self.safety_floor)
        horizon_exhausted = horizon <= self.e_stop_horizon
        accelerating_collapse = is_collapsing and m_alpha < (self.panic_threshold / 2.0)
        alpha_below_floor = self.tracker.ema_alpha < self.safety_floor
        in_warmup = self._step_count <= self.warmup_steps
        # During warmup, only fire e-stop on velocity collapse (not static floor check)
        should_e_stop = (
            (accelerating_collapse) or (horizon_exhausted and not in_warmup) or (alpha_below_floor and not in_warmup)
        )
        if should_e_stop and self._step_count > self.tracker.window_size:
            if not self._e_stop:
                reason = (
                    "horizon exhausted"
                    if horizon_exhausted
                    else "accelerating collapse (M_α)"
                    if accelerating_collapse
                    else "alpha below safety floor"
                )
                print(
                    f"[GRG] E-Stop: {reason} (α={self.tracker.ema_alpha:.3f}, Vα={v_alpha:.4f}, Mα={m_alpha:.4f}, H={horizon:.1f})"
                )
            self._e_stop = True

        return stats

    def should_stop(self) -> bool:
        """Return True if the GRG e-stop condition has fired."""
        return self._e_stop

    @property
    def current_lr(self) -> float:
        """Current LR of the first param group."""
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def is_slashed(self) -> bool:
        """True if LR has been reduced by GRG."""
        return self._slashed

    @property
    def panic_count(self) -> int:
        """Number of times GRG has triggered LR slashing."""
        return self._panic_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _slash_lr(self, v_alpha: float) -> None:
        new_lr = self.base_lr * self.lr_slash_factor
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        self._slashed = True
        print(f"[GRG] V_α={v_alpha:.4f} < panic={self.panic_threshold}. LR slashed {self.base_lr:.2e} → {new_lr:.2e}")

    def _restore_lr(self, v_alpha: float) -> None:
        new_lr = self.base_lr * self.lr_restore_factor
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        self._slashed = False
        print(f"[GRG] V_α={v_alpha:.4f} > stable={self.stability_threshold}. LR restored → {new_lr:.2e}")
