"""Loss functions and wrappers for Structural Geometry Optimization."""

from typing import Any

import numpy as np

from karakana.metrics import AlphaMomentumTracker, evaluate_structural_geometry


class SGOLoss:
    """
    A wrapper for integrating structural geometry metrics into training losses.

    This class collects outcomes (rewards/payoffs) and computes a regularization
    term that can be added to a standard policy or value loss. It includes
    trajectory-aware auto-tuning of the regularization weight (lambda) based on
    structural velocity and momentum via AlphaMomentumTracker.

    Sprint 17 upgrade: replaces StructuralTracker with AlphaMomentumTracker,
    adds rolling-window α computation, tiered λ response, and α-floor gate.
    """

    def __init__(
        self,
        base_lambda: float = 0.1,
        metric_key: str = "alpha_s_sil",
        utility_power: float = 1.1,
        auto_tune: bool = True,
        trajectory_window_size: int = 500,
        trajectory_window_step: int = 50,
        v_alpha_panic: float | None = None,
        alpha_floor: float | None = None,
    ):
        """
        Initialize the SGO Loss wrapper.

        Args:
            base_lambda: Base regularization weight for the structural term.
            metric_key: The specific karakana metric to use as the error signal.
                       Options: "alpha_s_sil", "normalized_mse_cofactors",
                                "relative_cofactors_fro_error", "normalized_structure_term"
            utility_power: Power for non-linear magnitude scaling.
            auto_tune: Whether to dynamically adjust lambda based on structural
                       velocity. When True, uses tiered AlphaMomentumTracker
                       response (Sprint 17). When False, lambda stays fixed.
            trajectory_window_size: Number of recent outcomes for rolling-window
                                    α computation.
            trajectory_window_step: Recompute α every N outcomes.
            v_alpha_panic: V_α threshold for collapse warning.
                           Defaults to config GRG value.
            alpha_floor: α below which structure is considered collapsed.
                         Defaults to config GRG value.
        """
        self.base_lambda = base_lambda
        self.current_lambda = base_lambda
        self.metric_key = metric_key
        self.utility_power = utility_power
        self.auto_tune = auto_tune

        # ── Sprint 17: Trajectory-Aware Audit ─────────────────────────
        self.trajectory_window_size = trajectory_window_size
        self.trajectory_window_step = trajectory_window_step

        # Resolve thresholds from config if not explicitly provided
        if v_alpha_panic is None or alpha_floor is None:
            from karakana.config import get_config

            cfg = get_config() if callable(get_config) else {}
            if v_alpha_panic is None:
                v_alpha_panic = cfg.get("v_alpha_panic_threshold", -0.02)
            if alpha_floor is None:
                alpha_floor = cfg.get("alpha_safety_floor", 0.10)

        self.v_alpha_panic = v_alpha_panic
        self.alpha_floor = alpha_floor

        # AlphaMomentumTracker with EMA smoothing (replaces StructuralTracker)
        self.tracker = AlphaMomentumTracker(window_size=20, alpha_ema=0.3)

        # Rolling window for recent outcomes (for α trajectory)
        self.outcomes: list[float] = []
        self._step_counter: int = 0

    def add_outcome(self, magnitude: float):
        """Record a single outcome magnitude (win or loss)."""
        self.outcomes.append(float(magnitude))

    def reset(self):
        """Clear recorded outcomes and reset tracker state."""
        self.outcomes = []
        self._step_counter = 0
        self.current_lambda = self.base_lambda
        self.tracker = AlphaMomentumTracker(window_size=20, alpha_ema=0.3)

    def _update_lambda(self):
        """Tiered λ adjustment based on AlphaMomentumTracker stats.

        Replaces the old binary tighten/relax with three response tiers:
        - Tier 3 (collapsing): α < floor or V_α < 2× panic → aggressive clamp
        - Tier 2 (degrading):  V_α < panic → fast tightening
        - Tier 1 (drift):      V_α < 0 → gentle tightening
        - Improving:           V_α > 0.01 → relax
        """
        if not self.auto_tune:
            return

        stats = self.tracker.get_stats()
        v_alpha = stats["v_alpha"]
        alpha = stats["alpha"]

        if alpha < self.alpha_floor or v_alpha < self.v_alpha_panic * 2:
            # Tier 3 — Collapsing: aggressive clamp + momentum check
            self.current_lambda = min(self.base_lambda * 20.0, self.current_lambda * 2.0)
        elif v_alpha < self.v_alpha_panic:
            # Tier 2 — Degrading fast
            self.current_lambda = min(self.base_lambda * 10.0, self.current_lambda * 1.5)
        elif v_alpha < 0:
            # Tier 1 — Slow structural drift
            self.current_lambda = min(self.base_lambda * 5.0, self.current_lambda * 1.10)
        elif v_alpha > 0.01:
            # Improving — relax regularization
            self.current_lambda = max(self.base_lambda * 0.1, self.current_lambda * 0.90)
        # else: dead zone near zero — no change

    def _maybe_update_trajectory(self):
        """Recompute rolling-window α and push to tracker every WINDOW_STEP steps."""
        if not self.auto_tune:
            return

        self._step_counter += 1
        if self._step_counter % self.trajectory_window_step != 0:
            return

        window = self.outcomes[-self.trajectory_window_size :]
        if len(window) < max(10, self.trajectory_window_size // 10):
            return  # not enough data yet

        positives = [o for o in window if o > 0]
        negatives = [o for o in window if o <= 0]

        if len(positives) == 0 or len(negatives) == 0:
            return

        try:
            metrics = evaluate_structural_geometry(
                np.array(positives),
                np.array(negatives),
                utility_power=self.utility_power,
            )
            alpha = float(metrics.get("alpha_s_sil", 0.0))
            self.tracker.push(alpha)
        except Exception:
            pass  # skip noisy windows

    def calculate_regularization(self) -> float:
        """
        Compute the structural regularization term based on collected outcomes.

        Returns:
            The weighted structural error: λ * structural_error
        """
        if not self.outcomes:
            return 0.0

        outcomes_np = np.array(self.outcomes)
        positives = outcomes_np[outcomes_np > 0]
        negatives = outcomes_np[outcomes_np <= 0]

        if positives.size == 0 or negatives.size == 0:
            # Cannot compute geometry with only one side; return high penalty
            # to encourage exploration of both win and loss states.
            penalty = self.current_lambda * 1.0
            self._maybe_update_trajectory()
            self._update_lambda()
            return penalty

        try:
            metrics = evaluate_structural_geometry(
                positives,
                negatives,
                utility_power=self.utility_power,
            )
            structural_error = metrics.get(self.metric_key, 1.0)

            # ── Sprint 17: trajectory-aware audit ─────────────────────
            self._maybe_update_trajectory()
            self._update_lambda()

            return float(self.current_lambda * structural_error)
        except Exception:
            self._maybe_update_trajectory()
            self._update_lambda()
            return self.current_lambda * 1.0

    def total_loss(self, base_loss: float) -> float:
        """Combine a base loss with the SGO regularization term."""
        return float(base_loss + self.calculate_regularization())

    def get_tracker_stats(self) -> dict[str, Any]:
        """Return current AlphaMomentumTracker stats for monitoring."""
        return self.tracker.get_stats()
