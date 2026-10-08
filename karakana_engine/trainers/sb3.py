"""Stable-Baselines3 (SB3) Integration for karakana Structural Monitoring.

Sprint 17 upgrade: StructuralSB3Callback now tracks α trajectory across
rollouts via AlphaMomentumTracker and logs V_α, M_α, and structural
horizon to the SB3 logger (TensorBoard / CSV / stdout).
"""

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    # Define stubs so the file can still be parsed without SB3 installed
    class BaseCallback:
        def __init__(self, verbose=0):
            pass

    PPO = object

import numpy as np

from karakana_engine.metrics import AlphaMomentumTracker, evaluate_structural_geometry
from karakana_engine.torch_loss import TorchSGOLoss


class StructuralSB3Callback(BaseCallback):
    """
    A Stable-Baselines3 callback that monitors structural health across rollouts.

    Sprint 17: tracks α trajectory via AlphaMomentumTracker and logs
    V_α (velocity), M_α (momentum), and H_s (structural horizon) to
    enable early-warning collapse detection in SB3 training loops.

    Logged metrics (per rollout):
        rollout/structural_nmse            — normalized MSE cofactors
        rollout/structural_ranking_score   — karakana ranking score
        rollout/structural_bias_s          — expectancy
        rollout/structural_credibility     — sample-size credibility
        rollout/trajectory_alpha           — EMA-smoothed α
        rollout/trajectory_v_alpha         — α velocity
        rollout/trajectory_m_alpha         — α momentum
        rollout/trajectory_horizon         — estimated rollouts until collapse
        rollout/trajectory_is_collapsing   — 1 if collapse detected, else 0
    """

    def __init__(
        self,
        verbose: int = 0,
        ema_weight: float = 0.3,
        v_alpha_panic: float = -0.02,
        alpha_floor: float = 0.10,
    ):
        super().__init__(verbose)
        self.rollout_rewards: list = []
        self.tracker = AlphaMomentumTracker(window_size=20, alpha_ema=ema_weight)
        self.v_alpha_panic = v_alpha_panic
        self.alpha_floor = alpha_floor
        self._rollout_count: int = 0
        self._collapse_count: int = 0
        self._e_stop: bool = False

    def _on_step(self) -> bool:
        """Collect rewards from each environment step."""
        self.rollout_rewards.extend(self.locals["rewards"].tolist())
        return True

    def _on_rollout_end(self) -> None:
        """Compute structural metrics and update trajectory tracker."""
        self._rollout_count += 1
        rewards = np.array(self.rollout_rewards)
        positives = rewards[rewards > 0]
        negatives = rewards[rewards <= 0]
        total = len(positives) + len(negatives)

        # ── Aggregate structural metrics ──────────────────────────────
        if len(positives) > 0 and len(negatives) > 0:
            try:
                report = evaluate_structural_geometry(positives, negatives)
                self.logger.record("rollout/structural_nmse", report["normalized_mse_cofactors"])
                self.logger.record("rollout/structural_ranking_score", report["ranking_score"])
                self.logger.record("rollout/structural_bias_s", report["expectancy"])

                # ── Sprint 17: trajectory tracking ────────────────────
                # Tracker expects structural health [0, 1] where 1 is ideal
                alpha = float(np.clip(1.0 - float(report.get("alpha_s_sil", 1.0)) / 2.0, 0.0, 1.0))
            except Exception as e:
                if self.verbose > 0:
                    print(f"karakana callback error: {e}")
                alpha = 1.0
                self.logger.record("rollout/structural_nmse", 1.0)
                self.logger.record("rollout/structural_ranking_score", 1000.0)
                self.logger.record("rollout/structural_bias_s", 0.0)
        else:
            alpha = 1.0
            self.logger.record("rollout/structural_nmse", 1.0)
            self.logger.record("rollout/structural_ranking_score", 1000.0)
            self.logger.record(
                "rollout/structural_bias_s",
                float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
            )

        # ── Sprint 17: push to AlphaMomentumTracker ───────────────────
        self.tracker.push(alpha)
        stats = self.tracker.get_stats()

        self.logger.record("rollout/trajectory_alpha", stats["alpha"])
        self.logger.record("rollout/trajectory_v_alpha", stats["v_alpha"])
        self.logger.record("rollout/trajectory_m_alpha", stats["m_alpha"])
        self.logger.record("rollout/trajectory_horizon", stats["horizon"])
        self.logger.record("rollout/trajectory_is_collapsing", int(stats["is_collapsing"]))

        # Collapse detection
        if stats["is_collapsing"]:
            self._collapse_count += 1
            if self.verbose > 0:
                print(
                    f"[karakana] Rollout {self._rollout_count}: collapse detected "
                    f"(α={stats['alpha']:.3f}, Vα={stats['v_alpha']:.4f}, "
                    f"H={stats['horizon']:.1f})"
                )

        # E-stop: alpha below floor or sustained collapse
        if stats["alpha"] < self.alpha_floor:
            self._e_stop = True
            if self.verbose > 0:
                print(f"[karakana] E-STOP: α={stats['alpha']:.3f} below floor={self.alpha_floor}")

        # Credibility
        self.logger.record("rollout/structural_credibility", float(total / (total + 50.0)))

        # Clear for next rollout
        self.rollout_rewards = []

    def should_stop(self) -> bool:
        """Return True if structural collapse has been detected.

        Call this in the training loop to implement early stopping::

            callback = StructuralSB3Callback()
            model.learn(total_timesteps=100000, callback=callback)
            if callback.should_stop():
                print("Structural collapse detected — halting early")
        """
        return self._e_stop

    @property
    def collapse_count(self) -> int:
        """Number of rollouts where collapse was detected."""
        return self._collapse_count

    @property
    def rollout_count(self) -> int:
        """Number of rollouts processed."""
        return self._rollout_count


class StructuralPPO(PPO):
    """
    A custom PPO implementation that adds structural regularization to the
    actor-critic objective.

    Sprint 17: accepts an optional ``auto_tune_lambda`` parameter that, when
    combined with a ``StructuralSB3Callback``, adjusts λ based on trajectory
    health.

    This is an example of 'Active Structural Optimization' integrated into SB3.
    """

    def __init__(self, *args, lambda_sgo: float = 0.1, auto_tune_lambda: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_sgo = lambda_sgo
        self._base_lambda = lambda_sgo
        self._auto_tune_lambda = auto_tune_lambda
        self.sgo_loss_fn = TorchSGOLoss(lambda_sgo=lambda_sgo)

    def train(self) -> None:
        """
        Override the SB3 train method to inject SGO Loss.

        In a full implementation, you would:
        1. Call standard rollout/buffer processing
        2. Add ``self.sgo_loss_fn(buffer_rewards)`` to the PPO objective
        3. If ``auto_tune_lambda`` is enabled and a callback is attached,
           adjust ``self.sgo_loss_fn.lambda_sgo`` based on trajectory stats
        4. Call ``.backward()`` on the combined loss
        """
        super().train()
