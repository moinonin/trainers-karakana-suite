"""Freqtrade Hyperopt integration for karakana_engine."""

from datetime import datetime
from typing import Any

import numpy as np
from pandas import DataFrame

try:
    from freqtrade.constants import Config
    from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
except ImportError:
    # Fallback for environments without freqtrade installed
    class IHyperOptLoss:
        pass

    Config = dict[str, Any]

from karakana_engine.config import load_config
from karakana_engine.metrics import AlphaMomentumTracker, compute_sgo_matrix, evaluate_structural_geometry

_config_defaults = load_config().get("global", {})


class KarakanaStructuralHyperOptLoss(IHyperOptLoss):
    """
    Base Hyperopt loss function using karakana structural metrics.

    This loss function combines traditional profit-based metrics with
    karakana's structural geometry analysis to find hyperparameters that
    are both profitable and structurally robust.

    The reviewer's within-unit gain-loss co-occurrence diagnostic
    (E[PN], Cov(P,N), rho_PN) can be optionally enabled with the
    ``co_occurrence_diag`` flag. When enabled, the loss also penalizes
    strategies where gains and losses co-occur within the same trading
    window (high rho_PN), which the structural geometry framework
    associates with fragile regime dynamics.

    See: docs/whitepaper_v3.tex, Section on co-occurrence and
    fragility (Section~\\ref{sec:probability}).

    Usage::

        # Enable co-occurrence diagnostic during hyperopt
        freqtrade hyperopt --strategy KarakanaSGOShieldStrategy \\
            --hyperopt-loss KarakanaStructuralHyperOptLoss \\
            --params co_occurrence_diag=true \\
            --params co_occurrence_penalty=1000

        # Or in config.json
        {
            "co_occurrence_diag": true,
            "co_occurrence_penalty": 1000.0
        }

        # Adjust penalty weight (default 1000)
        # Higher values penalize co-occurrence more aggressively
        "co_occurrence_penalty": 2000,

    The diagnostic is computed from the Freqtrade results DataFrame,
    which typically contains open_date and close_date columns. If
    present, trades are grouped by day before computing the SGO
    moment matrix. If date columns are absent, each trade is treated
    as a separate unit (per-trade basis).

    Notes:
    - The penalty is based on rho_PN (normalized correlation),
      which is unit-independent and bounded in [-1, 1].
    - rho_PN > 0 indicates gains and losses co-occur (fragile regime).
    - rho_PN = 0 indicates mutual exclusion of gains and losses.
    - rho_PN < 0 indicates gains and losses alternate (healthy).
    """

    # Override this in subclasses to change structural sensitivity
    # Options: "balanced", "conservative", "aggressive"
    ranking_profile: str = "conservative"

    # ── Co-occurrence diagnostic (reviewer framework) ────────────
    # When enabled, computes E[PN], Cov(P,N), rho_PN from the
    # SGO moment matrix and applies a penalty when co-occurrence
    # exceeds expected levels under independence.
    #
    # NOTE: This diagnostic is most meaningful when trades are
    # aggregated by day (each day = one unit). In hyperopt, the
    # results DataFrame contains per-trade data, so co-occurrence
    # within a single day requires access to open_date/close_date.
    # For meaningful results, use with a custom objective that
    # provides per-day aggregates, or enable via config with
    # co_occurrence_trade_dates set to True (see docs).
    #
    # Default: disabled. Requires >= co_occurrence_min_trades
    # trades to produce stable estimates.
    co_occurrence_diag: bool | None = _config_defaults.get("use_coupling_trajectory", False)
    co_occurrence_penalty: float = _config_defaults.get("lambda_coupling", 0.0) * 2000  # scaled relative
    co_occurrence_min_trades: int = _config_defaults.get("coupling_K", 10) * 3
    coupling_K: int = _config_defaults.get("coupling_K", 10)

    # If True, compute co-occurrence on per-trade basis
    # (each trade is a unit). Default False (per-day aggregation
    # preferred but requires date access not available in
    # standard Freqtrade hyperopt results).
    co_occurrence_per_trade: bool = False

    # ── Sprint 17: Trajectory-Aware Audit ─────────────────────────────
    # When enabled, the objective also runs a rolling-window structural
    # audit on the chronological trade history and applies collapse
    # penalties.  Disabled by default for backward compatibility;
    # KarakanaConservative enables it.
    trajectory_audit_enabled: bool | None = None

    # Audit parameters (overridable in subclasses)
    trajectory_window_size: int = 500
    trajectory_window_step: int = 50
    trajectory_ema_weight: float = 0.3
    trajectory_v_alpha_panic: float = -0.02
    trajectory_alpha_floor: float = 0.10

    # Penalty weights added to the aggregate ranking_score
    trajectory_penalty_collapsed: float = 5000.0
    trajectory_penalty_collapsing: float = 2000.0
    trajectory_penalty_degrading: float = 500.0
    trajectory_bonus_horizon: float = -200.0  # negative = reward
    trajectory_min_horizon_for_bonus: int = 100

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        *args,
        **kwargs,
    ) -> float:
        """
        Static method required by Freqtrade IHyperOptLoss interface.
        Delegates to the objective function of an instance.
        """
        # Create an instance to call the objective method with correct profile
        # Use the class that called this static method
        return KarakanaStructuralHyperOptLoss().objective(
            results, trade_count, min_date, max_date, *args, config=config, **kwargs
        )

    def _compute_co_occurrence_diag(self, profits: np.ndarray, dates: np.ndarray | None = None) -> float:
        """
        Compute within-unit gain-loss co-occurrence diagnostic.

        For each unit, computes P_u = sum(gains) and
        N_u = sum(losses), then evaluates E[PN], Cov(P,N), rho_PN.

        Parameters:
        - profits: array of profit ratios (one per trade)
        - dates: optional array of trade close dates (datetime-like).
          If provided, trades are aggregated by day before computing
          the moment matrix. If None, each trade is treated as a
          separate unit (per-trade basis).

        Returns a penalty (>= 0) when co-occurrence exceeds the
        independence baseline, or 0.0 when the diagnostic is disabled
        or insufficient data exists.

        The reviewer's framework (whitepaper_v3.tex):
        - E[PN] = 0 iff mutual exclusion of gains and losses
        - Cov(P,N) > 0 indicates a fragile regime (both gains and
          losses spike together)
        - rho_PN measures normalized co-occurrence strength
        """
        co_occ = getattr(self, "co_occurrence_diag", None)
        if co_occ is None:
            from karakana_engine.config import get_config as _cfg

            co_occ = _cfg.get("co_occurrence_diag", False)
        if not co_occ:
            return 0.0

        n = len(profits)
        if n < self.co_occurrence_min_trades:
            return 0.0

        try:
            # Aggregate by day if dates are provided
            if dates is not None:
                import pandas as pd

                dates_arr = np.asarray(dates)
                date_strs = pd.to_datetime(dates_arr).strftime("%Y-%m-%d").tolist()
                df_daily = pd.DataFrame({"profit": profits, "date": date_strs})
                # Group raw returns by day for compute_sgo_matrix
                # Each day is a unit; compute_sgo_matrix computes
                # P_u and N_u internally from the raw returns
                daily_units = []
                for _date_str, group in df_daily.groupby("date"):
                    unit = group["profit"].to_numpy()
                    daily_units.append(unit)
                num_units = len(daily_units)
            else:
                # Use configurable K chunking for the co-occurrence diagnostic
                # (matches the proposal's Step B: K=10 experience units)
                K = int(getattr(self, "coupling_K", 10))
                if K < 1:
                    K = 10
                # Chunk the profit sequence into non-overlapping K-sized units
                chunked_units = []
                for i in range(0, len(profits) - K + 1, K):
                    chunked_units.append(profits[i : i + K].astype(float))
                if len(chunked_units) < 2:
                    return 0.0
                daily_units = chunked_units
                num_units = len(chunked_units)

            if num_units < 2:
                return 0.0

            # Compute the moment matrix E[[1, P, N]^T [1, P, N]]
            # compute_sgo_matrix expects a list of arrays (raw returns per unit)
            matrix = compute_sgo_matrix(daily_units, utility_power=1.0)
            e_p = float(matrix[0, 1])
            e_n = float(matrix[0, 2])
            e_pn = float(matrix[1, 2])

            # Cov(P,N) = E[PN] - E[P]E[N]
            cov_pn = e_pn - e_p * e_n

            # rho_PN = Cov(P,N) / sqrt(Var(P) * Var(N))
            var_p = float(matrix[1, 1] - e_p**2)
            var_n = float(matrix[2, 2] - e_n**2)
            denom = np.sqrt(max(var_p * var_n, 1e-12))
            rho_pn = float(cov_pn / denom) if denom > 0 else 0.0

            # Penalty: penalize positive co-occurrence (fragile regime)
            # Use rho_PN (normalized) scaled by co_occurrence_penalty
            # so the penalty is unit-independent and tunable.
            excess_rho = max(rho_pn, 0.0)
            penalty = self.co_occurrence_penalty * excess_rho

            # Log diagnostic info
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Co-occurrence diag (units={num_units}): "
                f"E[P]={e_p:.4f}, E[N]={e_n:.4f}, "
                f"E[PN]={e_pn:.4f}, Cov(P,N)={cov_pn:.4f}, "
                f"rho_PN={rho_pn:.3f}, penalty={penalty:.2f}"
            )

            return float(penalty)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Co-occurrence diagnostic failed: {e}")
            return 0.0

    def objective(
        self, results: DataFrame, trade_count: int, min_date: datetime, max_date: datetime, *args, **kwargs
    ) -> float:
        """
        Objective function to minimize.

        Uses karakana ranking_score as the primary optimization target.
        Optionally applies co-occurrence diagnostic penalty when
        ``co_occurrence_diag`` is enabled.

        Usage in Freqtrade hyperopt::

            freqtrade hyperopt --strategy KarakanaSGOShieldStrategy \
                --hyperopt-loss KarakanaStructuralHyperOptLoss \
                --params co_occurrence_diag=true \
                --params co_occurrence_penalty=1000

        For per-day co-occurrence analysis (requires trade dates),
        use the post-trade analysis script:
        examples/freqtrade/sgo_diagnostics.py

        The co-occurrence diagnostic requires date columns in the
        results DataFrame. If "close_date" or "open_date" columns
        are present, trades are grouped by day. Otherwise, the
        diagnostic returns 0.0.

        Parameters:
        - results: DataFrame with profit_ratio column from Freqtrade.
          Optionally contains open_date/close_date columns for
          per-day co-occurrence aggregation.
        - trade_count: number of trades in results
        - min_date, max_date: date range of results
        - config: Freqtrade configuration dict
        """
        if results.empty or trade_count == 0:
            return 20000.0  # High penalty for no trades

        # Extract profit ratios as magnitudes
        profits = results["profit_ratio"].values
        wins = profits[profits > 0]
        losses = profits[profits <= 0]

        if wins.size == 0 or losses.size == 0:
            return 10000.0

        try:
            # Use the profile defined on the class
            profile = getattr(self, "ranking_profile", "conservative")

            report = evaluate_structural_geometry(wins, losses, utility_power=1.1, ranking_profile=profile)

            loss = float(report["ranking_score"])

            # ── Co-occurrence diagnostic (reviewer framework) ─────
            # Extract trade dates if available for per-day aggregation.
            # Freqtrade results DataFrames typically have open_date
            # and close_date columns; if present, trades are grouped
            # by day before computing the SGO moment matrix.
            trade_dates = None
            if "close_date" in results.columns:
                trade_dates = np.asarray(results["close_date"].values)
            elif "open_date" in results.columns:
                trade_dates = np.asarray(results["open_date"].values)

            co_penalty = self._compute_co_occurrence_diag(
                profits,
                dates=trade_dates,  # type: ignore[arg-type]
            )
            loss += co_penalty

            # ── Sprint 17: Trajectory-Aware Audit ─────────────
            traj = getattr(self, "trajectory_audit_enabled", None)
            if traj is None:
                from karakana_engine.config import get_config as _cfg

                traj = _cfg.get("trajectory_audit_enabled", False)
            if traj:
                trajectory_penalty = self._compute_trajectory_penalty(profits)
                loss += trajectory_penalty

            return loss

        except Exception:
            return 15000.0

    def _compute_trajectory_penalty(self, profits: np.ndarray) -> float:
        """Run a rolling-window structural audit and return a penalty.

        Feeds chronological profit ratios through a sliding-window SGO
        matrix, tracks α velocity via AlphaMomentumTracker, and applies
        tiered penalties for degrading / collapsing / collapsed strategies.

        Returns a positive penalty (higher loss) for structurally unsound
        trajectories, or a negative bonus for strategies with long
        structural horizons.
        """
        n = len(profits)
        win_sz = self.trajectory_window_size
        step = self.trajectory_window_step

        if n < win_sz:
            return 0.0  # not enough data for trajectory audit

        # ── Rolling-window α ──────────────────────────────────────────
        alpha_windows = []
        for start in range(0, max(1, n - win_sz + 1), step):
            window = profits[start : start + win_sz]
            pos = window[window > 0]
            neg = window[window <= 0]
            if pos.size == 0 or neg.size == 0:
                alpha_windows.append(None)
                continue
            try:
                m = evaluate_structural_geometry(
                    pos,
                    neg,
                    utility_power=1.1,
                    ranking_profile=getattr(self, "ranking_profile", "conservative"),
                )
                err = float(m.get("alpha_s_sil", 0.0))
                health = float(np.clip(1.0 - err / 2.0, 0.0, 1.0))
                alpha_windows.append(health)
            except Exception:
                alpha_windows.append(None)

        valid_alphas = [a for a in alpha_windows if a is not None]
        if not valid_alphas:
            return 0.0

        # ── AlphaMomentumTracker ──────────────────────────────────────
        tracker = AlphaMomentumTracker(
            window_size=20,
            alpha_ema=self.trajectory_ema_weight,
        )
        for a in valid_alphas:
            tracker.push(a)

        stats = tracker.get_stats()
        ema_alpha = stats["alpha"]
        v_alpha = stats["v_alpha"]
        horizon = stats["horizon"]
        v_panic = self.trajectory_v_alpha_panic
        floor = self.trajectory_alpha_floor

        # ── Tiered Penalty ────────────────────────────────────────────
        penalty = 0.0

        if ema_alpha < floor or v_alpha < v_panic * 2:
            penalty += self.trajectory_penalty_collapsed
        elif v_alpha < v_panic:
            penalty += self.trajectory_penalty_collapsing
        elif v_alpha < 0:
            penalty += self.trajectory_penalty_degrading

        # Horizon bonus: reward strategies with long structural runway
        if horizon > self.trajectory_min_horizon_for_bonus:
            penalty += self.trajectory_bonus_horizon

        return penalty


class KarakanaBalanced(KarakanaStructuralHyperOptLoss):
    """karakana balanced profile: Equal focus on stability and profit."""

    ranking_profile = "balanced"

    @staticmethod
    def hyperopt_loss_function(*args, **kwargs) -> float:
        return KarakanaBalanced().objective(*args, **kwargs)


class KarakanaAggressive(KarakanaStructuralHyperOptLoss):
    """karakana aggressive profile: Prioritize profit while retaining structural guardrails."""

    ranking_profile = "aggressive"

    @staticmethod
    def hyperopt_loss_function(*args, **kwargs) -> float:
        return KarakanaAggressive().objective(*args, **kwargs)


class KarakanaConservative(KarakanaStructuralHyperOptLoss):
    """karakana conservative profile: Prioritize structural integrity and safety.

    Enables Sprint 17 trajectory-aware audit by default.  Each hyperopt
    epoch runs a rolling-window structural audit on the chronological
    trade history and applies collapse penalties.  Strategies that look
    profitable but have degrading α trajectories are penalized heavily,
    steering hyperopt toward structurally durable parameter regions.
    """

    ranking_profile = "conservative"
    trajectory_audit_enabled = True

    @staticmethod
    def hyperopt_loss_function(*args, **kwargs) -> float:
        return KarakanaConservative().objective(*args, **kwargs)
