"""Model-agnostic streaming structural alpha monitoring.

This module lets any language model — regardless of framework — be audited
for structural collapse in real time.  The caller feeds per-token outcome
scores (e.g. log-probability deviations) and receives chunk-level alpha,
V_α drift, and collapse warnings without needing a HuggingFace model or
tokenizer.

Typical usage::

    from karakana_engine.streaming import StreamingAlphaMonitor

    monitor = StreamingAlphaMonitor(
        chunk_size=50,
        collapse_threshold=-0.02,
        loss_module="probability",
        ranking_profile="conservative",
    )

    for token_log_prob in model_stream:
        # Now supporting content-aware metrics: token_id and entropy
        monitor.push(token_log_prob, token_id=tid, entropy=ent)
        if monitor.should_stop():
            print("GRG E-Stop triggered — structural collapse detected")
            break

    report = monitor.report()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from karakana_engine.pytorch import SGOLossModule


@dataclass
class StreamingAlphaReport:
    """Complete structural audit for a streaming generation."""

    chunk_alphas: list[float] = field(default_factory=list)
    chunk_momentums: list[float] = field(default_factory=list)
    overall_alpha: float = 1.0
    v_alpha: float = 0.0
    m_alpha: float = 0.0
    min_chunk_alpha: float = 1.0
    max_alpha_drop: float = 0.0
    collapsed: bool = False
    collapse_chunk: int = -1
    num_chunks: int = 0
    num_tokens: int = 0
    horizon: float = 999.0
    horizon_chunks: float = 999.0
    horizon_tokens: float = 999.0
    e_stop_triggered: bool = False


class StreamingAlphaMonitor:
    """Real-time structural integrity monitor for streaming LLM output.

    Parameters
    ----------
    chunk_size: int
        Number of tokens per structural chunk (default 50).
    collapse_threshold: float
        V_α below which a chunk transition is flagged as collapse (default -0.02).
    loss_module: str
        SGO loss module — ``"moment"``, ``"probability_kl"``, or ``"probability_fro"``.
    ranking_profile: str
        SGO ranking profile — ``"balanced"``, ``"conservative"``, or ``"aggressive"``.
    safety_floor: float
        Alpha floor below which e-stop is triggered (default 0.4).
    repetition_decay: float
        Weight for unique-token ratio penalty (default 0.1).
    entropy_gate_threshold: float
        Entropy floor below which repetition is penalized more severely (default 0.1).
    """

    def __init__(
        self,
        chunk_size: int = 50,
        collapse_threshold: float = -0.02,
        loss_module: str = "probability_kl",
        ranking_profile: str = "conservative",
        safety_floor: float = 0.4,
        repetition_decay: float = 0.1,
        entropy_gate_threshold: float = 0.1,
    ):
        self.chunk_size = chunk_size
        self.collapse_threshold = collapse_threshold
        self.loss_module = loss_module
        self.ranking_profile = ranking_profile
        self.safety_floor = safety_floor
        self.repetition_decay = repetition_decay
        self.entropy_gate_threshold = entropy_gate_threshold

        self._buffer: list[float] = []
        self._token_id_buffer: list[int] = []
        self._entropy_buffer: list[float] = []

        self._chunk_alphas: list[float] = []
        self._chunk_momentums: list[float] = []
        self._prev_alpha: float | None = None
        self._total_tokens: int = 0
        self._collapsed: bool = False
        self._collapse_chunk: int = -1
        self._max_alpha_drop: float = 0.0
        self._e_stop: bool = False

        # EMA-based momentum tracking
        self._ema_alpha: float = 0.0
        self._ema_velocity: float = 0.0
        self._ema_momentum: float = 0.0
        self._ema_weight: float = 0.3
        self._ema_initialized: bool = False

        self._sgo = SGOLossModule(
            lambda_sgo=1.0,  # Unit weight for raw alpha
            metric_key="alpha_s_sil" if loss_module == "moment" else "normalized_structure_term",
            loss_module=loss_module,
            ranking_profile=ranking_profile,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, token_score: float, token_id: int | None = None, entropy: float | None = None) -> None:
        """Feed a single per-token outcome score.

        Optional parameters token_id and entropy enable content-aware penalties
        for repetition and low-entropy loops.
        """
        self._buffer.append(float(token_score))
        if token_id is not None:
            self._token_id_buffer.append(int(token_id))
        if entropy is not None:
            self._entropy_buffer.append(float(entropy))

        self._total_tokens += 1

        if len(self._buffer) >= self.chunk_size:
            self._flush_chunk()

    def should_stop(self) -> bool:
        """Return True if the GRG e-stop condition is met."""
        return self._e_stop or self._collapsed

    @property
    def num_tokens(self) -> int:
        """Total number of pushed observations."""
        return self._total_tokens

    @property
    def pending_tokens(self) -> int:
        """Observations collected toward the next chunk."""
        return len(self._buffer)

    @property
    def num_chunks(self) -> int:
        """Number of completed structural chunks."""
        return len(self._chunk_alphas)

    @property
    def has_completed_chunk(self) -> bool:
        """Return True after at least one structural chunk has been evaluated."""
        return bool(self._chunk_alphas)

    @property
    def in_warmup(self) -> bool:
        """Return True before the first structural chunk has been evaluated."""
        return not self.has_completed_chunk

    @property
    def current_alpha(self) -> float:
        """The most recent chunk alpha (or 1.0 before the first chunk)."""
        if self._chunk_alphas:
            return self._chunk_alphas[-1]
        return 1.0

    @property
    def current_v_alpha(self) -> float:
        """EMA-smoothed velocity (V_α)."""
        return self._ema_velocity

    @property
    def chunk_ready(self) -> bool:
        """Return True if the internal buffer was just flushed (completed a chunk)."""
        return len(self._buffer) == 0 and self._total_tokens > 0

    @property
    def current_horizon(self) -> float:
        """Predicted remaining chunks before hitting the safety floor."""
        dist = self._ema_alpha - self.safety_floor
        if dist <= 0:
            return 0.0
        if abs(self._ema_velocity) < 1e-8 or self._ema_velocity >= 0:
            return 999.0
        return float(dist / abs(self._ema_velocity))

    @property
    def current_horizon_tokens(self) -> float:
        """Predicted remaining tokens before hitting the safety floor."""
        horizon_chunks = self.current_horizon
        if horizon_chunks >= 999.0:
            return 999.0
        return float(horizon_chunks * self.chunk_size)

    def flush(self) -> None:
        """Force-flush any remaining tokens in the buffer to a final chunk."""
        if self._buffer:
            self._flush_chunk()

    def report(self) -> StreamingAlphaReport:
        """Produce a final structural audit report."""
        self.flush()
        overall = sum(self._chunk_alphas) / len(self._chunk_alphas) if self._chunk_alphas else 1.0
        horizon_chunks = self.current_horizon
        horizon_tokens = self.current_horizon_tokens

        return StreamingAlphaReport(
            chunk_alphas=list(self._chunk_alphas),
            chunk_momentums=list(self._chunk_momentums),
            overall_alpha=overall,
            v_alpha=self._ema_velocity,
            m_alpha=self._ema_momentum,
            min_chunk_alpha=min(self._chunk_alphas) if self._chunk_alphas else 1.0,
            max_alpha_drop=self._max_alpha_drop,
            collapsed=self._collapsed,
            collapse_chunk=self._collapse_chunk,
            num_chunks=len(self._chunk_alphas),
            num_tokens=self._total_tokens,
            horizon=horizon_tokens,
            horizon_chunks=horizon_chunks,
            horizon_tokens=horizon_tokens,
            e_stop_triggered=self._e_stop,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_chunk(self) -> None:
        scores = torch.tensor(self._buffer, dtype=torch.float32)

        # 1. Content-Aware Penalties
        content_penalty = 0.0

        # Repetition Penalty: Unique tokens / total tokens
        if self._token_id_buffer:
            unique_ratio = len(set(self._token_id_buffer)) / len(self._token_id_buffer)
            if unique_ratio < 0.5:  # Penalty kicks in below 50% diversity
                content_penalty += self.repetition_decay * (0.5 - unique_ratio) * 2.0

        # Entropy Gate: Low entropy (certainty) loops are penalized more
        if self._entropy_buffer:
            avg_entropy = sum(self._entropy_buffer) / len(self._entropy_buffer)
            if avg_entropy < self.entropy_gate_threshold:
                # Severity increases as entropy approaches zero
                multiplier = 1.0 + (self.entropy_gate_threshold - avg_entropy) / self.entropy_gate_threshold
                content_penalty *= multiplier

        self._buffer.clear()
        self._token_id_buffer.clear()
        self._entropy_buffer.clear()

        if scores.numel() < 2:
            return

        # 2. Compute SGO loss and alpha
        # Check for single-sign outcome streaks to avoid cofactor math degeneracies (only relevant for moment-based SGO)
        if self.loss_module == "moment":
            num_pos = int((scores > 0).sum().item())
            num_neg = int((scores <= 0).sum().item())

            if num_pos == scores.numel():
                # 100% win streak is structurally perfect
                alpha = max(0.0, 1.0 - content_penalty)
            elif num_neg == scores.numel():
                # 100% loss streak is completely collapsed
                alpha = 0.0
            else:
                with torch.no_grad():
                    sgo_loss = float(self._sgo(scores).item())

                # Apply content penalty to SGO loss (capped at 1.0)
                total_loss = min(1.0, sgo_loss + content_penalty)
                alpha = max(0.0, 1.0 - total_loss)
        else:
            with torch.no_grad():
                sgo_loss = float(self._sgo(scores).item())

            # Apply content penalty to SGO loss (capped at 1.0)
            total_loss = min(1.0, sgo_loss + content_penalty)
            alpha = max(0.0, 1.0 - total_loss)

        self._chunk_alphas.append(alpha)
        self._update_ema(alpha)
        self._chunk_momentums.append(self._ema_momentum)
        self._check_collapse(alpha)

        # E-stop check
        if self._ema_alpha < self.safety_floor:
            self._e_stop = True

    def _update_ema(self, alpha: float) -> None:
        if not self._ema_initialized:
            self._ema_alpha = alpha
            self._ema_velocity = 0.0
            self._ema_momentum = 0.0
            self._ema_initialized = True
            return

        prev_alpha = self._ema_alpha
        prev_velocity = self._ema_velocity

        w = self._ema_weight
        self._ema_alpha = w * alpha + (1 - w) * prev_alpha
        current_velocity = self._ema_alpha - prev_alpha
        self._ema_velocity = w * current_velocity + (1 - w) * prev_velocity
        current_momentum = self._ema_velocity - prev_velocity
        self._ema_momentum = w * current_momentum + (1 - w) * self._ema_momentum

    def _check_collapse(self, alpha: float) -> None:
        if self._prev_alpha is not None:
            drop = alpha - self._prev_alpha
            if drop < self._max_alpha_drop:
                self._max_alpha_drop = drop
            if self._ema_velocity < self.collapse_threshold and not self._collapsed:
                self._collapsed = True
                self._collapse_chunk = len(self._chunk_alphas) - 1
        self._prev_alpha = alpha


def token_scores_from_logprobs(
    target_log_probs: list[float],
    mean_log_probs: list[float],
    scale: float = 2.0,
) -> list[float]:
    """Convert token-level log-probabilities to SGO outcome scores."""
    if len(target_log_probs) != len(mean_log_probs):
        raise ValueError(f"Length mismatch: {len(target_log_probs)} vs {len(mean_log_probs)}")
    return [(t - m) * scale for t, m in zip(target_log_probs, mean_log_probs)]


def compute_streaming_alpha_static(
    scores: list[float],
    chunk_size: int = 50,
    collapse_threshold: float = -0.02,
    loss_module: str = "probability_kl",
    ranking_profile: str = "conservative",
    safety_floor: float = 0.4,
    token_ids: list[int] | None = None,
    entropies: list[float] | None = None,
) -> StreamingAlphaReport:
    """One-shot convenience: run streaming alpha on a pre-computed score list."""
    monitor = StreamingAlphaMonitor(
        chunk_size=chunk_size,
        collapse_threshold=collapse_threshold,
        loss_module=loss_module,
        ranking_profile=ranking_profile,
        safety_floor=safety_floor,
    )
    for i, score in enumerate(scores):
        tid = token_ids[i] if token_ids is not None else None
        ent = entropies[i] if entropies is not None else None
        monitor.push(score, token_id=tid, entropy=ent)
    return monitor.report()
