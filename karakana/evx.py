"""Canonical Excess Volume Index math and leakage-safe dataset preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPSILON = 1e-12
REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class EVXDatasetReport:
    input_rows: int
    output_rows: int
    invalid_ohlc_rows: int
    non_finite_rows: int
    irregular_intervals: int


def _as_vector(name: str, values: np.ndarray | Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return vector


def solve_evx_regime(
    previous_close: np.ndarray | Sequence[float],
    current_open: np.ndarray | Sequence[float],
    current_close: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return EVX = (y - x) / x; previous_close is retained for API compatibility."""
    y_n = _as_vector("previous_close", previous_close)
    x = _as_vector("current_open", current_open)
    y = _as_vector("current_close", current_close)
    if not (len(y_n) == len(x) == len(y)):
        raise ValueError("EVX price arrays must have equal lengths")

    result = np.full(len(y), np.nan, dtype=np.float64)
    valid = np.isfinite(y_n) & np.isfinite(x) & np.isfinite(y) & (x > EPSILON)
    result[valid] = (y[valid] - x[valid]) / x[valid]
    return result


def infer_evx_price(
    previous_close: np.ndarray | Sequence[float] | float,
    current_open: np.ndarray | Sequence[float] | float,
    regime: np.ndarray | Sequence[float] | float,
) -> np.ndarray:
    """Reconstruct y = x(1 + EVX); previous_close is retained for compatibility."""
    y_n, x, e = np.broadcast_arrays(
        np.asarray(previous_close, dtype=np.float64),
        np.asarray(current_open, dtype=np.float64),
        np.asarray(regime, dtype=np.float64),
    )
    result = np.full(y_n.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(y_n) & np.isfinite(x) & np.isfinite(e) & (x > EPSILON)
    result[valid] = x[valid] * (1.0 + e[valid])
    return result


def calculate_evx(
    open_p: np.ndarray | Sequence[float],
    close_p: np.ndarray | Sequence[float],
    high_p: np.ndarray | Sequence[float],
    low_p: np.ndarray | Sequence[float],
    volume: np.ndarray | Sequence[float],
    n_shift: int = 1,
) -> dict[str, np.ndarray]:
    """Calculate B=yV/(x+y), A=xV/(x+y), and EVX=(B-A)/A."""
    if n_shift < 1:
        raise ValueError("n_shift must be at least 1")
    arrays = {
        "open": _as_vector("open", open_p),
        "close": _as_vector("close", close_p),
        "high": _as_vector("high", high_p),
        "low": _as_vector("low", low_p),
        "volume": _as_vector("volume", volume),
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("EVX OHLCV arrays must have equal lengths")
    if n_shift >= len(arrays["close"]):
        raise ValueError("n_shift must be smaller than the input length")

    del n_shift  # Retained only for compatibility with earlier callers.
    x = arrays["open"]
    y = arrays["close"]
    denominator = x + y
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(arrays["volume"])
        & (x > EPSILON)
        & (y >= 0.0)
        & (arrays["volume"] > EPSILON)
        & (np.abs(denominator) > EPSILON)
    )
    asks = np.full(len(x), np.nan, dtype=np.float64)
    bids = np.full(len(x), np.nan, dtype=np.float64)
    regime = np.full(len(x), np.nan, dtype=np.float64)
    asks[valid] = x[valid] * arrays["volume"][valid] / denominator[valid]
    bids[valid] = y[valid] * arrays["volume"][valid] / denominator[valid]
    regime[valid] = (bids[valid] - asks[valid]) / asks[valid]
    return {"evx": regime, "bids": bids, "asks": asks, "spread": bids - asks}


def calculate_evx_coefficient(returns: np.ndarray, risk_ratios: Sequence[float]) -> float:
    """Calculate whitepaper equation 19: sum(risk ratios) / return CV."""
    values = _as_vector("returns", returns)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    mean_return = float(np.mean(values))
    std_return = float(np.std(values))
    if std_return < EPSILON:
        return 0.0
    coefficient_of_variation = std_return / (abs(mean_return) + EPSILON)
    return float(np.sum(np.asarray(risk_ratios, dtype=np.float64)) / coefficient_of_variation)


def prepare_evx_dataset(
    frame: pd.DataFrame,
    shifts: Sequence[int] = (1, 5, 20),
    threshold: float = 0.0025,
) -> tuple[pd.DataFrame, EVXDatasetReport]:
    """Build labels and candle-open features without future-data leakage."""
    missing = sorted(set(REQUIRED_OHLCV_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")
    normalized_shifts = tuple(sorted(set(int(value) for value in shifts)))
    if not normalized_shifts or normalized_shifts[0] < 1:
        raise ValueError("shifts must contain positive integers")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    df = frame.copy()
    for column in REQUIRED_OHLCV_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    finite = np.isfinite(df[list(REQUIRED_OHLCV_COLUMNS)]).all(axis=1)
    invalid_ohlc = finite & (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["volume"] <= EPSILON)
    )

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], unit="ms", utc=True, errors="coerce")
        deltas = dates.diff()
        if (deltas.dropna() <= pd.Timedelta(0)).any():
            raise ValueError("EVX rows must be strictly ordered by ascending date")
        positive_deltas = deltas.dropna()
        expected_delta = positive_deltas.mode().iloc[0] if not positive_deltas.empty else None
        irregular_intervals = int((positive_deltas != expected_delta).sum()) if expected_delta is not None else 0
        regular = deltas.eq(expected_delta)
        regular.iloc[0] = False
        regular_history = regular.rolling(max(normalized_shifts) + 1).min().fillna(0).astype(bool)
    else:
        irregular_intervals = 0
        regular_history = pd.Series(True, index=df.index)

    args = [df[column].to_numpy() for column in ("open", "close", "high", "low", "volume")]
    df["evx"] = calculate_evx(*args, n_shift=normalized_shifts[0])["evx"]
    df["mean_ideal_e"] = df["evx"]
    df["label"] = np.select(
        [df["mean_ideal_e"] > threshold, df["mean_ideal_e"] < -threshold],
        [1, -1],
        default=0,
    )

    # Inputs at row t use only the current candle open and candles closed before t.
    previous_close = df["close"].shift(1)
    df["open_gap"] = (df["open"] - previous_close) / previous_close.replace(0.0, np.nan)

    feature_columns = evx_feature_columns(normalized_shifts)
    usable = finite & ~invalid_ohlc & regular_history & np.isfinite(df[feature_columns + ["mean_ideal_e"]]).all(axis=1)
    prepared = df.loc[usable].copy().reset_index(drop=True)
    report = EVXDatasetReport(
        input_rows=len(df),
        output_rows=len(prepared),
        invalid_ohlc_rows=int(invalid_ohlc.sum()),
        non_finite_rows=int((~finite).sum()),
        irregular_intervals=irregular_intervals,
    )
    return prepared, report


def evx_feature_columns(shifts: Sequence[int]) -> list[str]:
    del shifts
    return ["open_gap"]


def prepare_evx_open_gap(previous_close: float, current_open: float) -> pd.DataFrame:
    """Create the minimal scale-independent EVX policy input."""
    previous_close = float(previous_close)
    current_open = float(current_open)
    if not np.isfinite(previous_close) or not np.isfinite(current_open):
        raise ValueError("previous_close and current_open must be finite")
    if previous_close == 0.0:
        raise ValueError("previous_close must be non-zero")
    return pd.DataFrame([{"open_gap": (current_open - previous_close) / previous_close}])


def prepare_evx_inference_record(
    closed_candles: pd.DataFrame,
    current_open: float,
    shifts: Sequence[int] = (1, 5, 20),
) -> pd.DataFrame:
    """Create one candle-open feature row from already-closed candles."""
    normalized_shifts = tuple(sorted(set(int(value) for value in shifts)))
    if not normalized_shifts or normalized_shifts[0] < 1:
        raise ValueError("shifts must contain positive integers")
    minimum_rows = 1
    if len(closed_candles) < minimum_rows:
        raise ValueError(f"Inference requires at least {minimum_rows} closed candles")
    if "close" not in closed_candles.columns:
        raise ValueError("Inference history requires a close column")

    history = closed_candles.copy()
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    tail = history.iloc[-minimum_rows:]
    if not np.isfinite(tail["close"]).all():
        raise ValueError("Inference history contains a non-finite close")
    if "date" in tail.columns and len(tail) > 1:
        dates = pd.to_datetime(tail["date"], unit="ms", utc=True, errors="coerce")
        deltas = dates.diff().dropna()
        if deltas.empty or (deltas <= pd.Timedelta(0)).any() or deltas.nunique() != 1:
            raise ValueError("Inference history must contain contiguous ascending candles")

    return prepare_evx_open_gap(history.iloc[-1]["close"], current_open)
