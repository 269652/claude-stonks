"""Classic technical indicators computed on a price series (numpy).

These are the "financial models" half of the hybrid: cheap, deterministic
signals the LLM/SLM layer is fused with.
"""
from __future__ import annotations

import numpy as np


def _as_array(prices: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    return arr[~np.isnan(arr)]


def sma(prices: list[float] | np.ndarray, window: int) -> float | None:
    arr = _as_array(prices)
    if len(arr) < window:
        return None
    return float(arr[-window:].mean())


def ema(prices: list[float] | np.ndarray, window: int) -> float | None:
    arr = _as_array(prices)
    if len(arr) < window:
        return None
    alpha = 2.0 / (window + 1)
    e = arr[0]
    for p in arr[1:]:
        e = alpha * p + (1 - alpha) * e
    return float(e)


def rsi(prices: list[float] | np.ndarray, window: int = 14) -> float | None:
    arr = _as_array(prices)
    if len(arr) < window + 1:
        return None
    deltas = np.diff(arr[-(window + 1):])
    gains = deltas[deltas > 0].sum() / window
    losses = -deltas[deltas < 0].sum() / window
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def macd(
    prices: list[float] | np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float] | None:
    arr = _as_array(prices)
    if len(arr) < slow + signal:
        return None
    ema_fast = ema(arr, fast)
    ema_slow = ema(arr, slow)
    if ema_fast is None or ema_slow is None:
        return None
    macd_line = ema_fast - ema_slow
    # approximate signal line from the tail
    tail = arr[-(signal + 1):]
    signal_line = ema(tail, signal) or macd_line
    hist = macd_line - (signal_line - ema_slow)
    return float(macd_line), float(signal_line), float(hist)


def trend(prices: list[float] | np.ndarray, window: int = 50) -> str | None:
    """Medium-term trend: 'up' when the last price is above its SMA(window),
    'down' when below, None on insufficient history. The classic SMA timing
    filter (Faber 2007) / time-series-momentum sign (Moskowitz-Ooi-Pedersen
    2012): the sign of medium-term drift is one of the most replicated
    return predictors, used here to confirm — never to generate — entries."""
    arr = _as_array(prices)
    mid = sma(arr, window)
    if mid is None or not len(arr):
        return None
    last = float(arr[-1])
    if last == mid:
        return None   # dead flat: no information, treat as unknown
    return "up" if last > mid else "down"


def volatility(prices: list[float] | np.ndarray, window: int = 20) -> float | None:
    arr = _as_array(prices)
    if len(arr) < window + 1:
        return None
    rets = np.diff(arr[-(window + 1):]) / arr[-(window + 1):-1]
    return float(np.std(rets))


def indicator_snapshot(prices: list[float] | np.ndarray) -> dict[str, float | None]:
    """Compute the full set used by the technical agent."""
    return {
        "last": float(_as_array(prices)[-1]) if len(_as_array(prices)) else None,
        "sma_fast": sma(prices, 10),
        "sma_slow": sma(prices, 30),
        "rsi": rsi(prices, 14),
        "vol": volatility(prices, 20),
        "trend": trend(prices, 50),
    }


def daily_atr(closes: list[float], window: int = 14) -> float | None:
    """Average absolute close-to-close move over the last `window` days (an
    ATR from closes only). None with too little data or no movement."""
    if window <= 0 or len(closes) < window + 1:
        return None
    tail = [float(x) for x in closes[-(window + 1):]]
    atr = sum(abs(b - a) for a, b in zip(tail, tail[1:])) / window
    return atr if atr > 0 else None
