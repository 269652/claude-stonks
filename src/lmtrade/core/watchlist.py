"""Entry timing for the watch list.

A trade signal is not entered at market. It waits on the watch list until the
price is stretched against it by at least k standard deviations from the
intraday SMA — below the average for buys (buy the dip), above it for sells
(sell the spike). Volatility-scaled on purpose: a 1-sigma move is ~0.5% for
SPY but several percent for crypto, so a fixed percentage would mean very
different things per asset."""
from __future__ import annotations

import statistics

MIN_BARS = 10   # fewer bars than this: no meaningful average, so no timing


def entry_z(price: float, history: list[float], window: int) -> float | None:
    """Standard score of `price` against the last `window` bars of history.
    None when there is too little data or the window is flat (std == 0)."""
    bars = history[-window:] if window > 0 else history
    if len(bars) < MIN_BARS:
        return None
    mean = statistics.fmean(bars)
    std = statistics.pstdev(bars)
    if std <= 0:
        return None
    return (price - mean) / std


def should_enter(direction: str, z: float | None, k: float) -> bool:
    """True when the stretch is in the signal's favour by at least k sigma."""
    if z is None:
        return False
    if direction == "buy":
        return z <= -k
    if direction == "sell":
        return z >= k
    return False
