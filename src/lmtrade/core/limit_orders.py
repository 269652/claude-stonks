"""Exchange-side limit entries for watched signals (helpers).

Instead of waiting on this machine for the price to reach the entry trigger
and then buying at market, a watched signal gets a resting limit order at
the price its instrument would have once the underlying reaches the trigger:
the intraday SMA minus k sigma for buys (call / long knockout) or plus k
sigma for sells (put / short knockout). Both are BUYs of a certificate, so
the order always fills at or below its limit. TP / SL exits stay market
orders."""
from __future__ import annotations

import statistics

from .watchlist import MIN_BARS


def target_spot(history: list[float], window: int, k: float, direction: str) -> float | None:
    """Underlying price at which a watched signal should be entered."""
    if direction not in ("buy", "sell"):
        return None
    bars = history[-window:] if window > 0 else history
    if len(bars) < MIN_BARS:
        return None
    std = statistics.pstdev(bars)
    if std <= 0:
        return None
    mean = statistics.fmean(bars)
    return mean - k * std if direction == "buy" else mean + k * std


def needs_reprice(old_limit: float, new_limit: float, drift: float) -> bool:
    """True when a resting order's limit is more than `drift` (fraction) off
    the freshly computed one."""
    if old_limit <= 0:
        return True
    return abs(new_limit - old_limit) / old_limit > drift
