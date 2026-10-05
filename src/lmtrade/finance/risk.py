"""Position sizing and risk guardrails. Deliberately conservative — the whole
account is ~10 EUR, so a single fat-fingered size can wipe it out."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import RiskConfig


@dataclass
class SizingResult:
    qty: float
    notional: float
    reason: str


def min_notional_for_fee(fee: float, max_fee_pct: float) -> float:
    """Smallest order value for which `fee` stays within `max_fee_pct`.
    0.0 when the guard is disabled (max_fee_pct <= 0)."""
    if max_fee_pct <= 0:
        return 0.0
    return max(0.0, fee) / max_fee_pct


def fee_ratio_ok(notional: float, fee: float, max_fee_pct: float) -> bool:
    """True when an order of `notional` with flat `fee` is acceptable."""
    if notional <= 0:
        return False
    if max_fee_pct <= 0 or fee <= 0:
        return True
    return fee / notional <= max_fee_pct + 1e-12


def fee_aware_budget(sized: float, ceiling: float, fee: float, max_fee_pct: float) -> float:
    """Order budget with the fee floor baked in. `sized` is the signal-scaled
    budget, `ceiling` the hard cap (position cap, deployable capital, cash).
    Below the fee-implied minimum the order is raised to that minimum when the
    ceiling allows it; otherwise the trade is infeasible (0.0)."""
    sized = max(0.0, min(sized, ceiling))
    floor = min_notional_for_fee(fee, max_fee_pct)
    if sized >= floor:
        return sized
    return floor if ceiling >= floor else 0.0


def trailing_exit(peak: float, pnl: float, min_profit: float,
                  pct: float) -> tuple[bool, float | None, bool]:
    """Trailing take-profit with a guaranteed minimum, on net EUR profit.

    Activates once the peak profit reaches min_profit / (1 - pct); from then
    on the stop sits pct below the peak, which by construction is never below
    min_profit. Returns (active, stop_level, hit)."""
    if min_profit <= 0 or not 0 <= pct < 1:
        return False, None, False
    if peak < min_profit / (1 - pct) - 1e-9:
        return False, None, False
    stop = max(min_profit, peak * (1 - pct))
    return True, stop, pnl <= stop + 1e-9


def size_position(
    *,
    price: float,
    cash: float,
    equity: float,
    confidence: float,
    cfg: RiskConfig,
    fee: float = 0.0,
) -> SizingResult:
    """Confidence-scaled fractional sizing, capped by max_position_fraction and
    available cash. Fractional share quantities are allowed (TR supports them)."""
    if price <= 0:
        return SizingResult(0.0, 0.0, "invalid price")
    if confidence < cfg.min_confidence:
        return SizingResult(0.0, 0.0, f"confidence {confidence:.2f} < {cfg.min_confidence}")

    # Fraction of equity to allocate, scaled by how far above the threshold we are.
    span = max(1e-6, 1.0 - cfg.min_confidence)
    scale = (confidence - cfg.min_confidence) / span      # 0..1
    fraction = cfg.max_position_fraction * scale
    notional = min(equity * fraction, cash)
    if notional < 1e-3:
        return SizingResult(0.0, 0.0, "notional below minimum")
    if not fee_ratio_ok(notional, fee, cfg.max_fee_pct):
        return SizingResult(
            0.0, 0.0,
            f"fee {fee:.2f} > {cfg.max_fee_pct:.1%} of notional {notional:.2f}")
    qty = notional / price
    return SizingResult(qty, notional, f"alloc {fraction:.0%} of equity @ conf {confidence:.2f}")


def should_exit(
    *, avg_price: float, last_price: float, cfg: RiskConfig
) -> tuple[bool, str]:
    """Hard stop-loss / take-profit check for an open long position."""
    if avg_price <= 0:
        return False, ""
    change = (last_price - avg_price) / avg_price
    if change <= -cfg.stop_loss_pct:
        return True, f"stop-loss hit ({change:.1%})"
    if change >= cfg.take_profit_pct:
        return True, f"take-profit hit ({change:.1%})"
    return False, ""
