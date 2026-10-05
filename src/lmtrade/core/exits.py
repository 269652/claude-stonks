"""Effective exit plan (take-profit / stop-loss / trailing) for an open
option or knockout position.

A position's plan comes from its own stops (set at open) and the global
OptionsConfig, unless the user saved a per-position override from the
dashboard (meta "exit_overrides", keyed by position id). Overrides are in
net EUR — what would actually be booked after the entry and exit fee — and a
None value disables that exit. Engine, API and dashboard all read the plan
through exit_plan() so they can never disagree."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import OptionsConfig


@dataclass
class ExitPlan:
    tp: float | None            # take-profit premium level (None = off)
    sl: float | None            # stop-loss premium level (None = off)
    trail_enabled: bool
    trail_min: float            # net EUR the trailing exit never goes below
    trail_pct: float            # drop from the peak profit that closes
    custom: bool                # True when a per-position override applies


def premium_for_pnl(entry: float, contracts: float, pnl: float, fee: float) -> float:
    """Premium at which closing books `pnl` net EUR (both fees included)."""
    return entry + (pnl + 2 * fee) / contracts


def pnl_for_premium(entry: float, contracts: float, premium: float, fee: float) -> float:
    return (premium - entry) * contracts - 2 * fee


def exit_plan(o: dict, cfg: OptionsConfig, override: dict | None, fee: float) -> ExitPlan:
    entry = float(o["entry_premium"])
    n = float(o["contracts"])
    if override:
        tp_pnl, sl_pnl = override.get("tp_pnl"), override.get("sl_pnl")
        return ExitPlan(
            tp=None if tp_pnl is None else premium_for_pnl(entry, n, tp_pnl, fee),
            sl=None if sl_pnl is None else max(0.0, premium_for_pnl(entry, n, sl_pnl, fee)),
            trail_enabled=bool(override.get("trail_enabled")),
            trail_min=float(override.get("trail_min_profit_eur") or cfg.trail_min_profit_eur),
            trail_pct=float(override.get("trail_pct")
                            if override.get("trail_pct") is not None else cfg.trail_pct),
            custom=True)
    tp = o.get("tp_premium") or entry * (1 + cfg.take_profit_pct)
    sl = o.get("sl_premium")
    if sl is None:
        sl = entry * (1 - cfg.stop_loss_pct)
    return ExitPlan(tp=tp, sl=sl, trail_enabled=cfg.trail_enabled,
                    trail_min=cfg.trail_min_profit_eur, trail_pct=cfg.trail_pct,
                    custom=False)


def _num(v) -> tuple[bool, float | None]:
    """(ok, value) for an optional finite number; bools are not numbers."""
    if v is None:
        return True, None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return False, None
    return True, float(v)


def validate_override(o: dict, payload: dict, fee: float) -> tuple[dict | None, str | None]:
    """Check a dashboard exit edit for position `o`. Returns (override, None)
    or (None, error message)."""
    ok_tp, tp = _num(payload.get("tp_pnl"))
    ok_sl, sl = _num(payload.get("sl_pnl"))
    if not (ok_tp and ok_sl):
        return None, "take-profit / stop-loss must be numbers (EUR) or empty"
    if tp is not None and sl is not None and sl >= tp:
        return None, "stop-loss must be below take-profit"
    entry, n = float(o["entry_premium"]), float(o["contracts"])
    if sl is not None and premium_for_pnl(entry, n, sl, fee) < 0:
        worst = pnl_for_premium(entry, n, 0.0, fee)
        return None, f"stop-loss can't be worse than losing the whole position ({worst:.2f} €)"
    trail_enabled = payload.get("trail_enabled", False)
    if not isinstance(trail_enabled, bool):
        return None, "trailing on/off must be true or false"
    ok_min, t_min = _num(payload.get("trail_min_profit_eur"))
    ok_pct, t_pct = _num(payload.get("trail_pct"))
    if trail_enabled:
        if not ok_min or t_min is None or t_min <= 0:
            return None, "trailing minimum profit must be a positive EUR amount"
        if not ok_pct or t_pct is None or not 0 <= t_pct < 1:
            return None, "trailing drop must be between 0 and 1 (e.g. 0.15 = 15%)"
    return {"tp_pnl": tp, "sl_pnl": sl, "trail_enabled": trail_enabled,
            "trail_min_profit_eur": t_min if ok_min else None,
            "trail_pct": t_pct if ok_pct else None}, None
