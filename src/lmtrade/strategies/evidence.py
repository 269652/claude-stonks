"""Evidence-based strategy rules on DAILY closes.

These are long-documented approaches, implemented on the horizon their
published evidence uses (daily bars, months of history) — not on minutes of
intraday data. Their historical results come from diversified portfolios with
low costs over decades; nothing here promises that they make money for this
bot (leveraged knock-outs, flat 1 EUR fees, one or two positions).

- tsmom_12m:     time-series momentum, sign of the 12-month return
                 (Moskowitz, Ooi & Pedersen 2012)
- xsmom_12_1:    cross-sectional momentum, 12-month return skipping the last
                 month, top / bottom third of the universe (Jegadeesh & Titman 1993)
- donchian:      55-day channel breakout (the "turtle" rules)
- rsi2_pullback: 2-day RSI extremes in the direction of the 200-day trend (Connors)

Each returns (direction, confidence) with direction in buy | sell | hold;
too little history means hold (abstain)."""
from __future__ import annotations

import math
import statistics

YEAR = 252
MONTH = 21


def tsmom_12m(closes: list[float]) -> tuple[str, float]:
    if len(closes) < YEAR + 1 or closes[-YEAR - 1] <= 0:
        return "hold", 0.0
    r = closes[-1] / closes[-YEAR - 1] - 1
    rets = [b / a - 1 for a, b in zip(closes[-YEAR - 1:-1], closes[-YEAR:]) if a > 0]
    vol = statistics.pstdev(rets) * math.sqrt(YEAR) if len(rets) > 1 else 0.0
    if r == 0:
        return "hold", 0.0
    conf = 1.0 if vol <= 0 else min(1.0, abs(r) / vol)    # return per unit of risk
    return ("buy" if r > 0 else "sell"), conf


def _mom_12_1(c: list[float]) -> float | None:
    if len(c) < YEAR + 1 or c[-YEAR - 1] <= 0:
        return None
    return c[-MONTH - 1] / c[-YEAR - 1] - 1


def xsmom_12_1(symbol: str, universe: dict[str, list[float]]) -> tuple[str, float]:
    scores = {s: m for s, c in universe.items() if (m := _mom_12_1(c)) is not None}
    if symbol not in scores or len(scores) < 6:
        return "hold", 0.0
    ranked = sorted(scores, key=scores.get)
    p = ranked.index(symbol) / (len(ranked) - 1)          # 0 = worst, 1 = best
    if p >= 2 / 3:
        return "buy", abs(p - 0.5) * 2
    if p <= 1 / 3:
        return "sell", abs(p - 0.5) * 2
    return "hold", 0.0


def donchian(closes: list[float], n: int = 55) -> tuple[str, float]:
    if len(closes) < n + 1:
        return "hold", 0.0
    prev = closes[-n - 1:-1]
    last = closes[-1]
    if last > max(prev):
        return "buy", 0.7
    if last < min(prev):
        return "sell", 0.7
    return "hold", 0.0


def _rsi(closes: list[float], n: int) -> float | None:
    if len(closes) < n + 1:
        return None
    diffs = [b - a for a, b in zip(closes[-n - 1:-1], closes[-n:])]
    gain = sum(d for d in diffs if d > 0) / n
    loss = -sum(d for d in diffs if d < 0) / n
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return 100 - 100 / (1 + gain / loss)


def rsi2_pullback(closes: list[float]) -> tuple[str, float]:
    if len(closes) < 201:
        return "hold", 0.0
    sma200 = statistics.fmean(closes[-200:])
    r = _rsi(closes, 2)
    last = closes[-1]
    if r is None:
        return "hold", 0.0
    if last > sma200 and r <= 10:
        return "buy", 0.5 + (10 - r) / 20
    if last < sma200 and r >= 90:
        return "sell", 0.5 + (r - 90) / 20
    return "hold", 0.0


def evidence_vote(symbol: str, universe: dict[str, list[float]]) -> tuple[str, float, str]:
    """Combine the four rules into one vote: mean signed confidence of the
    rules that take a side (holds abstain), with a per-rule rationale."""
    closes = universe.get(symbol)
    if not closes:
        return "hold", 0.0, "no daily data"
    votes = [("tsmom", tsmom_12m(closes)), ("xsmom", xsmom_12_1(symbol, universe)),
             ("donchian", donchian(closes)), ("rsi2", rsi2_pullback(closes))]
    sided = [(d, c) for _, (d, c) in votes if d != "hold"]
    why = ", ".join(f"{n} {d}" + (f" {c:.2f}" if d != "hold" else "") for n, (d, c) in votes)
    if not sided:
        return "hold", 0.0, why
    net = sum(c if d == "buy" else -c for d, c in sided) / len(sided)
    if abs(net) <= 0.1:
        return "hold", 0.0, why
    return ("buy" if net > 0 else "sell"), min(1.0, abs(net)), why
