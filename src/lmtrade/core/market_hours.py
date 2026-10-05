"""Trading hours of the venue the live certificates trade on.

Mon-Fri inside `hours` ("HH:MM-HH:MM", Europe/Berlin). Holidays aren't
listed: a TR `exchangeClosed` rejection supplies them at runtime (the engine
passes it as `closed_until`). An empty or unparsable `hours` means always
open — never block trading on a config typo."""
from __future__ import annotations

from datetime import datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Berlin")


def _parse(hours: str) -> tuple[dtime, dtime] | None:
    try:
        a, b = (x.strip() for x in hours.split("-"))
        start, end = dtime.fromisoformat(a), dtime.fromisoformat(b)
    except (ValueError, AttributeError):
        return None
    return (start, end) if start < end else None


def is_open(ts: float, hours: str) -> bool:
    span = _parse(hours or "")
    if span is None:
        return True
    t = datetime.fromtimestamp(ts, TZ)
    return t.weekday() < 5 and span[0] <= t.time() < span[1]


def next_open(ts: float, hours: str) -> float | None:
    """The first opening at or after ts (None when always open)."""
    span = _parse(hours or "")
    if span is None:
        return None
    day = datetime.fromtimestamp(ts, TZ).date()
    for i in range(8):
        d = day + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        opening = datetime.combine(d, span[0], TZ).timestamp()
        if opening >= ts:
            return opening
    return None


def _closes(ts: float, hours: str) -> float | None:
    span = _parse(hours or "")
    if span is None:
        return None
    d = datetime.fromtimestamp(ts, TZ).date()
    return datetime.combine(d, span[1], TZ).timestamp()


def market_status(ts: float, hours: str, closed_until: float | None = None) -> dict:
    """{"open", "next_open", "closes"} at ts; closed_until (learned from an
    exchangeClosed rejection) overrides the schedule until it passes."""
    if closed_until and closed_until > ts:
        return {"open": False, "next_open": closed_until, "closes": None}
    if is_open(ts, hours):
        return {"open": True, "next_open": None, "closes": _closes(ts, hours)}
    return {"open": False, "next_open": next_open(ts, hours), "closes": None}
