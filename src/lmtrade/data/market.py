"""Market data providers.

`auto` (and the legacy `yfinance` provider name) fetch real closes directly
from Yahoo Finance's public chart JSON endpoint via httpx. The `yfinance`
PyPI package is deliberately NOT used: its `curl_cffi` backend impersonates a
browser's TLS fingerprint, which this project's sandboxed proxy environments
cannot pass through (the handshake is reset) — plain httpx against the same
Yahoo endpoint works fine. Falls back to a deterministic synthetic
random-walk generator on any fetch failure, so the bot always runs — offline,
in CI, or on a fresh machine before network/keys are set.
"""
from __future__ import annotations

import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

# (symbol, range_, interval) -> list of close prices, oldest -> newest
Fetcher = Callable[[str, str, str], list[float]]


@dataclass
class Quote:
    symbol: str
    price: float
    history: list[float]      # recent close prices, oldest -> newest
    source: str


def default_yahoo_fetcher(symbol: str, range_: str, interval: str) -> list[float]:
    """Fetch closes from Yahoo Finance's public chart API via plain httpx.

    Regular exchange sessions are closed most of the day (and EU sessions
    close even earlier than the US), so 1-minute intraday requests ask for
    extended-hours (pre/post market) bars too — otherwise the feed goes
    stale outside 9:30-16:00 ET and the bot would be trading on frozen
    prices without knowing it. Daily bars are unaffected either way.
    """
    import httpx

    params = {"range": range_, "interval": interval}
    if interval == "1m":
        params["includePrePost"] = "true"

    r = httpx.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15.0,
    )
    r.raise_for_status()
    payload = r.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise ValueError(f"no chart data for {symbol}")
    closes = result[0]["indicators"]["quote"][0]["close"]
    return [float(c) for c in closes if c is not None]


class MarketData:
    def __init__(
        self, provider: str = "auto", lookback: int = 60, intraday: bool = False,
        fetcher: Fetcher | None = None,
    ):
        self.lookback = lookback
        self.intraday = intraday
        self.provider = self._resolve(provider)
        self.fetcher = fetcher or default_yahoo_fetcher
        self._daily_cache: dict[str, tuple[float, list[float]]] = {}

    def _resolve(self, provider: str) -> str:
        if provider == "synthetic":
            return "synthetic"
        return "yahoo"   # "auto" and the legacy "yfinance" name both mean live data

    def quote(self, symbol: str) -> Quote:
        if self.provider == "yahoo":
            try:
                return self._yahoo_quote(symbol)
            except Exception:
                pass  # fall through to synthetic on any data hiccup
        return self._synthetic_quote(symbol)

    def quotes_concurrent(self, symbols: list[str], max_workers: int = 6) -> dict[str, Quote]:
        """Fetch multiple symbols in parallel, bounded by max_workers so a
        larger universe doesn't burst-hammer Yahoo (the direct cause of the
        rate-limiting that triggered the synthetic-fallback data-integrity
        bug — see test_data_source_safety.py). A failing symbol falls back to
        synthetic individually without affecting the others, same as quote()."""
        if not symbols:
            return {}
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            results = list(pool.map(self.quote, symbols))
        return dict(zip(symbols, results))

    DAILY_TTL_S = 6 * 3600

    def daily_closes(self, symbol: str) -> list[float] | None:
        """~2 years of daily closes for the daily-horizon strategies, cached
        for DAILY_TTL_S. Live mode returns None on failure — never synthetic
        stand-ins for a real symbol."""
        hit = self._daily_cache.get(symbol)
        if hit and time.time() - hit[0] < self.DAILY_TTL_S:
            return hit[1]
        if self.provider == "synthetic":
            closes = self._synthetic_daily(symbol)
        else:
            try:
                # 2y: one Yahoo year is ~251 trading days — fewer than the 253
                # closes the 12-month momentum rules need.
                closes = self.fetcher(symbol, "2y", "1d")
            except Exception:  # noqa: BLE001
                return None
            if not closes:
                return None
        self._daily_cache[symbol] = (time.time(), closes)
        return closes

    def _synthetic_daily(self, symbol: str, n: int = 260) -> list[float]:
        seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % 10_000
        price = 50.0 + seed % 200
        day = int(time.time() // 86400)
        out = []
        for i in range(n):
            h = int(hashlib.sha256(f"{symbol}:d{day - n + i}".encode()).hexdigest(), 16)
            price = max(1.0, price * (1 + 0.015 * ((h % 2000) / 1000.0 - 1.0)))
            out.append(round(price, 4))
        return out

    # -- live data (Yahoo, via httpx) ------------------------------------------
    def _yahoo_quote(self, symbol: str) -> Quote:
        if self.intraday:
            closes = self.fetcher(symbol, "1d", "1m")
            if len(closes) < 2:   # market closed / no intraday bars -> daily fallback
                closes = self.fetcher(symbol, "3mo", "1d")
        else:
            closes = self.fetcher(symbol, "3mo", "1d")
        closes = closes[-self.lookback:]
        if not closes:
            raise ValueError("no data")
        return Quote(symbol, closes[-1], closes, "yahoo")

    # -- synthetic ------------------------------------------------------------
    def _synthetic_quote(self, symbol: str) -> Quote:
        """Deterministic-ish random walk seeded by symbol + coarse time, so the
        series evolves between loops but is reproducible within a bucket
        (10s buckets intraday, 60s otherwise)."""
        seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % 10_000
        base = 50 + (seed % 200)
        bucket = 10 if self.intraday else 60
        t0 = int(time.time() // bucket)
        closes: list[float] = []
        price = float(base)
        for i in range(self.lookback):
            # pseudo-random step from a hash of (symbol, tick)
            h = int(hashlib.sha256(f"{symbol}:{t0 - self.lookback + i}".encode()).hexdigest(), 16)
            step = ((h % 2000) / 1000.0 - 1.0)          # -1..1
            drift = 0.02 * math.sin((t0 + i) / 7.0)
            price = max(1.0, price * (1 + 0.01 * step + drift * 0.01))
            closes.append(round(price, 4))
        return Quote(symbol, closes[-1], closes, "synthetic")
