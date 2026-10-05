"""Regression tests for a live-outage/synthetic-fallback data-integrity bug
found during a multi-hour real-data dry run.

What happened: MarketData.quote() silently falls back to synthetic prices
(data/market.py) whenever the live Yahoo fetch fails for any reason (network
hiccup, rate limiting after sustained hourly volume, etc). Over a long
unattended run, this flip-flopped repeatedly for the same symbol — e.g. SPY
alternated between real strikes (~744) and synthetic ones (~196-208). When an
option opened against a REAL strike was later marked against a SYNTHETIC spot
price (or vice versa), the Black-Scholes marker saw a nonsensical spot-vs-
strike gap and computed absurd "take-profits" (observed: +11,082%, +12,351%).
Reinvesting that fake gain compounded a 100 EUR paper account to over 66,000
EUR in about 6.5 hours.

The fix: the engine must never open, close, or mark a position using a quote
whose data source doesn't match what the configured provider promises (e.g. a
"synthetic" quote arriving while `data.provider` expects live data) — such a
quote must be treated as a data outage for that symbol this cycle, not used
for trading. A secondary circuit breaker halts trading outright if net worth
ever implies an implausible multiple of the starting budget, as defense in
depth against any other bug producing similar runaway behavior.

Written before the fix per strict TDD (CLAUDE.md: bug fixes start with a
regression test that reproduces the bug).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.economics.cost_accounting import CostAccountant


class ScriptedMarket:
    """Test double satisfying Engine's `self.market.quote(symbol)` interface.
    Returns a scripted sequence of Quotes per symbol, repeating the last one
    once exhausted."""

    def __init__(self, scripts: dict[str, list[Quote]]):
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._idx: dict[str, int] = {k: 0 for k in scripts}

    def quote(self, symbol: str) -> Quote:
        script = self._scripts.get(symbol)
        if not script:
            # Neutral default for symbols we don't care about in a given test
            # (e.g. the benchmark symbol).
            return Quote(symbol, 100.0, [100.0] * 60, "yahoo")
        i = min(self._idx[symbol], len(script) - 1)
        self._idx[symbol] += 1
        return script[i]


def real_history(base: float, n: int = 60) -> list[float]:
    return [base] * n


def buy_signal_history(base: float, n: int = 60) -> list[float]:
    """A noisy uptrend that reliably triggers the heuristic's 'buy': a
    perfectly monotonic line pins RSI at 100 (overbought), which the
    heuristic discounts, netting to 'hold' — real (and realistic synthetic)
    series aren't perfectly monotonic, so this adds mild noise."""
    import math
    return [base + i * 0.6 + 3 * math.sin(i * 0.7) for i in range(n)]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    # provider != "synthetic": the engine expects LIVE data, so any quote that
    # comes back with source="synthetic" is a fallback/outage, not the truth.
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.risk.min_confidence = 0.5
    s.risk.max_fee_pct = 0.0   # fee guard has its own tests (test_fee_guard.py)
    s.options.enabled = True
    s.learning.enabled = False
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make_engine(settings, store, market) -> Engine:
    broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
    return Engine(settings, store, broker, market=market)


class TestDegradedQuoteBlocksNewEntries:
    def test_synthetic_fallback_quote_does_not_open_a_position(self, settings, store):
        # A strong buy signal, but the ONLY quote available this cycle is a
        # synthetic fallback (source mismatch) — must not trade on it.
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "synthetic")],
        })
        engine = make_engine(settings, store, market)
        engine.run_cycle()
        assert store.open_options() == [], "must not open a position on an untrusted quote"

    def test_trustworthy_quote_opens_normally(self, settings, store):
        # Sanity check: the same rising series, but genuinely from "yahoo" —
        # trading must proceed as before the fix.
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = make_engine(settings, store, market)
        engine.run_cycle()
        assert store.open_options(), "a trustworthy quote must still be tradeable"


class TestDegradedQuoteBlocksPositionManagement:
    def test_position_opened_on_real_price_is_not_corrupted_by_fake_mark(
        self, settings, store
    ):
        """This is the exact bug: open at a real strike (~313), then the next
        cycle's quote silently falls back to a synthetic price from a totally
        different regime (~51). Without the fix, the option is marked/closed
        against the fake price and produces an absurd "profit". With the fix,
        the position must be left untouched — no mark, no close, no P&L."""
        market = ScriptedMarket({
            "AAPL": [
                Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo"),
                Quote("AAPL", 51.0, real_history(45.0), "synthetic"),
            ],
        })
        engine = make_engine(settings, store, market)
        engine.run_cycle()  # opens a real position at K~313
        opened = store.open_options()
        assert opened, "setup: first cycle should have opened a position"
        opened_id = opened[0]["id"]

        engine.run_cycle()  # degraded quote arrives — must be a no-op for this symbol
        still_open = store.open_options()
        assert len(still_open) == 1
        assert still_open[0]["id"] == opened_id
        assert store.closed_options() == [], "must not close/mark using the fake price"

        # Net worth must not have exploded from a bogus mark.
        econ = store.get_meta("economics")
        assert econ["net_worth_eur"] < settings.budget * 3


class TestSanityCircuitBreaker:
    def test_halts_when_net_worth_implausibly_exceeds_starting_budget(
        self, settings, store
    ):
        store.set_meta("starting_cash", settings.budget)
        accountant = CostAccountant(settings, store)
        snap = accountant.snapshot(cash_eur=settings.budget, positions_value_eur=0.0)
        assert not snap.halt_trading  # normal state: no breach

        # Simulate a runaway valuation (what the bug produced).
        runaway_snap = accountant.snapshot(
            cash_eur=settings.budget, positions_value_eur=settings.budget * 500)
        assert runaway_snap.halt_trading
        assert runaway_snap.sanity_breached

    def test_normal_growth_within_bounds_does_not_halt(self, settings, store):
        store.set_meta("starting_cash", settings.budget)
        accountant = CostAccountant(settings, store)
        # A generous but plausible 3x gain must not trip the breaker.
        snap = accountant.snapshot(
            cash_eur=settings.budget, positions_value_eur=settings.budget * 2)
        assert not snap.sanity_breached

    def test_engine_stops_opening_new_positions_once_breached(self, settings, store):
        """End-to-end: once the sanity ceiling is breached, no further trades
        should be opened, regardless of how strong the signal is."""
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        # Directly inflate cash to simulate a prior runaway (or a real, if
        # unlikely, huge win) — the breaker must still catch it going forward.
        broker.adjust_cash(settings.budget * 1000)
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = Engine(settings, store, broker, market=market)
        engine.run_cycle()
        econ = store.get_meta("economics")
        assert econ["halt_trading"] is True
        assert store.open_options() == []
