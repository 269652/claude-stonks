"""Tests for scaling the engine to a larger universe: concurrent (bounded)
quote fetching so cycle time doesn't grow linearly with universe size, and
signal-ranked entry selection so limited position slots go to the strongest
candidates rather than whichever symbols happen to come first in the
universe list. Written before implementation per strict TDD."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings, load_settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import MarketData, Quote


def buy_signal_history(base: float, n: int = 60) -> list[float]:
    return [base + i * 0.6 + 3 * math.sin(i * 0.7) for i in range(n)]


class TestConcurrentQuoteFetching:
    def test_fetches_all_symbols_exactly_once(self):
        calls: list[str] = []
        lock = threading.Lock()

        def fetcher(symbol, range_, interval):
            with lock:
                calls.append(symbol)
            return [100.0 + i for i in range(80)]

        md = MarketData("yahoo", lookback=30, fetcher=fetcher)
        symbols = [f"SYM{i}" for i in range(12)]
        quotes = md.quotes_concurrent(symbols, max_workers=4)
        assert set(quotes) == set(symbols)
        assert sorted(calls) == sorted(symbols)
        assert all(q.source == "yahoo" for q in quotes.values())

    def test_respects_concurrency_cap(self):
        concurrent = 0
        max_seen = 0
        lock = threading.Lock()

        def fetcher(symbol, range_, interval):
            nonlocal concurrent, max_seen
            with lock:
                concurrent += 1
                max_seen = max(max_seen, concurrent)
            time.sleep(0.05)
            with lock:
                concurrent -= 1
            return [100.0 + i for i in range(80)]

        md = MarketData("yahoo", lookback=30, fetcher=fetcher)
        symbols = [f"SYM{i}" for i in range(15)]
        md.quotes_concurrent(symbols, max_workers=3)
        assert max_seen <= 3

    def test_faster_than_sequential_for_slow_fetcher(self):
        def slow_fetcher(symbol, range_, interval):
            time.sleep(0.05)
            return [100.0 + i for i in range(80)]

        md = MarketData("yahoo", lookback=30, fetcher=slow_fetcher)
        symbols = [f"SYM{i}" for i in range(10)]

        start = time.time()
        md.quotes_concurrent(symbols, max_workers=8)
        elapsed = time.time() - start
        # Sequential would take ~0.5s (10 * 0.05s); concurrent with 8 workers
        # should be well under that.
        assert elapsed < 0.3

    def test_one_failing_symbol_falls_back_without_breaking_others(self):
        def flaky_fetcher(symbol, range_, interval):
            if symbol == "BAD":
                raise RuntimeError("boom")
            return [100.0 + i for i in range(80)]

        md = MarketData("yahoo", lookback=30, fetcher=flaky_fetcher)
        quotes = md.quotes_concurrent(["GOOD1", "BAD", "GOOD2"], max_workers=3)
        assert quotes["GOOD1"].source == "yahoo"
        assert quotes["GOOD2"].source == "yahoo"
        assert quotes["BAD"].source == "synthetic"  # graceful per-symbol fallback


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0,
                 universe=["AAA", "BBB", "CCC", "DDD"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.loop.max_positions = 2       # fewer slots than candidates
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


class TestRankedEntrySelection:
    def test_strongest_signals_win_over_universe_order(self, settings, store):
        """AAA and BBB are listed first but have weak signals; CCC and DDD
        come later but have the strongest confidence. With only 2 slots, the
        engine must pick the strongest candidates, not just list order.

        The heuristic model's confidence is coarse (a handful of discrete
        tiers driven by threshold crosses, not a smooth function of trend
        magnitude), so reverse-engineering price series to hit exact
        confidence values is fragile. Instead, drive the scenario directly
        through `_decide` — this is what's actually under test (the ranking
        and slot-allocation logic), not the heuristic's signal generation,
        which has its own coverage elsewhere."""
        from lmtrade.agents.fusion import Decision

        def fetcher(symbol, range_, interval):
            return buy_signal_history(300.0)

        market = MarketData("yahoo", lookback=60, fetcher=fetcher)
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        engine = Engine(settings, store, broker, market=market)

        confidences = {"AAA": 0.55, "BBB": 0.60, "CCC": 0.95, "DDD": 0.90}

        def fake_decide(quote):
            conf = confidences[quote.symbol]
            return Decision(quote.symbol, "buy", conf, "scripted"), None

        engine._decide = fake_decide
        engine.run_cycle()

        opened = {o["underlying"] for o in store.open_options()}
        assert len(opened) == 2
        assert "CCC" in opened and "DDD" in opened, \
            f"expected the strongest-confidence symbols, got {opened}"

    def test_never_opens_more_than_max_positions(self, settings, store):
        def fetcher(symbol, range_, interval):
            return buy_signal_history(300.0 + hash(symbol) % 50, 60)

        market = MarketData("yahoo", lookback=60, fetcher=fetcher)
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        engine = Engine(settings, store, broker, market=market)
        engine.run_cycle()
        assert len(store.open_options()) <= settings.loop.max_positions


class TestDefaultUniverse:
    def test_universe_is_meaningfully_larger_than_five(self):
        s = load_settings()
        assert len(s.universe) >= 20

    def test_universe_spans_multiple_asset_classes(self):
        s = load_settings()
        universe = set(s.universe)
        us_equities = {"AAPL", "MSFT"} & universe
        eu_equities = {s for s in universe if s.endswith((".DE", ".PA", ".AS", ".SW"))}
        etfs = {"SPY", "QQQ", "IWM"} & universe
        crypto = {s for s in universe if s.endswith("-USD")}
        assert us_equities, "expected US equities in the default universe"
        assert eu_equities, "expected EU equities in the default universe"
        assert etfs, "expected ETFs in the default universe"
        assert crypto, "expected crypto in the default universe"

    def test_max_positions_smaller_than_universe(self):
        s = load_settings()
        assert s.loop.max_positions < len(s.universe)

    def test_position_sizing_scaled_down_for_larger_universe(self):
        s = load_settings()
        # With more concurrent positions possible, per-position sizing must be
        # more conservative than the old 5-symbol default (0.3) so the budget
        # isn't wildly over-committed if many slots fill.
        assert s.options.max_option_fraction <= 0.2
