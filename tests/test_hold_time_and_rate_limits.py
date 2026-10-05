"""Tests for min/max hold-time limits on option positions and a per-cycle cap
on new entries ("how many trades it makes each hour"). Written before
implementation per strict TDD."""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


def buy_signal_history(base: float, n: int = 60) -> list[float]:
    return [base + i * 0.6 + 3 * math.sin(i * 0.7) for i in range(n)]


class ScriptedMarket:
    def __init__(self, quotes: dict[str, Quote]):
        self._quotes = quotes

    def quote(self, symbol: str) -> Quote:
        return self._quotes.get(symbol, Quote(symbol, 100.0, [100.0] * 60, "yahoo"))


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.risk.min_confidence = 1.1
    s.options.flip_exit_confidence = 0.0   # isolate time-based exits from signal exits
    s.options.enabled = True
    s.learning.enabled = False
    s.risk.max_fee_pct = 0.0   # these tests are about slot caps, not fee drag
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make_engine(settings, store, market) -> Engine:
    broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
    return Engine(settings, store, broker, market=market)


class TestMinHoldHours:
    def test_blocks_premature_take_profit(self, settings, store):
        settings.options.min_hold_hours = 6.0
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        engine = make_engine(settings, store, market)
        # Deep ITM call opened "now" -> would hit take-profit immediately,
        # but min_hold_hours must block the exit.
        store.open_option("AAPL", "call", strike=0.01, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=0.01, entry_premium=0.01, genome_id=None)
        engine.run_cycle()
        assert store.closed_options() == []
        assert len(store.open_options()) == 1

    def test_allows_exit_once_min_hold_elapsed(self, settings, store):
        settings.options.min_hold_hours = 1.0
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        opened_ts = time.time() - 2 * 3600   # opened 2h ago
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        store.open_option("AAPL", "call", strike=0.01, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=0.01, entry_premium=0.01, genome_id=None)
        # Backdate opened_ts directly since open_option always stamps "now".
        with store._lock:
            store._conn.execute("UPDATE option_positions SET opened_ts=?", (opened_ts,))
            store._conn.commit()
        engine = Engine(settings, store, broker, market=market)
        engine.run_cycle()
        assert store.closed_options(), "min_hold_hours has elapsed, exit must proceed"

    def test_expiry_force_close_ignores_min_hold(self, settings, store):
        """The expiry-window force-close is a hard constraint of the
        synthetic option itself — min_hold_hours must not block it."""
        settings.options.min_hold_hours = 1000.0   # effectively "never" for TP/SL
        settings.options.min_hours_to_expiry = 24.0
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        engine = make_engine(settings, store, market)
        store.open_option("AAPL", "call", strike=313.0, expiry_ts=time.time() + 3600,
                          iv=0.4, contracts=1.0, entry_premium=1.0, genome_id=None)
        engine.run_cycle()
        assert store.closed_options(), "expiry force-close must fire regardless of min_hold"


class TestMaxHoldHours:
    def test_force_closes_after_max_hold_even_without_tp_sl(self, settings, store):
        settings.options.max_hold_hours = 1.0
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        opened_ts = time.time() - 2 * 3600  # opened 2h ago, past max_hold
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        # ATM-ish, unlikely to have hit TP/SL from the tiny synthetic move
        store.open_option("AAPL", "call", strike=313.0, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=1.0, entry_premium=5.0, genome_id=None)
        with store._lock:
            store._conn.execute("UPDATE option_positions SET opened_ts=?", (opened_ts,))
            store._conn.commit()
        engine = Engine(settings, store, broker, market=market)
        engine.run_cycle()
        closed = store.closed_options()
        assert closed, "must force-close once max_hold_hours has elapsed"
        trades = store.recent_trades(5)
        close_trade = next(t for t in trades if t["symbol"] == "AAPL" and t["side"] == "sell")
        assert "max hold" in close_trade["reason"].lower()

    def test_disabled_by_default_zero(self, settings, store):
        assert settings.options.max_hold_hours == 0.0
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 57 + [313.0] * 3, "yahoo"),
        })
        opened_ts = time.time() - 1000 * 3600  # ancient position
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        store.open_option("AAPL", "call", strike=313.0, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=1.0, entry_premium=5.0, genome_id=None)
        with store._lock:
            store._conn.execute("UPDATE option_positions SET opened_ts=?", (opened_ts,))
            store._conn.commit()
        engine = Engine(settings, store, broker, market=market)
        engine.run_cycle()
        assert store.open_options(), "max_hold_hours=0 (disabled) must not force-close"


class TestMaxNewPositionsPerCycle:
    def test_caps_new_opens_even_with_more_slots_and_candidates(self, settings, store):
        settings.universe = ["AAA", "BBB", "CCC", "DDD"]
        settings.loop.max_positions = 4
        settings.loop.max_new_positions_per_cycle = 2
        settings.risk.min_confidence = 0.5

        market = ScriptedMarket({
            "AAA": Quote("AAA", 300.0, buy_signal_history(300.0), "yahoo"),
            "BBB": Quote("BBB", 300.0, buy_signal_history(300.0), "yahoo"),
            "CCC": Quote("CCC", 300.0, buy_signal_history(300.0), "yahoo"),
            "DDD": Quote("DDD", 300.0, buy_signal_history(300.0), "yahoo"),
        })
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        engine = Engine(settings, store, broker, market=market)

        def fake_decide(quote):
            return Decision(quote.symbol, "buy", 0.9, "scripted"), None

        engine._decide = fake_decide
        engine.run_cycle()
        assert len(store.open_options()) == 2, \
            "max_new_positions_per_cycle=2 must cap opens even with 4 slots free"

    def test_zero_means_unlimited_up_to_slots(self, settings, store):
        settings.universe = ["AAA", "BBB", "CCC"]
        settings.loop.max_positions = 3
        settings.loop.max_new_positions_per_cycle = 0   # default: no extra cap
        settings.risk.min_confidence = 0.5

        market = ScriptedMarket({
            "AAA": Quote("AAA", 300.0, buy_signal_history(300.0), "yahoo"),
            "BBB": Quote("BBB", 300.0, buy_signal_history(300.0), "yahoo"),
            "CCC": Quote("CCC", 300.0, buy_signal_history(300.0), "yahoo"),
        })
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        engine = Engine(settings, store, broker, market=market)

        def fake_decide(quote):
            return Decision(quote.symbol, "buy", 0.9, "scripted"), None

        engine._decide = fake_decide
        engine.run_cycle()
        assert len(store.open_options()) == 3
