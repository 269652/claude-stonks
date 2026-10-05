"""Signal-driven exits: a held position is closed when the fused signal for its
underlying points AGAINST it with confidence >= options.flip_exit_confidence
(0 = off). Hard exits (TP / SL / trailing / knock-out) count as confidence-1.0
exit signals; every exit is logged as "[exit-signal]". Also: the position
commentary runs during decision passes, not only between them.
Written before implementation (TDD). Offline: market and decisions faked."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.base import OrderResult
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import OptionsConfig, Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


class Market:
    def quote(self, symbol):
        return Quote(symbol, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], "yahoo")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 0.5
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.options.min_hold_hours = 0
    s.options.trail_enabled = False
    s.research.commentary_minutes = 0
    s.tr.market_hours = ""             # always open: runs at any hour
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def engine(settings, store, broker=None):
    e = Engine(settings, store, broker or PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
               market=Market(), tr_derivatives=None)
    e._mark_position = lambda o, spot: 2.0
    return e


def hold(store, kind="call", opened_ago_h=1.0):
    oid = store.open_option("AAPL", kind, strike=100.0, expiry_ts=time.time() + 5 * 86400,
                            iv=0.4, contracts=10.0, entry_premium=2.0, genome_id=None,
                            tp_premium=1e9, sl_premium=0.0)
    store._conn.execute("UPDATE option_positions SET opened_ts=? WHERE id=?",
                        (time.time() - opened_ago_h * 3600, oid))
    store._conn.commit()
    return oid


def force(e, direction, conf):
    e._decide = lambda q: (Decision(q.symbol, direction, conf, "forced"), None)


def logs(store):
    return " ".join(r["message"] for r in store.recent_logs(200))


class TestConfig:
    def test_default(self):
        assert OptionsConfig().flip_exit_confidence == pytest.approx(0.8)

    def test_editable(self):
        from lmtrade.web.settings_editor import EDITABLE
        assert "options.flip_exit_confidence" in {k for k, _, _ in EDITABLE}


class TestFlipExit:
    def test_strong_opposite_signal_closes_long(self, settings, store):
        e = engine(settings, store)
        hold(store, "call")
        force(e, "sell", 0.9)
        e.run_cycle()
        assert store.open_options() == []
        assert "signal exit" in store.recent_trades(5)[0]["reason"]
        assert "[exit-signal]" in logs(store)

    def test_strong_opposite_signal_closes_short(self, settings, store):
        e = engine(settings, store)
        hold(store, "put")
        force(e, "buy", 0.85)
        e.run_cycle()
        assert store.open_options() == []

    def test_weak_opposite_signal_keeps(self, settings, store):
        e = engine(settings, store)
        hold(store, "call")
        force(e, "sell", 0.6)
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_same_direction_signal_keeps(self, settings, store):
        e = engine(settings, store)
        hold(store, "call")
        force(e, "buy", 1.0)
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_disabled(self, settings, store):
        settings.options.flip_exit_confidence = 0.0
        e = engine(settings, store)
        hold(store, "call")
        force(e, "sell", 1.0)
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_min_hold_respected(self, settings, store):
        settings.options.min_hold_hours = 2.0
        e = engine(settings, store)
        hold(store, "call", opened_ago_h=1.0)
        force(e, "sell", 1.0)
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_live_flip_exit_sends_a_real_sell(self, settings, store):
        class FakeTR(PaperBroker):
            mode = "live"

            def __init__(self, s):
                super().__init__(s, starting_cash=300.0, fee=OPTION_FEE)
                self.armed, self.sells = True, []

            def place_order(self, isin, side, size, exchange="LSX"):
                self.sells.append((isin, side, size))
                return OrderResult(True, isin, side, size, 0.0, 1.0, "ok", order_id="S1")

        tr = FakeTR(store)
        e = engine(settings, store, broker=tr)
        oid = hold(store, "call")
        store._conn.execute("UPDATE option_positions SET isin='DE000X1' WHERE id=?", (oid,))
        store._conn.commit()
        force(e, "sell", 0.9)
        e.run_cycle()
        assert tr.sells == [("DE000X1", "sell", 10.0)]
        assert store.open_options() == []


class TestHardExitsAreExitSignals:
    def test_take_profit_logged_as_full_confidence_exit_signal(self, settings, store):
        e = engine(settings, store)
        oid = hold(store, "call")
        store._conn.execute("UPDATE option_positions SET tp_premium=1.5 WHERE id=?", (oid,))
        store._conn.commit()
        force(e, "hold", 0.0)
        e.run_cycle()
        assert store.open_options() == []
        assert "[exit-signal] AAPL conf 1.00" in logs(store)


class TestCommentaryDuringPass:
    def test_commentary_tick_runs_inside_the_decision_loop(self, settings, store):
        settings.research.commentary_minutes = 5
        settings.universe = ["AAPL", "MSFT"]
        e = engine(settings, store)
        e.commentary_async = False
        e.commentary_caller = lambda p: "positions steady"
        hold(store, "call")
        force(e, "hold", 0.0)
        e.run_cycle()
        assert (store.get_meta("position_commentary") or {}).get("text") == "positions steady"
