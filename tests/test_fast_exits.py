"""Fast exit checks: exits used to be evaluated once per full engine cycle
(~3 min with 30 symbols), so a trailing stop at +22.25 EUR closed at +17.26
after a drop between two checks. Open positions are now re-checked every
loop.exit_check_seconds (fresh quotes for held symbols only), between symbol
decisions and during the idle wait. Written before implementation (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import LoopConfig, Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


class CountingMarket:
    def __init__(self, source="yahoo"):
        self.calls: list[str] = []
        self.source = source

    def quote(self, symbol: str) -> Quote:
        self.calls.append(symbol)
        return Quote(symbol, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], self.source)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL", "MSFT"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.options.min_hold_hours = 0
    s.loop.exit_check_seconds = 0          # no throttle unless a test sets one
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store, market):
    e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=1.0),
               market=market, tr_derivatives=None)
    return e


def open_call(store, tp=3.0):
    return store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                             iv=0.4, contracts=10.0, entry_premium=2.0, genome_id=None,
                             tp_premium=tp, sl_premium=0.0)


class TestFastExitCheck:
    def test_default_interval(self):
        assert LoopConfig().exit_check_seconds == pytest.approx(10.0)

    def test_closes_tp_without_full_cycle_quoting_only_held_symbols(self, settings, store):
        m = CountingMarket()
        e = make(settings, store, m)
        e._mark_position = lambda o, spot: 3.5
        open_call(store)
        e._check_exits_fast()
        assert store.closed_options()
        assert m.calls == ["AAPL"]                 # not the whole universe

    def test_no_positions_no_quotes(self, settings, store):
        m = CountingMarket()
        make(settings, store, m)._check_exits_fast()
        assert m.calls == []

    def test_throttled(self, settings, store):
        settings.loop.exit_check_seconds = 60
        m = CountingMarket()
        e = make(settings, store, m)
        e._mark_position = lambda o, spot: 2.0     # nothing to close
        open_call(store)
        e._check_exits_fast()
        e._check_exits_fast()
        assert m.calls == ["AAPL"]

    def test_untrustworthy_quote_never_closes(self, settings, store):
        e = make(settings, store, CountingMarket(source="synthetic"))
        e._mark_position = lambda o, spot: 3.5
        open_call(store)
        e._check_exits_fast()
        assert len(store.open_options()) == 1

    def test_quote_error_is_survived(self, settings, store):
        class Broken:
            def quote(self, symbol):
                raise RuntimeError("network down")
        e = make(settings, store, Broken())
        open_call(store)
        e._check_exits_fast()                      # must not raise
        assert len(store.open_options()) == 1

    def test_marks_refreshed_for_dashboard(self, settings, store):
        e = make(settings, store, CountingMarket())
        e._mark_position = lambda o, spot: 2.5
        oid = open_call(store)
        e._check_exits_fast()
        assert store.get_meta("open_option_marks")[str(oid)]["mark_premium"] == 2.5

    def test_runs_during_idle_wait(self, settings, store):
        e = make(settings, store, CountingMarket())
        e._mark_position = lambda o, spot: 3.5
        open_call(store)
        e._idle(1.2)
        assert store.closed_options()

    def test_runs_between_symbol_decisions(self, settings, store):
        settings.risk.min_confidence = 0.5
        e = make(settings, store, CountingMarket())
        open_call(store)
        seen = []
        real = e._decide

        def decide(q):
            seen.append(len(store.open_options()))
            return real(q)
        e._decide = decide
        e._mark_position = lambda o, spot: 2.0
        e.run_cycle()
        calls_before = e._fast_exit_checks
        assert calls_before >= len(settings.universe)
