"""Exit decisions use a de-spiked underlying price (median of the last 3
observations), so a single outlier print — live incident: a thin after-hours
Yahoo tick valued GOOGL ~24% higher for one check, recorded a +35.80 EUR
trailing peak and fired a trailing take-profit on the next normal print —
cannot arm the trailing stop or trigger TP / SL / knock-out. Sustained moves
still trigger after a confirmation. Written before the fix (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


class Market:
    """One 1-minute bar per tick; the quote's price is the latest bar."""

    def __init__(self):
        self.bars = [99.0, 101.0] * 29 + [100.0, 100.0]

    @property
    def price(self):
        return self.bars[-1]

    @price.setter
    def price(self, p):
        self.bars.append(p)

    def quote(self, symbol):
        return Quote(symbol, self.bars[-1], list(self.bars), "yahoo")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.loop.exit_check_seconds = 0
    s.options.min_hold_hours = 0
    s.options.flip_exit_confidence = 0.0
    s.options.trail_enabled = True
    s.options.trail_min_profit_eur = 20.0
    s.options.trail_pct = 0.15
    s.research.commentary_minutes = 0
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def setup(settings, store, tp=1e9, sl=0.0):
    m = Market()
    e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
               market=m, tr_derivatives=None)
    e._mark_position = lambda o, spot: spot / 50.0        # 100 -> 2.00 per contract
    store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                      iv=0.4, contracts=25.0, entry_premium=2.0, genome_id=None,
                      tp_premium=tp, sl_premium=sl)
    return e, m


def tick(e, m, price):
    m.price = price
    e._check_exits_fast()


class TestRepeatedChecks:
    def test_spike_bar_seen_by_many_fast_checks_still_ignored(self, settings, store):
        # the fast check runs every ~10 s; a spiky 1-minute bar is seen ~6x
        e, m = setup(settings, store, tp=2.3)
        m.price = 124.0
        for _ in range(6):
            e._check_exits_fast()
        assert len(store.open_options()) == 1


class TestDespike:
    def test_single_spike_does_not_arm_or_fire_trailing(self, settings, store):
        e, m = setup(settings, store)
        for p in (100.0, 100.0, 124.0, 100.0, 100.0):      # one outlier print
            tick(e, m, p)
        assert len(store.open_options()) == 1
        peak = (store.get_meta("trail_peaks") or {}).get("1")
        assert peak is not None and peak < 5.0               # never saw the spike

    def test_single_spike_does_not_hit_take_profit(self, settings, store):
        e, m = setup(settings, store, tp=2.3)
        for p in (100.0, 100.0, 124.0, 100.0):
            tick(e, m, p)
        assert len(store.open_options()) == 1

    def test_single_dip_does_not_hit_stop_loss(self, settings, store):
        e, m = setup(settings, store, sl=1.8)
        for p in (100.0, 100.0, 80.0, 100.0):
            tick(e, m, p)
        assert len(store.open_options()) == 1

    def test_sustained_move_still_triggers(self, settings, store):
        e, m = setup(settings, store, tp=2.3)
        for p in (100.0, 124.0, 124.0):                     # confirmed by a 2nd print
            tick(e, m, p)
        assert store.open_options() == []
