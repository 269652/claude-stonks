"""Tests for profit stashing: a configurable fraction of realized profit on
each winning close is moved into a reserve excluded from future trading
capital (net worth/alpha still reflect it — it's still the account's money,
just protected from being re-risked). Written before implementation per
strict TDD."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


class ScriptedMarket:
    def __init__(self, quotes: dict[str, Quote]):
        self._quotes = quotes

    def quote(self, symbol: str) -> Quote:
        # Falls back to a neutral default for symbols not scripted (e.g. the
        # engine always fetches the benchmark symbol too).
        return self._quotes.get(symbol, Quote(symbol, 100.0, [100.0] * 60, "yahoo"))


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.risk.min_confidence = 1.1   # block new entries; isolate the exit path
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


class TestProfitStash:
    def test_winning_close_stashes_configured_fraction(self, settings, store):
        settings.economics.profit_stash_pct = 0.5
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        engine = make_engine(settings, store, market)
        # Deep ITM call at a tiny premium -> huge mark -> take-profit next cycle
        store.open_option("AAPL", "call", strike=0.01, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=0.01, entry_premium=0.01, genome_id=None)
        engine.run_cycle()

        closed = store.closed_options()
        assert closed and closed[0]["pnl"] > 0
        pnl = closed[0]["pnl"]
        expected_stash = pnl * 0.5
        assert store.reserve_balance() == pytest.approx(expected_stash, rel=1e-6)

    def test_losing_close_stashes_nothing(self, settings, store):
        settings.economics.profit_stash_pct = 0.5
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        engine = make_engine(settings, store, market)
        # Deep OTM call, will lose value -> stop-loss, negative pnl
        store.open_option("AAPL", "call", strike=100000.0,
                          expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=1.0, entry_premium=5.0, genome_id=None)
        engine.run_cycle()
        closed = store.closed_options()
        assert closed and closed[0]["pnl"] < 0
        assert store.reserve_balance() == pytest.approx(0.0)

    def test_zero_stash_pct_stashes_nothing(self, settings, store):
        settings.economics.profit_stash_pct = 0.0
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        engine = make_engine(settings, store, market)
        store.open_option("AAPL", "call", strike=0.01, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=0.01, entry_premium=0.01, genome_id=None)
        engine.run_cycle()
        assert store.closed_options()[0]["pnl"] > 0
        assert store.reserve_balance() == pytest.approx(0.0)

    def test_reserve_excluded_from_tradeable_cash(self, settings, store):
        settings.economics.profit_stash_pct = 0.7
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        engine = make_engine(settings, store, market)
        cash_before = engine.broker.cash()
        store.open_option("AAPL", "call", strike=0.01, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=1.0, entry_premium=0.01, genome_id=None)
        engine.run_cycle()
        pnl = store.closed_options()[0]["pnl"]
        stash = store.reserve_balance()
        # tradeable cash gained (proceeds - stash), not the full proceeds. The
        # planted position never paid its entry fee out of cash, hence the
        # + OPTION_FEE allowance (pnl already deducts it).
        from lmtrade.core.engine import OPTION_FEE
        assert engine.broker.cash() < cash_before + pnl + OPTION_FEE
        assert stash > 0

    def test_reserve_included_in_reported_net_worth(self, settings, store):
        settings.economics.profit_stash_pct = 1.0  # stash everything
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 313.0, [300.0] * 60, "yahoo"),
        })
        engine = make_engine(settings, store, market)
        store.open_option("AAPL", "call", strike=0.01, expiry_ts=time.time() + 5 * 86400,
                          iv=0.4, contracts=0.01, entry_premium=0.01, genome_id=None)
        engine.run_cycle()
        econ = store.get_meta("economics")
        # Net worth must include the stashed reserve, not just tradeable cash.
        assert econ["net_worth_eur"] == pytest.approx(
            engine.broker.cash() + store.reserve_balance(), abs=0.05)
        assert econ["net_worth_eur"] > settings.budget  # the win is still counted

    def test_reserve_persists_across_store_reopen(self, settings, store):
        store.add_reserve(12.5)
        assert store.reserve_balance() == pytest.approx(12.5)
        db_path = store.db_path
        store.close()
        reopened = Store(db_path)
        assert reopened.reserve_balance() == pytest.approx(12.5)
        reopened.close()
