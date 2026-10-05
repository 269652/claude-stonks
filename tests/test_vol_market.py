"""Volatility rule for limit-mode entries: when the intraday sigma relative to
the price is at or above entry.market_above_sigma_pct, no resting limit order
is created — the signal is entered with a market order once triggered (a
resting one is cancelled when volatility rises). 0 disables the rule.
Written before implementation (TDD). Offline: market and decisions faked."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import EntryConfig, Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.limit_orders import relative_sigma_pct
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


class TestRelativeSigma:
    def test_value(self):
        assert relative_sigma_pct([99.0, 101.0] * 30, 60) == pytest.approx(1.0)
        assert relative_sigma_pct([97.0, 103.0] * 30, 60) == pytest.approx(3.0)

    def test_unknown(self):
        assert relative_sigma_pct([100.0] * 5, 60) is None
        assert relative_sigma_pct([], 60) is None


class TestConfig:
    def test_default_and_editable(self):
        assert EntryConfig().market_above_sigma_pct == pytest.approx(0.0)   # off in code
        from lmtrade.web.settings_editor import EDITABLE
        assert "entry.market_above_sigma_pct" in {k for k, _, _ in EDITABLE}


class Market:
    """Price around SMA 100 with a configurable half-range (sigma = amp)."""
    def __init__(self):
        self.amp = 0.2           # 0.2% sigma: calm
        self.price = 100.0

    def quote(self, symbol):
        a = self.amp
        return Quote(symbol, self.price, [100 - a, 100 + a] * 29 + [100 - a, self.price], "yahoo")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 0.5
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.loop.exit_check_seconds = 1e9
    s.sizing.vol_target_annual = 100.0
    s.sizing.regime_filter_enabled = False
    s.options.max_option_fraction = 2 / 3
    s.options.cash_reserve_pct = 1 / 3
    s.options.trail_enabled = False
    s.entry.watch_enabled = True
    s.entry.watch_k = 1.0
    s.entry.limit_orders = True
    s.entry.market_above_sigma_pct = 1.5
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store):
    m = Market()
    e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
               market=m, tr_derivatives=None)
    e._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "forced"), None)
    return e, m


def pending(store):
    return store.get_meta("pending_entries") or {}


class TestVolatilityRule:
    def test_calm_market_still_uses_limit_orders(self, settings, store):
        e, m = make(settings, store)
        e.run_cycle()
        assert "AAPL" in pending(store)

    def test_volatile_untriggered_gets_no_order(self, settings, store):
        e, m = make(settings, store)
        m.amp = 2.0                                    # 2% sigma >= 1.5%
        e.run_cycle()
        assert pending(store) == {}
        assert store.open_options() == []
        w = store.get_meta("watchlist")["AAPL"]
        assert w["high_vol"] is True and w["sigma_pct"] == pytest.approx(2.0, rel=0.05)

    def test_volatile_triggered_enters_at_market(self, settings, store):
        e, m = make(settings, store)
        m.amp = 2.0
        m.price = 97.0                                 # well below SMA - 1 sigma
        e.run_cycle()
        assert len(store.open_options()) == 1
        assert pending(store) == {}

    def test_resting_order_cancelled_when_volatility_rises(self, settings, store):
        e, m = make(settings, store)
        e.run_cycle()
        assert "AAPL" in pending(store)
        m.amp = 2.0
        e.run_cycle()
        assert pending(store) == {}
        assert "volatility" in " ".join(r["message"] for r in store.recent_logs(100))

    def test_zero_threshold_disables_rule(self, settings, store):
        settings.entry.market_above_sigma_pct = 0.0
        e, m = make(settings, store)
        m.amp = 2.0
        e.run_cycle()
        assert "AAPL" in pending(store)

    def test_watchlist_api_flags_market_entry(self, settings, store):
        from fastapi.testclient import TestClient

        from lmtrade.web.app import create_app
        e, m = make(settings, store)
        m.amp = 2.0
        e.run_cycle()
        row = TestClient(create_app(settings)).get("/api/watchlist").json()[0]
        assert row["high_vol"] is True
        assert row["sigma_pct"] == pytest.approx(2.0, rel=0.05)
