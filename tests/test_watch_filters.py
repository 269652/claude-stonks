"""Watch-list admission and instant entries:
- Only symbols TR can actually trade (a knockout of the signal's direction
  within the leverage band) are watched when the TR client is up; cached.
- Signals at or above entry.instant_confidence skip the timing: entered at
  the current price (limit mode: a marketable limit, +instant_slippage).
Written before implementation (TDD). Offline: market, TR and decisions faked."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.brokers.tr_derivatives import FakeTRDerivatives, TRDerivativeQuote
from lmtrade.config import EntryConfig, Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote

BASE = [99.0, 101.0] * 30


class Market:
    def __init__(self):
        self.price: dict[str, float] = {}

    def quote(self, symbol: str) -> Quote:
        p = self.price.get(symbol, 100.0)
        return Quote(symbol, p, BASE[:-1] + [p], "yahoo")


def ko(sym, kind, lev=5.0):
    return TRDerivativeQuote(isin=f"DE{sym}{kind}", underlying=sym, kind=kind,
                             strike=80.0 if kind == "ko_call" else 120.0,
                             barrier=80.0 if kind == "ko_call" else 120.0, ratio=10.0,
                             price=2.0, leverage=lev)


class CountingDerivs(FakeTRDerivatives):
    def __init__(self, catalog):
        super().__init__(catalog)
        self.searches: list[str] = []

    def search(self, underlying, direction):
        self.searches.append(underlying)
        return super().search(underlying, direction)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL", "MSFT"],
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
    s.entry.watch_window = 60
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store, tr=None):
    m = Market()
    e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
               market=m, tr_derivatives=tr)
    return e, m


def force(e, conf=0.9, direction="buy"):
    e._decide = lambda q: (Decision(q.symbol, direction, conf, "forced"), None)


def watched(store):
    return set(store.get_meta("watchlist") or {})


class TestConfig:
    def test_defaults(self):
        c = EntryConfig()
        assert c.instant_confidence == pytest.approx(1.0)
        assert c.instant_slippage == pytest.approx(0.005)

    def test_editable(self):
        from lmtrade.web.settings_editor import EDITABLE
        keys = {k for k, _, _ in EDITABLE}
        assert {"entry.instant_confidence", "entry.instant_slippage"} <= keys


class TestOnlyTradableWatched:
    def test_symbol_without_tr_knockout_is_not_watched(self, settings, store):
        tr = CountingDerivs({"AAPL": [ko("AAPL", "ko_call")]})
        e, _ = make(settings, store, tr)
        force(e)
        e.run_cycle()
        assert watched(store) == {"AAPL"}
        assert "not tradable on TR" in " ".join(r["message"] for r in store.recent_logs(200))

    def test_wrong_direction_only_is_not_tradable(self, settings, store):
        tr = CountingDerivs({"AAPL": [ko("AAPL", "ko_put")]})     # puts only
        e, _ = make(settings, store, tr)
        force(e)                                                   # buy -> needs a call
        e.run_cycle()
        assert "AAPL" not in watched(store)

    def test_leverage_outside_band_is_not_tradable(self, settings, store):
        tr = CountingDerivs({"AAPL": [ko("AAPL", "ko_call", lev=80.0)]})
        e, _ = make(settings, store, tr)
        force(e)
        e.run_cycle()
        assert "AAPL" not in watched(store)

    def test_tradability_cached(self, settings, store):
        tr = CountingDerivs({"AAPL": [ko("AAPL", "ko_call")]})
        e, _ = make(settings, store, tr)
        force(e)
        e.run_cycle()
        e.run_cycle()
        assert tr.searches.count("MSFT") == 1

    def test_without_tr_client_everything_may_be_watched(self, settings, store):
        e, _ = make(settings, store, None)
        force(e)
        e.run_cycle()
        assert watched(store) == {"AAPL", "MSFT"}


class TestInstantEntries:
    def test_market_mode_enters_immediately_despite_stretch(self, settings, store):
        settings.universe = ["AAPL"]
        settings.entry.limit_orders = False
        e, m = make(settings, store)
        m.price["AAPL"] = 101.5                    # stretched AGAINST a buy
        force(e, conf=1.0)
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_below_threshold_still_waits(self, settings, store):
        settings.universe = ["AAPL"]
        settings.entry.limit_orders = False
        e, m = make(settings, store)
        m.price["AAPL"] = 101.5
        force(e, conf=0.95)
        e.run_cycle()
        assert store.open_options() == []

    def test_threshold_is_configurable(self, settings, store):
        settings.universe = ["AAPL"]
        settings.entry.limit_orders = False
        settings.entry.instant_confidence = 0.9
        e, m = make(settings, store)
        m.price["AAPL"] = 101.5
        force(e, conf=0.95)
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_limit_mode_places_marketable_limit_and_fills(self, settings, store):
        settings.universe = ["AAPL"]
        settings.entry.limit_orders = True
        e, m = make(settings, store)
        m.price["AAPL"] = 101.5
        force(e, conf=1.0)
        e.run_cycle()
        p = (store.get_meta("pending_entries") or {})["AAPL"]
        assert p["target_spot"] == pytest.approx(101.5)
        assert p.get("instant") is True
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_watchlist_api_shows_instant_as_ready(self, settings, store):
        from fastapi.testclient import TestClient

        from lmtrade.web.app import create_app
        settings.universe = ["AAPL"]
        settings.entry.limit_orders = True
        e, m = make(settings, store)
        m.price["AAPL"] = 101.5
        force(e, conf=1.0)
        e.run_cycle()
        row = TestClient(create_app(settings)).get("/api/watchlist").json()[0]
        assert row["instant"] is True and row["sigma_to_go"] == 0
