"""The low-balance guard (net worth < LOW_BALANCE_EUR blocks real orders unless
double-armed) must measure NET WORTH — cash + positions after exit fees — not
cash alone. Live incident: 22.77 EUR cash with ~272 EUR in positions showed
"NOT executing real orders (net worth under 100)". Written before
implementation (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.control import ControlState
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.web.app import create_app


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1
    s.learning.enabled = False
    return s


def open_ko(store, contracts=25.0, entry=6.0):
    return store.open_option("AAPL", "ko_call", 80.0, time.time() + 3e7, 0.0, contracts, entry,
                             None, instrument_type="knockout", barrier=80.0, ratio=10.0,
                             isin="DE000KO1")


class TestDashboardGuard:
    def test_control_uses_cash_plus_positions(self, settings):
        live = Store(settings.live_db_path)
        oid = open_ko(live)
        live.set_meta("open_option_marks", {str(oid): {"mark_premium": 6.0, "value": 150.0,
                                                       "unrealized_pnl": -1.0, "spot": 100.0}})
        live.set_meta("tr_account_cash", 22.77)
        live.close()
        c = ControlState(settings.control_path)
        c.set_mode("live")
        ctrl = TestClient(create_app(settings)).get("/api/control").json()
        assert ctrl["net_worth"] == pytest.approx(22.77 + 150.0 - OPTION_FEE)
        assert ctrl["low_balance"] is False

    def test_really_low_net_worth_still_flagged(self, settings):
        live = Store(settings.live_db_path)
        live.set_meta("tr_account_cash", 40.0)
        live.close()
        ControlState(settings.control_path).set_mode("live")
        assert TestClient(create_app(settings)).get("/api/control").json()["low_balance"] is True


class Market:
    def quote(self, symbol):
        return Quote(symbol, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], "yahoo")


class CashBroker(PaperBroker):
    mode = "live"

    def __init__(self, store, cash):
        super().__init__(store, starting_cash=cash, fee=OPTION_FEE)
        self.armed = True

    def place_order(self, *a, **k):
        raise AssertionError("not used")


class TestEngineGuard:
    def test_engine_net_worth_includes_positions(self, settings):
        store = Store(settings.db_path)
        try:
            e = Engine(settings, store, CashBroker(store, 22.77), market=Market(),
                       tr_derivatives=None)
            open_ko(store)
            e._last_good_price["AAPL"] = 100.0
            e._mark_position = lambda o, spot: 6.0          # 25 x 6 = 150
            assert e._guard_net_worth() == pytest.approx(22.77 + 150.0 - OPTION_FEE)
        finally:
            store.close()

    def test_engine_uses_it_for_the_guard(self):
        src = (Path(__file__).parents[1] / "src" / "lmtrade" / "core" / "engine.py") \
            .read_text(encoding="utf-8")
        assert "low_balance_blocks(\n                    self.broker.cash())" not in src
        assert "net_worth = self.broker.cash()" not in src
        assert "low_balance_blocks(self._guard_net_worth())" in src     # limit entries
        assert "net_worth = self._guard_net_worth()" in src              # market entries
