"""Net worth is the liquidation value: cash + positions' value minus the exit
fee each open position still costs to sell. Equivalently cash + sum(cost +
unrealized P&L) over the rows (row P&L already nets the entry fee). It used
to be cash + value, overstating net worth by 1 EUR per open position.
Written before implementation (TDD)."""
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


class Market:
    def quote(self, symbol):
        return Quote(symbol, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], "yahoo")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    return s


def open_call(store, contracts=10.0, entry=2.0):
    return store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                             iv=0.4, contracts=contracts, entry_premium=entry, genome_id=None,
                             tp_premium=1e9, sl_premium=0.0)


class TestEngineNetWorth:
    def test_recorded_equity_and_economics_net_of_exit_fees(self, settings):
        store = Store(settings.db_path)
        try:
            e = Engine(settings, store, PaperBroker(store, starting_cash=279.0, fee=OPTION_FEE),
                       market=Market(), tr_derivatives=None)
            open_call(store)                         # 10 x 2.00 = 20 of value
            e._mark_position = lambda o, spot: 2.0
            e.run_cycle()
            expected = 279.0 + 20.0 - OPTION_FEE
            assert store.equity_curve()[-1]["equity"] == pytest.approx(expected)
            assert store.get_meta("economics")["net_worth_eur"] == pytest.approx(expected)
        finally:
            store.close()


class TestLiveSummaryNetWorth:
    @pytest.fixture()
    def summary(self, settings):
        live = Store(settings.live_db_path)
        ids = [open_call(live, 10.0, 2.0), open_call(live, 5.0, 4.0)]
        live.set_meta("open_option_marks", {
            str(ids[0]): {"mark_premium": 2.5, "value": 25.0, "unrealized_pnl": 4.0, "spot": 100.0},
            str(ids[1]): {"mark_premium": 3.0, "value": 15.0, "unrealized_pnl": -6.0, "spot": 100.0},
        })
        live.set_meta("tr_account_cash", 100.0)
        live.close()
        ControlState(settings.control_path).set_mode("live")
        return TestClient(create_app(settings)).get("/api/summary").json()

    def test_rows_carry_liquidation_value(self, summary):
        assert sorted(r["liquidation_value"] for r in summary["positions"]) == \
            pytest.approx(sorted([25.0 - OPTION_FEE, 15.0 - OPTION_FEE]))

    def test_equity_is_cash_plus_cost_plus_pnl_of_all_rows(self, summary):
        rows = summary["positions"]
        identity = summary["cash"] + sum(r["qty"] * r["avg_price"] + r["unrealized_pnl"]
                                         for r in rows)
        assert summary["equity"] == pytest.approx(identity)
        assert summary["equity"] == pytest.approx(100.0 + 40.0 - 2 * OPTION_FEE)

    def test_dashboard_uses_liquidation_value(self):
        js = (Path(__file__).parents[1] / "src" / "lmtrade" / "web" / "static"
              / "app.js").read_text(encoding="utf-8")
        assert "liquidation_value" in js
