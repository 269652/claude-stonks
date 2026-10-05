"""Equity curve tracking: the live curve showed a flat line because points were
recorded only once per ~2.5-minute cycle and failed TR cash reads were
recorded as 0 EUR (spikes stretching the axis from 0). Now: the TR cash read
matches its own subscription with a timeout and reports success; failed reads
are never recorded; the fast exit check adds a point every EQUITY_POINT_S
while positions are open; the live view hides old zero-cash spikes.
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
from test_tr_broker import FakeOrderApi, make_broker


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
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


class TestTrCashRead:
    def test_success_sets_flag(self, store, settings, monkeypatch):
        b = make_broker(store, settings, armed=False, api=FakeOrderApi(), monkeypatch=monkeypatch)
        assert b.cash() == pytest.approx(42.5) or b.cash() > 0
        assert b.last_cash_ok is True

    def test_crosstalk_frame_is_skipped(self, store, settings, monkeypatch):
        api = FakeOrderApi(cash_payload=[{"currencyId": "EUR", "amount": 88.0}],
                           crosstalk=[("sub-other", {}, {"some": "ticker"})])
        b = make_broker(store, settings, armed=False, api=api, monkeypatch=monkeypatch)
        assert b.cash() == pytest.approx(88.0)
        assert b.last_cash_ok is True

    def test_unreachable_flags_failure(self, store, settings, monkeypatch):
        b = make_broker(store, settings, armed=False, api=FakeOrderApi(resume_ok=False),
                        monkeypatch=monkeypatch)
        assert b.cash() == 0.0
        assert b.last_cash_ok is False


class Market:
    def __init__(self):
        self.price = 100.0

    def quote(self, symbol):
        p = self.price
        return Quote(symbol, p, [p - 1, p + 1] * 29 + [p - 1, p], "yahoo")


class FlakyCashBroker(PaperBroker):
    def __init__(self, store):
        super().__init__(store, starting_cash=300.0, fee=OPTION_FEE)
        self.last_cash_ok = False

    def cash(self):
        return 0.0                            # what TR returns when the read failed


def open_call(store):
    return store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                             iv=0.4, contracts=10.0, entry_premium=2.0, genome_id=None,
                             tp_premium=1e9, sl_premium=0.0)


class TestRecording:
    def test_failed_cash_read_records_no_point(self, settings, store):
        e = Engine(settings, store, FlakyCashBroker(store), market=Market(), tr_derivatives=None)
        e.run_cycle()
        assert store.equity_curve() == []

    def test_fast_check_adds_points_with_fresh_marks(self, settings, store):
        e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
                   market=Market(), tr_derivatives=None)
        e.EQUITY_POINT_S = 0
        open_call(store)
        e._mark_position = lambda o, spot: 2.0
        e.run_cycle()
        n = len(store.equity_curve())
        e._mark_position = lambda o, spot: 2.37
        e._check_exits_fast()
        curve = store.equity_curve()
        assert len(curve) == n + 1
        assert curve[-1]["equity"] - curve[-2]["equity"] == pytest.approx(3.7)   # 10 x 0.37

    def test_fast_points_throttled(self, settings, store):
        e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
                   market=Market(), tr_derivatives=None)
        e.EQUITY_POINT_S = 3600
        open_call(store)
        e._mark_position = lambda o, spot: 2.0
        e.run_cycle()
        n = len(store.equity_curve())
        e._check_exits_fast()
        e._check_exits_fast()
        assert len(store.equity_curve()) == n


class TestLiveCurveView:
    def test_live_view_hides_zero_cash_spikes(self, settings):
        live = Store(settings.live_db_path)
        for cash, equity in [(177.77, 299.9), (0.0, 0.0), (0.0, 122.2), (177.77, 300.1)]:
            live.record_equity(cash, equity, 0.0)
            time.sleep(0.01)
        live.close()
        ControlState(settings.control_path).set_mode("live")
        curve = TestClient(create_app(settings)).get("/api/equity").json()
        assert [round(p["equity"], 1) for p in curve] == [299.9, 300.1]

    def test_paper_view_keeps_everything(self, settings):
        paper = Store(settings.db_path)
        for cash, equity in [(300.0, 300.0), (0.0, 280.0)]:
            paper.record_equity(cash, equity, 0.0)
            time.sleep(0.01)
        paper.close()
        assert len(TestClient(create_app(settings)).get("/api/equity").json()) == 2
