"""Manual close from the dashboard: the web app queues a close request, the
engine executes it at the current price through the normal exit bookkeeping
(cash, exit fee, fee-inclusive P&L, trade record), and picks requests up
within seconds, not just once per cycle. Armed live behaviour (real TR
sell orders) is covered in test_live_exits.py. Written before implementation
(TDD). Offline: market is faked."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.web.app import create_app

HIST = [99.0, 101.0] * 30


class Market:
    def __init__(self, source="yahoo"):
        self.source = source

    def quote(self, symbol: str) -> Quote:
        return Quote(symbol, 100.0, HIST[:-1] + [100.0], self.source)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1        # no new entries: isolate the close
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store, market=None, broker=None):
    broker = broker or PaperBroker(store, starting_cash=settings.budget, fee=1.0)
    return Engine(settings, store, broker, market=market or Market(), tr_derivatives=None)


def open_call(store, contracts=10.0, entry=2.0):
    return store.open_option("AAPL", "call", strike=100.0,
                             expiry_ts=time.time() + 5 * 86400, iv=0.4,
                             contracts=contracts, entry_premium=entry, genome_id=None,
                             tp_premium=1e9, sl_premium=0.0)


def queue(store, **req):
    store.set_meta("close_requests", (store.get_meta("close_requests") or []) + [req])


class TestEngineManualClose:
    def test_option_closed_at_current_mark_with_fees(self, settings, store, monkeypatch):
        e = make(settings, store)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 3.0)
        oid = open_call(store)
        cash = e.broker.cash()
        queue(store, kind="option", id=oid)
        assert e.process_close_requests() == 1
        assert store.open_options() == []
        closed = store.closed_options()[0]
        assert closed["pnl"] == pytest.approx((3.0 - 2.0) * 10 - 2 * OPTION_FEE)
        assert e.broker.cash() == pytest.approx(cash + 3.0 * 10 - OPTION_FEE)
        assert "manual" in store.recent_trades(5)[0]["reason"]
        assert not store.get_meta("close_requests")

    def test_equity_position_sold(self, settings, store):
        e = make(settings, store)
        assert e.broker.buy("AAPL", 1.0, 100.0).ok
        queue(store, kind="equity", symbol="AAPL")
        assert e.process_close_requests() == 1
        assert store.position("AAPL") is None

    def test_unknown_or_already_closed_request_is_dropped(self, settings, store):
        e = make(settings, store)
        queue(store, kind="option", id=999)
        assert e.process_close_requests() == 0
        assert not store.get_meta("close_requests")

    def test_no_trustworthy_price_keeps_request_pending(self, settings, store):
        e = make(settings, store, market=Market(source="synthetic"))
        oid = open_call(store)
        queue(store, kind="option", id=oid)
        assert e.process_close_requests() == 0
        assert len(store.open_options()) == 1
        assert store.get_meta("close_requests") == [{"kind": "option", "id": oid}]

    def test_requests_processed_during_idle_wait(self, settings, store, monkeypatch):
        e = make(settings, store)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 2.0)
        oid = open_call(store)
        queue(store, kind="option", id=oid)
        e._idle(1.2)
        assert store.open_options() == []

    def test_cycle_processes_requests(self, settings, store, monkeypatch):
        e = make(settings, store)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 2.0)
        oid = open_call(store)
        queue(store, kind="option", id=oid)
        e.run_cycle()
        assert store.open_options() == []


class TestCloseApi:
    @pytest.fixture()
    def client(self, settings):
        st = Store(settings.db_path)
        self.oid = open_call(st)
        st.close()
        return TestClient(create_app(settings))

    def test_summary_rows_carry_close_handle(self, client):
        rows = client.get("/api/summary").json()["positions"]
        assert rows[0]["id"] == self.oid
        assert rows[0]["close_pending"] is False

    def test_post_queues_request(self, client, settings):
        r = client.post("/api/positions/close", json={"kind": "option", "id": self.oid})
        assert r.status_code == 200 and r.json()["queued"] is True
        rows = client.get("/api/summary").json()["positions"]
        assert rows[0]["close_pending"] is True
        st = Store(settings.db_path)
        try:
            assert st.get_meta("close_requests") == [{"kind": "option", "id": self.oid}]
        finally:
            st.close()

    def test_duplicate_request_not_queued_twice(self, client, settings):
        for _ in range(2):
            client.post("/api/positions/close", json={"kind": "option", "id": self.oid})
        st = Store(settings.db_path)
        try:
            assert len(st.get_meta("close_requests")) == 1
        finally:
            st.close()

    @pytest.mark.parametrize("body", [{}, {"kind": "option"}, {"kind": "equity"},
                                      {"kind": "bogus", "id": 1}])
    def test_invalid_body_rejected(self, client, body):
        assert client.post("/api/positions/close", json=body).status_code == 400


class TestCloseButtonWiring:
    def _read(self, *parts):
        return (Path(__file__).parents[1].joinpath("src", "lmtrade", "web", *parts)
                .read_text(encoding="utf-8"))

    def test_positions_table_posts_close_requests(self):
        js = self._read("static", "app.js")
        assert "/api/positions/close" in js
        assert "close-btn" in js

    def test_positions_header_has_action_column(self):
        html = self._read("templates", "dashboard.html")
        head = html[html.index("<th>Symbol</th><th>Type</th>"):]
        head = head[:head.index("</tr>")]
        assert head.count("<th") == 10   # + Exits and Price columns
