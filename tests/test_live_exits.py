"""Live exits: in ARMED live mode every exit (take-profit, stop-loss, stale
eviction, manual close) must send a real TR sell order for the position's
ISIN and only book the close once TR confirms it. Before this, exits only
updated the local records while the certificate stayed open at TR.
Written before implementation (TDD). Offline: TR is a fake broker."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.brokers.base import OrderResult
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote

HIST = [99.0, 101.0] * 30
ISIN = "DE000KO12345"


class Market:
    def __init__(self, price=100.0):
        self.price = price

    def quote(self, symbol: str) -> Quote:
        p = self.price
        return Quote(symbol, p, [p - 1.0, p + 1.0] * 29 + [p - 1.0, p], "yahoo")


class FakeTR(PaperBroker):
    """Armed live broker double: records real-order calls."""
    mode = "live"

    def __init__(self, store, ok=True):
        super().__init__(store, starting_cash=300.0, fee=1.0)
        self.armed = True
        self.ok = ok
        self.orders: list[tuple] = []

    def place_order(self, isin, side, size, exchange="LSX"):
        self.orders.append((isin, side, size))
        msg = "live order placed (ORD-1)" if self.ok else "TR rejected order: [x]"
        return OrderResult(self.ok, isin, side, size, 0.0, 1.0, msg)


class Clock:
    def __init__(self):
        self.t = time.time() + 5.0   # positions are opened after the engine is built

    def __call__(self):
        return self.t


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1          # no entries: isolate exits
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.options.min_hold_hours = 0
    s.tr.market_hours = ""             # always open: these tests run at any hour
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store, broker, market=None, clock=None):
    return Engine(settings, store, broker, market=market or Market(),
                  tr_derivatives=None, now=clock or Clock())


def open_ko(store, isin=ISIN, barrier=80.0, tp=1e9, sl=0.0, opened_ago_h=0.0):
    oid = store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=time.time() + 365 * 86400,
                            iv=0.0, contracts=10.0, entry_premium=2.0, genome_id=None,
                            tp_premium=tp, sl_premium=sl, instrument_type="knockout",
                            barrier=barrier, ratio=0.1, isin=isin)
    if opened_ago_h:
        store._conn.execute("UPDATE option_positions SET opened_ts = ? WHERE id = ?",
                            (time.time() - opened_ago_h * 3600, oid))
        store._conn.commit()
    return oid


def warns(store) -> str:
    return " ".join(r["message"] for r in store.recent_logs(100))


class TestAutomaticExitsLive:
    def test_take_profit_sends_real_sell_then_books(self, settings, store, monkeypatch):
        tr = FakeTR(store)
        e = make(settings, store, tr)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 3.0)
        open_ko(store, tp=2.5)
        e.run_cycle()
        assert tr.orders == [(ISIN, "sell", 10.0)]
        assert store.open_options() == []
        assert "live sell" in store.recent_trades(5)[0]["reason"]

    def test_rejected_sell_keeps_position_open(self, settings, store, monkeypatch):
        tr = FakeTR(store, ok=False)
        e = make(settings, store, tr)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 3.0)
        open_ko(store, tp=2.5)
        e.run_cycle()
        assert len(tr.orders) == 1
        assert len(store.open_options()) == 1
        assert "LIVE exit" in warns(store) and "rejected" in warns(store)

    def test_failed_sell_backs_off_then_retries(self, settings, store, monkeypatch):
        tr = FakeTR(store, ok=False)
        clock = Clock()
        e = make(settings, store, tr, clock=clock)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 3.0)
        open_ko(store, tp=2.5)
        e.run_cycle()
        e.run_cycle()                       # within backoff: no second order
        assert len(tr.orders) == 1
        clock.t += 5 * 60 + 1
        tr.ok = True
        e.run_cycle()
        assert len(tr.orders) == 2
        assert store.open_options() == []

    def test_knocked_out_books_loss_without_order(self, settings, store):
        tr = FakeTR(store)
        e = make(settings, store, tr, market=Market(price=79.0))   # below barrier
        open_ko(store, barrier=80.0)
        e.run_cycle()
        assert tr.orders == []
        closed = store.closed_options()
        assert closed and closed[0]["exit_premium"] == 0.0

    def test_missing_isin_closes_locally_with_warning(self, settings, store, monkeypatch):
        tr = FakeTR(store)
        e = make(settings, store, tr)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 3.0)
        open_ko(store, isin=None, tp=2.5)
        e.run_cycle()
        assert tr.orders == []
        assert store.open_options() == []
        assert "no ISIN" in warns(store)

    def test_stale_eviction_sells_live(self, settings, store, monkeypatch):
        settings.loop.stale_evict_enabled = True
        settings.loop.stale_hours = 1
        tr = FakeTR(store)
        e = make(settings, store, tr)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 2.0)   # flat
        open_ko(store, opened_ago_h=2)
        e.run_cycle()
        assert tr.orders == [(ISIN, "sell", 10.0)]
        assert store.open_options() == []

    def test_unarmed_never_sends_orders(self, settings, store, monkeypatch):
        tr = FakeTR(store)
        tr.armed = False
        e = make(settings, store, tr)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 3.0)
        open_ko(store, tp=2.5)
        e.run_cycle()
        assert tr.orders == []
        assert store.open_options() == []


class TestManualCloseLive:
    def test_manual_close_sells_live(self, settings, store):
        tr = FakeTR(store)
        e = make(settings, store, tr)
        oid = open_ko(store)
        store.set_meta("close_requests", [{"kind": "option", "id": oid}])
        assert e.process_close_requests() == 1
        assert tr.orders == [(ISIN, "sell", 10.0)]
        assert store.open_options() == []

    def test_manual_close_ignores_backoff(self, settings, store, monkeypatch):
        tr = FakeTR(store, ok=False)
        e = make(settings, store, tr)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: 3.0)
        oid = open_ko(store, tp=2.5)
        e.run_cycle()                          # automatic attempt fails -> backoff
        tr.ok = True
        store.set_meta("close_requests", [{"kind": "option", "id": oid}])
        assert e.process_close_requests() == 1  # user click retries right away
        assert len(tr.orders) == 2

    def test_rejected_manual_close_drops_request_and_warns(self, settings, store):
        tr = FakeTR(store, ok=False)
        e = make(settings, store, tr)
        oid = open_ko(store)
        store.set_meta("close_requests", [{"kind": "option", "id": oid}])
        assert e.process_close_requests() == 0
        assert len(store.open_options()) == 1
        assert not store.get_meta("close_requests")
        assert "rejected" in warns(store)

    def test_manual_equity_close_refused_live(self, settings, store):
        tr = FakeTR(store)
        e = make(settings, store, tr)
        assert PaperBroker.buy(tr, "AAPL", 1.0, 100.0).ok
        store.set_meta("close_requests", [{"kind": "equity", "symbol": "AAPL"}])
        assert e.process_close_requests() == 0
        assert store.position("AAPL") is not None
        assert tr.orders == []
        assert "TR app" in warns(store)
