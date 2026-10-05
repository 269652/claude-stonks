"""Live fill reconciliation against the real TR portfolio.

Live incident: a resting JNJ knockout order FILLED at TR, the bot missed it
(order-status format unverified), then "cancelled" it when the signal flipped
— TR accepted the cancel without error — and forgot it. On restart the
portfolio sync imported the certificate as a bare equity row: no price, no
P&L, no TP/SL. Now every placed order is remembered (order_history), and any
remembered certificate showing up in the TR portfolio is booked / adopted as
a managed knockout position. Written before implementation (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.base import OrderResult
from lmtrade.brokers.paper import PaperBroker
from lmtrade.brokers.tr_derivatives import TRDerivativeQuote
from lmtrade.config import Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Position, Store
from lmtrade.data.market import Quote

ISIN = "DE000FC2KLH7"
BASE = [99.0, 101.0] * 30


class Market:
    def __init__(self):
        self.price = 100.0

    def quote(self, symbol):
        return Quote(symbol, self.price, BASE[:-1] + [self.price], "yahoo")


class FakeTR(PaperBroker):
    mode = "live"

    def __init__(self, store):
        super().__init__(store, starting_cash=300.0, fee=OPTION_FEE)
        self.armed = True
        self.placed, self.cancelled = [], []
        self.status = {"state": "open", "fill_price": None}

    def cash(self):
        return 300.0

    def place_order(self, isin, side, size, exchange="LSX"):
        return OrderResult(True, isin, side, size, 0.0, 1.0, "ok", order_id="M1")

    def place_limit_order(self, isin, side, size, limit, exchange="LSX"):
        self.placed.append((isin, size, limit, exchange))
        return OrderResult(True, isin, side, size, limit, 1.0, "ok", order_id="L1")

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True                      # TR accepts even when already filled

    def order_status(self, order_id):
        return self.status


class FakeDerivs:
    def __init__(self):
        self.holdings: list[dict] | None = []

    def available(self):
        return True

    def _ko(self, underlying, kind):
        return TRDerivativeQuote(isin=ISIN if kind == "ko_call" else "DE000PUT0001",
                                 underlying=underlying, kind=kind,
                                 strike=80.0 if kind == "ko_call" else 120.0,
                                 barrier=80.0 if kind == "ko_call" else 120.0, ratio=10.0,
                                 price=4.5, leverage=5.0, currency="EUR", exchange="SGL")

    def find_knockout(self, underlying, direction, spot, leverage):
        return self._ko(underlying, "ko_call" if direction == "buy" else "ko_put")

    def search(self, underlying, direction):
        return [self._ko(underlying, "ko_call"), self._ko(underlying, "ko_put")]

    def portfolio(self):
        return self.holdings

    def account_cash(self):
        return None


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["JNJ"],
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
    s.tr.market_hours = ""             # always open: these tests run at any hour
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


@pytest.fixture()
def live(settings, store, monkeypatch):
    from lmtrade.core import control as control_mod
    monkeypatch.setattr(control_mod.ControlState, "low_balance_blocks", lambda self, nw: False)
    tr, derivs, market = FakeTR(store), FakeDerivs(), Market()
    e = Engine(settings, store, tr, market=market, tr_derivatives=derivs)
    e.PORTFOLIO_CHECK_S = 0               # no throttling in tests
    e._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "forced"), None)
    return e, tr, derivs


def opts(store):
    return store.open_options()


class TestOrderHistory:
    def test_placed_order_is_remembered(self, live, store):
        e, tr, _ = live
        e.run_cycle()
        rec = store.get_meta("order_history")[ISIN]
        assert rec["underlying"] == "JNJ" and rec["direction"] == "buy"
        assert rec["instrument"]["exchange"] == "SGL"
        assert rec["target_spot"] < 100.0


class TestFillFromPortfolio:
    def test_fill_seen_in_portfolio_is_booked_even_if_status_says_open(self, live, store):
        e, tr, derivs = live
        e.run_cycle()
        derivs.holdings = [{"isin": ISIN, "size": 27.0, "avg_price": 4.527}]
        e.run_cycle()
        o = opts(store)
        assert len(o) == 1 and o[0]["isin"] == ISIN
        assert o[0]["contracts"] == pytest.approx(27.0)
        assert o[0]["entry_premium"] == pytest.approx(4.527)
        assert o[0]["tp_premium"] and o[0]["sl_premium"]          # exits set
        assert (store.get_meta("pending_entries") or {}) == {}

    def test_cancel_of_an_already_filled_order_books_it(self, live, store):
        e, tr, derivs = live
        e.run_cycle()
        derivs.holdings = [{"isin": ISIN, "size": 27.0, "avg_price": 4.527}]
        e._decide = lambda q: (Decision(q.symbol, "hold", 0.0, "forced"), None)
        e.run_cycle()
        assert len(opts(store)) == 1
        assert tr.cancelled == []          # filled: nothing to cancel

    def test_portfolio_unavailable_changes_nothing(self, live, store):
        e, tr, derivs = live
        e.run_cycle()
        derivs.holdings = None
        e.run_cycle()
        assert opts(store) == []
        assert "JNJ" in (store.get_meta("pending_entries") or {})


class TestAdoptImportedRow:
    def seed(self, store, with_instrument=False):
        rec = {"underlying": "JNJ", "direction": "buy", "target_spot": 99.0,
               "confidence": 0.8, "genome_id": None, "placed_ts": time.time()}
        if with_instrument:
            rec["instrument"] = {"instrument_type": "knockout", "kind": "ko_call",
                                 "strike": 80.0, "barrier": 80.0, "ratio": 10.0, "isin": ISIN,
                                 "expiry_ts": time.time() + 3e7, "iv": 0.0, "size": 0.1,
                                 "currency": "EUR", "exchange": "SGL"}
        store.set_meta("order_history", {ISIN: rec})
        store.upsert_position(Position(ISIN, 27.0, 4.527, 0.0))

    def test_imported_equity_row_adopted_as_knockout(self, live, store):
        e, tr, derivs = live
        self.seed(store, with_instrument=True)
        derivs.holdings = [{"isin": ISIN, "size": 27.0, "avg_price": 4.527}]
        e._decide = lambda q: (Decision(q.symbol, "hold", 0.0, "forced"), None)
        e.run_cycle()
        assert store.position(ISIN) is None
        o = opts(store)
        assert len(o) == 1 and o[0]["underlying"] == "JNJ" and o[0]["kind"] == "ko_call"
        assert e._mark_position(o[0], 99.0) == pytest.approx(4.527)   # anchored at fill

    def test_history_without_instrument_resolved_via_search(self, live, store):
        e, tr, derivs = live
        self.seed(store, with_instrument=False)
        derivs.holdings = [{"isin": ISIN, "size": 27.0, "avg_price": 4.527}]
        e._decide = lambda q: (Decision(q.symbol, "hold", 0.0, "forced"), None)
        e.run_cycle()
        o = opts(store)
        assert len(o) == 1 and o[0]["isin"] == ISIN and o[0]["strike"] == 80.0

    def test_unknown_certificate_left_alone(self, live, store):
        e, tr, derivs = live
        store.upsert_position(Position("DE000OTHER01", 5.0, 1.0, 0.0))
        derivs.holdings = [{"isin": "DE000OTHER01", "size": 5.0, "avg_price": 1.0}]
        e._decide = lambda q: (Decision(q.symbol, "hold", 0.0, "forced"), None)
        e.run_cycle()
        assert store.position("DE000OTHER01") is not None
        assert opts(store) == []

    def test_dashboard_row_of_adopted_position_has_exits(self, live, store, settings):
        from fastapi.testclient import TestClient

        from lmtrade.core.control import ControlState
        from lmtrade.web.app import create_app
        e, tr, derivs = live
        self.seed(store, with_instrument=True)
        derivs.holdings = [{"isin": ISIN, "size": 27.0, "avg_price": 4.527}]
        e._decide = lambda q: (Decision(q.symbol, "hold", 0.0, "forced"), None)
        e.run_cycle()
        assert opts(store)[0]["tp_premium"] > 4.527 > opts(store)[0]["sl_premium"]
