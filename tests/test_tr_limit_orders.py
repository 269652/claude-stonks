"""TR limit-order primitives for exchange-side entries (brokers/trade_republic.py):
place a REAL limit order (armed only, TR must confirm with an order id),
cancel it, and read its status from the orders overview. Fully offline: the
pytr session is a fake. TR's orders-overview payload shape is not verified
against the real API here, so parsing is defensive and unknown -> None.
Written before implementation (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.config import Settings
from lmtrade.core.state import Store


class FakeLimitApi:
    def __init__(self, order_payload=None, overview=None, cancel_payload=None):
        self.order_payload = order_payload if order_payload is not None else {"orderId": "L1"}
        self.overview = overview if overview is not None else {"orders": []}
        self.cancel_payload = cancel_payload if cancel_payload is not None else {}
        self.limit_orders: list[dict] = []
        self.cancelled: list[str] = []
        self._last = None

    def resume_websession(self) -> bool:
        return True

    async def limit_order(self, isin, exchange, order_type, size, limit, expiry,
                          expiry_date=None, warnings_shown=None):
        self.limit_orders.append({"isin": isin, "exchange": exchange, "side": order_type,
                                  "size": size, "limit": limit, "expiry": expiry})
        self._last = ("sub-limit", self.order_payload)
        return "sub-limit"

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        self._last = ("sub-cancel", self.cancel_payload)
        return "sub-cancel"

    async def order_overview(self):
        self._last = ("sub-orders", self.overview)
        return "sub-orders"

    async def recv(self):
        sub, payload = self._last
        return (sub, {}, payload)

    async def unsubscribe(self, sub_id):
        pass


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="live", budget=100.0, universe=["AAPL"])
    s.data_dir = tmp_path
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.live_db_path)
    yield st
    st.close()


def broker(store, settings, monkeypatch, api, armed=True):
    monkeypatch.setenv("TR_PHONE", "+491234")
    monkeypatch.setenv("TR_PIN", "1234")
    from lmtrade.brokers.trade_republic import TradeRepublicBroker
    return TradeRepublicBroker(store, settings, armed=armed, api_factory=lambda: api)


class TestPlaceLimit:
    def test_places_gfd_limit_buy_and_returns_order_id(self, store, settings, monkeypatch):
        api = FakeLimitApi()
        res = broker(store, settings, monkeypatch, api).place_limit_order(
            "DE000KO1", "buy", 12, 2.345)
        assert res.ok and res.order_id == "L1"
        assert api.limit_orders == [{"isin": "DE000KO1", "exchange": "LSX", "side": "buy",
                                     "size": 12, "limit": 2.35, "expiry": "gfd"}]

    def test_unarmed_refuses(self, store, settings, monkeypatch):
        api = FakeLimitApi()
        res = broker(store, settings, monkeypatch, api, armed=False).place_limit_order(
            "DE000KO1", "buy", 12, 2.3)
        assert not res.ok and api.limit_orders == []

    def test_unconfirmed_is_failure(self, store, settings, monkeypatch):
        api = FakeLimitApi(order_payload={"warnings": []})
        res = broker(store, settings, monkeypatch, api).place_limit_order(
            "DE000KO1", "buy", 12, 2.3)
        assert not res.ok and res.order_id is None

    def test_rejected_is_failure(self, store, settings, monkeypatch):
        api = FakeLimitApi(order_payload={"errors": [{"errorCode": "X"}]})
        res = broker(store, settings, monkeypatch, api).place_limit_order(
            "DE000KO1", "buy", 12, 2.3)
        assert not res.ok

    @pytest.mark.parametrize("size,limit", [(0, 2.0), (3, 0.0), (-1, 2.0)])
    def test_invalid_size_or_limit_refused(self, store, settings, monkeypatch, size, limit):
        api = FakeLimitApi()
        res = broker(store, settings, monkeypatch, api).place_limit_order(
            "DE000KO1", "buy", size, limit)
        assert not res.ok and api.limit_orders == []


class TestCancel:
    def test_cancel_ok(self, store, settings, monkeypatch):
        api = FakeLimitApi()
        assert broker(store, settings, monkeypatch, api).cancel_order("L1") is True
        assert api.cancelled == ["L1"]

    def test_cancel_error_payload_is_false(self, store, settings, monkeypatch):
        api = FakeLimitApi(cancel_payload={"errors": [{"errorCode": "ORDER_NOT_FOUND"}]})
        assert broker(store, settings, monkeypatch, api).cancel_order("L1") is False


class TestOrderStatus:
    def status(self, store, settings, monkeypatch, overview, oid="L1"):
        return broker(store, settings, monkeypatch,
                      FakeLimitApi(overview=overview)).order_status(oid)

    def test_open(self, store, settings, monkeypatch):
        st = self.status(store, settings, monkeypatch,
                         {"orders": [{"id": "L1", "status": "open"}]})
        assert st == {"state": "open", "fill_price": None}

    def test_filled_with_price(self, store, settings, monkeypatch):
        st = self.status(store, settings, monkeypatch,
                         {"orders": [{"id": "L1", "status": "executed", "averagePrice": 2.31}]})
        assert st == {"state": "filled", "fill_price": 2.31}

    @pytest.mark.parametrize("raw", ["cancelled", "expired", "rejected"])
    def test_gone(self, store, settings, monkeypatch, raw):
        st = self.status(store, settings, monkeypatch,
                         {"orders": [{"id": "L1", "status": raw}]})
        assert st["state"] == "gone"

    def test_not_in_overview_is_unknown(self, store, settings, monkeypatch):
        assert self.status(store, settings, monkeypatch, {"orders": []}) is None

    def test_unrecognised_shape_is_unknown(self, store, settings, monkeypatch):
        assert self.status(store, settings, monkeypatch, "garbage") is None
