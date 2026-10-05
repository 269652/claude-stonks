"""Knockout pricing anchored to TR's real quote: a certificate moves 1:1 with
the underlying per `size` (TR's underlying-per-certificate), converted from
the underlying's currency to EUR. Limit prices and the marks of booked
knockouts both use it, so a fresh position is marked at its fill price (no
phantom P&L that could trip TP / SL) and USD / CHF underlyings are priced in
EUR. Written before implementation (TDD). Offline: market is faked."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


class Market:
    def __init__(self, fx=None, source="yahoo"):
        self.fx = fx or {}          # "EURUSD=X" -> rate
        self.source = source
        self.price = 100.0

    def quote(self, symbol: str) -> Quote:
        if symbol in self.fx:
            r = self.fx[symbol]
            return Quote(symbol, r, [r] * 60, self.source)
        p = self.price
        return Quote(symbol, p, [p - 1.0, p + 1.0] * 29 + [p - 1.0, p], self.source)


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


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def engine(settings, store, market):
    return Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
                  market=market, tr_derivatives=None)


def inst(kind="ko_call", currency="EUR", ref_price=2.0, ref_spot=100.0, size=0.1):
    level = 80.0 if kind == "ko_call" else 120.0      # barrier on the losing side
    return {"instrument_type": "knockout", "kind": kind, "strike": level, "barrier": level,
            "ratio": 1 / size, "isin": "DE000KO1", "expiry_ts": time.time() + 3e7, "iv": 0.0,
            "ref_price": ref_price, "ref_spot": ref_spot, "size": size, "currency": currency}


class TestAnchoredInstrumentPrice:
    def test_call_and_put_move_with_underlying_per_size(self, settings, store):
        e = engine(settings, store, Market())
        assert e._instrument_price(inst("ko_call"), 99.0) == pytest.approx(1.9)
        assert e._instrument_price(inst("ko_put", ref_price=2.0), 99.0) == pytest.approx(2.1)

    def test_converted_from_underlying_currency(self, settings, store):
        e = engine(settings, store, Market(fx={"EURUSD=X": 1.25}))
        # 1 USD move * 0.1 size = 0.10 USD = 0.08 EUR
        assert e._instrument_price(inst(currency="USD"), 99.0) == pytest.approx(1.92)

    def test_no_fx_rate_means_no_price(self, settings, store):
        e = engine(settings, store, Market())               # no EURUSD quote available
        e.market.quote = lambda s: Quote(s, 1.1, [1.1] * 60, "synthetic")
        assert e._instrument_price(inst(currency="USD"), 99.0) is None

    def test_knocked_out_is_worthless(self, settings, store):
        e = engine(settings, store, Market())
        assert e._instrument_price(inst("ko_call"), 79.0) == 0.0


class TestAnchoredMarks:
    def book(self, e, store, kind="ko_call", currency="EUR", price=1.87, spot=100.0):
        p = {"direction": "buy" if kind == "ko_call" else "sell", "confidence": 0.8,
             "genome_id": None, "instrument": inst(kind, currency, price, spot),
             "limit": price, "size": 50.0, "notional": 50 * price, "target_spot": spot,
             "placed_ts": time.time(), "order_id": None}
        e._book_limit_fill("AAPL", p, price, spot=spot)
        return store.open_options()[0]

    def test_fresh_position_marked_at_fill_price(self, settings, store):
        e = engine(settings, store, Market())
        o = self.book(e, store)
        assert e._mark_position(o, 100.0) == pytest.approx(1.87)
        assert e._mark_position(o, 101.0) == pytest.approx(1.97)

    def test_usd_underlying_marked_in_eur(self, settings, store):
        e = engine(settings, store, Market(fx={"EURUSD=X": 1.25}))
        o = self.book(e, store, currency="USD")
        assert e._mark_position(o, 102.5) == pytest.approx(1.87 + 0.2)

    def test_refs_dropped_on_close(self, settings, store):
        e = engine(settings, store, Market())
        o = self.book(e, store)
        e._book_option_close(o, 1.9, "test")
        assert str(o["id"]) not in (store.get_meta("ko_refs") or {})

    def test_legacy_knockout_without_ref_uses_old_model(self, settings, store):
        e = engine(settings, store, Market())
        oid = store.open_option("AAPL", "ko_call", 80.0, time.time() + 3e7, 0.0, 10.0, 2.0,
                                None, instrument_type="knockout", barrier=80.0, ratio=10.0,
                                isin="DE000OLD")
        o = [x for x in store.open_options() if x["id"] == oid][0]
        assert e._mark_position(o, 100.0) > 0
