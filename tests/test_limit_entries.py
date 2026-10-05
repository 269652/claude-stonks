"""Exchange-side limit entries: a watched signal gets a resting limit BUY for
its instrument at the price that instrument would have when the underlying
reaches the entry trigger (intraday SMA -/+ k sigma). The order is re-priced
when that target drifts > reprice_drift, cancelled when the signal goes away
or the watch expires, and booked as a position only when it fills. Capital
goes to the signals closest to their entry first. TP / SL stay market orders.
Paper simulates fills; armed live uses real TR limit orders.
Written before implementation (TDD). Offline: market, TR and decisions faked."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.base import OrderResult
from lmtrade.brokers.paper import PaperBroker
from lmtrade.brokers.tr_derivatives import TRDerivativeQuote
from lmtrade.config import EntryConfig, Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.limit_orders import needs_reprice, target_spot
from lmtrade.core.state import Store
from lmtrade.data.market import Quote

BASE = [99.0, 101.0] * 30     # SMA 100, std 1


class TestPureHelpers:
    def test_target_spot_buy_below_sell_above(self):
        assert target_spot(BASE, 60, 1.0, "buy") == pytest.approx(99.0)
        assert target_spot(BASE, 60, 0.7, "sell") == pytest.approx(100.7)

    def test_target_spot_needs_data(self):
        assert target_spot([100.0] * 5, 60, 1.0, "buy") is None
        assert target_spot([100.0] * 60, 60, 1.0, "buy") is None   # flat: no sigma
        assert target_spot(BASE, 60, 1.0, "hold") is None

    def test_reprice_threshold(self):
        assert needs_reprice(2.00, 2.02, 0.005) is True
        assert needs_reprice(2.00, 2.005, 0.005) is False
        assert needs_reprice(0.0, 1.0, 0.005) is True

    def test_config_defaults(self):
        c = EntryConfig()
        assert c.limit_orders is False and c.reprice_drift == pytest.approx(0.005)


class Market:
    """Per-symbol price over a fixed SMA-100 / sigma-1 history; `shift` moves
    the whole history (and so the SMA / target)."""
    def __init__(self):
        self.price: dict[str, float] = {}
        self.shift = 0.0

    def quote(self, symbol: str) -> Quote:
        p = self.price.get(symbol, 100.0)
        hist = [x + self.shift for x in BASE[:-1]] + [p]
        return Quote(symbol, p, hist, "yahoo")


class Clock:
    def __init__(self):
        self.t = time.time() + 5

    def __call__(self):
        return self.t


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 0.5
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.loop.exit_check_seconds = 1e9            # keep fast checks out of the way
    s.sizing.vol_target_annual = 100.0
    s.sizing.regime_filter_enabled = False
    s.options.max_option_fraction = 2 / 3
    s.options.cash_reserve_pct = 1 / 3
    s.options.trail_enabled = False
    s.entry.watch_enabled = True
    s.entry.watch_k = 1.0
    s.entry.watch_window = 60
    s.entry.limit_orders = True
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store, broker=None, tr=None, clock=None):
    market, clock = Market(), clock or Clock()
    broker = broker or PaperBroker(store, starting_cash=settings.budget, fee=OPTION_FEE)
    e = Engine(settings, store, broker, market=market, tr_derivatives=tr, now=clock)
    return e, market, clock


def force(e, direction="buy", conf=0.9, only=None):
    def decide(q):
        if only is not None and q.symbol not in only:
            return Decision(q.symbol, "hold", 0.0, "forced"), None
        return Decision(q.symbol, direction, conf, "forced"), None
    e._decide = decide


def pending(store) -> dict:
    return store.get_meta("pending_entries") or {}


def logs(store) -> str:
    return "\n".join(r["message"] for r in store.recent_logs(300))


class TestPaperLimitEntries:
    def test_order_rests_at_fair_price_without_booking(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()
        p = pending(store)["AAPL"]
        assert p["size"] > 0 and p["limit"] > 0
        assert p["target_spot"] < 100.0                 # buy -> below the SMA
        assert p["size"] * p["limit"] >= 100.0 - 1e-6    # fee minimum (1 EUR <= 1%)
        assert store.open_options() == []
        assert e.broker.cash() == pytest.approx(300.0)   # nothing debited yet
        assert "[limit]" in logs(store)

    def test_fills_when_target_reached_and_books_position(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()
        p = pending(store)["AAPL"]
        market.price["AAPL"] = 98.5                      # below the 99 target
        e.run_cycle()
        opts = store.open_options()
        assert len(opts) == 1
        o = opts[0]
        assert o["contracts"] == pytest.approx(p["size"])
        assert o["entry_premium"] <= p["limit"] + 1e-9   # filled at or better than limit
        assert "AAPL" not in pending(store)
        cost = o["contracts"] * o["entry_premium"] + OPTION_FEE
        assert e.broker.cash() == pytest.approx(300.0 - cost)

    def test_sell_signal_buys_put_on_spike(self, settings, store):
        e, market, _ = make(settings, store)
        force(e, "sell")
        e.run_cycle()
        assert pending(store)["AAPL"]["target_spot"] > 100.0
        market.price["AAPL"] = 101.5
        e.run_cycle()
        assert store.open_options()[0]["kind"] == "put"

    def test_cancelled_when_signal_goes(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()
        force(e, "hold", 0.0)
        e.run_cycle()
        assert pending(store) == {}
        assert "cancel" in logs(store)

    def test_cancelled_when_watch_expires(self, settings, store):
        e, market, clock = make(settings, store)
        force(e)
        e.run_cycle()
        clock.t += 4 * 3600 + 1
        e.run_cycle()
        assert pending(store) == {}
        assert store.open_options() == []

    def test_repriced_on_drift_only(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()
        first = pending(store)["AAPL"]["limit"]
        market.shift = 0.01                              # ~0.01% move: keep
        market.price["AAPL"] = 100.01
        e.run_cycle()
        assert pending(store)["AAPL"]["limit"] == pytest.approx(first)
        market.shift = 3.0                               # SMA +3%: re-price
        market.price["AAPL"] = 103.0
        e.run_cycle()
        assert pending(store)["AAPL"]["limit"] != pytest.approx(first)

    def test_capital_goes_to_closest_signal_first(self, settings, store):
        settings.universe = ["AAPL", "MSFT"]
        settings.budget = 160.0                          # deployable ~106: one order
        e, market, _ = make(settings, store)
        market.price["AAPL"] = 100.8                     # far from 99 target
        market.price["MSFT"] = 99.3                      # close to 99 target
        force(e)
        e.run_cycle()
        assert list(pending(store)) == ["MSFT"]

    def test_disabled_falls_back_to_market_entry_on_trigger(self, settings, store):
        settings.entry.limit_orders = False
        e, market, _ = make(settings, store)
        force(e)
        market.price["AAPL"] = 98.5                      # already triggered
        e.run_cycle()
        assert len(store.open_options()) == 1
        assert pending(store) == {}


class FakeTR(PaperBroker):
    mode = "live"

    def __init__(self, store):
        super().__init__(store, starting_cash=300.0, fee=OPTION_FEE)
        self.armed = True
        self.placed: list[tuple] = []
        self.cancelled: list[str] = []
        self.status: dict | None = {"state": "open", "fill_price": None}
        self.cancel_ok = True
        self._n = 0

    def cash(self):
        return 300.0

    def place_order(self, isin, side, size, exchange="LSX"):
        return OrderResult(True, isin, side, size, 0.0, 1.0, "ok", order_id="M1")

    def place_limit_order(self, isin, side, size, limit, exchange="LSX"):
        self._n += 1
        self.placed.append((isin, side, size, limit))
        return OrderResult(True, isin, side, size, limit, 1.0, "ok", order_id=f"L{self._n}")

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return self.cancel_ok

    def order_status(self, order_id):
        return self.status


class FakeDerivs:
    def available(self):
        return True

    def find_knockout(self, underlying, direction, spot, leverage):
        kind = "ko_call" if direction == "buy" else "ko_put"
        strike = 80.0 if kind == "ko_call" else 120.0
        return TRDerivativeQuote(isin="DE000KO0001", underlying=underlying, kind=kind,
                                 strike=strike, barrier=strike, ratio=10.0,   # (S-K)/ratio ~ 2 EUR
                                 price=2.0, leverage=5.0)

    def account_cash(self):
        return None


@pytest.fixture()
def live(settings, store, monkeypatch):
    from lmtrade.core import control as control_mod
    monkeypatch.setattr(control_mod.ControlState, "low_balance_blocks", lambda self, nw: False)
    tr = FakeTR(store)
    e, market, clock = make(settings, store, broker=tr, tr=FakeDerivs())
    force(e)
    return e, market, tr


class TestLiveLimitEntries:
    def test_places_real_limit_with_isin_and_whole_size(self, live, store):
        e, market, tr = live
        e.run_cycle()
        assert len(tr.placed) == 1
        isin, side, size, limit = tr.placed[0]
        assert (isin, side) == ("DE000KO0001", "buy")
        assert size == int(size) and size >= 1
        assert pending(store)["AAPL"]["order_id"] == "L1"
        assert store.open_options() == []

    def test_books_only_on_confirmed_fill_at_fill_price(self, live, store):
        e, market, tr = live
        e.run_cycle()
        size = tr.placed[0][2]
        tr.status = {"state": "filled", "fill_price": 1.87}
        e.run_cycle()
        o = store.open_options()[0]
        assert o["isin"] == "DE000KO0001" and o["contracts"] == pytest.approx(size)
        assert o["entry_premium"] == pytest.approx(1.87)
        assert pending(store) == {}

    def test_unknown_status_books_nothing_and_keeps_order(self, live, store):
        e, market, tr = live
        e.run_cycle()
        tr.status = None
        e.run_cycle()
        assert store.open_options() == []
        assert "AAPL" in pending(store)

    def test_cancels_on_exchange_when_signal_goes(self, live, store):
        e, market, tr = live
        e.run_cycle()
        force(e, "hold", 0.0)
        e.run_cycle()
        assert tr.cancelled == ["L1"]
        assert pending(store) == {}

    def test_filled_while_cancelling_is_booked(self, live, store):
        e, market, tr = live
        e.run_cycle()
        force(e, "hold", 0.0)
        tr.cancel_ok = False
        tr.status = {"state": "filled", "fill_price": 1.9}
        e.run_cycle()
        assert len(store.open_options()) == 1
        assert pending(store) == {}

    def test_reprice_cancels_then_replaces(self, live, store):
        e, market, tr = live
        e.run_cycle()
        market.shift = 3.0
        market.price["AAPL"] = 103.0
        e.run_cycle()
        assert tr.cancelled == ["L1"]
        assert len(tr.placed) == 2
        assert pending(store)["AAPL"]["order_id"] == "L2"


class TestWatchlistShowsOrders:
    def test_api_rows_carry_resting_order(self, settings, store):
        from fastapi.testclient import TestClient

        from lmtrade.web.app import create_app
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()
        p = pending(store)["AAPL"]
        row = TestClient(create_app(settings)).get("/api/watchlist").json()[0]
        assert row["order"]["limit"] == pytest.approx(p["limit"])
        assert row["order"]["size"] == pytest.approx(p["size"])
        assert row["order"]["target_spot"] == pytest.approx(p["target_spot"])

    def test_tab_has_order_column(self):
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        head = html[html.index('<section id="tab-watchlist"'):]
        assert "<th>Order</th>" in head[:head.index("</tr>")]
