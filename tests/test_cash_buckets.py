"""Cash buckets: a fixed reserve for the user's manual actions, a dip budget
for averaging down held positions, the rest for new entries.

- budget_split.reserve_eur: never used by the bot.
- budget_split.dips_eur: only for dip buys (adding to a held position when its
  underlying dipped >= dips.k_sigma below the intraday SMA while the fused
  signal still backs the position). Money spent on dip buys of positions that
  are still open counts against it; it is released when they close.
- Entries get what is left.
Dip buys go through the same live guards as every order (market hours,
Claude sign-off). Written before implementation (TDD). Offline."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import BudgetConfig, DipConfig, Settings
from lmtrade.core.budget import split_cash
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


class TestSplit:
    def test_normal(self):
        assert split_cash(300.0, 50.0, 50.0) == {"entry": 200.0, "dips": 50.0, "reserve": 50.0}

    def test_low_cash_fills_reserve_first_then_dips(self):
        assert split_cash(70.0, 50.0, 50.0) == {"entry": 0.0, "dips": 20.0, "reserve": 50.0}
        assert split_cash(30.0, 50.0, 50.0) == {"entry": 0.0, "dips": 0.0, "reserve": 30.0}

    def test_spent_dips_release_cash_to_nobody_else(self):
        # 30 of the 50 dip budget is in open positions: 20 left for dips
        assert split_cash(300.0, 50.0, 50.0, dip_spent=30.0) == \
            {"entry": 230.0, "dips": 20.0, "reserve": 50.0}

    def test_overspent_and_zero_or_negative_cash(self):
        assert split_cash(300.0, 50.0, 50.0, dip_spent=80.0)["dips"] == 0.0
        assert split_cash(0.0, 50.0, 50.0) == {"entry": 0.0, "dips": 0.0, "reserve": 0.0}
        assert split_cash(-5.0, 50.0, 50.0) == {"entry": 0.0, "dips": 0.0, "reserve": 0.0}

    def test_no_buckets_everything_is_entry(self):
        assert split_cash(120.0, 0.0, 0.0) == {"entry": 120.0, "dips": 0.0, "reserve": 0.0}


class TestConfig:
    def test_code_defaults_off_yaml_on(self):
        import yaml
        assert BudgetConfig().reserve_eur == 0.0 and BudgetConfig().dips_eur == 0.0
        assert DipConfig().enabled is False
        assert DipConfig().max_fee_pct == pytest.approx(0.03)
        cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text(
            encoding="utf-8"))
        assert cfg["budget_split"] == {"reserve_eur": 50, "dips_eur": 50}
        assert cfg["dips"]["enabled"] is True

    def test_editable(self):
        from lmtrade.web.settings_editor import EDITABLE
        keys = {k for k, _, _ in EDITABLE}
        assert {"budget_split.reserve_eur", "budget_split.dips_eur", "dips.enabled", "dips.k_sigma",
                "dips.min_confidence", "dips.max_fee_pct", "dips.max_adds"} <= keys


# ------------------------------------------------------------ engine
BASE = [99.0, 101.0] * 30          # SMA 100, sigma 1


class Market:
    def __init__(self, price=100.0):
        self.price = price

    def quote(self, s):
        return Quote(s, self.price, BASE[:-3] + [self.price] * 3, "yahoo")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.options.min_hold_hours = 0
    s.options.flip_exit_confidence = 0.0
    s.options.max_option_fraction = 1.0
    s.options.cash_reserve_pct = 0.0
    s.research.commentary_minutes = 0
    s.tr.market_hours = ""
    s.budget_split.reserve_eur = 50.0
    s.budget_split.dips_eur = 50.0
    s.dips.enabled = True
    s.dips.k_sigma = 1.0
    s.dips.min_confidence = 0.6
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store, broker=None, price=97.0, decision=("buy", 0.8)):
    market = Market(price)
    broker = broker or PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE)
    e = Engine(settings, store, broker, market=market, tr_derivatives=None)
    e._mark_position = lambda o, spot: spot / 50.0          # 97 -> 1.94 per cert
    e._decide = lambda q: (Decision(q.symbol, decision[0], decision[1], "forced"), None)
    store.open_option("AAPL", "ko_call", strike=80.0, expiry_ts=time.time() + 3e7, iv=0.0,
                      contracts=50.0, entry_premium=2.0, genome_id=None,
                      instrument_type="knockout", barrier=80.0, ratio=10.0,
                      isin="DE000KO0001")
    return e, market


def dip(e, market):
    q = market.quote("AAPL")
    e._dip_buys({"AAPL": q}, {"AAPL"})


class TestEntryBudget:
    def test_entries_only_get_cash_minus_reserve_and_dips(self, settings, store):
        e, market = make(settings, store, price=100.0)
        for o in store.open_options():
            store.close_option(o["id"], 2.0, 0.0)
        d = Decision("AAPL", "buy", 1.0, "x")
        _, ceiling = e._budget_bounds(market.quote("AAPL"), d, None)
        assert ceiling == pytest.approx(200.0 - OPTION_FEE)


class TestDipBuysPaper:
    def test_adds_to_the_dipped_position_and_averages_the_entry(self, settings, store):
        e, market = make(settings, store)
        dip(e, market)
        o = store.open_options()[0]
        # floor(50 / 1.94) = 25 certificates @ 1.94 (48.50 EUR, fee 2.1% <= 3%)
        assert o["contracts"] == pytest.approx(75.0)
        assert o["entry_premium"] == pytest.approx((50 * 2.0 + 25 * 1.94) / 75)
        assert e._cash() == pytest.approx(300.0 - 25 * 1.94 - OPTION_FEE)
        adds = store.get_meta("dip_adds")[str(o["id"])]
        assert len(adds) == 1 and adds[0]["size"] == 25
        assert any("DIP" in a["summary"] for a in store.recent_activity(20))

    def test_bucket_shrinks_and_entries_unaffected(self, settings, store):
        e, market = make(settings, store)
        dip(e, market)
        b = e._buckets()
        assert b["dips"] == pytest.approx(50.0 - 25 * 1.94)
        assert b["reserve"] == pytest.approx(50.0)

    def test_released_when_the_position_closes(self, settings, store):
        e, market = make(settings, store)
        dip(e, market)
        o = store.open_options()[0]
        store.close_option(o["id"], 2.0, 0.0)
        assert e._buckets()["dips"] == pytest.approx(50.0)

    def test_max_adds_per_position(self, settings, store):
        settings.budget_split.dips_eur = 200.0
        e, market = make(settings, store)
        dip(e, market)
        dip(e, market)
        assert len(store.get_meta("dip_adds")["1"]) == 1

    @pytest.mark.parametrize("price,decision", [
        (99.8, ("buy", 0.8)),      # no dip (z ~ -0.2)
        (97.0, ("buy", 0.4)),      # signal too weak
        (97.0, ("sell", 0.9)),     # signal against the position
        (97.0, ("hold", 0.0)),
    ])
    def test_no_add_without_dip_and_support(self, settings, store, price, decision):
        e, market = make(settings, store, price=price, decision=decision)
        dip(e, market)
        assert store.open_options()[0]["contracts"] == 50.0

    def test_no_add_when_position_is_in_profit(self, settings, store):
        # averaging DOWN: only into a losing position
        e, market = make(settings, store)
        e._mark_position = lambda o, spot: 2.5
        dip(e, market)
        assert store.open_options()[0]["contracts"] == 50.0

    def test_no_add_close_to_the_barrier(self, settings, store):
        settings.dips.min_barrier_distance = 0.20      # 97 is only 17.5% above 80
        e, market = make(settings, store)
        dip(e, market)
        assert store.open_options()[0]["contracts"] == 50.0

    def test_fee_limit_and_empty_bucket(self, settings, store):
        settings.budget_split.dips_eur = 20.0            # 10 certs = 19.40, fee 5% > 3%
        e, market = make(settings, store)
        dip(e, market)
        assert store.open_options()[0]["contracts"] == 50.0
        settings.budget_split.dips_eur = 0.0
        dip(e, market)
        assert store.open_options()[0]["contracts"] == 50.0

    def test_disabled(self, settings, store):
        settings.dips.enabled = False
        e, market = make(settings, store)
        dip(e, market)
        assert store.open_options()[0]["contracts"] == 50.0


class LiveBroker:
    armed = True
    fee = 1.0
    mode = "live"

    def __init__(self, store):
        self._p = PaperBroker(store, starting_cash=300.0, fee=1.0)
        self.orders = []

    def __getattr__(self, name):
        return getattr(self._p, name)

    def place_order(self, isin, side, size, exchange="LSX", **kw):
        self.orders.append((isin, side, size, exchange))
        return SimpleNamespace(ok=True, message="ok", order_id="o1", price=1.95)


class TestDipBuysLive:
    def live(self, settings, store, reply='{"approve": true, "reason": "ok"}'):
        from lmtrade.core.signoff import SignOff
        settings.signoff.enabled = True
        b = LiveBroker(store)
        e, market = make(settings, store, broker=b)
        store.set_meta("ko_refs", {"1": {"price": 2.0, "spot": 100.0, "size": 0.1,
                                         "currency": "EUR", "exchange": "SGL"}})
        e.signoff = SignOff(settings, caller=lambda p: reply)
        return e, market, b

    def test_real_order_on_the_certificates_exchange_at_fill_price(self, settings, store):
        e, market, b = self.live(settings, store)
        dip(e, market)
        assert b.orders == [("DE000KO0001", "buy", 25.0, "SGL")]
        o = store.open_options()[0]
        assert o["entry_premium"] == pytest.approx((50 * 2.0 + 25 * 1.95) / 75)

    def test_sign_off_rejection_blocks(self, settings, store):
        e, market, b = self.live(settings, store, '{"approve": false, "reason": "no"}')
        dip(e, market)
        assert b.orders == [] and store.open_options()[0]["contracts"] == 50.0

    def test_market_closed_blocks(self, settings, store):
        settings.tr.market_hours = "08:00-22:00"
        e, market, b = self.live(settings, store)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        sat = datetime(2026, 10, 10, 12, tzinfo=ZoneInfo("Europe/Berlin")).timestamp()
        e.now = lambda: sat
        dip(e, market)
        assert b.orders == []


class TestDashboard:
    def test_summary_shows_the_split_and_js_paints_it(self, tmp_path):
        from fastapi.testclient import TestClient

        from lmtrade.web.app import create_app
        s = Settings(mode="paper", budget=300.0, universe=["AAPL"])
        s.data_dir = tmp_path
        s.budget_split.reserve_eur = 50.0
        s.budget_split.dips_eur = 50.0
        st = Store(s.db_path)
        st.set_meta("cash", 120.0)
        st.close()
        b = TestClient(create_app(s)).get("/api/summary").json()["cash_buckets"]
        assert b == {"entry": 20.0, "dips": 50.0, "reserve": 50.0}
        js = (Path(__file__).parents[1] / "src" / "lmtrade" / "web" / "static"
              / "app.js").read_text(encoding="utf-8")
        assert "cash_buckets" in js
