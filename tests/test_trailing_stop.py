"""Trailing take-profit with a guaranteed minimum: once a position's net
profit (after entry + exit fee) peaks at >= min_profit / (1 - pct), it is
closed as soon as profit falls pct below that peak — which by construction is
never below min_profit. Written before implementation (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import OptionsConfig, Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.finance.risk import trailing_exit


class TestTrailingRule:
    def test_inactive_below_activation(self):
        active, stop, hit = trailing_exit(peak=22.0, pnl=5.0, min_profit=20.0, pct=0.15)
        assert (active, hit) == (False, False)

    def test_activation_threshold_is_min_over_one_minus_pct(self):
        active, stop, _ = trailing_exit(peak=20.0 / 0.85, pnl=23.0, min_profit=20.0, pct=0.15)
        assert active and stop == pytest.approx(20.0)

    def test_stop_follows_peak(self):
        active, stop, hit = trailing_exit(peak=40.0, pnl=35.0, min_profit=20.0, pct=0.15)
        assert active and stop == pytest.approx(34.0) and not hit

    def test_hit_when_falling_to_stop(self):
        assert trailing_exit(peak=40.0, pnl=34.0, min_profit=20.0, pct=0.15)[2] is True
        assert trailing_exit(peak=40.0, pnl=10.0, min_profit=20.0, pct=0.15)[2] is True

    def test_stop_never_below_min_profit(self):
        _, stop, _ = trailing_exit(peak=23.6, pnl=23.0, min_profit=20.0, pct=0.15)
        assert stop >= 20.0

    def test_zero_pct_degenerates_to_fixed_floor(self):
        active, stop, hit = trailing_exit(peak=20.0, pnl=20.0, min_profit=20.0, pct=0.0)
        assert active and stop == pytest.approx(20.0) and hit

    def test_invalid_inputs_never_trigger(self):
        assert trailing_exit(peak=50.0, pnl=0.0, min_profit=20.0, pct=1.0) == (False, None, False)
        assert trailing_exit(peak=50.0, pnl=0.0, min_profit=0.0, pct=0.15) == (False, None, False)


class TestConfig:
    def test_code_defaults_off_with_example_values(self):
        c = OptionsConfig()
        assert c.trail_enabled is False
        assert c.trail_min_profit_eur == pytest.approx(20.0)
        assert c.trail_pct == pytest.approx(0.15)

    def test_editable_in_dashboard(self):
        from lmtrade.web.settings_editor import EDITABLE
        keys = {k for k, _, _ in EDITABLE}
        assert {"options.trail_enabled", "options.trail_min_profit_eur",
                "options.trail_pct"} <= keys


class Market:
    def quote(self, symbol: str) -> Quote:
        return Quote(symbol, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], "yahoo")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1           # no entries: isolate exits
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.options.min_hold_hours = 0
    s.options.trail_enabled = True
    s.options.trail_min_profit_eur = 20.0
    s.options.trail_pct = 0.15
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def engine(settings, store):
    return Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=1.0),
                  market=Market(), tr_derivatives=None)


def open_call(store):
    # 10 contracts @ 2.00, no fixed TP / SL in the way.
    return store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                             iv=0.4, contracts=10.0, entry_premium=2.0, genome_id=None,
                             tp_premium=1e9, sl_premium=0.0)


def step(e, mark):
    e._mark_position = lambda o, spot: mark
    e.run_cycle()


# net P&L = (mark - 2.0) * 10 - 2 * 1 EUR fee
class TestEngineTrailing:
    def test_closes_after_pullback_from_peak_at_or_above_minimum(self, settings, store):
        e = engine(settings, store)
        open_call(store)
        step(e, 4.6)                 # +24 net -> active, stop 20.4
        step(e, 4.3)                 # +21 net -> hold
        assert len(store.open_options()) == 1
        step(e, 4.2)                 # +20 net <= 20.4 -> close
        closed = store.closed_options()
        assert closed and closed[0]["pnl"] == pytest.approx(20.0)
        assert "trailing" in store.recent_trades(5)[0]["reason"]

    def test_not_active_below_activation_even_on_big_drop(self, settings, store):
        e = engine(settings, store)
        open_call(store)
        step(e, 4.4)                 # +22 net: below 23.53 activation
        step(e, 2.5)                 # +3 net: no trailing exit
        assert len(store.open_options()) == 1

    def test_peak_survives_engine_restart(self, settings, store):
        open_call(store)
        step(engine(settings, store), 5.0)       # +28 net -> stop 23.8
        step(engine(settings, store), 4.5)       # fresh engine: +23 net -> close
        assert store.closed_options()

    def test_disabled_does_nothing(self, settings, store):
        settings.options.trail_enabled = False
        e = engine(settings, store)
        open_call(store)
        step(e, 5.0)
        step(e, 3.0)
        assert len(store.open_options()) == 1

    def test_peak_state_cleaned_up_after_close(self, settings, store):
        e = engine(settings, store)
        oid = open_call(store)
        step(e, 5.0)
        step(e, 4.0)
        assert store.closed_options()
        assert str(oid) not in (store.get_meta("trail_peaks") or {})

    def test_marks_expose_trailing_state_for_dashboard(self, settings, store):
        e = engine(settings, store)
        oid = open_call(store)
        step(e, 5.0)                 # +28 net, stop 23.8
        m = store.get_meta("open_option_marks")[str(oid)]
        assert m["trail_active"] is True
        assert m["trail_stop_pnl"] == pytest.approx(23.8)
        assert m["peak_pnl"] == pytest.approx(28.0)


class TestPositionRowExits:
    """The Positions table shows each position's TP / SL (premium level and
    the net EUR result if hit) and its trailing take-profit state."""

    @pytest.fixture()
    def client(self, settings, store):
        from fastapi.testclient import TestClient

        from lmtrade.web.app import create_app
        # 10 @ 2.00, TP 3.00, SL 1.20; trailing armed with peak +28 -> stop +23.8
        self.oid = store.open_option("AAPL", "call", strike=100.0,
                                     expiry_ts=time.time() + 5 * 86400, iv=0.4,
                                     contracts=10.0, entry_premium=2.0, genome_id=None,
                                     tp_premium=3.0, sl_premium=1.2)
        store.set_meta("open_option_marks", {str(self.oid): {
            "mark_premium": 4.6, "value": 46.0, "unrealized_pnl": 25.0, "spot": 100.0,
            "peak_pnl": 28.0, "trail_active": True, "trail_stop_pnl": 23.8}})
        return TestClient(create_app(settings))

    def test_row_has_tp_sl_levels_and_net_eur(self, client):
        row = client.get("/api/summary").json()["positions"][0]
        assert row["tp_price"] == pytest.approx(3.0)
        assert row["tp_pnl"] == pytest.approx((3.0 - 2.0) * 10 - 2.0)
        assert row["sl_price"] == pytest.approx(1.2)
        assert row["sl_pnl"] == pytest.approx((1.2 - 2.0) * 10 - 2.0)

    def test_row_has_trailing_state(self, client):
        t = client.get("/api/summary").json()["positions"][0]["trail"]
        assert t["enabled"] is True
        assert t["arm_pnl"] == pytest.approx(20.0 / 0.85)
        assert t["active"] is True
        assert t["stop_pnl"] == pytest.approx(23.8)
        assert t["peak_pnl"] == pytest.approx(28.0)

    def test_trailing_disabled_reported(self, settings, client):
        settings.options.trail_enabled = False
        t = client.get("/api/summary").json()["positions"][0]["trail"]
        assert t["enabled"] is False and t["active"] is False

    def test_table_renders_exit_column(self):
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        js = (web / "static" / "app.js").read_text(encoding="utf-8")
        assert "exitsCell" in js and "tp_pnl" in js and "stop_pnl" in js
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        head = html[html.index("<th>Symbol</th><th>Type</th>"):]
        assert "<th>Exits</th>" in head[:head.index("</tr>")]
