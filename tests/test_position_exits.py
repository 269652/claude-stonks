"""Per-position exit configuration (TP / SL / trailing), editable after the
position is open via the dashboard's + button. Values are in net EUR (both
1 EUR fees out); None disables that exit. Positions without an override keep
the global config. Written before implementation (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import OptionsConfig, Settings
from lmtrade.core.engine import Engine
from lmtrade.core.exits import exit_plan, premium_for_pnl, validate_override
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.web.app import create_app

FEE = 1.0
POS = {"id": 1, "entry_premium": 2.0, "contracts": 10.0,
       "tp_premium": 3.0, "sl_premium": 1.2}


class TestPremiumConversion:
    def test_net_eur_to_premium(self):
        # +20 net on 10 contracts @ 2.00 with 2 EUR total fees -> 4.20
        assert premium_for_pnl(2.0, 10.0, 20.0, FEE) == pytest.approx(4.2)
        assert premium_for_pnl(2.0, 10.0, -12.0, FEE) == pytest.approx(1.0)


class TestExitPlan:
    def test_no_override_uses_position_and_config(self):
        cfg = OptionsConfig(trail_enabled=True)
        plan = exit_plan(POS, cfg, None, FEE)
        assert plan.tp == pytest.approx(3.0) and plan.sl == pytest.approx(1.2)
        assert plan.trail_enabled and plan.trail_min == 20.0 and plan.trail_pct == 0.15
        assert plan.custom is False

    def test_legacy_row_without_stops_falls_back_to_config_pct(self):
        cfg = OptionsConfig(take_profit_pct=0.5, stop_loss_pct=0.4)
        plan = exit_plan({**POS, "tp_premium": None, "sl_premium": None}, cfg, None, FEE)
        assert plan.tp == pytest.approx(3.0) and plan.sl == pytest.approx(1.2)

    def test_override_in_eur(self):
        ov = {"tp_pnl": 20.0, "sl_pnl": -12.0, "trail_enabled": True,
              "trail_min_profit_eur": 5.0, "trail_pct": 0.5}
        plan = exit_plan(POS, OptionsConfig(), ov, FEE)
        assert plan.tp == pytest.approx(4.2) and plan.sl == pytest.approx(1.0)
        assert plan.trail_enabled and plan.trail_min == 5.0 and plan.trail_pct == 0.5
        assert plan.custom is True

    def test_none_disables(self):
        ov = {"tp_pnl": None, "sl_pnl": None, "trail_enabled": False,
              "trail_min_profit_eur": 20.0, "trail_pct": 0.15}
        plan = exit_plan(POS, OptionsConfig(trail_enabled=True), ov, FEE)
        assert plan.tp is None and plan.sl is None and plan.trail_enabled is False


class TestValidateOverride:
    def ok(self, **kw):
        base = {"tp_pnl": 30.0, "sl_pnl": -10.0, "trail_enabled": True,
                "trail_min_profit_eur": 20.0, "trail_pct": 0.15}
        return validate_override(POS, {**base, **kw}, FEE)

    def test_valid(self):
        ov, err = self.ok()
        assert err is None and ov["tp_pnl"] == 30.0 and ov["sl_pnl"] == -10.0

    def test_nulls_allowed(self):
        ov, err = self.ok(tp_pnl=None, sl_pnl=None)
        assert err is None and ov["tp_pnl"] is None

    @pytest.mark.parametrize("kw", [
        {"sl_pnl": 5.0, "tp_pnl": 4.0},          # SL above TP
        {"sl_pnl": -40.0},                        # more than the whole position (2*10 + fees)
        {"trail_pct": 1.0}, {"trail_pct": -0.1},
        {"trail_min_profit_eur": 0.0},
        {"tp_pnl": "abc"}, {"trail_enabled": "yes"},
        {"tp_pnl": float("nan")},
    ])
    def test_invalid(self, kw):
        ov, err = self.ok(**kw)
        assert ov is None and err

    def test_trail_params_ignored_when_trailing_off(self):
        ov, err = self.ok(trail_enabled=False, trail_pct=5.0)
        assert err is None and ov["trail_enabled"] is False


class Market:
    def quote(self, symbol: str) -> Quote:
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
    s.options.min_hold_hours = 0
    s.options.trail_enabled = False
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def open_call(store, tp=3.0, sl=1.2):
    return store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                             iv=0.4, contracts=10.0, entry_premium=2.0, genome_id=None,
                             tp_premium=tp, sl_premium=sl)


def step(settings, store, mark):
    e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=FEE),
               market=Market(), tr_derivatives=None)
    e._mark_position = lambda o, spot: mark
    e.run_cycle()


def override(store, oid, **kw):
    base = {"tp_pnl": None, "sl_pnl": None, "trail_enabled": False,
            "trail_min_profit_eur": 20.0, "trail_pct": 0.15}
    store.set_meta("exit_overrides", {str(oid): {**base, **kw}})


class TestEngineUsesOverrides:
    def test_custom_tp_in_eur_closes(self, settings, store):
        oid = open_call(store, tp=100.0)              # column TP far away
        override(store, oid, tp_pnl=20.0)             # -> premium 4.20
        step(settings, store, 4.25)
        assert store.closed_options()

    def test_disabled_tp_never_closes(self, settings, store):
        oid = open_call(store)                        # column TP 3.00
        override(store, oid, tp_pnl=None)
        step(settings, store, 9.0)
        assert len(store.open_options()) == 1

    def test_disabled_sl_never_closes(self, settings, store):
        oid = open_call(store)
        override(store, oid, sl_pnl=None)
        step(settings, store, 0.5)
        assert len(store.open_options()) == 1

    def test_per_position_trailing_params(self, settings, store):
        oid = open_call(store, tp=100.0, sl=0.0)
        override(store, oid, trail_enabled=True, trail_min_profit_eur=5.0, trail_pct=0.5)
        step(settings, store, 3.2)                    # +10 net: active (>=10), stop 5
        step(settings, store, 2.65)                   # +4.5 net <= 5 -> close
        assert store.closed_options()

    def test_override_dropped_after_close(self, settings, store):
        oid = open_call(store, tp=100.0)
        override(store, oid, tp_pnl=20.0)
        step(settings, store, 5.0)
        assert str(oid) not in (store.get_meta("exit_overrides") or {})


class TestExitsApi:
    @pytest.fixture()
    def client(self, settings, store):
        self.oid = open_call(store)
        return TestClient(create_app(settings))

    def body(self, **kw):
        return {"id": self.oid, "tp_pnl": 25.0, "sl_pnl": -8.0, "trail_enabled": True,
                "trail_min_profit_eur": 10.0, "trail_pct": 0.2, **kw}

    def test_save_and_summary_reflects_it(self, client, store):
        r = client.post("/api/positions/exits", json=self.body())
        assert r.status_code == 200 and r.json()["ok"] is True
        row = client.get("/api/summary").json()["positions"][0]
        assert row["exits_custom"] is True
        assert row["tp_pnl"] == pytest.approx(25.0)
        assert row["sl_pnl"] == pytest.approx(-8.0)
        assert row["trail"]["min_pnl"] == 10.0 and row["trail"]["pct"] == 0.2

    def test_invalid_rejected(self, client):
        r = client.post("/api/positions/exits", json=self.body(sl_pnl=30.0))
        assert r.status_code == 400 and r.json()["ok"] is False

    def test_unknown_position_404(self, client):
        assert client.post("/api/positions/exits",
                           json=self.body(id=999)).status_code == 404

    def test_reset_restores_defaults(self, client):
        client.post("/api/positions/exits", json=self.body())
        r = client.post("/api/positions/exits", json={"id": self.oid, "reset": True})
        assert r.status_code == 200
        row = client.get("/api/summary").json()["positions"][0]
        assert row["exits_custom"] is False
        assert row["tp_price"] == pytest.approx(3.0)


class TestModalWiring:
    def test_modal_and_button_present(self):
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        js = (web / "static" / "app.js").read_text(encoding="utf-8")
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        assert 'id="exits-overlay"' in html
        assert "/api/positions/exits" in js and "exits-btn" in js


class TestExitsCellLayout:
    def test_pencil_button_beside_the_lines(self):
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        js = (web / "static" / "app.js").read_text(encoding="utf-8")
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        cell = js[js.index("function exitsCell"):js.index("function closeButton")]
        assert "exits-lines" in cell                     # lines in their own block
        assert '"✎ custom" : "+"' not in cell and "✎ custom" not in cell
        assert "✎" in cell                               # always a pencil
        assert ".exits-wrap" in html and "display:flex" in html[html.index(".exits-wrap"):][:120]
