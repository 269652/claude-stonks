"""TR trading hours for the certificates (issuer venues, e.g. Société
Générale): Mon-Fri inside tr.market_hours (Europe/Berlin). An exchangeClosed
rejection from TR also marks the market closed until the next scheduled open
(holidays). While closed and armed live, automatic exits wait (no overnight
sell retries on thin after-hours prints); the dashboard disables its close
buttons and notifies when the market opens again. Written before
implementation (TDD)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lmtrade.config import Settings, TRConfig
from lmtrade.core.market_hours import is_open, market_status, next_open

BER = ZoneInfo("Europe/Berlin")


def ts(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=BER).timestamp()


HOURS = "08:00-22:00"


class TestSchedule:
    def test_weekday_inside_and_outside(self):
        assert is_open(ts(2026, 10, 5, 12), HOURS)          # Monday noon
        assert not is_open(ts(2026, 10, 5, 22, 57), HOURS)  # live incident time
        assert not is_open(ts(2026, 10, 5, 7, 59), HOURS)
        assert is_open(ts(2026, 10, 5, 8, 0), HOURS)

    def test_weekend_closed(self):
        assert not is_open(ts(2026, 10, 10, 12), HOURS)     # Saturday

    def test_next_open_same_day_next_day_and_after_weekend(self):
        assert next_open(ts(2026, 10, 5, 7), HOURS) == ts(2026, 10, 5, 8)
        assert next_open(ts(2026, 10, 5, 23), HOURS) == ts(2026, 10, 6, 8)
        assert next_open(ts(2026, 10, 9, 23), HOURS) == ts(2026, 10, 12, 8)   # Fri -> Mon

    def test_empty_hours_means_always_open(self):
        assert is_open(ts(2026, 10, 10, 3), "")

    def test_bad_hours_string_degrades_to_open(self):
        assert is_open(ts(2026, 10, 10, 3), "nonsense")

    def test_status_with_learned_closure(self):
        now = ts(2026, 10, 5, 12)
        st = market_status(now, HOURS, closed_until=now + 3600)
        assert st["open"] is False and st["next_open"] == now + 3600
        st = market_status(now, HOURS, closed_until=now - 1)
        assert st["open"] is True and st["closes"] == ts(2026, 10, 5, 22)

    def test_default_config(self):
        assert TRConfig().market_hours == "08:00-22:00"
        from lmtrade.web.settings_editor import EDITABLE
        assert "tr.market_hours" in {k for k, _, _ in EDITABLE}


# ------------------------------------------------------------ engine wiring
class FakeLiveBroker:
    armed = True
    fee = 1.0
    mode = "live"

    def __init__(self, store, message="ok", ok=True):
        from lmtrade.brokers.paper import PaperBroker
        self._p = PaperBroker(store, starting_cash=300.0, fee=1.0)
        self.orders, self.ok, self.message = [], ok, message

    def __getattr__(self, name):
        return getattr(self._p, name)

    def place_order(self, isin, side, size, exchange="LSX", **kw):
        from types import SimpleNamespace
        self.orders.append((isin, side, size))
        return SimpleNamespace(ok=self.ok, message=self.message)


class Market:
    def __init__(self, price=100.0):
        self.price = price

    def quote(self, s):
        p = self.price
        return __import__("lmtrade.data.market", fromlist=["Quote"]).Quote(
            s, p, [p - 1, p + 1] * 29 + [p - 1, p], "yahoo")


@pytest.fixture()
def env(tmp_path: Path):
    from lmtrade.core.engine import Engine
    from lmtrade.core.state import Store
    s = Settings(mode="live", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.loop.exit_check_seconds = 0
    s.options.min_hold_hours = 0
    s.options.flip_exit_confidence = 0.0
    s.research.commentary_minutes = 0
    store = Store(s.db_path)
    yield s, store, Engine
    store.close()


def make(env, now, ok=True, message="ok"):
    s, store, Engine = env
    broker = FakeLiveBroker(store, ok=ok, message=message)
    e = Engine(s, store, broker, market=Market(), tr_derivatives=None)
    e.now = lambda: now
    e._mark_position = lambda o, spot: spot / 50.0
    store.open_option("AAPL", "call", strike=100.0, expiry_ts=now + 5 * 86400,
                      iv=0.4, contracts=25.0, entry_premium=2.0, genome_id=None,
                      tp_premium=1.5, sl_premium=0.0, isin="DE000TEST0001")
    with store._lock:   # opened an hour before the simulated clock
        store._conn.execute("UPDATE option_positions SET opened_ts = ?", (now - 3600,))
        store._conn.commit()
    return e, broker, store


class TestEngineGate:
    def test_no_automatic_live_exit_while_closed(self, env):
        e, broker, store = make(env, ts(2026, 10, 5, 22, 57))   # TP (1.5) is hit
        e._manage_options({"AAPL": 100.0}, {"AAPL"})
        assert broker.orders == [] and len(store.open_options()) == 1

    def test_exits_run_when_open(self, env):
        e, broker, store = make(env, ts(2026, 10, 5, 12))
        e._manage_options({"AAPL": 100.0}, {"AAPL"})
        assert broker.orders and store.open_options() == []

    def test_exchange_closed_rejection_learns_closure(self, env):
        now = ts(2026, 10, 5, 12)                                # holiday-like
        e, broker, store = make(env, now, ok=False,
                                message="{'code': 'exchangeClosed'} Exchange is closed")
        e._manage_options({"AAPL": 100.0}, {"AAPL"})
        assert len(broker.orders) == 1
        st = e._market_status()
        assert st["open"] is False and st["next_open"] == ts(2026, 10, 6, 8)
        e._manage_options({"AAPL": 100.0}, {"AAPL"})            # no retry while closed
        assert len(broker.orders) == 1

    def test_status_published_and_open_transition_announced(self, env):
        e, broker, store = make(env, ts(2026, 10, 5, 7, 50))
        e._publish_market_status()
        assert store.get_meta("market_status")["open"] is False
        e.now = lambda: ts(2026, 10, 5, 8, 1)
        e._publish_market_status()
        st = store.get_meta("market_status")
        assert st["open"] is True and st["opened_ts"] == pytest.approx(ts(2026, 10, 5, 8, 1))
        assert any("market open" in a["summary"].lower() for a in store.recent_activity(20))

    def test_paper_mode_is_never_gated(self, env):
        s, store, Engine = env
        from lmtrade.brokers.paper import PaperBroker
        e = Engine(s, store, PaperBroker(store, starting_cash=300.0), market=Market(),
                   tr_derivatives=None)
        e.now = lambda: ts(2026, 10, 10, 3)                     # Saturday night
        assert e._market_status()["open"] is True


# ------------------------------------------------------------ dashboard
class TestDashboard:
    def client(self, tmp_path, monkeypatch, now, armed=True):
        from fastapi.testclient import TestClient

        from lmtrade.core.control import ControlState
        from lmtrade.core.state import Store
        from lmtrade.web import app as webapp
        s = Settings(mode="live", budget=300.0, universe=["AAPL"])
        s.data_dir = tmp_path
        live = Store(s.live_db_path)
        live.open_option("AAPL", "ko_call", 90.0, now + 3e7, 0.0, 10.0, 2.0, None,
                         instrument_type="knockout", barrier=90.0, ratio=10.0,
                         isin="DE000TEST0001")
        live.close()
        c = ControlState(s.control_path)
        c.set_mode("live")
        if armed:
            assert c.arm(confirm=True)
        monkeypatch.setattr(webapp.time, "time", lambda: now)
        return TestClient(webapp.create_app(s))

    def test_closed_market_in_summary_and_rows_not_closable(self, tmp_path, monkeypatch):
        cl = self.client(tmp_path, monkeypatch, ts(2026, 10, 5, 23))
        s = cl.get("/api/summary").json()
        assert s["market"]["open"] is False
        assert s["market"]["next_open"] == ts(2026, 10, 6, 8)
        assert s["positions"][0]["close_blocked"]

    def test_open_market_rows_closable(self, tmp_path, monkeypatch):
        cl = self.client(tmp_path, monkeypatch, ts(2026, 10, 5, 12))
        s = cl.get("/api/summary").json()
        assert s["market"]["open"] is True and not s["positions"][0]["close_blocked"]

    def test_close_refused_while_closed(self, tmp_path, monkeypatch):
        cl = self.client(tmp_path, monkeypatch, ts(2026, 10, 5, 23))
        r = cl.post("/api/positions/close", json={"kind": "option", "id": 1})
        assert r.status_code == 409 and r.json()["queued"] is False
        assert "closed" in r.json()["error"]

    def test_unarmed_view_never_blocked(self, tmp_path, monkeypatch):
        cl = self.client(tmp_path, monkeypatch, ts(2026, 10, 5, 23), armed=False)
        assert cl.get("/api/summary").json()["market"]["open"] is True

    def test_js_disables_buttons_and_notifies_on_reopen(self):
        js = (Path(__file__).parents[1] / "src" / "lmtrade" / "web" / "static"
              / "app.js").read_text(encoding="utf-8")
        assert "close_blocked" in js and "Market closed" in js
        assert "Notification" in js and "Market open" in js


# ------------------------------------------------------------ paused while closed
class TestPausedWhileClosed:
    """While the venue is closed (armed live): no signal generation, no news /
    daily analysis, no position commentary — nothing to act on, so no LLM
    or research spend. Valuation, heartbeat and the status keep running."""

    def engine(self, env, now):
        s, store, Engine = env
        s.universe = ["AAPL", "MSFT"]
        s.loop.max_positions = 10
        s.research.commentary_minutes = 5
        e, broker, store = make(env, now)
        calls = {"decide": 0, "jobs": 0, "commentary": 0}

        def decide(q):
            calls["decide"] += 1
            from lmtrade.agents.fusion import Decision
            return Decision(q.symbol, "hold", 0.0, "x"), None
        e._decide = decide
        real_jobs = e._run_scheduled_jobs

        def jobs():
            calls["jobs"] += 1
            return None
        e._run_scheduled_jobs = jobs
        assert real_jobs

        def caller(prompt):
            calls["commentary"] += 1
            return "Fine: ok"
        e.commentary_caller = caller
        e.commentary_async = False
        return e, store, calls

    def test_closed_pauses_everything(self, env):
        e, store, calls = self.engine(env, ts(2026, 10, 5, 23))
        e.run_cycle()
        e._position_commentary_tick()
        assert calls == {"decide": 0, "jobs": 0, "commentary": 0}
        assert "paused" in " ".join(r["message"] for r in store.recent_logs(30)).lower()
        assert store.get_meta("market_status")["open"] is False

    def test_open_runs_them(self, env):
        e, store, calls = self.engine(env, ts(2026, 10, 5, 12))
        e._position_commentary_tick()          # before the cycle's take-profit closes it
        e.run_cycle()
        assert calls["decide"] >= 1 and calls["jobs"] == 1 and calls["commentary"] >= 1
