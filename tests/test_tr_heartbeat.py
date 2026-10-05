"""TR heartbeat: every tr.heartbeat_seconds the engine checks that TR answers
(a bounded cash read). When it doesn't, the session is dropped and re-logged
in immediately (bypassing the retry backoff); if that fails too (expired
saved session -> manual `pytr login` with 2FA), the dashboard shows it.
Written before implementation (TDD). Offline: TR is faked."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings, TRConfig
from lmtrade.core.control import ControlState
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.web.app import create_app
from test_tr_broker import FakeOrderApi


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 1.1
    s.learning.enabled = False
    s.tr.heartbeat_seconds = 0.0001
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


class TestConfig:
    def test_default_and_editable(self):
        assert TRConfig().heartbeat_seconds == pytest.approx(30.0)
        from lmtrade.web.settings_editor import EDITABLE
        assert "tr.heartbeat_seconds" in {k for k, _, _ in EDITABLE}


class Switchable(FakeOrderApi):
    """resume_websession() result can be changed between calls."""


class TestBrokerHeartbeat:
    def broker(self, store, settings, monkeypatch, api):
        monkeypatch.setenv("TR_PHONE", "+491234")
        monkeypatch.setenv("TR_PIN", "1234")
        from lmtrade.brokers.trade_republic import TradeRepublicBroker
        return TradeRepublicBroker(store, settings, armed=False, api_factory=lambda: api)

    def test_responsive(self, store, settings, monkeypatch):
        b = self.broker(store, settings, monkeypatch, Switchable())
        assert b.heartbeat() is True

    def test_unresponsive(self, store, settings, monkeypatch):
        b = self.broker(store, settings, monkeypatch, Switchable(resume_ok=False))
        assert b.heartbeat() is False

    def test_relogin_bypasses_backoff(self, store, settings, monkeypatch):
        api = Switchable(resume_ok=False)
        b = self.broker(store, settings, monkeypatch, api)
        assert b.heartbeat() is False          # failure -> 60s retry backoff armed
        api.resume_ok = True
        assert b.relogin() is True             # immediately, despite the backoff
        assert b.heartbeat() is True


class FakeClient:
    def __init__(self, alive=True, relogin_ok=True):
        self.alive, self.relogin_ok = alive, relogin_ok
        self.beats = 0
        self.relogins = 0

    def heartbeat(self):
        self.beats += 1
        return self.alive

    def relogin(self):
        self.relogins += 1
        if self.relogin_ok:
            self.alive = True
        return self.relogin_ok


class HBBroker(PaperBroker):
    def __init__(self, store, client):
        super().__init__(store, starting_cash=300.0, fee=OPTION_FEE)
        self._c = client

    def heartbeat(self):
        return self._c.heartbeat()

    def relogin(self):
        return self._c.relogin()


class Market:
    def quote(self, symbol):
        return Quote(symbol, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], "yahoo")


def engine(settings, store, client):
    return Engine(settings, store, HBBroker(store, client), market=Market(),
                  tr_derivatives=None)


class TestEngineHeartbeat:
    def test_responsive_no_relogin(self, settings, store):
        c = FakeClient()
        e = engine(settings, store, c)
        e._tr_heartbeat()
        assert c.beats == 1 and c.relogins == 0
        assert store.get_meta("tr_session")["ok"] is True

    def test_unresponsive_triggers_relogin(self, settings, store):
        c = FakeClient(alive=False)
        e = engine(settings, store, c)
        e._tr_heartbeat()
        assert c.relogins == 1
        assert store.get_meta("tr_session")["ok"] is True
        assert "re-logged in" in " ".join(r["message"] for r in store.recent_logs(50))

    def test_failed_relogin_reported(self, settings, store):
        c = FakeClient(alive=False, relogin_ok=False)
        e = engine(settings, store, c)
        e._tr_heartbeat()
        st = store.get_meta("tr_session")
        assert st["ok"] is False and "pytr login" in st["msg"]

    def test_throttled(self, settings, store):
        settings.tr.heartbeat_seconds = 3600
        c = FakeClient()
        e = engine(settings, store, c)
        e._tr_heartbeat()
        e._tr_heartbeat()
        assert c.beats == 1

    def test_disabled(self, settings, store):
        settings.tr.heartbeat_seconds = 0
        c = FakeClient()
        engine(settings, store, c)._tr_heartbeat()
        assert c.beats == 0

    def test_no_tr_client_is_noop(self, settings, store):
        e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
                   market=Market(), tr_derivatives=None)
        e._tr_heartbeat()
        assert store.get_meta("tr_session") is None

    def test_runs_during_idle(self, settings, store):
        c = FakeClient()
        e = engine(settings, store, c)
        e._idle(1.2)
        assert c.beats >= 1


class TestDashboard:
    def test_summary_exposes_session_and_banner_wired(self, settings):
        live = Store(settings.live_db_path)
        live.set_meta("tr_session", {"ok": False, "ts": 1.0, "msg": "run pytr login"})
        live.close()
        ControlState(settings.control_path).set_mode("live")
        s = TestClient(create_app(settings)).get("/api/summary").json()
        assert s["tr_session"]["ok"] is False
        js = (Path(__file__).parents[1] / "src" / "lmtrade" / "web" / "static"
              / "app.js").read_text(encoding="utf-8")
        assert "tr_session" in js and "TR connection" in js


class TestHeartbeatTimestamps:
    def test_sent_and_received_recorded(self, settings, store):
        c = FakeClient()
        e = engine(settings, store, c)
        e._tr_heartbeat()
        st = store.get_meta("tr_session")
        assert st["ok"] is True and st["sent_ts"] <= st["ts"]
        assert st["interval"] == pytest.approx(settings.tr.heartbeat_seconds)

    def test_failure_keeps_sent_ts(self, settings, store):
        c = FakeClient(alive=False, relogin_ok=False)
        e = engine(settings, store, c)
        e._tr_heartbeat()
        st = store.get_meta("tr_session")
        assert st["ok"] is False and st["sent_ts"] > 0


import json
import shutil
import subprocess

CHART_JS = Path(__file__).parents[1] / "src" / "lmtrade" / "web" / "static" / "chart.js"
NODE = shutil.which("node")


def js_state(trs, now):
    script = (f"const C = require({json.dumps(str(CHART_JS))});"
              f"process.stdout.write(JSON.stringify(C.heartbeatState({json.dumps(trs)}, {now})));")
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node not installed")
class TestHeartbeatDotState:
    def test_none(self):
        assert js_state(None, 100) == "none"

    def test_ok(self):
        assert js_state({"ok": True, "ts": 95, "sent_ts": 94, "interval": 30}, 100) == "ok"

    def test_pending_while_waiting_for_answer(self):
        assert js_state({"ok": True, "ts": 60, "sent_ts": 99, "interval": 30}, 100) == "pending"

    def test_stale_when_no_answer_for_two_intervals(self):
        assert js_state({"ok": True, "ts": 20, "sent_ts": 19, "interval": 30}, 100) == "stale"

    def test_down(self):
        assert js_state({"ok": False, "ts": 99, "sent_ts": 99, "interval": 30}, 100) == "down"


class TestHeaderDotWired:
    def test_dot_has_id_and_is_driven_by_heartbeat(self):
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        js = (web / "static" / "app.js").read_text(encoding="utf-8")
        assert 'id="hb-dot"' in html and ".hb-blink" in html
        assert "heartbeatState" in js and "hb-blink" in js
