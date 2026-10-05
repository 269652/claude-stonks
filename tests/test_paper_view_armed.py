"""Switching the dashboard to PAPER no longer disarms live trading: the badge
selects the VIEW, `armed` alone decides whether the engine trades real money.
The dashboard asks whether to disarm when switching to paper (and can
remember the answer); while viewing paper with live armed it says so.
Written before implementation (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import lmtrade.cli as cli
from lmtrade.config import Settings
from lmtrade.core.control import ControlState
from lmtrade.web.app import create_app


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    return s


def armed_then_paper(settings, disarm=False) -> ControlState:
    c = ControlState.load(settings.control_path)
    c.set_mode("live")
    c.arm(confirm=True)
    c.set_mode("paper", disarm=disarm)
    return c


class TestEngineKeepsTradingLive:
    def test_paper_view_with_live_armed_trades_the_live_book(self, settings, monkeypatch):
        monkeypatch.setenv("TR_PHONE", "+49123")
        monkeypatch.setenv("TR_PIN", "1234")
        engine, store = cli._build_engine_for_control(settings, armed_then_paper(settings))
        assert store.db_path == settings.live_db_path
        assert engine.broker.mode == "live" and engine.broker.armed is True
        store.close()

    def test_paper_view_after_disarm_trades_paper(self, settings):
        engine, store = cli._build_engine_for_control(
            settings, armed_then_paper(settings, disarm=True))
        assert store.db_path == settings.db_path
        assert engine.broker.mode == "paper"
        store.close()

    def test_view_switch_alone_does_not_rebuild_the_engine(self, settings):
        c = ControlState.load(settings.control_path)
        c.set_mode("live")
        c.arm(confirm=True)
        key = cli._control_key(c)
        c.set_mode("paper")
        assert cli._control_key(c) == key           # same trading state: no rebuild
        c.disarm()
        assert cli._control_key(c) != key


class TestModeApi:
    def test_switch_to_paper_keeps_armed(self, settings):
        c = ControlState.load(settings.control_path)
        c.set_mode("live")
        c.arm(confirm=True)
        r = TestClient(create_app(settings)).post("/api/control/mode", json={"mode": "paper"})
        body = r.json()
        assert body["mode"] == "paper" and body["armed"] is True and body["live_armed"] is True

    def test_switch_to_paper_with_disarm(self, settings):
        c = ControlState.load(settings.control_path)
        c.set_mode("live")
        c.arm(confirm=True)
        body = TestClient(create_app(settings)).post(
            "/api/control/mode", json={"mode": "paper", "disarm": True}).json()
        assert body["armed"] is False and body["live_armed"] is False


class TestDashboardPrompt:
    def test_prompt_remember_and_banner_wired(self):
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        js = (web / "static" / "app.js").read_text(encoding="utf-8")
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        assert 'id="paper-switch-overlay"' in html and 'id="ps-remember"' in html
        assert "paperSwitchChoice" in js                 # remembered choice
        assert "disarm: " in js                          # sent with the mode switch
        assert "still armed" in js                       # banner while viewing paper
