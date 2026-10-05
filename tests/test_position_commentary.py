"""Live position commentary: every research.commentary_minutes (default 5)
while positions are open, Claude writes ~300 characters on how they are
developing; the dashboard shows it in an info box above the positions.
Written before implementation (TDD). Offline: the LLM caller is faked."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import ResearchConfig, Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.research.commentary import build_prompt, clean_commentary
from lmtrade.web.app import create_app

ROWS = [{"symbol": "JNJ", "direction": "long", "entry": 4.527, "price": 4.61,
         "pnl_eur": 1.24, "pnl_pct": 1.8, "tp": 6.79, "sl": 2.72,
         "trail": "arms at +23.53", "held_min": 95, "spot": 254.3, "spot_move_pct": -0.4}]


class TestPure:
    def test_prompt_mentions_each_position_and_limit(self):
        p = build_prompt(ROWS, max_chars=300)
        assert "JNJ" in p and "long" in p and "+1.24" in p and "300" in p

    def test_clean_collapses_and_truncates(self):
        assert clean_commentary("  JNJ  is\n\n holding.  ", 300) == "JNJ is holding."
        out = clean_commentary("x" * 500, 300)
        assert len(out) <= 300 and out.endswith("…")

    def test_clean_empty_is_none(self):
        assert clean_commentary("   ", 300) is None
        assert clean_commentary(None, 300) is None

    def test_config_defaults_and_editable(self):
        c = ResearchConfig()
        assert c.commentary_minutes == 5
        assert c.commentary_model == "claude-opus-5-5"
        from lmtrade.web.settings_editor import EDITABLE
        keys = {k for k, _, _ in EDITABLE}
        assert {"research.commentary_minutes", "research.commentary_model"} <= keys


class Market:
    def quote(self, symbol):
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
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def engine(settings, store, caller):
    e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
               market=Market(), tr_derivatives=None)
    e.commentary_caller = caller
    e.commentary_async = False            # run inline in tests
    return e


def open_call(store):
    return store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                             iv=0.4, contracts=10.0, entry_premium=2.0, genome_id=None,
                             tp_premium=3.0, sl_premium=1.2)


class TestEngine:
    def test_writes_commentary_for_open_positions(self, settings, store):
        prompts = []
        e = engine(settings, store, lambda p: (prompts.append(p), "AAPL call is flat, "
                                               "holding near entry.")[1])
        open_call(store)
        e._mark_position = lambda o, spot: 2.1
        e.run_cycle()
        e._position_commentary_tick()
        c = store.get_meta("position_commentary")
        assert c["text"] == "AAPL call is flat, holding near entry."
        assert c["ts"] > 0 and "AAPL" in prompts[0]

    def test_throttled_to_interval(self, settings, store):
        calls = []
        e = engine(settings, store, lambda p: (calls.append(1), "ok")[1])
        open_call(store)
        e._position_commentary_tick()
        e._position_commentary_tick()
        assert len(calls) == 1

    def test_no_positions_no_call_and_cleared(self, settings, store):
        calls = []
        store.set_meta("position_commentary", {"text": "old", "ts": 1.0})
        e = engine(settings, store, lambda p: (calls.append(1), "x")[1])
        e._position_commentary_tick()
        assert calls == [] and store.get_meta("position_commentary") is None

    def test_caller_failure_keeps_previous_text(self, settings, store):
        def boom(p):
            raise RuntimeError("cli down")
        store.set_meta("position_commentary", {"text": "earlier view", "ts": 1.0})
        e = engine(settings, store, boom)
        open_call(store)
        e._position_commentary_tick()                     # must not raise
        c = store.get_meta("position_commentary")
        assert c["text"] == "earlier view" and "cli down" in c.get("error", "")

    def test_disabled(self, settings, store):
        settings.research.commentary_minutes = 0
        calls = []
        e = engine(settings, store, lambda p: (calls.append(1), "x")[1])
        open_call(store)
        e._position_commentary_tick()
        assert calls == []


class TestDashboard:
    def test_summary_carries_commentary_and_box_wired(self, settings, store):
        store.set_meta("position_commentary", {"text": "Both positions drift up.", "ts": 5.0})
        s = TestClient(create_app(settings)).get("/api/summary").json()
        assert s["position_commentary"]["text"] == "Both positions drift up."
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        js = (web / "static" / "app.js").read_text(encoding="utf-8")
        pos = html.index('<section id="tab-positions"')
        assert html.index('id="pos-commentary"', pos) < html.index('<tbody id="positions">', pos)
        assert "position_commentary" in js


class TestVerdict:
    """The commentary opens with a verdict (fine / watch / going badly) and
    the prompt explains what an un-armed trailing stop means."""

    def test_prompt_asks_for_verdict_and_explains_trailing(self):
        from lmtrade.research.commentary import build_prompt
        p = build_prompt(ROWS, 300)
        assert "Fine" in p and "Watch" in p and "Going badly" in p
        assert "not active yet" in p

    @pytest.mark.parametrize("text,verdict", [
        ("Fine: both positions drift up.", "fine"),
        ("Watch — JNJ nears its stop-loss.", "watch"),
        ("Going badly: both falling toward SL.", "bad"),
        ("No clear verdict here.", None),
    ])
    def test_parse_verdict(self, text, verdict):
        from lmtrade.research.commentary import parse_verdict
        assert parse_verdict(text) == verdict

    def test_engine_stores_verdict(self, settings, store):
        e = engine(settings, store, lambda p: "Watch: JNJ drifting toward its stop.")
        open_call(store)
        e._position_commentary_tick()
        assert store.get_meta("position_commentary")["verdict"] == "watch"

    def test_box_colored_by_verdict(self):
        web = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
        html = (web / "templates" / "dashboard.html").read_text(encoding="utf-8")
        js = (web / "static" / "app.js").read_text(encoding="utf-8")
        assert ".pos-commentary.v-bad" in html and ".pos-commentary.v-fine" in html
        assert "v-" in js and "verdict" in js
