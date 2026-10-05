"""Inline TP/SL editing in the Positions row: clicking a TP/SL value turns it
into an input; saving posts the position's full exit settings with just that
field changed (empty = off). The request body is built by a pure function in
web/static/exits.js, exercised through Node (skipped without Node).
Written before implementation (TDD)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).parents[1] / "src" / "lmtrade" / "web"
EXITS_JS = WEB / "static" / "exits.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def run(expr: str):
    script = (f"const X = require({json.dumps(str(EXITS_JS))});"
              f"process.stdout.write(JSON.stringify({expr}));")
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


ROW = {"id": 7, "tp_pnl": 78.0, "sl_pnl": -66.0,
       "trail": {"enabled": True, "min_pnl": 20.0, "pct": 0.15}}


@needs_node
class TestInlineExitBody:
    def test_changes_only_the_edited_field(self):
        body = run(f'X.inlineExitBody({json.dumps(ROW)}, "tp_pnl", "40")')
        assert body == {"id": 7, "tp_pnl": 40.0, "sl_pnl": -66.0, "trail_enabled": True,
                        "trail_min_profit_eur": 20.0, "trail_pct": 0.15}

    def test_sl_edit(self):
        body = run(f'X.inlineExitBody({json.dumps(ROW)}, "sl_pnl", "-25.5")')
        assert body["sl_pnl"] == -25.5 and body["tp_pnl"] == 78.0

    def test_empty_turns_exit_off(self):
        assert run(f'X.inlineExitBody({json.dumps(ROW)}, "tp_pnl", "  ")')["tp_pnl"] is None

    def test_not_a_number_is_rejected(self):
        assert run(f'X.inlineExitBody({json.dumps(ROW)}, "tp_pnl", "abc")') is None

    def test_unknown_field_is_rejected(self):
        assert run(f'X.inlineExitBody({json.dumps(ROW)}, "trail_pct", "1")') is None

    def test_missing_trail_info_defaults_off(self):
        row = {"id": 3, "tp_pnl": 10.0, "sl_pnl": -5.0}
        body = run(f'X.inlineExitBody({json.dumps(row)}, "tp_pnl", "12")')
        assert body["trail_enabled"] is False


class TestInlineWiring:
    def test_scripts_and_hooks(self):
        js = (WEB / "static" / "app.js").read_text(encoding="utf-8")
        assert "inline-exit" in js and "inlineExitBody" in js
        assert "_inlineEditing" in js            # repaint paused while editing
        assert "TP off" in js and "SL off" in js

    def test_dashboard_loads_exits_js_cache_busted(self, tmp_path):
        from fastapi.testclient import TestClient

        from lmtrade.config import Settings
        from lmtrade.web.app import create_app

        s = Settings(mode="paper", budget=100.0, universe=["AAPL"])
        s.data_dir = tmp_path
        html = TestClient(create_app(s)).get("/").text
        assert "/static/exits.js?v=" in html
        assert html.index("exits.js") < html.index("app.js")
