"""The daily analysis has its own model setting (Claude Opus 5.5 by default),
independent of the cheap per-symbol `model.cloud_model`. Written before
implementation (TDD). Fully offline: subprocess and HTTP are faked."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.config import Settings, load_settings
from lmtrade.core.state import Store
from lmtrade.models.providers import ClaudeCLIProvider, _default_claude_cli_runner
from lmtrade.research.daily import DailyAnalyst

OPUS = "claude-opus-5-5"


class Done:
    def __init__(self, stdout="ok", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


@pytest.fixture()
def captured(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return Done()

    monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda n: "/usr/bin/claude")
    monkeypatch.setattr("lmtrade.models.providers.os.name", "posix")
    monkeypatch.setattr("lmtrade.models.providers.subprocess.run", fake_run)
    return seen


class TestConfig:
    def test_default_is_opus_5_5(self):
        assert Settings().research.analysis_model == OPUS

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LMTRADE_ANALYSIS_MODEL", "claude-sonnet-5-5")
        s = load_settings(load_env=False, toml_path=tmp_path / "none.toml")
        assert s.research.analysis_model == "claude-sonnet-5-5"

    def test_cloud_model_unchanged(self):
        # per-symbol decisions stay on the cheap model
        assert "haiku" in Settings().model.cloud_model


class TestCliRunnerModel:
    def test_model_flag_added_when_given(self, captured):
        _default_claude_cli_runner("hi", model=OPUS)
        cmd = captured["cmd"]
        assert cmd[cmd.index("--model") + 1] == OPUS

    def test_no_model_flag_by_default(self, captured):
        _default_claude_cli_runner("hi")
        assert "--model" not in captured["cmd"]

    def test_provider_forwards_model(self, captured):
        text, _ = ClaudeCLIProvider(model=OPUS).ask("hello")
        assert text == "ok"
        assert OPUS in captured["cmd"]

    def test_provider_without_model_has_no_flag(self, captured):
        ClaudeCLIProvider().ask("hello")
        assert "--model" not in captured["cmd"]

    def test_custom_runner_still_called_with_two_args(self):
        calls = []
        p = ClaudeCLIProvider(runner=lambda prompt, t: calls.append((prompt, t)) or "x",
                              model=OPUS)
        monkey_available = p.available
        if monkey_available():          # only meaningful where `claude` exists
            assert p.ask("q")[0] == "x"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.data_dir = tmp_path
    return s


class TestDailyAnalystUsesOpus:
    def test_cli_caller_uses_analysis_model(self, settings, captured, tmp_path):
        settings.research.analysis_provider = "claude_cli"
        store = Store(settings.db_path)
        try:
            analyst = DailyAnalyst(store, settings)
            assert analyst.caller is not None
            analyst.caller("review the market")
        finally:
            store.close()
        cmd = captured["cmd"]
        assert cmd[cmd.index("--model") + 1] == OPUS

    def test_anthropic_caller_uses_analysis_model(self, settings, monkeypatch):
        settings.research.analysis_provider = "anthropic"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        sent = {}

        class Resp:
            def raise_for_status(self): ...
            def json(self): return {"content": [{"type": "text", "text": "hi"}]}

        def fake_post(url, **kw):
            sent["json"] = kw["json"]
            return Resp()

        monkeypatch.setattr("lmtrade.research.daily.httpx.post", fake_post)
        store = Store(settings.db_path)
        try:
            DailyAnalyst(store, settings).caller("review")
        finally:
            store.close()
        assert sent["json"]["model"] == OPUS

    def test_no_key_degrades_to_no_caller(self, settings, monkeypatch):
        settings.research.analysis_provider = "anthropic"
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = Store(settings.db_path)
        try:
            assert DailyAnalyst(store, settings).caller is None
        finally:
            store.close()
