"""Vast.ai provisioning was removed: the bot runs locally (Claude CLI for the
LLM work). The generic GPU-cost economics (economics.gpu_usd_per_hour) stay.
Written before the removal (TDD)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_vast_module_and_script_removed():
    with pytest.raises(ImportError):
        importlib.import_module("lmtrade.infra.vast")
    assert not (ROOT / "scripts" / "deploy_vast.sh").exists()


def test_deploy_command_removed():
    from typer.testing import CliRunner

    from lmtrade.cli import app
    result = CliRunner().invoke(app, ["deploy"])
    assert result.exit_code != 0                     # no such command


def test_no_vast_mentions_left():
    hits = []
    for p in [*ROOT.glob("*.md"), ROOT / ".env.example", *ROOT.glob("config/*.yaml"),
              *ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*"), *ROOT.glob("notebooks/*.ipynb"),
              *ROOT.glob("docs/*.md")]:
        if p.is_file() and "vast" in p.read_text(encoding="utf-8", errors="ignore").lower():
            hits.append(str(p.relative_to(ROOT)))
    assert hits == []


def test_zero_gpu_rate_is_supported_for_local_runs():
    from lmtrade.config import Settings
    s = Settings()
    s.economics.gpu_usd_per_hour = 0.0
    from lmtrade.economics.cost_accounting import CostAccountant
    assert CostAccountant is not None and s.economics.gpu_usd_per_hour == 0.0
