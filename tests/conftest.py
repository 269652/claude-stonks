"""Hermetic test environment: the suite must run offline with no keys, no matter
what the developer has in their real `.env` or shell (TR credentials, API keys,
LMTRADE_* overrides)."""
from __future__ import annotations

import os

import pytest

import lmtrade.config as config

_SECRET_VARS = (
    "TR_PHONE", "TR_PIN", "TR_LOCALE",
    "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path):
    # The developer's local config.toml (dashboard-edited overrides) must not
    # change what the suite sees; tests that need a toml pass their own path.
    monkeypatch.setattr(config, "DEFAULT_TOML_PATH", tmp_path / "no-config.toml")

    for name in list(os.environ):
        if name.startswith("LMTRADE_") or name in _SECRET_VARS:
            monkeypatch.delenv(name, raising=False)

    real_loader = config._load_dotenv
    real_env_file = config.REPO_ROOT / ".env"

    def guarded(path):
        # Never read the developer's real .env; tests that point the loader at
        # their own temporary file still work.
        if path == real_env_file:
            return
        real_loader(path)

    monkeypatch.setattr(config, "_load_dotenv", guarded)
