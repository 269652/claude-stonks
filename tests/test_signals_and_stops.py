"""Tests for: per-position TP/SL persisted at open time, imported hourly news
as signals (with freshness), and imported daily market analysis as a signal
provider. Written before implementation per strict TDD.

Context: the hourly Claude Routine session gathers news / compiles the daily
analysis itself (it IS a Claude model with web search) and imports them via
CLI commands; the engine then uses them as weighted fusion signals. TP/SL are
stored on each option position when opened so every position always carries
explicit, immutable stops — config changes never retroactively alter them.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lmtrade.cli as cli
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote

runner = CliRunner()


def buy_signal_history(base: float, n: int = 60) -> list[float]:
    return [base + i * 0.6 + 3 * math.sin(i * 0.7) for i in range(n)]


class ScriptedMarket:
    def __init__(self, scripts: dict[str, list[Quote]]):
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._idx: dict[str, int] = {k: 0 for k in scripts}

    def quote(self, symbol: str) -> Quote:
        script = self._scripts.get(symbol)
        if not script:
            return Quote(symbol, 100.0, [100.0] * 60, "yahoo")
        i = min(self._idx[symbol], len(script) - 1)
        self._idx[symbol] += 1
        return script[i]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.risk.min_confidence = 0.5
    s.risk.max_fee_pct = 0.0   # fee guard has its own tests (test_fee_guard.py)
    s.options.enabled = True
    s.learning.enabled = False
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make_engine(settings, store, market) -> Engine:
    broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
    return Engine(settings, store, broker, market=market)


# --------------------------------------------------------------------------- #
# Per-position TP/SL
# --------------------------------------------------------------------------- #
class TestPersistedStops:
    def test_open_option_persists_tp_sl(self, store):
        oid = store.open_option("AAPL", "call", 100.0, time.time() + 7 * 86400,
                                iv=0.3, contracts=1.0, entry_premium=2.0,
                                genome_id=None, tp_premium=3.0, sl_premium=1.2)
        row = [o for o in store.open_options() if o["id"] == oid][0]
        assert row["tp_premium"] == pytest.approx(3.0)
        assert row["sl_premium"] == pytest.approx(1.2)

    def test_schema_migration_adds_columns_to_old_db(self, tmp_path):
        """A bot-state DB created before this change lacks the tp/sl columns;
        opening it must migrate in place, not crash."""
        db = tmp_path / "old.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""CREATE TABLE option_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            underlying TEXT NOT NULL, kind TEXT NOT NULL, strike REAL NOT NULL,
            expiry_ts REAL NOT NULL, iv REAL NOT NULL, contracts REAL NOT NULL,
            entry_premium REAL NOT NULL, opened_ts REAL NOT NULL,
            genome_id TEXT, status TEXT NOT NULL DEFAULT 'open',
            exit_premium REAL, closed_ts REAL, pnl REAL)""")
        conn.execute("INSERT INTO option_positions(underlying,kind,strike,"
                     "expiry_ts,iv,contracts,entry_premium,opened_ts) "
                     "VALUES('AAPL','call',100,1e12,0.3,1.0,2.0,0)")
        conn.commit()
        conn.close()

        st = Store(db)
        rows = st.open_options()
        st.close()
        assert rows and rows[0]["tp_premium"] is None  # legacy row, columns exist

    def test_engine_sets_stops_on_every_open(self, settings, store):
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = make_engine(settings, store, market)
        engine.run_cycle()
        opts = store.open_options()
        assert opts, "setup: expected an open position"
        o = opts[0]
        cfg = settings.options
        assert o["tp_premium"] == pytest.approx(
            o["entry_premium"] * (1 + cfg.take_profit_pct), rel=1e-6)
        assert o["sl_premium"] == pytest.approx(
            o["entry_premium"] * (1 - cfg.stop_loss_pct), rel=1e-6)

    def test_stored_stops_win_over_later_config_changes(self, settings, store):
        """Tighten config after opening — the position's own stored stops must
        still govern, proving stops are placed AT OPEN, not derived at check."""
        market = ScriptedMarket({
            "AAPL": [
                Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo"),
                Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo"),
            ],
        })
        engine = make_engine(settings, store, market)
        engine.run_cycle()
        assert store.open_options()
        # Make config absurdly tight; if exits recomputed from config, the
        # unchanged mark (~entry) would instantly "hit" a 0%-wide stop band.
        settings.options.take_profit_pct = 0.0
        settings.options.stop_loss_pct = 0.0
        engine.run_cycle()
        assert store.open_options(), \
            "position must still be open — its stored stops govern, not config"


# --------------------------------------------------------------------------- #
# Imported hourly news as signals
# --------------------------------------------------------------------------- #
class TestNewsImport:
    def _news_file(self, tmp_path, items) -> str:
        p = tmp_path / "news.json"
        p.write_text(json.dumps(items))
        return str(p)

    def test_cli_import_news(self, settings, store, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        store.close()
        f = self._news_file(tmp_path, [
            {"symbol": "AAPL", "text": "Strong iPhone demand. SENTIMENT: bullish"},
            {"symbol": "MSFT", "text": "Cloud growth slows.", "sentiment": "bearish"},
        ])
        result = runner.invoke(cli.app, ["import-news", f])
        assert result.exit_code == 0, result.output
        st = Store(settings.db_path)
        aapl = st.latest_news("AAPL")
        msft = st.latest_news("MSFT")
        st.close()
        assert aapl["sentiment"] == "bullish"     # parsed from text
        assert msft["sentiment"] == "bearish"     # explicit field respected

    def test_cli_import_news_rejects_malformed(self, settings, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        p = tmp_path / "bad.json"
        p.write_text("not json")
        result = runner.invoke(cli.app, ["import-news", str(p)])
        assert result.exit_code != 0

    def test_fresh_news_becomes_a_signal(self, settings, store):
        store.add_news("AAPL", "record earnings. SENTIMENT: bullish", "bullish")
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = make_engine(settings, store, market)
        decision, _ = engine._decide(engine.market.quote("AAPL"))
        assert any(s.provider == "news" for s in decision.signals)

    def test_stale_news_is_ignored(self, settings, store):
        two_days_ago = time.time() - 2 * 86400
        store.add_news("AAPL", "old story. SENTIMENT: bullish", "bullish",
                       ts=two_days_ago)
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = make_engine(settings, store, market)
        decision, _ = engine._decide(engine.market.quote("AAPL"))
        assert not any(s.provider == "news" for s in decision.signals), \
            "stale news must not keep influencing decisions"


# --------------------------------------------------------------------------- #
# Imported daily market analysis as a signal provider
# --------------------------------------------------------------------------- #
class TestAnalysisImport:
    def _analysis_file(self, tmp_path, symbols, valid_hours=24) -> str:
        p = tmp_path / "analysis.json"
        p.write_text(json.dumps({"valid_hours": valid_hours, "symbols": symbols}))
        return str(p)

    def test_cli_import_analysis(self, settings, store, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        store.close()
        f = self._analysis_file(tmp_path, {
            "AAPL": {"bias": "bullish", "confidence": 0.7, "notes": "supply chain easing"},
            "SPY": {"bias": "neutral", "confidence": 0.5, "notes": "range-bound"},
        })
        result = runner.invoke(cli.app, ["import-analysis", f])
        assert result.exit_code == 0, result.output
        st = Store(settings.db_path)
        stored = st.get_meta("market_analysis")
        st.close()
        assert stored["symbols"]["AAPL"]["bias"] == "bullish"
        assert stored["ts"] > 0

    def test_cli_import_analysis_clamps_confidence(self, settings, store, tmp_path,
                                                   monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        store.close()
        f = self._analysis_file(tmp_path, {
            "AAPL": {"bias": "bearish", "confidence": 7.5, "notes": "x"},
        })
        result = runner.invoke(cli.app, ["import-analysis", f])
        assert result.exit_code == 0
        st = Store(settings.db_path)
        stored = st.get_meta("market_analysis")
        st.close()
        assert 0.0 <= stored["symbols"]["AAPL"]["confidence"] <= 1.0

    def test_cli_import_analysis_rejects_bad_bias(self, settings, tmp_path,
                                                  monkeypatch):
        monkeypatch.setattr(cli, "load_settings", lambda: settings)
        f = self._analysis_file(tmp_path, {
            "AAPL": {"bias": "to-the-moon", "confidence": 0.9, "notes": "x"},
        })
        result = runner.invoke(cli.app, ["import-analysis", f])
        assert result.exit_code != 0

    def test_fresh_analysis_becomes_a_signal(self, settings, store):
        store.set_meta("market_analysis", {
            "ts": time.time(), "valid_hours": 24,
            "symbols": {"AAPL": {"bias": "bearish", "confidence": 0.8,
                                 "notes": "rate fears"}},
        })
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = make_engine(settings, store, market)
        decision, _ = engine._decide(engine.market.quote("AAPL"))
        analysis_sigs = [s for s in decision.signals if s.provider == "analysis"]
        assert analysis_sigs
        assert analysis_sigs[0].direction == "sell"
        assert analysis_sigs[0].confidence == pytest.approx(0.8)

    def test_expired_analysis_is_ignored(self, settings, store):
        store.set_meta("market_analysis", {
            "ts": time.time() - 30 * 3600, "valid_hours": 24,
            "symbols": {"AAPL": {"bias": "bullish", "confidence": 0.9, "notes": "x"}},
        })
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = make_engine(settings, store, market)
        decision, _ = engine._decide(engine.market.quote("AAPL"))
        assert not any(s.provider == "analysis" for s in decision.signals)

    def test_neutral_bias_adds_no_signal(self, settings, store):
        store.set_meta("market_analysis", {
            "ts": time.time(), "valid_hours": 24,
            "symbols": {"AAPL": {"bias": "neutral", "confidence": 0.9, "notes": "x"}},
        })
        market = ScriptedMarket({
            "AAPL": [Quote("AAPL", 313.0, buy_signal_history(300.0), "yahoo")],
        })
        engine = make_engine(settings, store, market)
        decision, _ = engine._decide(engine.market.quote("AAPL"))
        assert not any(s.provider == "analysis" for s in decision.signals)
