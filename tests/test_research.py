"""Tests for the scheduler (core/scheduler.py), hourly Perplexity news service
(research/news.py) and daily Claude strategy analyst (research/daily.py).
Written before implementation per strict TDD. All offline — external calls are
injected fakes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmtrade.config import Settings
from lmtrade.core.scheduler import Scheduler
from lmtrade.core.state import Store
from lmtrade.research.daily import DailyAnalyst
from lmtrade.research.news import NewsService


@pytest.fixture()
def store(tmp_path: Path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=10.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.data_dir = tmp_path
    return s


class TestScheduler:
    def test_first_call_is_due_then_not(self, store):
        clock = [1000.0]
        sched = Scheduler(store, now=lambda: clock[0])
        assert sched.due("news", 3600) is True
        assert sched.due("news", 3600) is False      # just ran
        clock[0] += 3601
        assert sched.due("news", 3600) is True       # interval elapsed

    def test_jobs_are_independent(self, store):
        clock = [1000.0]
        sched = Scheduler(store, now=lambda: clock[0])
        assert sched.due("a", 60) is True
        assert sched.due("b", 60) is True            # different job, own timer

    def test_persists_across_instances(self, store):
        clock = [1000.0]
        Scheduler(store, now=lambda: clock[0]).due("news", 3600)
        sched2 = Scheduler(store, now=lambda: clock[0])
        assert sched2.due("news", 3600) is False     # last-run survived


class TestNewsService:
    def test_fetches_and_caches(self, store, settings):
        calls = []

        def fake_fetch(symbol):
            calls.append(symbol)
            return f"{symbol} rallies on earnings. SENTIMENT: bullish", 0.005

        clock = [1000.0]
        svc = NewsService(store, settings, fetcher=fake_fetch, now=lambda: clock[0])
        item = svc.get("AAPL")
        assert item["sentiment"] == "bullish"
        assert "rallies" in item["text"]
        assert calls == ["AAPL"]

        # Within the freshness window: served from cache, no second fetch.
        item2 = svc.get("AAPL")
        assert calls == ["AAPL"]
        assert item2["text"] == item["text"]

        # After the window: refetched.
        clock[0] += settings.research.news_interval_minutes * 60 + 1
        svc.get("AAPL")
        assert calls == ["AAPL", "AAPL"]

    def test_sentiment_parsing(self, store, settings):
        svc = NewsService(store, settings,
                          fetcher=lambda s: ("bad quarter. SENTIMENT: bearish", 0.0))
        assert svc.get("AAPL")["sentiment"] == "bearish"

    def test_no_fetcher_degrades_gracefully(self, store, settings):
        svc = NewsService(store, settings, fetcher=None)
        assert svc.get("AAPL") is None               # no key, no crash

    def test_corrupt_cache_is_invalidated_and_refetched(self, store, settings):
        # A broken entry (no SENTIMENT marker) stored by an older build must be
        # ignored and refetched, not served for the whole freshness window.
        store.add_news("AAPL", "Execution errorBatchvorgang abbrechen (J/N)?",
                       "neutral", ts=1000.0)
        calls = []

        def fetch(sym):
            calls.append(sym)
            return f"{sym} rallies hard. SENTIMENT: bullish", 0.0

        svc = NewsService(store, settings, fetcher=fetch, now=lambda: 1001.0)
        item = svc.get("AAPL")
        assert calls == ["AAPL"]                     # refetched despite fresh ts
        assert item["sentiment"] == "bullish"

    def test_corrupt_cache_with_no_fetcher_returns_none(self, store, settings):
        store.add_news("AAPL", "garbage without marker", "neutral", ts=1000.0)
        svc = NewsService(store, settings, fetcher=None, now=lambda: 1001.0)
        assert svc.get("AAPL") is None               # never serve the garbage

    def test_valid_cache_still_served(self, store, settings):
        store.add_news("AAPL", "steady. SENTIMENT: neutral", "neutral", ts=1000.0)
        calls = []
        svc = NewsService(store, settings,
                          fetcher=lambda s: (calls.append(s), ("x. SENTIMENT: bullish", 0.0))[1],
                          now=lambda: 1001.0)
        item = svc.get("AAPL")
        assert calls == []                           # fresh + valid -> cache
        assert item["sentiment"] == "neutral"


class TestNewsServiceConcurrentFetch:
    """get_many fetches the whole universe in parallel, bounded by
    max_workers — a sequential per-symbol loop is impractical once a fetcher
    has real per-call latency (e.g. a claude CLI subprocess with web search)
    across a 30-symbol universe."""

    def test_get_many_fetches_all_symbols(self, store, settings):
        calls = []

        def fake_fetch(symbol):
            calls.append(symbol)
            return f"{symbol} update. SENTIMENT: neutral", 0.0

        svc = NewsService(store, settings, fetcher=fake_fetch)
        results = svc.get_many(["AAPL", "MSFT", "SPY"])
        assert set(results.keys()) == {"AAPL", "MSFT", "SPY"}
        assert all(results[s]["sentiment"] == "neutral" for s in results)
        assert sorted(calls) == ["AAPL", "MSFT", "SPY"]

    def test_get_many_runs_concurrently_not_sequentially(self, store, settings):
        import time

        def slow_fetch(symbol):
            time.sleep(0.2)
            return f"{symbol}. SENTIMENT: neutral", 0.0

        svc = NewsService(store, settings, fetcher=slow_fetch)
        symbols = [f"SYM{i}" for i in range(6)]
        start = time.time()
        svc.get_many(symbols, max_workers=6)
        elapsed = time.time() - start
        assert elapsed < 0.8   # sequential would be ~1.2s; concurrent ~0.2s

    def test_get_many_empty_list(self, store, settings):
        svc = NewsService(store, settings, fetcher=lambda s: ("x", 0.0))
        assert svc.get_many([]) == {}

    def test_get_many_handles_missing_fetcher(self, store, settings):
        svc = NewsService(store, settings, fetcher=None)
        assert svc.get_many(["AAPL", "MSFT"]) == {"AAPL": None, "MSFT": None}

    def test_get_many_reports_live_progress(self, store, settings):
        """A slow real fetcher (claude CLI web search) makes get_many block for
        minutes; an on_progress callback fired per completion lets the engine
        show live progress instead of appearing hung."""
        svc = NewsService(store, settings,
                          fetcher=lambda s: (f"{s}. SENTIMENT: bullish", 0.0))
        seen = []
        svc.get_many(["AAPL", "MSFT", "SPY"], max_workers=3,
                     on_progress=lambda sym, item, done, total: seen.append((done, total, sym)))
        assert len(seen) == 3
        assert [d for d, _, _ in seen] == [1, 2, 3]        # monotonic completion count
        assert all(t == 3 for _, t, _ in seen)
        assert {s for _, _, s in seen} == {"AAPL", "MSFT", "SPY"}

    def test_get_many_progress_optional(self, store, settings):
        svc = NewsService(store, settings, fetcher=lambda s: ("x. SENTIMENT: neutral", 0.0))
        # No callback provided — must still work.
        assert set(svc.get_many(["AAPL"]).keys()) == {"AAPL"}

    def test_fetch_cost_recorded(self, store, settings):
        svc = NewsService(store, settings,
                          fetcher=lambda s: ("x. SENTIMENT: neutral", 0.005))
        svc.get("AAPL")
        assert store.total_costs().get("inference", 0) == pytest.approx(0.005)


class TestDailyAnalyst:
    def test_applies_bounded_adjustments(self, store, settings):
        response = json.dumps({
            "risk": {"stop_loss_pct": 0.04, "take_profit_pct": 0.10,
                     "min_confidence": 0.6},
            "notes": "tighten stops, demand higher conviction",
        })
        analyst = DailyAnalyst(store, settings, caller=lambda prompt: (response, 0.01))
        result = analyst.run()
        assert result is not None
        assert settings.risk.stop_loss_pct == pytest.approx(0.04)
        assert settings.risk.min_confidence == pytest.approx(0.6)
        # Insight is persisted for the dashboard.
        kinds = [a["kind"] for a in store.recent_activity(10)]
        assert "insight" in kinds

    def test_out_of_bounds_values_are_clamped(self, store, settings):
        response = json.dumps({"risk": {"stop_loss_pct": 0.9, "min_confidence": 0.01}})
        analyst = DailyAnalyst(store, settings, caller=lambda p: (response, 0.0))
        analyst.run()
        # Clamped into the sane envelope, not applied raw.
        assert settings.risk.stop_loss_pct <= 0.15
        assert settings.risk.min_confidence >= 0.4

    def test_malformed_response_is_a_noop(self, store, settings):
        before = settings.risk.stop_loss_pct
        analyst = DailyAnalyst(store, settings, caller=lambda p: ("not json at all", 0.0))
        assert analyst.run() is None
        assert settings.risk.stop_loss_pct == before

    def test_no_caller_and_no_key_degrades(self, store, settings):
        analyst = DailyAnalyst(store, settings, caller=None)
        assert analyst.run() is None


class TestCompileAnalysis:
    """Compiling the per-symbol market_analysis the Analysis tab + decision
    engine read — the important signal that was previously only importable via
    the hourly routine, never generated by a local run."""

    def test_compiles_and_stores_market_analysis(self, store, settings):
        response = json.dumps({"symbols": {
            "AAPL": {"bias": "bullish", "confidence": 0.7, "notes": "chip rally"},
            "SPY": {"bias": "neutral", "confidence": 0.4, "notes": "rangebound"}}})
        analyst = DailyAnalyst(store, settings, caller=lambda p: (response, 0.0))
        out = analyst.compile_analysis(["AAPL", "SPY"], now_ts=1000.0)
        assert out is not None
        stored = store.get_meta("market_analysis")
        assert stored["symbols"]["AAPL"]["bias"] == "bullish"
        assert stored["ts"] == 1000.0

    def test_clamps_and_sanitizes(self, store, settings):
        response = json.dumps({"symbols": {
            "AAPL": {"bias": "MOON", "confidence": 5.0},
            "SPY": {"bias": "bearish", "confidence": "oops"}}})
        analyst = DailyAnalyst(store, settings, caller=lambda p: (response, 0.0))
        out = analyst.compile_analysis(["AAPL", "SPY"], now_ts=1.0)
        assert out["symbols"]["AAPL"]["bias"] == "neutral"      # invalid bias
        assert out["symbols"]["AAPL"]["confidence"] == 1.0      # clamped
        assert out["symbols"]["SPY"]["confidence"] == 0.4       # bad -> default

    def test_no_caller_returns_none(self, store, settings):
        analyst = DailyAnalyst(store, settings, caller=None)
        assert analyst.compile_analysis(["AAPL"], now_ts=1.0) is None

    def test_malformed_response_returns_none(self, store, settings):
        analyst = DailyAnalyst(store, settings, caller=lambda p: ("not json", 0.0))
        assert analyst.compile_analysis(["AAPL"], now_ts=1.0) is None


class TestNewsServiceClaudeCLI:
    """research.news_provider = 'claude_cli' runs research through a local
    `claude` CLI instead of the Perplexity API — no key needed, everything
    local. Offline: the CLI subprocess is always faked."""

    def test_uses_claude_cli_when_configured_and_available(self, store, settings, monkeypatch):
        settings.research.news_provider = "claude_cli"
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        monkeypatch.setattr(
            "lmtrade.models.providers._default_claude_cli_runner",
            lambda prompt, timeout: "AAPL steady. SENTIMENT: neutral",
        )
        svc = NewsService(store, settings)
        item = svc.get("AAPL")
        assert item is not None
        assert item["sentiment"] == "neutral"

    def test_claude_cli_configured_but_binary_missing_degrades(self, store, settings, monkeypatch):
        settings.research.news_provider = "claude_cli"
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda name: None)
        svc = NewsService(store, settings)
        assert svc.get("AAPL") is None


class TestDailyAnalystClaudeCLI:
    """research.analysis_provider = 'claude_cli' runs the daily review through
    a local `claude` CLI instead of the Anthropic API — no key needed."""

    def test_uses_claude_cli_when_configured(self, store, settings, monkeypatch):
        settings.research.analysis_provider = "claude_cli"
        monkeypatch.setattr(
            "lmtrade.models.providers.shutil.which", lambda name: "/usr/bin/claude"
        )
        response = json.dumps({"risk": {"stop_loss_pct": 0.04}, "notes": "trim risk"})
        monkeypatch.setattr(
            "lmtrade.models.providers._default_claude_cli_runner",
            lambda prompt, timeout, model=None: response,
        )
        analyst = DailyAnalyst(store, settings)
        result = analyst.run()
        assert result is not None
        assert settings.risk.stop_loss_pct == pytest.approx(0.04)

    def test_claude_cli_configured_but_missing_binary_degrades(self, store, settings, monkeypatch):
        settings.research.analysis_provider = "claude_cli"
        monkeypatch.setattr("lmtrade.models.providers.shutil.which", lambda name: None)
        analyst = DailyAnalyst(store, settings, caller=None)
        assert analyst.run() is None
