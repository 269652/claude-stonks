"""Opus as the decision model, pre-screened and cached: the fusion collects the
cheap votes first (heuristics, evidence rules, strategy, news, analysis); an
expensive provider (claude_cli / cloud) is only consulted when the cheap
pre-screen is strong enough or the symbol is held, and its answer is cached
for model.decision_cache_minutes. The LLM prompt carries the other votes,
the news and whether the position is held. Written before implementation
(TDD). Offline: providers are faked."""
from __future__ import annotations

import pytest

from lmtrade.agents.fusion import FusionEngine
from lmtrade.config import ModelConfig, Settings
from lmtrade.data.market import Quote
from lmtrade.models.base import ModelProvider, Signal
from lmtrade.models.providers import _prompt

QUOTE = Quote("AAPL", 100.0, [99.0, 101.0] * 30, "yahoo")


class Counting(ModelProvider):
    name = "claude_cli"

    def __init__(self, direction="buy", conf=0.9):
        self.calls, self.contexts = 0, []
        self.direction, self.conf = direction, conf

    def analyze(self, symbol, context):
        self.calls += 1
        self.contexts.append(context)
        return Signal(self.name, self.direction, self.conf, "opus view", 0.0)


def fusion(provider, **model):
    s = Settings()
    s.model.weights = {"claude_cli": 2.0, "evidence": 2.0}
    for k, v in model.items():
        setattr(s.model, k, v)
    return FusionEngine(s, {"claude_cli": provider})


def strong():
    return [Signal("evidence", "buy", 0.9, "tsmom buy", 0.0)]


def weak():
    return [Signal("evidence", "buy", 0.2, "weak", 0.0)]


class TestConfig:
    def test_defaults(self):
        m = ModelConfig()
        assert m.decision_model == "claude-opus-5-5"
        assert m.prescreen_confidence == pytest.approx(0.5)
        assert m.decision_cache_minutes == 30
        assert "claude_cli" in m.screened_providers

    def test_cli_provider_built_with_decision_model(self):
        from lmtrade.models.providers import build_providers
        s = Settings()
        s.model.stack = ["claude_cli"]
        assert build_providers(s)["claude_cli"].model == "claude-opus-5-5"


class TestPrescreen:
    def test_weak_prescreen_skips_expensive_provider(self):
        p = Counting()
        d = fusion(p).decide(QUOTE, extra_signals=weak())
        assert p.calls == 0
        assert all(s.provider != "claude_cli" for s in d.signals)

    def test_strong_prescreen_consults_it(self):
        p = Counting()
        d = fusion(p).decide(QUOTE, extra_signals=strong())
        assert p.calls == 1
        assert any(s.provider == "claude_cli" for s in d.signals)

    def test_held_symbol_always_consulted(self):
        p = Counting()
        fusion(p).decide(QUOTE, extra_signals=weak(), context_extra={"held": "long"})
        assert p.calls == 1

    def test_answer_cached(self):
        p = Counting()
        f = fusion(p)
        f.decide(QUOTE, extra_signals=strong())
        d2 = f.decide(QUOTE, extra_signals=strong())
        assert p.calls == 1
        assert any(s.provider == "claude_cli" for s in d2.signals)

    def test_cache_disabled_calls_again(self):
        p = Counting()
        f = fusion(p, decision_cache_minutes=0)
        f.decide(QUOTE, extra_signals=strong())
        f.decide(QUOTE, extra_signals=strong())
        assert p.calls == 2

    def test_llm_sees_the_other_votes(self):
        p = Counting()
        fusion(p).decide(QUOTE, extra_signals=strong(), context_extra={"research": "Apple beat"})
        ctx = p.contexts[0]
        assert any(v[0] == "evidence" for v in ctx["votes"])
        assert ctx["research"] == "Apple beat"


class TestPrompt:
    def test_prompt_includes_votes_news_and_held(self):
        ctx = {"indicators": {"last": 100.0}, "research": "Apple beat earnings",
               "votes": [("evidence", "buy", 0.9, "tsmom buy 1.00")], "held": "long"}
        p = _prompt("AAPL", ctx)
        assert "Apple beat earnings" in p and "evidence" in p and "tsmom" in p
        assert "held" in p.lower() and "long" in p
        assert "~10 EUR" not in p              # stale account description removed


class TestEngineHeldContext:
    def test_signal_exit_pass_tells_the_llm_the_position_is_held(self, tmp_path):
        import time

        from lmtrade.brokers.paper import PaperBroker
        from lmtrade.core.engine import Engine
        from lmtrade.core.state import Store

        class Market:
            def quote(self, s):
                return Quote(s, 100.0, [99.0, 101.0] * 30, "yahoo")

        s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                     data={"provider": "yahoo", "intraday": True})
        s.model.stack = ["heuristic"]
        s.data_dir = tmp_path
        s.learning.enabled = False
        s.options.min_hold_hours = 0
        s.research.commentary_minutes = 0
        store = Store(s.db_path)
        try:
            e = Engine(s, store, PaperBroker(store, starting_cash=300.0), market=Market(),
                       tr_derivatives=None)
            p = Counting(direction="buy", conf=0.4)
            e.fusion.providers["claude_cli"] = p
            e.fusion.llm_async = False          # inline, to inspect the context
            store.open_option("AAPL", "ko_put", 120.0, time.time() + 3e7, 0.0, 10.0, 2.0, None,
                              instrument_type="knockout", barrier=120.0, ratio=10.0)
            e._signal_exits({"AAPL": Market().quote("AAPL")}, {"AAPL"})
            assert p.contexts and p.contexts[0]["held"] == "short"
        finally:
            store.close()


class TestNonBlocking:
    """In production the LLM is asked in the background: a decision never
    waits for it (it used to block exits / heartbeat for 20-60 s per call);
    the answer counts from the next decision on."""

    def test_async_first_call_returns_without_llm_then_uses_it(self):
        p = Counting()
        f = fusion(p)
        f.llm_async = True
        d1 = f.decide(QUOTE, extra_signals=strong())
        assert all(s.provider != "claude_cli" for s in d1.signals)
        f.wait_llm(5)
        assert p.calls == 1
        d2 = f.decide(QUOTE, extra_signals=strong())
        assert any(s.provider == "claude_cli" for s in d2.signals)
        assert p.calls == 1

    def test_engine_runs_llm_in_background(self, tmp_path):
        from lmtrade.brokers.paper import PaperBroker
        from lmtrade.core.engine import Engine
        from lmtrade.core.state import Store
        s = Settings(mode="paper", budget=100.0, universe=["AAPL"])
        s.data_dir = tmp_path
        store = Store(s.db_path)
        try:
            e = Engine(s, store, PaperBroker(store, starting_cash=100.0), tr_derivatives=None)
            assert e.fusion.llm_async is True
        finally:
            store.close()
