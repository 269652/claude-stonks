"""Strategy revision, step 1 — "setup -> trigger -> execution".

- Direction and confidence come from DAILY information: the evidence rules,
  the daily analysis and the LLM decision. Intraday / duplicated sources
  (model.non_voting: the 1-minute heuristic, the SLM on 1-minute
  indicators, Perplexity's keyword scan of the stored news, the news
  sentiment itself) are still computed and shown, but don't vote.
- An entry needs a daily setup: the evidence vote must point the same way
  (entry.require_daily_setup).
- News is a veto: a bullish/bearish news read against the trade blocks it.
- Symbols TR has no knockout for are skipped before any analysis.
Off in code (callers building Settings() keep the old behaviour);
config/default.yaml turns it on. Written before implementation (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from lmtrade.agents.fusion import Decision, FusionEngine
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import EntryConfig, ModelConfig, Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.models.base import ModelProvider, Signal

YAML = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text(
    encoding="utf-8"))
QUOTE = Quote("AAPL", 100.0, [99.0, 101.0] * 30, "yahoo")


class TestConfig:
    def test_code_defaults_keep_old_behaviour(self):
        assert EntryConfig().require_daily_setup is False
        assert ModelConfig().non_voting == []

    def test_yaml_turns_it_on(self):
        assert YAML["entry"]["require_daily_setup"] is True
        assert {"heuristic", "slm", "perplexity", "news"} <= set(YAML["model"]["non_voting"])
        assert YAML["learning"]["enabled"] is False        # genomes off until rebuilt daily
        assert YAML["loop"]["max_positions"] == 2

    def test_editable(self):
        from lmtrade.web.settings_editor import EDITABLE
        keys = {k for k, _, _ in EDITABLE}
        assert {"entry.require_daily_setup", "model.non_voting"} <= keys


class Fixed(ModelProvider):
    def __init__(self, name, direction, conf):
        self.name, self.direction, self.conf = name, direction, conf

    def analyze(self, symbol, context):
        return Signal(self.name, self.direction, self.conf, "fixed", 0.0)


class TestFusionNonVoting:
    def fusion(self, non_voting):
        s = Settings()
        s.model.non_voting = non_voting
        s.model.weights = {"heuristic": 1.0, "evidence": 2.0}
        return FusionEngine(s, {"heuristic": Fixed("heuristic", "sell", 1.0)})

    def test_non_voting_is_shown_but_does_not_vote(self):
        d = self.fusion(["heuristic"]).decide(
            QUOTE, extra_signals=[Signal("evidence", "buy", 0.8, "tsmom", 0.0)])
        assert d.direction == "buy" and d.confidence == pytest.approx(0.8)
        assert any(s.provider == "heuristic" for s in d.signals)

    def test_voting_by_default(self):
        d = self.fusion([]).decide(
            QUOTE, extra_signals=[Signal("evidence", "buy", 0.8, "tsmom", 0.0)])
        assert d.confidence < 0.8                          # the 1-minute sell pulls it down

    def test_only_non_voting_signals_means_hold(self):
        d = self.fusion(["heuristic"]).decide(QUOTE)
        assert d.direction == "hold"


class TestPerplexityDoubleCount:
    def test_without_key_it_does_not_rescan_the_stored_news(self):
        from lmtrade.models.providers import PerplexityProvider
        p = PerplexityProvider(Settings())
        assert not p.available()
        sig = p.analyze("AAPL", {"research": "SENTIMENT: bullish — strong quarter"})
        assert sig.direction == "hold"


# ------------------------------------------------------------ engine
@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 0.5
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.options.max_option_fraction = 1.0
    s.options.cash_reserve_pct = 0.0
    s.sizing.vol_target_annual = 100.0
    s.research.commentary_minutes = 0
    s.entry.require_daily_setup = True
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


class Market:
    def quote(self, s):
        return Quote(s, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], "yahoo")


def engine(settings, store, evidence, direction="buy", tr=None):
    e = Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
               market=Market(), tr_derivatives=tr)
    calls = []

    def decide(q):
        calls.append(q.symbol)
        sigs = [Signal("analysis", direction, 0.9, "x", 0.0)]
        if evidence:
            sigs.append(Signal("evidence", evidence, 0.7, "daily rules", 0.0))
        return Decision(q.symbol, direction, 0.9, "forced", sigs), None
    e._decide = decide
    return e, calls


class TestSetupGate:
    def test_matching_daily_setup_enters(self, settings, store):
        e, _ = engine(settings, store, evidence="buy")
        e.run_cycle()
        assert len(store.open_options()) == 1

    @pytest.mark.parametrize("evidence", [None, "hold", "sell"])
    def test_no_or_opposite_setup_does_not_enter(self, settings, store, evidence):
        e, _ = engine(settings, store, evidence=evidence)
        e.run_cycle()
        assert store.open_options() == []
        assert "no daily setup" in " ".join(r["message"] for r in store.recent_logs(30))

    def test_gate_off_keeps_old_behaviour(self, settings, store):
        settings.entry.require_daily_setup = False
        e, _ = engine(settings, store, evidence=None)
        e.run_cycle()
        assert len(store.open_options()) == 1


class TestNewsVeto:
    def test_opposite_news_blocks(self, settings, store):
        store.add_news("AAPL", "SENTIMENT: bearish", "bearish", time.time())
        e, _ = engine(settings, store, evidence="buy")
        e.run_cycle()
        assert store.open_options() == []
        assert "news" in " ".join(r["message"] for r in store.recent_logs(30)).lower()

    def test_agreeing_or_neutral_news_is_fine(self, settings, store):
        store.add_news("AAPL", "SENTIMENT: bullish", "bullish", time.time())
        e, _ = engine(settings, store, evidence="buy")
        e.run_cycle()
        assert len(store.open_options()) == 1


class NoKnockouts:
    def available(self):
        return True

    def search(self, underlying, direction):
        return []

    def find_knockout(self, *a, **k):
        return None

    def account_cash(self):
        return None


class TestTradableFirst:
    def test_untradable_symbols_are_not_analysed(self, settings, store):
        e, calls = engine(settings, store, evidence="buy", tr=NoKnockouts())
        e.run_cycle()
        assert calls == []
