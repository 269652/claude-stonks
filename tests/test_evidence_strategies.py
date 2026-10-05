"""Evidence-based strategy rules on DAILY bars (where their published results
come from), combined into one "evidence" vote for the fusion:
- 12-month time-series momentum (Moskowitz, Ooi & Pedersen 2012)
- 12-1 cross-sectional momentum (Jegadeesh & Titman 1993)
- Donchian 55-day breakout (turtle rules)
- RSI(2) pullback with the 200-day trend filter (Connors)
Pure functions + MarketData.daily_closes + engine wiring. Written before
implementation (TDD). Offline: data is constructed / faked."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.strategies.evidence import (
    donchian, evidence_vote, rsi2_pullback, tsmom_12m, xsmom_12_1,
)


def ramp(n, start=100.0, step=0.1):
    return [start + i * step for i in range(n)]


class TestTsmom:
    def test_uptrend_buys_downtrend_sells(self):
        assert tsmom_12m(ramp(260, 100, 0.2))[0] == "buy"
        assert tsmom_12m(ramp(260, 200, -0.2))[0] == "sell"

    def test_needs_a_year(self):
        assert tsmom_12m(ramp(200)) == ("hold", 0.0)

    def test_confidence_in_range(self):
        _, c = tsmom_12m(ramp(260, 100, 0.2))
        assert 0 < c <= 1


class TestXsmom:
    def test_top_and_bottom_of_the_universe(self):
        universe = {f"S{i}": ramp(260, 100, 0.05 * i - 0.2) for i in range(9)}
        assert xsmom_12_1("S8", universe)[0] == "buy"
        assert xsmom_12_1("S0", universe)[0] == "sell"
        assert xsmom_12_1("S4", universe)[0] == "hold"

    def test_too_few_symbols_or_data_abstains(self):
        assert xsmom_12_1("A", {"A": ramp(260), "B": ramp(260)}) == ("hold", 0.0)
        assert xsmom_12_1("A", {f"S{i}": ramp(50) for i in range(6)} | {"A": ramp(50)}) == ("hold", 0.0)


class TestDonchian:
    def test_breakout_up_and_down(self):
        flat = [100.0] * 60
        assert donchian(flat + [105.0])[0] == "buy"
        assert donchian(flat + [95.0])[0] == "sell"
        assert donchian(flat + [100.0])[0] == "hold"

    def test_needs_history(self):
        assert donchian([100.0] * 30) == ("hold", 0.0)


class TestRsi2Pullback:
    def test_dip_in_uptrend_buys(self):
        closes = ramp(220, 50, 0.5) + [158.0, 156.0, 154.0]
        assert rsi2_pullback(closes)[0] == "buy"

    def test_spike_in_downtrend_sells(self):
        closes = ramp(220, 200, -0.5) + [92.0, 94.0, 96.0]
        assert rsi2_pullback(closes)[0] == "sell"

    def test_dip_in_downtrend_is_not_bought(self):
        closes = ramp(220, 200, -0.5) + [88.0, 86.0, 84.0]
        assert rsi2_pullback(closes)[0] == "hold"


class TestEnsemble:
    def test_votes_combined_with_rationale(self):
        universe = {f"S{i}": ramp(260, 100, 0.05 * i - 0.2) for i in range(9)}
        d, c, why = evidence_vote("S8", universe)
        assert d == "buy" and 0 < c <= 1
        assert "tsmom" in why and "xsmom" in why

    def test_no_data_abstains(self):
        assert evidence_vote("X", {})[0] == "hold"


class TestDailyData:
    def test_yahoo_daily_closes_cached(self):
        from lmtrade.data.market import MarketData
        calls = []

        def fetcher(sym, range_, interval):
            calls.append((sym, range_, interval))
            return ramp(252)

        m = MarketData(provider="yahoo", fetcher=fetcher)
        assert len(m.daily_closes("AAPL")) == 252
        m.daily_closes("AAPL")
        # 2 years: one Yahoo year is only ~251 trading days, fewer than the
        # 253 closes 12-month momentum needs (live bug: tsmom/xsmom always held).
        assert calls == [("AAPL", "2y", "1d")]

    def test_yahoo_failure_is_none_not_fake(self):
        from lmtrade.data.market import MarketData

        def boom(*a):
            raise RuntimeError("down")
        assert MarketData(provider="yahoo", fetcher=boom).daily_closes("AAPL") is None

    def test_synthetic_provider_gives_a_series(self):
        from lmtrade.data.market import MarketData
        s = MarketData(provider="synthetic").daily_closes("AAPL")
        assert s is not None and len(s) >= 253


class TestEngineWiring:
    def test_decision_includes_evidence_vote(self, tmp_path):
        from lmtrade.brokers.paper import PaperBroker
        from lmtrade.config import Settings
        from lmtrade.core.engine import Engine
        from lmtrade.core.state import Store
        from lmtrade.data.market import Quote

        class Market:
            def quote(self, s):
                return Quote(s, 100.0, [99.0, 101.0] * 30, "yahoo")

            def daily_closes(self, s):
                return ramp(260, 100, 0.2)

        s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                     data={"provider": "yahoo", "intraday": True})
        s.model.stack = ["heuristic"]
        s.data_dir = tmp_path
        s.learning.enabled = False
        store = Store(s.db_path)
        try:
            e = Engine(s, store, PaperBroker(store, starting_cash=300.0), market=Market(),
                       tr_derivatives=None)
            d, _ = e._decide(Market().quote("AAPL"))
            ev = [x for x in d.signals if x.provider == "evidence"]
            assert ev and ev[0].direction == "buy"
        finally:
            store.close()

    def test_default_weight(self):
        import yaml
        cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text())
        assert cfg["model"]["weights"]["evidence"] == pytest.approx(2.0)


class TestYahooYearIsTooShort:
    def test_one_yahoo_year_cannot_feed_12m_momentum(self):
        # documents the bug: ~251 daily closes -> tsmom abstains
        assert tsmom_12m(ramp(251, 100, 0.2)) == ("hold", 0.0)

    def test_two_years_can(self):
        assert tsmom_12m(ramp(502, 100, 0.2))[0] == "buy"
