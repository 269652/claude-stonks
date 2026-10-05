"""Strategy revision, step 3 — sizing on daily data.

The volatility target (annualised with sqrt(252)) and the calm/storm regime
filter (needs > 60 bars) were fed 60 one-minute bars: vol targeting never
scaled anything and the regime was always "calm". They now use the
underlying's daily closes when available (intraday as fallback). Also: a dip
buy's trade record carried the genome id in its confidence field.
Written before the fix (TDD). Offline."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


def series(daily_vol, n=300, tail_vol=None, tail=10):
    """Closes alternating +-r so the realised daily vol is ~daily_vol."""
    out, p = [], 100.0
    for i in range(n):
        v = tail_vol if (tail_vol and i >= n - tail) else daily_vol
        p *= 1 + (v if i % 2 else -v)
        out.append(p)
    return out


class Market:
    def __init__(self, daily):
        self.daily = daily

    def quote(self, s):
        return Quote(s, 100.0, [99.9, 100.1] * 29 + [99.9, 100.0], "yahoo")

    def daily_closes(self, s):
        return self.daily


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"])
    s.data_dir = tmp_path
    s.learning.enabled = False
    s.options.max_option_fraction = 1.0
    s.options.cash_reserve_pct = 0.0
    s.sizing.vol_target_annual = 0.20
    s.research.commentary_minutes = 0
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def engine(settings, store, daily):
    return Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
                  market=Market(daily), tr_derivatives=None)


class TestVolTarget:
    def test_daily_vol_scales_the_budget(self, settings, store):
        # ~60% annualised daily vol vs a 20% target: about a third of the size
        e = engine(settings, store, series(0.6 / math.sqrt(252)))
        sized, _ = e._budget_bounds(e.market.quote("AAPL"), Decision("AAPL", "buy", 1.0, "x"),
                                    None)
        assert sized == pytest.approx(300.0 / 3, rel=0.1)

    def test_quiet_stock_full_size(self, settings, store):
        e = engine(settings, store, series(0.1 / math.sqrt(252)))
        sized, _ = e._budget_bounds(e.market.quote("AAPL"), Decision("AAPL", "buy", 1.0, "x"),
                                    None)
        assert sized == pytest.approx(300.0 - OPTION_FEE)   # capped at cash - fee

    def test_no_daily_data_falls_back_to_intraday(self, settings, store):
        e = engine(settings, store, None)
        sized, _ = e._budget_bounds(e.market.quote("AAPL"), Decision("AAPL", "buy", 1.0, "x"),
                                    None)
        assert sized == pytest.approx(300.0 - OPTION_FEE)   # capped at cash - fee


class TestRegime:
    def test_storm_on_daily_data_raises_the_bar(self, settings, store):
        e = engine(settings, store, series(0.005, tail_vol=0.03))
        q = e.market.quote("AAPL")
        assert e._entry_min_confidence(q) == pytest.approx(
            settings.risk.min_confidence + settings.sizing.storm_extra_confidence)

    def test_calm(self, settings, store):
        e = engine(settings, store, series(0.01))
        assert e._entry_min_confidence(e.market.quote("AAPL")) == pytest.approx(
            settings.risk.min_confidence)


class TestDipTradeRecord:
    def test_confidence_field_is_the_signal_confidence(self, settings, store):
        import time
        settings.budget_split.dips_eur = 50.0
        settings.dips.enabled = True
        settings.dips.k_sigma = 1.0
        settings.tr.market_hours = ""
        e = engine(settings, store, None)
        e._mark_position = lambda o, spot: 1.94
        e._decide = lambda q: (Decision(q.symbol, "buy", 0.8, "forced"), None)
        store.open_option("AAPL", "ko_call", 80.0, time.time() + 3e7, 0.0, 50.0, 2.0, "g-1",
                          instrument_type="knockout", barrier=80.0, ratio=10.0, isin="DE1")
        q = Quote("AAPL", 97.0, [99.0, 101.0] * 28 + [97.0] * 4, "yahoo")
        e._dip_buys({"AAPL": q}, {"AAPL"})
        t = store.recent_trades(5)[0]
        assert t["confidence"] == pytest.approx(0.8)
