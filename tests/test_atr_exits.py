"""Strategy revision, step 2 — exits sized to the underlying's daily range.

New positions get TP / SL at the certificate price the position would have
when the underlying moves atr_tp_mult x ATR in favour / atr_sl_mult x ATR
against (ATR: average absolute daily close-to-close move over atr_window
days). That matches the daily-setup entries instead of fixed +50% / -40% of
a leveraged certificate. Falls back to the % levels without daily data or
when the levels don't make sense. Off in code, on in config/default.yaml.
Written before implementation (TDD). Offline."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lmtrade.brokers.paper import PaperBroker
from lmtrade.brokers.tr_derivatives import TRDerivativeQuote
from lmtrade.agents.fusion import Decision
from lmtrade.config import OptionsConfig, Settings
from lmtrade.core.engine import OPTION_FEE, Engine
from lmtrade.core.exits import atr_exit_levels
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.finance.indicators import daily_atr


class TestDailyAtr:
    def test_mean_absolute_daily_move(self):
        closes = [100.0 + (2.0 if i % 2 else 0.0) for i in range(30)]   # +-2 every day
        assert daily_atr(closes, 14) == pytest.approx(2.0)

    def test_uses_the_last_window_only(self):
        closes = [100.0 + i * 5 for i in range(20)] + [200.0 + i for i in range(15)]
        assert daily_atr(closes, 14) == pytest.approx(1.0)

    def test_too_short_or_flat(self):
        assert daily_atr([100.0] * 5, 14) is None
        assert daily_atr([], 14) is None
        assert daily_atr([100.0] * 30, 14) is None          # no movement: no ATR


class TestLevels:
    def lin(self, s):                                       # long KO: 0.1 per point
        return 2.0 + (s - 100.0) * 0.1

    def test_long(self):
        tp, sl = atr_exit_levels("ko_call", 100.0, 2.0, self.lin, 2.0, 3.0, 2.0)
        assert tp == pytest.approx(2.6) and sl == pytest.approx(1.6)

    def test_short(self):
        def short(s):
            return 2.0 - (s - 100.0) * 0.1
        tp, sl = atr_exit_levels("ko_put", 100.0, 2.0, short, 2.0, 3.0, 2.0)
        assert tp == pytest.approx(2.6) and sl == pytest.approx(1.6)

    def test_sl_beyond_the_barrier_is_floored_at_zero(self):
        tp, sl = atr_exit_levels("ko_call", 100.0, 20.0, lambda s: max(0.0, self.lin(s)),
                                 2.0, 3.0, 2.0)
        assert sl == 0.0 and tp > 2.0

    @pytest.mark.parametrize("price_at", [lambda s: None, lambda s: 2.0])
    def test_unusable_levels(self, price_at):
        assert atr_exit_levels("ko_call", 100.0, 2.0, price_at, 2.0, 3.0, 2.0) is None

    def test_no_atr(self):
        assert atr_exit_levels("ko_call", 100.0, None, self.lin, 2.0, 3.0, 2.0) is None


class TestConfig:
    def test_defaults(self):
        o = OptionsConfig()
        assert o.atr_exits is False
        assert (o.atr_tp_mult, o.atr_sl_mult, o.atr_window) == (3.0, 2.0, 14)
        cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text(
            encoding="utf-8"))
        assert cfg["options"]["atr_exits"] is True

    def test_editable(self):
        from lmtrade.web.settings_editor import EDITABLE
        keys = {k for k, _, _ in EDITABLE}
        assert {"options.atr_exits", "options.atr_tp_mult", "options.atr_sl_mult"} <= keys


# ------------------------------------------------------------ engine
DAILY = [100.0 + (2.0 if i % 2 else 0.0) for i in range(300)]   # ATR 2.0


class Market:
    def __init__(self, daily=True):
        self.daily = daily

    def quote(self, s):
        return Quote(s, 100.0, [99.0, 101.0] * 29 + [99.0, 100.0], "yahoo")

    def daily_closes(self, s):
        return DAILY if self.daily else None


class Derivs:
    def available(self):
        return True

    def find_knockout(self, underlying, direction, spot, leverage):
        kind = "ko_call" if direction == "buy" else "ko_put"
        strike = 80.0 if kind == "ko_call" else 120.0
        return TRDerivativeQuote(isin="DE000KO0001", underlying=underlying, kind=kind,
                                 strike=strike, barrier=strike, ratio=10.0, price=2.0,
                                 leverage=5.0, exchange="SGL")

    def search(self, underlying, direction):
        return [self.find_knockout(underlying, direction, 100.0, 5.0)]

    def account_cash(self):
        return None


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.learning.enabled = False
    s.options.max_option_fraction = 1.0
    s.options.cash_reserve_pct = 0.0
    s.sizing.vol_target_annual = 100.0
    s.research.commentary_minutes = 0
    s.options.atr_exits = True
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def engine(settings, store, daily=True):
    return Engine(settings, store, PaperBroker(store, starting_cash=300.0, fee=OPTION_FEE),
                  market=Market(daily), tr_derivatives=Derivs())


class TestEngine:
    def test_knockout_entry_gets_atr_levels(self, settings, store):
        e = engine(settings, store)
        q = e.market.quote("AAPL")
        assert e._enter_knockout(q, Decision("AAPL", "buy", 0.9, "x"), None)
        o = store.open_options()[0]
        # 1 point of AAPL = 0.1 EUR per certificate (ratio 10, EUR)
        assert o["tp_premium"] == pytest.approx(2.6)
        assert o["sl_premium"] == pytest.approx(1.6)

    def test_short_knockout(self, settings, store):
        e = engine(settings, store)
        q = e.market.quote("AAPL")
        e._enter_knockout(q, Decision("AAPL", "sell", 0.9, "x"), None)
        o = store.open_options()[0]
        assert o["tp_premium"] == pytest.approx(2.6) and o["sl_premium"] == pytest.approx(1.6)

    def test_limit_fill_anchors_at_the_fill(self, settings, store):
        e = engine(settings, store)
        inst = {"instrument_type": "knockout", "kind": "ko_call", "strike": 80.0,
                "barrier": 80.0, "ratio": 10.0, "isin": "DE000KO0001",
                "expiry_ts": e.now() + 3e7, "iv": 0.0, "ref_price": 2.0, "ref_spot": 100.0,
                "size": 0.1, "currency": "EUR", "exchange": "SGL"}
        e._book_limit_fill("AAPL", {"instrument": inst, "size": 50.0}, price=1.9, spot=99.0)
        o = store.open_options()[0]
        assert o["tp_premium"] == pytest.approx(1.9 + 0.6)
        assert o["sl_premium"] == pytest.approx(1.9 - 0.4)

    def test_without_daily_data_falls_back_to_percent(self, settings, store):
        e = engine(settings, store, daily=False)
        e._enter_knockout(e.market.quote("AAPL"), Decision("AAPL", "buy", 0.9, "x"), None)
        o = store.open_options()[0]
        assert o["tp_premium"] == pytest.approx(2.0 * (1 + settings.options.take_profit_pct))
        assert o["sl_premium"] == pytest.approx(2.0 * (1 - settings.options.stop_loss_pct))

    def test_off_uses_percent(self, settings, store):
        settings.options.atr_exits = False
        e = engine(settings, store)
        e._enter_knockout(e.market.quote("AAPL"), Decision("AAPL", "buy", 0.9, "x"), None)
        assert store.open_options()[0]["tp_premium"] == pytest.approx(
            2.0 * (1 + settings.options.take_profit_pct))
