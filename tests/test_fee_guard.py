"""Fee-drag guard: only open positions whose fee is <= max_fee_pct of notional.
Written before implementation (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import RiskConfig, Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote
from lmtrade.finance.risk import fee_ratio_ok, min_notional_for_fee, size_position


class TestFeeHelpers:
    def test_default_threshold_is_one_percent(self):
        assert RiskConfig().max_fee_pct == pytest.approx(0.01)

    def test_min_notional(self):
        assert min_notional_for_fee(1.0, 0.01) == pytest.approx(100.0)

    def test_min_notional_disabled_when_pct_zero(self):
        assert min_notional_for_fee(1.0, 0.0) == 0.0

    def test_ratio_ok_boundary_inclusive(self):
        assert fee_ratio_ok(100.0, 1.0, 0.01) is True
        assert fee_ratio_ok(99.0, 1.0, 0.01) is False

    def test_zero_or_negative_notional_rejected(self):
        assert fee_ratio_ok(0.0, 1.0, 0.01) is False
        assert fee_ratio_ok(-5.0, 1.0, 0.01) is False

    def test_zero_fee_always_ok(self):
        assert fee_ratio_ok(0.5, 0.0, 0.01) is True

    def test_disabled_guard_allows_anything_positive(self):
        assert fee_ratio_ok(2.0, 1.0, 0.0) is True


class TestSizePositionFee:
    cfg = RiskConfig(min_confidence=0.5, max_position_fraction=0.5)

    def test_skips_when_fee_too_large(self):
        r = size_position(price=10.0, cash=50.0, equity=50.0, confidence=1.0,
                          cfg=self.cfg, fee=1.0)
        assert r.qty == 0 and "fee" in r.reason

    def test_allows_when_notional_big_enough(self):
        r = size_position(price=10.0, cash=1000.0, equity=1000.0, confidence=1.0,
                          cfg=self.cfg, fee=1.0)
        assert r.qty > 0 and r.notional >= 100.0

    def test_no_fee_arg_keeps_old_behavior(self):
        r = size_position(price=10.0, cash=50.0, equity=50.0, confidence=1.0,
                          cfg=self.cfg)
        assert r.qty > 0


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "synthetic"})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 0.5
    s.options.enabled = True
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def engine_for(settings, store, fee):
    broker = PaperBroker(store, starting_cash=settings.budget, fee=fee)
    return Engine(settings, store, broker)


QUOTE = Quote("AAPL", 100.0, [100.0] * 60, "synthetic")
BUY = Decision("AAPL", "buy", 0.9, "test")


class TestEngineGuard:
    def test_option_skipped_when_fee_exceeds_pct(self, settings, store, monkeypatch):
        import lmtrade.core.engine as eng
        monkeypatch.setattr(eng, "OPTION_FEE", 5.0)    # 5 EUR fee on a small budget
        e = engine_for(settings, store, fee=1.0)
        cash = e.broker.cash()
        e._enter_option(QUOTE, BUY, None)
        assert not store.open_options()
        assert e.broker.cash() == pytest.approx(cash)

    def test_option_opens_when_guard_disabled(self, settings, store, monkeypatch):
        import lmtrade.core.engine as eng
        monkeypatch.setattr(eng, "OPTION_FEE", 5.0)
        settings.risk.max_fee_pct = 0.0
        e = engine_for(settings, store, fee=1.0)
        e._enter_option(QUOTE, BUY, None)
        assert store.open_options()

    def test_option_opens_when_fee_small(self, settings, store):
        settings.risk.max_fee_pct = 0.05
        e = engine_for(settings, store, fee=0.1)      # 0.1 fee, budget ~15 -> <1%
        e._enter_option(QUOTE, BUY, None)
        assert store.open_options()

    def test_equity_skipped_when_fee_exceeds_pct(self, settings, store):
        settings.options.enabled = False
        e = engine_for(settings, store, fee=1.0)       # 1 EUR on <=50 EUR -> >1%
        econ = type("E", (), {"net_worth_eur": 100.0})()
        e._enter_equity(QUOTE, BUY, econ)
        assert store.position("AAPL") is None
        assert e.broker.cash() == pytest.approx(100.0)

    def test_equity_opens_when_guard_disabled(self, settings, store):
        settings.options.enabled = False
        settings.risk.max_fee_pct = 0.0
        e = engine_for(settings, store, fee=1.0)
        econ = type("E", (), {"net_worth_eur": 100.0})()
        e._enter_equity(QUOTE, BUY, econ)
        assert store.position("AAPL") is not None

    def test_sells_never_blocked_by_guard(self, settings, store):
        settings.options.enabled = False
        settings.risk.max_fee_pct = 0.0
        e = engine_for(settings, store, fee=1.0)
        econ = type("E", (), {"net_worth_eur": 100.0})()
        e._enter_equity(QUOTE, BUY, econ)
        settings.risk.max_fee_pct = 0.0001               # now absurdly strict
        e._enter_equity(QUOTE, Decision("AAPL", "sell", 0.9, "t"), econ)
        assert store.position("AAPL") is None


class TestFeeAwarePnl:
    """P&L must include the trading fee, not just the premium move."""

    def test_option_fee_models_tr_flat_fee(self):
        import lmtrade.core.engine as eng
        assert eng.OPTION_FEE == pytest.approx(1.0)

    def test_unrealized_pnl_includes_entry_fee(self, settings, store, monkeypatch):
        import time
        import lmtrade.core.engine as eng
        e = engine_for(settings, store, fee=1.0)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: o["entry_premium"])
        store.open_option("AAPL", "call", strike=100.0,
                          expiry_ts=time.time() + 5 * 86400, iv=0.4,
                          contracts=10.0, entry_premium=2.0, genome_id=None)
        e._persist_option_marks({"AAPL": 100.0})
        mark = next(iter(store.get_meta("open_option_marks").values()))
        assert mark["unrealized_pnl"] == pytest.approx(-eng.OPTION_FEE)

    def test_realized_pnl_includes_both_fees(self, settings, store, monkeypatch):
        import time
        import lmtrade.core.engine as eng
        settings.risk.min_confidence = 1.1     # no new entries; isolate the exit
        e = engine_for(settings, store, fee=1.0)
        monkeypatch.setattr(e, "_mark_position", lambda o, spot: o["entry_premium"] * 1.6)
        store.open_option("AAPL", "call", strike=100.0,
                          expiry_ts=time.time() + 5 * 86400, iv=0.4,
                          contracts=10.0, entry_premium=2.0, genome_id=None,
                          tp_premium=3.0, sl_premium=1.0)
        e.run_cycle()
        closed = store.closed_options()
        assert closed
        gross = (3.2 - 2.0) * 10.0
        assert closed[0]["pnl"] == pytest.approx(gross - 2 * eng.OPTION_FEE)


class TestFeeAwareBudget:
    """Pure sizing rule: the fee floor is part of the decision, not a late veto."""

    def test_sized_above_minimum_is_kept(self):
        from lmtrade.finance.risk import fee_aware_budget
        assert fee_aware_budget(150.0, 200.0, 1.0, 0.01) == pytest.approx(150.0)

    def test_sized_below_minimum_is_raised_when_caps_allow(self):
        from lmtrade.finance.risk import fee_aware_budget
        assert fee_aware_budget(40.0, 200.0, 1.0, 0.01) == pytest.approx(100.0)

    def test_infeasible_when_ceiling_below_minimum(self):
        from lmtrade.finance.risk import fee_aware_budget
        assert fee_aware_budget(40.0, 60.0, 1.0, 0.01) == 0.0

    def test_guard_disabled_keeps_sized(self):
        from lmtrade.finance.risk import fee_aware_budget
        assert fee_aware_budget(40.0, 60.0, 1.0, 0.0) == pytest.approx(40.0)

    def test_zero_or_negative_inputs(self):
        from lmtrade.finance.risk import fee_aware_budget
        assert fee_aware_budget(0.0, 0.0, 1.0, 0.01) == 0.0
        assert fee_aware_budget(-5.0, -1.0, 1.0, 0.0) == 0.0


def _force_decision(engine, direction="buy", conf=0.9):
    engine._decide = lambda q: (Decision(q.symbol, direction, conf, "forced"), None)


class TestFeeAwareSelection:
    def _caps(self, settings):
        settings.options.max_option_fraction = 2 / 3
        settings.options.cash_reserve_pct = 1 / 3

    def test_infeasible_buy_never_reaches_execution(self, settings, store):
        settings.budget = 50.0          # can never reach the 100 EUR minimum
        self._caps(settings)
        e = engine_for(settings, store, fee=1.0)
        _force_decision(e)
        entered = []
        e._enter_knockout = lambda *a, **k: entered.append("ko") or True
        e._enter_option = lambda *a, **k: entered.append("opt")
        e.run_cycle()
        assert entered == []
        logs = " ".join(r["message"] for r in store.recent_logs())
        assert "[execution] entering" not in logs
        assert "[fee-guard]" in logs and "AAPL" in logs

    def test_small_signal_is_raised_to_fee_minimum(self, settings, store):
        settings.budget = 300.0
        self._caps(settings)
        settings.risk.min_confidence = 0.05
        e = engine_for(settings, store, fee=1.0)
        _force_decision(e, conf=0.2)    # confidence sizing alone would be < 100
        e.run_cycle()
        opts = store.open_options()
        assert opts, "should open at the fee-floor size instead of skipping"
        notional = opts[0]["contracts"] * opts[0]["entry_premium"]
        assert notional == pytest.approx(100.0, rel=1e-3)

    def test_large_signal_capped_at_two_thirds_of_equity(self, settings, store):
        settings.budget = 300.0
        self._caps(settings)
        settings.sizing.vol_target_annual = 100.0   # no vol haircut
        settings.sizing.regime_filter_enabled = False
        e = engine_for(settings, store, fee=1.0)
        _force_decision(e, conf=1.0)
        e.run_cycle()
        opts = store.open_options()
        assert opts
        notional = opts[0]["contracts"] * opts[0]["entry_premium"]
        assert 100.0 <= notional <= 200.0 + 1e-6


class TestFeeRecheckAtExecution:
    """Regression: feasibility was only checked before the cycle's entries, so
    once an earlier entry consumed the deployable capital the later candidates
    still logged '[execution] entering ...' and then silently did nothing."""

    def test_later_candidates_dropped_once_capital_is_used(self, settings, store):
        settings.universe = ["AAPL", "MSFT"]
        settings.budget = 300.0
        settings.options.max_option_fraction = 2 / 3
        settings.options.cash_reserve_pct = 1 / 3
        settings.sizing.vol_target_annual = 100.0
        settings.sizing.regime_filter_enabled = False
        e = engine_for(settings, store, fee=1.0)
        _force_decision(e, conf=1.0)
        e.run_cycle()
        assert len(store.open_options()) == 1
        msgs = [r["message"] for r in store.recent_logs()]
        assert sum("[execution] entering" in m for m in msgs) == 1
        assert any("[fee-guard]" in m and "dropped" in m for m in msgs)
