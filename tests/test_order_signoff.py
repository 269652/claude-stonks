"""Every automatic LIVE order is signed off by Claude (Opus) before it goes to
TR: the reviewer sees the order, why the bot wants it, the position and the
recent price bars, and answers approve / reject. Bogus orders (live incident:
a trailing take-profit fired off a one-tick after-hours spike) are stopped.
If the reviewer is unavailable, entries are blocked (no new risk without a
review) and exits go through (never trap a position behind an outage).
Manual closes from the dashboard are the user's own sign-off. Written before
implementation (TDD). Offline: the reviewer is faked."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmtrade.config import Settings, SignoffConfig
from lmtrade.core.signoff import SignOff, build_prompt, parse_reply

ORDER = {"side": "sell", "symbol": "GOOGL", "isin": "DE000FE3CKE9", "size": 25,
         "kind": "ko_call", "reason": "trailing take-profit (peak +35.80 €, now -2.70 €)",
         "price": 6.17, "entry": 6.2, "pnl_eur": -2.7, "spot": 346.2,
         "bars": [346.0, 346.5, 405.0, 346.2]}


class TestConfig:
    def test_defaults_off_in_code_on_in_yaml(self):
        import yaml
        assert SignoffConfig().enabled is False
        assert SignoffConfig().model == "claude-opus-5-5"
        cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text(
            encoding="utf-8"))
        assert cfg["signoff"]["enabled"] is True

    def test_editable(self):
        from lmtrade.web.settings_editor import EDITABLE
        keys = {k for k, _, _ in EDITABLE}
        assert {"signoff.enabled", "signoff.model"} <= keys


class TestPromptAndParse:
    def test_prompt_carries_the_order_and_the_bars(self):
        p = build_prompt(ORDER)
        assert "GOOGL" in p and "SELL" in p and "trailing take-profit" in p
        assert "405.0" in p and "approve" in p

    @pytest.mark.parametrize("text,ok", [
        ('{"approve": true, "reason": "real move"}', True),
        ('sure: {"approve": false, "reason": "spike"} thanks', False),
        ('```json\n{"approve": false, "reason": "x"}\n```', False),
    ])
    def test_parse(self, text, ok):
        assert parse_reply(text)[0] is ok

    def test_unparseable_is_none(self):
        assert parse_reply("I think it's fine")[0] is None
        assert parse_reply(None)[0] is None


def signoff(reply=None, exc=None, enabled=True):
    s = Settings()
    s.signoff.enabled = enabled
    calls = []

    def caller(prompt):
        calls.append(prompt)
        if exc:
            raise exc
        return reply
    return SignOff(s, caller=caller), calls


class TestReview:
    def test_approved(self):
        so, calls = signoff('{"approve": true, "reason": "ok"}')
        ok, why = so.review(ORDER)
        assert ok and why == "ok" and len(calls) == 1

    def test_rejected(self):
        so, _ = signoff('{"approve": false, "reason": "one-tick spike"}')
        assert so.review(ORDER) == (False, "one-tick spike")

    def test_disabled_approves_without_asking(self):
        so, calls = signoff(enabled=False)
        assert so.review(ORDER)[0] is True and calls == []

    def test_outage_blocks_entries_but_lets_exits_through(self):
        so, _ = signoff(exc=TimeoutError("cli timeout"))
        assert so.review({**ORDER, "side": "buy"})[0] is False
        assert so.review(ORDER)[0] is True

    def test_unparseable_treated_like_outage(self):
        so, _ = signoff("hmm")
        assert so.review({**ORDER, "side": "buy"})[0] is False
        assert so.review(ORDER)[0] is True

    def test_no_reviewer_available(self):
        s = Settings()
        s.signoff.enabled = True
        so = SignOff(s, caller=None)
        so._caller = None
        assert so.review({**ORDER, "side": "buy"})[0] is False


# ------------------------------------------------------------ engine wiring
class LiveBroker:
    armed = True
    fee = 1.0
    mode = "live"

    def __init__(self, store):
        from lmtrade.brokers.paper import PaperBroker
        self._p = PaperBroker(store, starting_cash=300.0, fee=1.0)
        self.orders = []

    def __getattr__(self, name):
        return getattr(self._p, name)

    def place_order(self, isin, side, size, exchange="LSX", **kw):
        self.orders.append((isin, side, size))
        return SimpleNamespace(ok=True, message="ok", order_id="o1", price=None)


@pytest.fixture()
def env(tmp_path: Path):
    from lmtrade.core.engine import Engine
    from lmtrade.core.state import Store
    s = Settings(mode="live", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.learning.enabled = False
    s.loop.stale_evict_enabled = False
    s.options.min_hold_hours = 0
    s.options.flip_exit_confidence = 0.0
    s.research.commentary_minutes = 0
    s.tr.market_hours = ""
    s.signoff.enabled = True
    store = Store(s.db_path)
    broker = LiveBroker(store)
    e = Engine(s, store, broker, tr_derivatives=None)
    e._mark_position = lambda o, spot: spot / 50.0
    store.open_option("AAPL", "call", strike=100.0, expiry_ts=time.time() + 5 * 86400,
                      iv=0.4, contracts=25.0, entry_premium=2.0, genome_id=None,
                      tp_premium=1.5, sl_premium=0.0, isin="DE000TEST0001")
    yield e, broker, store
    store.close()


def reviewer(e, reply):
    seen = []

    def caller(prompt):
        seen.append(prompt)
        return reply
    e.signoff = SignOff(e.settings, caller=caller)
    return seen


class TestEngine:
    def test_rejected_exit_is_not_sent_and_backs_off(self, env):
        e, broker, store = env
        seen = reviewer(e, '{"approve": false, "reason": "bogus spike"}')
        e._manage_options({"AAPL": 100.0}, {"AAPL"})
        assert broker.orders == [] and len(store.open_options()) == 1
        assert seen and "take-profit" in seen[0]
        assert any("sign-off" in r["message"].lower() and "bogus spike" in r["message"]
                   for r in store.recent_logs(20))
        e._manage_options({"AAPL": 100.0}, {"AAPL"})            # backoff: not re-asked
        assert len(seen) == 1

    def test_approved_exit_is_sent(self, env):
        e, broker, store = env
        reviewer(e, '{"approve": true, "reason": "real"}')
        e._manage_options({"AAPL": 100.0}, {"AAPL"})
        assert broker.orders and store.open_options() == []

    def test_manual_close_skips_the_review(self, env):
        e, broker, store = env
        seen = reviewer(e, '{"approve": false, "reason": "no"}')
        o = store.open_options()[0]
        assert e._exit_option(o, 2.0, "manual close from dashboard", manual=True)
        assert seen == [] and broker.orders


from test_limit_entries import live, pending, settings, store  # noqa: E402,F401  (fixtures)


class TestLiveEntries:
    def test_rejected_limit_entry_is_not_placed(self, live, store):
        e, market, tr = live
        e.settings.signoff.enabled = True
        seen = reviewer(e, '{"approve": false, "reason": "no edge"}')
        e.run_cycle()
        assert tr.placed == [] and "AAPL" not in pending(store)
        assert seen and "BUY" in seen[0] and "DE000KO0001" in seen[0]

    def test_approved_limit_entry_is_placed(self, live, store):
        e, market, tr = live
        e.settings.signoff.enabled = True
        reviewer(e, '{"approve": true, "reason": "fine"}')
        e.run_cycle()
        assert len(tr.placed) == 1

    def test_rejected_market_entry_books_nothing(self, live, store):
        e, market, tr = live
        e.settings.signoff.enabled = True
        e.settings.entry.limit_orders = False
        market.price["AAPL"] = 98.5                       # watch trigger reached
        seen = reviewer(e, '{"approve": false, "reason": "no"}')
        e.run_cycle()
        assert seen and store.open_options() == []
