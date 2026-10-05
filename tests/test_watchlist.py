"""Entry timing via a watch list: a signal is not entered at market; it waits
until price is stretched k standard deviations from the intraday SMA in the
signal's favour (below for buys, above for sells), is dropped when the
signal disappears or flips, and expires after max_hours.
Written before implementation (TDD). Offline: market and decisions are faked."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import EntryConfig, Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.core.watchlist import entry_z, should_enter
from lmtrade.data.market import Quote

# Alternating 99/101: SMA 100, population std 1 -> z == price - 100.
HIST = [99.0, 101.0] * 30


class TestEntryZ:
    def test_z_relative_to_sma_and_std(self):
        assert entry_z(98.5, HIST, window=60) == pytest.approx(-1.5)
        assert entry_z(102.0, HIST, window=60) == pytest.approx(2.0)

    def test_window_uses_most_recent_bars(self):
        hist = [500.0] * 40 + HIST[-20:]
        assert entry_z(100.0, hist, window=20) == pytest.approx(0.0)

    def test_too_little_history_is_none(self):
        assert entry_z(100.0, [100.0, 101.0], window=60) is None
        assert entry_z(100.0, [], window=60) is None

    def test_flat_history_is_none(self):
        assert entry_z(100.0, [100.0] * 60, window=60) is None


class TestShouldEnter:
    def test_buy_needs_dip(self):
        assert should_enter("buy", -1.2, 1.0) is True
        assert should_enter("buy", -1.0, 1.0) is True      # boundary inclusive
        assert should_enter("buy", -0.5, 1.0) is False
        assert should_enter("buy", 1.5, 1.0) is False

    def test_sell_needs_spike(self):
        assert should_enter("sell", 1.2, 1.0) is True
        assert should_enter("sell", -1.5, 1.0) is False

    def test_unknown_z_never_enters(self):
        assert should_enter("buy", None, 1.0) is False

    def test_hold_never_enters(self):
        assert should_enter("hold", -3.0, 1.0) is False


class TestConfigDefaults:
    def test_shipped_defaults(self):
        c = EntryConfig()
        assert c.watch_k == pytest.approx(1.0)
        assert c.watch_max_hours == pytest.approx(4.0)


class MutableMarket:
    def __init__(self):
        self.price = 100.0

    def quote(self, symbol: str) -> Quote:
        return Quote(symbol, self.price, HIST[:-1] + [self.price], "yahoo")


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.risk.min_confidence = 0.5
    s.options.enabled = True
    s.options.max_option_fraction = 2 / 3
    s.options.cash_reserve_pct = 1 / 3
    s.learning.enabled = False
    s.entry.watch_enabled = True
    s.entry.watch_k = 1.0
    s.entry.watch_window = 60
    s.entry.watch_max_hours = 4.0
    return s


@pytest.fixture()
def store(settings):
    st = Store(settings.db_path)
    yield st
    st.close()


def make(settings, store):
    market, clock = MutableMarket(), Clock()
    e = Engine(settings, store, PaperBroker(store, starting_cash=settings.budget, fee=1.0),
               market=market, tr_derivatives=None, now=clock)
    return e, market, clock


def force(e, direction="buy", conf=0.9):
    e._decide = lambda q: (Decision(q.symbol, direction, conf, "forced"), None)


def logs(store) -> str:
    return "\n".join(r["message"] for r in store.recent_logs(500))


class TestEngineWatchlist:
    def test_signal_is_watched_not_bought_at_fair_price(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        market.price = 100.0                 # z == 0: no edge yet
        e.run_cycle()
        assert store.open_options() == []
        assert "AAPL" in (store.get_meta("watchlist") or {})
        assert "[watch]" in logs(store)

    def test_buy_entered_on_dip_and_removed_from_watchlist(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()                        # watched at fair price
        market.price = 98.5                  # z == -1.5 -> buy the dip
        e.run_cycle()
        opts = store.open_options()
        assert len(opts) == 1 and opts[0]["kind"] == "call"
        assert "AAPL" not in (store.get_meta("watchlist") or {})

    def test_buy_not_entered_on_spike(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()
        market.price = 102.0                 # stretched the wrong way
        e.run_cycle()
        assert store.open_options() == []
        assert "AAPL" in (store.get_meta("watchlist") or {})

    def test_sell_entered_on_spike(self, settings, store):
        e, market, _ = make(settings, store)
        force(e, "sell")
        e.run_cycle()
        market.price = 101.5
        e.run_cycle()
        opts = store.open_options()
        assert len(opts) == 1 and opts[0]["kind"] == "put"

    def test_immediate_entry_when_already_stretched(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        market.price = 98.0
        e.run_cycle()
        assert len(store.open_options()) == 1

    def test_expires_after_max_hours(self, settings, store):
        e, market, clock = make(settings, store)
        force(e)
        e.run_cycle()
        clock.t += 4 * 3600 + 1
        market.price = 98.0                  # would trigger, but too late
        e.run_cycle()
        assert store.open_options() == []
        assert "AAPL" not in (store.get_meta("watchlist") or {})
        assert "expired" in logs(store)

    def test_expiry_does_not_reset_while_signal_persists(self, settings, store):
        e, market, clock = make(settings, store)
        force(e)
        e.run_cycle()
        added = store.get_meta("watchlist")["AAPL"]["added_ts"]
        clock.t += 3600
        e.run_cycle()
        assert store.get_meta("watchlist")["AAPL"]["added_ts"] == added

    def test_dropped_when_signal_disappears(self, settings, store):
        e, market, _ = make(settings, store)
        force(e)
        e.run_cycle()
        force(e, "hold", 0.0)
        e.run_cycle()
        assert "AAPL" not in (store.get_meta("watchlist") or {})

    def test_direction_flip_replaces_entry(self, settings, store):
        e, market, _ = make(settings, store)
        force(e, "buy")
        e.run_cycle()
        force(e, "sell")
        e.run_cycle()
        assert store.get_meta("watchlist")["AAPL"]["direction"] == "sell"

    def test_disabled_enters_immediately(self, settings, store):
        settings.entry.watch_enabled = False
        e, market, _ = make(settings, store)
        force(e)
        market.price = 100.0
        e.run_cycle()
        assert len(store.open_options()) == 1
        assert not store.get_meta("watchlist")
