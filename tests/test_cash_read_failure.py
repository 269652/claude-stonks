"""A failed TR cash read returns 0.0 with broker.last_cash_ok False. The
engine must not take that as a real zero (live incident 20:04: net worth
dropped from ~299 to 122 EUR for one cycle and order sizing went to 0) — it
keeps the last good cash instead. Written before the fix (TDD)."""
from __future__ import annotations

from pathlib import Path

from lmtrade.brokers.paper import PaperBroker
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store


class FlakyCash(PaperBroker):
    def __init__(self, store):
        super().__init__(store, starting_cash=300.0)
        self.fail = False
        self.last_cash_ok = True

    def cash(self):
        if self.fail:
            self.last_cash_ok = False
            return 0.0
        self.last_cash_ok = True
        return 150.0


def make(tmp_path: Path):
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"])
    s.data_dir = tmp_path
    store = Store(s.db_path)
    b = FlakyCash(store)
    return Engine(s, store, b, tr_derivatives=None), b, store


def test_failed_read_uses_last_good_cash(tmp_path):
    e, b, store = make(tmp_path)
    try:
        assert e._cash() == 150.0
        b.fail = True
        assert e._cash() == 150.0
    finally:
        store.close()


def test_failed_read_without_history_is_zero(tmp_path):
    e, b, store = make(tmp_path)
    try:
        b.fail = True
        assert e._cash() == 0.0
    finally:
        store.close()


def test_brokers_without_the_flag_pass_through(tmp_path):
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"])
    s.data_dir = tmp_path
    store = Store(s.db_path)
    try:
        e = Engine(s, store, PaperBroker(store, starting_cash=300.0), tr_derivatives=None)
        assert e._cash() == 300.0
    finally:
        store.close()
