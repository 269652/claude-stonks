"""LIVE dashboard view shows only real money: positions come from the live
book (which mirrors the real TR account), never from the paper book, and net
worth is real TR cash + real positions. Paper positions used to appear in the
live view (unarmed live fell back to the paper book). While live is not
armed the engine doesn't trade the live book, so live rows carry no close /
exit-edit handles. Written before implementation (TDD)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmtrade.config import Settings
from lmtrade.core.control import ControlState
from lmtrade.core.state import Store
from lmtrade.web.app import create_app


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=300.0, universe=["AAPL"])
    s.data_dir = tmp_path
    return s


def open_ko(store: Store, isin: str | None, underlying="AAPL") -> int:
    return store.open_option(underlying, "ko_call", strike=80.0,
                             expiry_ts=time.time() + 365 * 86400, iv=0.0,
                             contracts=10.0, entry_premium=2.0, genome_id=None,
                             instrument_type="knockout", barrier=80.0, ratio=0.1,
                             isin=isin)


@pytest.fixture()
def books(settings):
    paper, live = Store(settings.db_path), Store(settings.live_db_path)
    # paper: a simulated position with a mark worth 50
    pid = open_ko(paper, None, "MSFT")
    paper.set_meta("open_option_marks", {str(pid): {"mark_premium": 5.0, "value": 50.0,
                                                    "unrealized_pnl": 29.0, "spot": 100.0}})
    # live: one real certificate worth 30, real TR cash 120
    lid = open_ko(live, "DE000REAL001")
    live.set_meta("open_option_marks", {str(lid): {"mark_premium": 3.0, "value": 30.0,
                                                   "unrealized_pnl": 9.0, "spot": 100.0}})
    live.set_meta("tr_account_cash", 120.0)
    paper.close(); live.close()
    return pid, lid


def view(settings, mode, armed=False):
    c = ControlState(settings.control_path)
    c.set_mode(mode)
    if armed:
        c.armed = True
        c._save()
    return TestClient(create_app(settings)).get("/api/summary").json()


class TestLiveViewBook:
    def test_unarmed_live_view_shows_only_real_positions(self, settings, books):
        s = view(settings, "live")
        assert [p["isin"] for p in s["positions"]] == ["DE000REAL001"]

    def test_live_net_worth_is_real_cash_plus_real_positions(self, settings, books):
        s = view(settings, "live")
        assert s["cash"] == pytest.approx(120.0)
        # real cash + real position net of its 1 EUR exit fee; not + the paper 50
        assert s["equity"] == pytest.approx(149.0)

    def test_unarmed_live_rows_have_no_action_handles(self, settings, books):
        row = view(settings, "live")["positions"][0]
        assert row["close_kind"] is None

    def test_armed_live_rows_are_actionable(self, settings, books):
        row = view(settings, "live", armed=True)["positions"][0]
        assert row["close_kind"] == "option"

    def test_paper_view_unchanged(self, settings, books):
        s = view(settings, "paper")
        assert [p["symbol"] for p in s["positions"]] == ["MSFT ko_call"]
        assert s["positions"][0]["close_kind"] == "option"

    def test_live_view_without_tr_cash_has_no_fabricated_net_worth(self, settings, books):
        live = Store(settings.live_db_path)
        live.set_meta("tr_account_cash", None)
        live.close()
        paper = Store(settings.db_path)
        paper.set_meta("tr_account_cash", None)
        paper.close()
        s = view(settings, "live")
        assert s["cash"] is None


class TestLiveRealized:
    @pytest.fixture()
    def closed(self, settings):
        paper, live = Store(settings.db_path), Store(settings.live_db_path)
        pid = open_ko(paper, None, "MSFT")
        paper.close_option(pid, 3.0, 10.0)            # simulated +10
        lid = open_ko(live, "DE000REAL001")
        live.close_option(lid, 2.3, 3.0)              # real +3
        paper.close(); live.close()

    def realized(self, settings, mode):
        ControlState(settings.control_path).set_mode(mode)
        return TestClient(create_app(settings)).get("/api/realized").json()

    def test_live_view_realized_is_real_money_only(self, settings, closed):
        r = self.realized(settings, "live")
        assert r["total_pnl"] == pytest.approx(3.0)
        assert [x["isin"] for x in r["rows"]] == ["DE000REAL001"]

    def test_paper_view_realized_unchanged(self, settings, closed):
        assert self.realized(settings, "paper")["total_pnl"] == pytest.approx(10.0)


class TestResearchFallbackInLive:
    """News, analysis and signals are market research, not money: right after
    arming live, the fresh live book has none yet, so the dashboard shows the
    paper book's research until the live book has its own."""

    @pytest.fixture()
    def research(self, settings):
        paper = Store(settings.db_path)
        paper.add_news("AAPL", "Apple up on earnings", "bullish")
        paper.set_meta("market_analysis", {"ts": 1.0, "symbols": {"AAPL": {"bias": "buy"}}})
        paper.add_activity("decision", "AAPL: BUY conf 0.80", "AAPL",
                           {"direction": "buy", "confidence": 0.8, "signals": []})
        paper.close()

    def client(self, settings, armed=True):
        c = ControlState(settings.control_path)
        c.set_mode("live")
        c.armed = armed
        c._save()
        return TestClient(create_app(settings))

    def test_empty_live_book_falls_back_to_paper_research(self, settings, research):
        c = self.client(settings)
        assert [n["symbol"] for n in c.get("/api/news").json()] == ["AAPL"]
        assert "AAPL" in c.get("/api/analysis").json()["symbols"]
        assert [s["symbol"] for s in c.get("/api/signals?min_confidence=0.5").json()] == ["AAPL"]

    def test_live_book_research_wins_once_present(self, settings, research):
        live = Store(settings.live_db_path)
        live.add_news("MSFT", "Microsoft live news", "neutral")
        live.set_meta("market_analysis", {"ts": 2.0, "symbols": {"MSFT": {"bias": "sell"}}})
        live.add_activity("decision", "MSFT: SELL conf 0.70", "MSFT",
                          {"direction": "sell", "confidence": 0.7, "signals": []})
        live.close()
        c = self.client(settings)
        assert [n["symbol"] for n in c.get("/api/news").json()] == ["MSFT"]
        assert list(c.get("/api/analysis").json()["symbols"]) == ["MSFT"]
        assert [s["symbol"] for s in c.get("/api/signals?min_confidence=0.5").json()] == ["MSFT"]

    def test_money_endpoints_never_fall_back(self, settings, research):
        s = self.client(settings).get("/api/summary").json()
        assert s["positions"] == []
