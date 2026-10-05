"""Tests for the optional Trade Republic derivatives adapter
(brokers/tr_derivatives.py) and its engine integration: when TR auth is
configured AND the client is reachable, paper trading uses TR-realistic
knockout instruments (real ISINs when the live client provides them); when
not, the engine falls back to the existing synthetic Black-Scholes options —
unchanged behavior. All offline: the live pytr client is faked. Written
before implementation per strict TDD."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lmtrade.agents.fusion import Decision
from lmtrade.brokers.paper import PaperBroker
from lmtrade.brokers.tr_derivatives import (
    WS_MAX_SIZE,
    FakeTRDerivatives,
    PytrDerivatives,
    TRDerivativeQuote,
    _patch_ws_max_size,
    build_tr_derivatives,
)
from lmtrade.config import Settings
from lmtrade.core.engine import Engine
from lmtrade.core.state import Store
from lmtrade.data.market import Quote


def buy_signal_history(base: float, n: int = 60) -> list[float]:
    return [base + i * 0.6 + 3 * math.sin(i * 0.7) for i in range(n)]


class ScriptedMarket:
    def __init__(self, quotes: dict[str, Quote]):
        self._quotes = quotes

    def quote(self, symbol: str) -> Quote:
        return self._quotes.get(symbol, Quote(symbol, 100.0, [100.0] * 60, "yahoo"))


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(mode="paper", budget=100.0, universe=["AAPL"],
                 data={"provider": "yahoo", "intraday": True})
    s.model.stack = ["heuristic"]
    s.data_dir = tmp_path
    s.loop.interval_seconds = 1
    s.risk.min_confidence = 0.5
    s.risk.max_fee_pct = 0.0   # fee guard has its own tests (test_fee_guard.py)
    s.options.enabled = True
    s.learning.enabled = False
    s.tr.use_derivatives = True
    return s


@pytest.fixture()
def store(settings: Settings):
    st = Store(settings.db_path)
    yield st
    st.close()


class TestFactory:
    def test_no_credentials_returns_none(self, settings, monkeypatch):
        monkeypatch.delenv("TR_PHONE", raising=False)
        monkeypatch.delenv("TR_PIN", raising=False)
        assert build_tr_derivatives(settings) is None

    def test_disabled_in_config_returns_none(self, settings, monkeypatch):
        monkeypatch.setenv("TR_PHONE", "+491234")
        monkeypatch.setenv("TR_PIN", "1234")
        settings.tr.use_derivatives = False
        assert build_tr_derivatives(settings) is None


class FakeAsyncTRApi:
    """Minimal async fake mirroring the actual pytr>=0.4 `TradeRepublicApi`
    surface PytrDerivatives relies on (resume_websession/search/
    search_derivative/recv/unsubscribe) — verified against the real
    installed pytr 0.4.9 source, not guessed. Lets PytrDerivatives' own glue
    logic be tested without the optional pytr dependency installed and
    without ever touching TR's real (websocket) API."""

    def __init__(self, resume_ok=True, search_results=None, derivative_results=None,
                 cash_payload=None):
        self.resume_ok = resume_ok
        self._search_results = (
            search_results if search_results is not None else [{"isin": "US0378331005"}])
        self._derivative_results = derivative_results if derivative_results is not None else []
        self._cash_payload = (cash_payload if cash_payload is not None
                              else [{"currencyId": "EUR", "amount": 42.5}])
        self.calls: list[tuple] = []

    def resume_websession(self) -> bool:
        return self.resume_ok

    async def search(self, query, asset_type="stock"):
        self.calls.append(("search", query, asset_type))
        return "sub-search"

    async def search_derivative(self, isin, product_type):
        self.calls.append(("search_derivative", isin, product_type))
        return "sub-deriv"

    async def cash(self):
        self.calls.append(("cash",))
        return "sub-cash"

    async def recv(self):
        kind = self.calls[-1][0]
        if kind == "search":
            return ("sub-search", {}, {"results": self._search_results})
        if kind == "cash":
            return ("sub-cash", {}, self._cash_payload)
        return ("sub-deriv", {}, {"results": self._derivative_results})

    async def unsubscribe(self, sub_id):
        self.calls.append(("unsubscribe", sub_id))


class FakeWebsocketsModule:
    """Stand-in for the `websockets` module referenced inside pytr.api, so
    the max_size patch can be tested without pytr or a real socket."""

    def __init__(self):
        self.connect_calls: list[dict] = []

        def connect(uri, **kwargs):
            self.connect_calls.append({"uri": uri, **kwargs})
            return ("fake-ws", uri, kwargs)

        self.connect = connect


class TestWebsocketMaxSizePatch:
    """TR returns >1 MiB of instruments for a knockout search on a liquid
    underlying; pytr opens its websocket with the library-default 1 MiB
    max_size and reconnects on every search, so the frame is rejected with
    a 1009 'message too big'. We raise the cap on pytr's own connect()."""

    def test_patch_injects_larger_max_size(self):
        ws = FakeWebsocketsModule()
        _patch_ws_max_size(ws)
        ws.connect("wss://api.traderepublic.com", ssl=None)
        assert ws.connect_calls[-1]["max_size"] == WS_MAX_SIZE
        assert WS_MAX_SIZE > 1024 * 1024   # bigger than the 1 MiB default

    def test_patch_preserves_explicit_max_size(self):
        ws = FakeWebsocketsModule()
        _patch_ws_max_size(ws)
        ws.connect("wss://x", max_size=999)
        assert ws.connect_calls[-1]["max_size"] == 999

    def test_patch_is_idempotent(self):
        ws = FakeWebsocketsModule()
        original = ws.connect
        _patch_ws_max_size(ws)
        once = ws.connect
        _patch_ws_max_size(ws)
        assert ws.connect is once            # not re-wrapped
        assert ws.connect is not original    # but was wrapped the first time
        ws.connect("wss://x")
        assert ws.connect_calls[-1]["max_size"] == WS_MAX_SIZE

    def test_patch_forwards_other_kwargs_and_return(self):
        ws = FakeWebsocketsModule()
        _patch_ws_max_size(ws)
        result = ws.connect("wss://y", ssl="ctx", additional_headers={"Cookie": "x"})
        assert result[0] == "fake-ws"
        call = ws.connect_calls[-1]
        assert call["ssl"] == "ctx"
        assert call["additional_headers"] == {"Cookie": "x"}


class TestPytrLoginFlow:
    """The installed pytr>=0.4 TradeRepublicApi has no `.login()` method —
    confirmed by inspecting the actual installed package. Non-interactive use
    (from inside the engine loop) must resume a cached session via
    resume_websession() and degrade — never block on interactive 2FA input —
    when there isn't one to resume."""

    def test_resumes_cached_session_without_interactive_login(self):
        api = FakeAsyncTRApi(resume_ok=True)
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.available() is True

    def test_unresumable_session_degrades_without_blocking(self):
        api = FakeAsyncTRApi(resume_ok=False)
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.available() is False
        assert client.search("AAPL", "buy") == []

    def test_factory_exception_degrades_gracefully(self):
        def boom():
            raise RuntimeError("no pytr installed")

        client = PytrDerivatives("+491234", "1234", api_factory=boom)
        assert client.available() is False

    def test_login_result_is_cached_not_repeated(self):
        calls = []

        def factory():
            calls.append(1)
            return FakeAsyncTRApi(resume_ok=True)

        client = PytrDerivatives("+491234", "1234", api_factory=factory)
        client.available()
        client.available()
        assert len(calls) == 1


class TestPytrSearchGlue:
    """search() must resolve the underlying ticker to an ISIN first (TR's
    search_derivative takes an ISIN, not a ticker) then query derivatives —
    the previous code called a nonexistent `derivative_search` method with
    the wrong signature entirely."""

    def test_resolves_isin_then_queries_derivatives(self):
        api = FakeAsyncTRApi(
            search_results=[{"isin": "US0378331005"}],
            derivative_results=[{
                "isin": "DE000ABC123", "strike": 180.0, "barrier": 180.0,
                "ratio": 10.0, "ask": 2.5, "leverage": 5.0,
                "issuerDisplayName": "TestBank",
            }])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        quotes = client.search("AAPL", "buy")
        assert len(quotes) == 1
        assert quotes[0].isin == "DE000ABC123"
        assert quotes[0].kind == "ko_call"
        assert quotes[0].leverage == pytest.approx(5.0)
        assert ("search", "AAPL", "stock") in api.calls
        assert ("search_derivative", "US0378331005", "knockOutProduct") in api.calls

    def test_sell_direction_maps_to_ko_put(self):
        api = FakeAsyncTRApi(
            derivative_results=[{"isin": "DE1", "strike": 100.0, "ask": 1.0, "leverage": 4.0}])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        quotes = client.search("AAPL", "sell")
        assert quotes[0].kind == "ko_put"

    def test_no_isin_match_returns_empty(self):
        api = FakeAsyncTRApi(search_results=[])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.search("UNKNOWN", "buy") == []
        assert not any(c[0] == "search_derivative" for c in api.calls)

    def test_malformed_derivative_items_are_skipped_not_crashed(self):
        api = FakeAsyncTRApi(
            search_results=[{"isin": "US0378331005"}],
            derivative_results=[
                {"isin": "DE1"},  # missing strike -> skipped
                {"isin": "DE2", "strike": 50.0, "ask": 1.0, "leverage": 3.0},
            ])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        quotes = client.search("AAPL", "buy")
        assert len(quotes) == 1
        assert quotes[0].isin == "DE2"

    def test_unavailable_client_short_circuits_search(self):
        api = FakeAsyncTRApi(resume_ok=False)
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        assert client.search("AAPL", "buy") == []
        assert api.calls == []


class TestPytrAccountCash:
    """Live TR account cash balance for the dashboard (replaces the GPU/
    compute card). Parses TR's per-currency cash payload; degrades to None."""

    def test_parses_eur_amount(self):
        api = FakeAsyncTRApi(cash_payload=[{"currencyId": "EUR", "amount": 123.45}])
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.account_cash() == pytest.approx(123.45)

    def test_prefers_eur_over_other_currencies(self):
        api = FakeAsyncTRApi(cash_payload=[
            {"currencyId": "USD", "amount": 10.0},
            {"currencyId": "EUR", "amount": 55.0}])
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.account_cash() == pytest.approx(55.0)

    def test_none_when_unavailable(self):
        api = FakeAsyncTRApi(resume_ok=False)
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.account_cash() is None

    def test_none_on_malformed_payload(self):
        api = FakeAsyncTRApi(cash_payload={"unexpected": "shape"})
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.account_cash() is None

    def test_base_and_fake_default_to_none(self):
        assert FakeTRDerivatives(catalog={}).account_cash() is None


class TestSessionRecovery:
    """A lost session (e.g. HTTP 401 on the websocket) must NOT disable TR
    permanently — after a backoff the client re-resumes, so once the user
    re-runs `pytr login` and the cookie refreshes, the bot recovers without a
    restart."""

    def test_failed_login_retries_after_backoff(self):
        clock = [1000.0]
        states = iter([False, True])   # first resume fails, second succeeds

        class FlakyApi(FakeAsyncTRApi):
            def resume_websession(self):
                return next(states)

        made = []

        def factory():
            made.append(1)
            return FlakyApi()

        client = PytrDerivatives("+49", "1", api_factory=factory,
                                 now=lambda: clock[0], retry_backoff_s=60.0)
        assert client.available() is False        # first attempt fails
        assert client.available() is False        # still in backoff, no new attempt
        assert len(made) == 1
        clock[0] += 61                            # backoff elapsed
        assert client.available() is True         # re-resumes successfully
        assert len(made) == 2

    def test_search_error_drops_session_for_reresume(self):
        clock = [1000.0]

        class Boom(FakeAsyncTRApi):
            async def recv(self):
                raise RuntimeError("server rejected WebSocket connection: HTTP 401")

        client = PytrDerivatives("+49", "1", api_factory=lambda: Boom(),
                                 now=lambda: clock[0], retry_backoff_s=60.0)
        assert client.search("AAPL", "buy") == []
        # Session invalidated + in backoff now.
        assert client._api is None
        assert client._next_retry > clock[0]


class TestPytrSearchDiagnostics:
    """When TR returns instruments but none parse (guessed field names don't
    match TR's real schema), the result is a silent fall-back to synthetic
    options with no ISIN. The search must log LOUDLY what TR actually
    returned so the field mapping can be corrected against real data."""

    def test_warns_with_field_names_when_none_parse(self, caplog):
        import logging

        # Realistic-but-different schema: strike is under "strikePrice", etc.
        api = FakeAsyncTRApi(
            search_results=[{"isin": "US0378331005"}],
            derivative_results=[
                {"isin": "DE1", "strikePrice": 100.0, "leverageFactor": 5.0, "askPrice": 2.0},
                {"isin": "DE2", "strikePrice": 90.0, "leverageFactor": 6.0, "askPrice": 1.5},
            ])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        with caplog.at_level(logging.WARNING, logger="lmtrade.tr"):
            out = client.search("AAPL", "buy")
        assert out == []
        blob = " ".join(r.message for r in caplog.records)
        assert "2" in blob                     # reported the raw count
        assert "strikePrice" in blob           # dumped the real field names

    def test_info_reports_usable_count_on_success(self, caplog):
        import logging

        api = FakeAsyncTRApi(
            search_results=[{"isin": "US0378331005"}],
            derivative_results=[
                {"isin": "DE1", "strike": 100.0, "ask": 2.0, "leverage": 5.0}])
        client = PytrDerivatives("+491234", "1234", api_factory=lambda: api)
        with caplog.at_level(logging.INFO, logger="lmtrade.tr"):
            out = client.search("AAPL", "buy")
        assert len(out) == 1
        blob = " ".join(r.message for r in caplog.records)
        assert "AAPL" in blob


class FakeTRError(ValueError):
    """Mirrors pytr.api.TradeRepublicError: a rejected subscription surfaces as
    an exception carrying the error payload on `.error` (it has no message
    args, so str(exc) is empty — the code must inspect `.error`)."""

    def __init__(self, sub_id, subscription, error):
        self.subscription_id = sub_id
        self.subscription = subscription
        self.error = error


class FakePortfolioApi:
    """Fake pytr session whose holdings-topic support is configurable, to
    model the live incident where TR rejected BOTH `compactPortfolio` and
    `portfolio` with BAD_SUBSCRIPTION_TYPE (newer accounts serve holdings only
    via `compactPortfolioByType`). Subscribes are made by raw topic via
    subscribe({"type": ...}), mirroring PytrDerivatives.portfolio()."""

    def __init__(self, supported=("compactPortfolio",), positions=None,
                 resume_ok=True, recv_error=None, payload_by_topic=None):
        self.supported = set(supported)
        self.positions = positions if positions is not None else []
        self.resume_ok = resume_ok
        self._recv_error = recv_error         # a non-topic error (e.g. 401)
        self._payload_by_topic = payload_by_topic or {}
        self.topics: list[str] = []           # topics subscribed, in order
        self.calls: list = []
        self._counter = 0
        self._pending: dict[str, str] = {}    # sub_id -> topic
        self._last_sub: str | None = None

    def resume_websession(self):
        return self.resume_ok

    async def subscribe(self, payload):
        topic = payload.get("type")
        self.topics.append(topic)
        self.calls.append(("subscribe", topic))
        self._counter += 1
        sub_id = f"sub-{self._counter}"
        self._pending[sub_id] = topic
        self._last_sub = sub_id
        return sub_id

    async def recv(self):
        if self._recv_error is not None:
            raise self._recv_error
        sub_id = self._last_sub
        topic = self._pending.get(sub_id)
        if topic not in self.supported:
            raise FakeTRError(
                sub_id, {"type": topic},
                {"errors": [{"errorCode": "BAD_SUBSCRIPTION_TYPE",
                             "errorMessage": f"Unknown topic type: {topic}.31"}]})
        payload = self._payload_by_topic.get(topic, {"positions": self.positions})
        return (sub_id, {}, payload)

    async def unsubscribe(self, sub_id):
        self._pending.pop(sub_id, None)
        self.calls.append(("unsubscribe", sub_id))


class TestPytrPortfolio:
    """portfolio() reconciles the live book against the real TR account. The
    holdings list has moved between subscription topics across account
    versions: newer accounts REMOVED `compactPortfolio` and `portfolio` (both
    rejected with BAD_SUBSCRIPTION_TYPE) and serve holdings only via
    `compactPortfolioByType`. A rejected TOPIC is not a dead SESSION, so the
    client tries the next candidate rather than tearing the session down."""

    POSITIONS = [{"instrumentId": "DE000KO1", "netSize": "5", "averageBuyIn": "2.5"}]

    def test_parses_positions_from_compact_topic(self):
        api = FakePortfolioApi(supported=("compactPortfolio",), positions=self.POSITIONS)
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        out = client.portfolio()
        assert out == [{"isin": "DE000KO1", "size": 5.0, "avg_price": 2.5}]
        assert "compactPortfolio" in api.topics       # classic topic accepted here

    def test_falls_back_to_portfolio_topic_when_compact_rejected(self):
        # Both compact topics unknown to this backend; legacy portfolio accepted.
        api = FakePortfolioApi(supported=("portfolio",), positions=self.POSITIONS)
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        out = client.portfolio()
        assert out == [{"isin": "DE000KO1", "size": 5.0, "avg_price": 2.5}]
        # Modern topic tried first, then the classic ones, until one is accepted.
        assert api.topics[:3] == ["compactPortfolioByType", "compactPortfolio", "portfolio"]

    def test_newer_account_uses_compact_portfolio_by_type(self):
        # The live regression: both classic topics rejected, holdings served
        # via compactPortfolioByType, which groups positions under categories.
        api = FakePortfolioApi(
            supported=("compactPortfolioByType",),
            payload_by_topic={"compactPortfolioByType": {"categories": [
                {"categoryType": "derivative", "positions": [
                    {"instrumentId": "DE000KO1", "netSize": "5", "averageBuyIn": "2.5"}]},
                {"categoryType": "stock", "positions": [
                    {"instrumentId": "US0378331005", "netSize": "2", "averageBuyIn": "180"}]},
            ]}})
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        out = client.portfolio()
        assert {p["isin"] for p in out} == {"DE000KO1", "US0378331005"}
        assert client._api is api            # classic-topic rejections kept the session

    def test_rejected_topic_does_not_drop_session(self):
        api = FakePortfolioApi(supported=("portfolio",), positions=self.POSITIONS)
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        client.portfolio()
        assert client._api is api             # session preserved, not invalidated

    def test_parses_alternate_field_names(self):
        api = FakePortfolioApi(
            supported=("portfolio",),
            positions=[{"isin": "DE000X", "amount": "3", "avg_price": "9.0"}])
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.portfolio() == [{"isin": "DE000X", "size": 3.0, "avg_price": 9.0}]

    def test_none_when_all_topics_rejected(self):
        api = FakePortfolioApi(supported=())     # no topic accepted
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.portfolio() is None

    def test_real_session_error_returns_none_and_drops_session(self):
        boom = RuntimeError("server rejected WebSocket connection: HTTP 401")
        api = FakePortfolioApi(recv_error=boom)
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.portfolio() is None
        assert client._api is None            # a real session failure DOES drop it

    def test_none_when_unavailable(self):
        api = FakePortfolioApi(resume_ok=False)
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.portfolio() is None

    def test_skips_unparseable_entries(self):
        api = FakePortfolioApi(supported=("compactPortfolio",), positions=[
            {"instrumentId": "DE000A", "netSize": "bad", "averageBuyIn": "1"},  # bad size
            {"netSize": "2", "averageBuyIn": "1"},                              # no isin
            {"instrumentId": "DE000B", "netSize": "0", "averageBuyIn": "1"},    # zero size
            {"instrumentId": "DE000C", "netSize": "4", "averageBuyIn": "3"},    # good
        ])
        client = PytrDerivatives("+49", "1", api_factory=lambda: api)
        assert client.portfolio() == [{"isin": "DE000C", "size": 4.0, "avg_price": 3.0}]


class TestFakeClient:
    def test_search_returns_ko_quotes_sorted_by_leverage_fit(self):
        fake = FakeTRDerivatives(catalog={
            "AAPL": [
                TRDerivativeQuote(isin="DE000A1", underlying="AAPL",
                                  kind="ko_call", strike=80.0, barrier=80.0,
                                  ratio=10.0, price=2.05, leverage=5.0,
                                  issuer="FakeBank"),
                TRDerivativeQuote(isin="DE000A2", underlying="AAPL",
                                  kind="ko_call", strike=95.0, barrier=95.0,
                                  ratio=10.0, price=0.55, leverage=20.0,
                                  issuer="FakeBank"),
            ]})
        best = fake.find_knockout("AAPL", direction="buy", spot=100.0,
                                  target_leverage=6.0)
        assert best.isin == "DE000A1"     # 5x is closer to 6x than 20x
        assert fake.available() is True

    def test_no_instrument_for_symbol_returns_none(self):
        fake = FakeTRDerivatives(catalog={})
        assert fake.find_knockout("MSFT", "buy", 100.0, 5.0) is None


class TestEngineIntegration:
    def _engine(self, settings, store, client):
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        market = ScriptedMarket({
            "AAPL": Quote("AAPL", 100.0, buy_signal_history(80.0), "yahoo"),
        })
        eng = Engine(settings, store, broker, market=market,
                     tr_derivatives=client)
        eng._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        return eng

    def test_with_tr_client_opens_knockout_position(self, settings, store):
        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000KO1", underlying="AAPL",
                                       kind="ko_call", strike=80.0, barrier=80.0,
                                       ratio=10.0, price=2.05, leverage=5.0,
                                       issuer="FakeBank")]})
        engine = self._engine(settings, store, client)
        engine.run_cycle()
        opts = store.open_options()
        assert opts, "expected a knockout position"
        o = opts[0]
        assert o["instrument_type"] == "knockout"
        assert o["isin"] == "DE000KO1"
        assert o["barrier"] == pytest.approx(80.0)
        assert o["tp_premium"] and o["sl_premium"]   # stops still placed

    def test_without_tr_client_falls_back_to_bs_options(self, settings, store):
        engine = self._engine(settings, store, client=None)
        engine.run_cycle()
        opts = store.open_options()
        assert opts
        assert opts[0]["instrument_type"] == "option"   # legacy path unchanged
        assert opts[0]["isin"] is None

    def test_run_cycle_persists_tr_account_cash(self, settings, store):
        class CashClient(FakeTRDerivatives):
            def account_cash(self):
                return 314.15

        engine = self._engine(settings, store, CashClient(catalog={}))
        engine.run_cycle()
        assert store.get_meta("tr_account_cash") == pytest.approx(314.15)

    def test_armed_live_broker_places_real_knockout_order(self, settings, store):
        from lmtrade.brokers.base import OrderResult

        placed = []

        class ArmedBroker:
            mode = "live"
            armed = True

            def cash(self):
                return 1000.0

            def place_order(self, isin, side, size, exchange="LSX"):
                placed.append({"isin": isin, "side": side, "size": size})
                return OrderResult(True, isin, side, size, 0.0, 1.0, "live order placed")

        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000LIVE", underlying="AAPL",
                                       kind="ko_call", strike=80.0, barrier=80.0,
                                       ratio=10.0, price=2.0, leverage=5.0,
                                       issuer="Bank")]})
        market = ScriptedMarket({"AAPL": Quote("AAPL", 100.0, buy_signal_history(80.0), "yahoo")})
        engine = Engine(settings, store, ArmedBroker(), market=market, tr_derivatives=client)
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert placed, "a real order should have been placed"
        assert placed[0]["isin"] == "DE000LIVE"
        assert placed[0]["side"] == "buy"
        # Position recorded in the (live) book with the real ISIN.
        opts = store.open_options()
        assert opts and opts[0]["isin"] == "DE000LIVE"

    def test_low_balance_blocks_real_order_without_second_guard(self, settings, store):
        from lmtrade.brokers.base import OrderResult
        from lmtrade.core.control import ControlState

        # First guard armed, but balance under 100 and second guard NOT armed.
        c = ControlState.load(settings.control_path)
        c.set_mode("live")
        c.arm(confirm=True)
        placed = []

        class PoorBroker:
            mode = "live"
            armed = True

            def cash(self):
                return 42.0                      # under LOW_BALANCE_EUR

            def place_order(self, isin, side, size, exchange="LSX"):
                placed.append(isin)
                return OrderResult(True, isin, side, size, 0.0, 1.0, "placed")

        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000X", underlying="AAPL",
                                       kind="ko_call", strike=80.0, barrier=80.0,
                                       ratio=10.0, price=2.0, leverage=5.0, issuer="B")]})
        market = ScriptedMarket({"AAPL": Quote("AAPL", 100.0, buy_signal_history(80.0), "yahoo")})
        engine = Engine(settings, store, PoorBroker(), market=market, tr_derivatives=client)
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert placed == []                      # blocked by the low-balance guard
        assert store.open_options() == []

    def test_low_balance_allowed_with_second_guard(self, settings, store):
        from lmtrade.brokers.base import OrderResult
        from lmtrade.core.control import ControlState

        c = ControlState.load(settings.control_path)
        c.set_mode("live")
        c.arm(confirm=True)
        c.set_double_armed(confirm=True)         # second guard armed
        placed = []

        class PoorBroker:
            mode = "live"
            armed = True

            def cash(self):
                return 42.0

            def place_order(self, isin, side, size, exchange="LSX"):
                placed.append(isin)
                return OrderResult(True, isin, side, size, 0.0, 1.0, "placed")

        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000Y", underlying="AAPL",
                                       kind="ko_call", strike=80.0, barrier=80.0,
                                       ratio=10.0, price=2.0, leverage=5.0, issuer="B")]})
        market = ScriptedMarket({"AAPL": Quote("AAPL", 100.0, buy_signal_history(80.0), "yahoo")})
        engine = Engine(settings, store, PoorBroker(), market=market, tr_derivatives=client)
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert placed == ["DE000Y"]              # second guard permits it

    def test_armed_live_no_knockout_does_not_book_synthetic(self, settings, store):
        # THE phantom-trade bug: in armed live mode, when no real TR knockout
        # is tradeable, the engine must SKIP — never book a synthetic option,
        # which would appear on the dashboard as a trade the TR app never made.
        from lmtrade.brokers.base import OrderResult

        class ArmedBroker:
            mode = "live"
            armed = True

            def cash(self):
                return 1000.0

            def place_order(self, isin, side, size, exchange="LSX"):
                return OrderResult(True, isin, side, size, 0.0, 1.0, "placed")

        # Empty catalog -> find_knockout returns None -> _enter_knockout False.
        client = FakeTRDerivatives(catalog={})
        market = ScriptedMarket({"AAPL": Quote("AAPL", 100.0, buy_signal_history(80.0), "yahoo")})
        engine = Engine(settings, store, ArmedBroker(), market=market, tr_derivatives=client)
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert store.open_options() == []   # no phantom synthetic option booked

    def test_rejected_live_order_opens_no_position(self, settings, store):
        from lmtrade.brokers.base import OrderResult

        class RejectBroker:
            mode = "live"
            armed = True

            def cash(self):
                return 1000.0

            def place_order(self, isin, side, size, exchange="LSX"):
                return OrderResult(False, isin, side, size, 0.0, 1.0, "TR rejected")

        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000X", underlying="AAPL",
                                       kind="ko_call", strike=80.0, barrier=80.0,
                                       ratio=10.0, price=2.0, leverage=5.0, issuer="B")]})
        market = ScriptedMarket({"AAPL": Quote("AAPL", 100.0, buy_signal_history(80.0), "yahoo")})
        engine = Engine(settings, store, RejectBroker(), market=market, tr_derivatives=client)
        engine._decide = lambda q: (Decision(q.symbol, "buy", 0.9, "scripted"), None)
        engine.run_cycle()
        assert store.open_options() == []   # no fallback to synthetic when live-armed

    def test_knockout_position_knocked_out_when_barrier_touched(self, settings, store):
        client = FakeTRDerivatives(catalog={
            "AAPL": [TRDerivativeQuote(isin="DE000KO1", underlying="AAPL",
                                       kind="ko_call", strike=95.0, barrier=95.0,
                                       ratio=10.0, price=0.55, leverage=20.0,
                                       issuer="FakeBank")]})
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        market = ScriptedMarket({
            # Spot crashes through the 95 barrier
            "AAPL": Quote("AAPL", 94.0, [100.0] * 60, "yahoo"),
        })
        # Plant an open KO position directly, then run a cycle to manage it.
        store.open_option("AAPL", "ko_call", strike=95.0, expiry_ts=4e12,
                          iv=0.0, contracts=10.0, entry_premium=0.55,
                          genome_id=None, tp_premium=0.85, sl_premium=0.30,
                          instrument_type="knockout", barrier=95.0,
                          ratio=10.0, isin="DE000KO1")
        engine = Engine(settings, store, broker, market=market,
                        tr_derivatives=client)
        engine.run_cycle()
        closed = store.closed_options()
        assert closed, "barrier touch must close the position"
        assert closed[0]["exit_premium"] == pytest.approx(0.0, abs=1e-9)
        # Total loss of premium (that's what a knockout IS)
        assert closed[0]["pnl"] < 0

    def test_min_hold_does_not_block_knockout(self, settings, store):
        """A knockout is an involuntary event — min_hold_hours must never
        keep a dead instrument on the books."""
        settings.options.min_hold_hours = 1000.0
        client = FakeTRDerivatives(catalog={})
        broker = PaperBroker(store, starting_cash=settings.budget, fee=0.1)
        market = ScriptedMarket({"AAPL": Quote("AAPL", 94.0, [100.0] * 60, "yahoo")})
        store.open_option("AAPL", "ko_call", strike=95.0, expiry_ts=4e12,
                          iv=0.0, contracts=10.0, entry_premium=0.55,
                          genome_id=None, tp_premium=0.85, sl_premium=0.30,
                          instrument_type="knockout", barrier=95.0,
                          ratio=10.0, isin="DE000KO1")
        engine = Engine(settings, store, broker, market=market,
                        tr_derivatives=client)
        engine.run_cycle()
        assert store.closed_options(), "knockout fires regardless of min_hold"
