"""Optional Trade Republic derivatives catalog adapter.

When TR credentials (TR_PHONE/TR_PIN) are configured AND tr.use_derivatives
is on, paper trading selects REAL Trade Republic knockout certificates (real
ISINs from the issuer catalog) instead of synthetic instruments — fills are
still simulated (paper), but on instruments that actually exist on TR, priced
with the knockout model in finance/knockouts.py against live underlying data.

Without credentials the factory returns None and the engine falls back to the
synthetic Black-Scholes options layer, exactly as before — TR login stays
strictly optional.

Honest operational caveats, written down so nobody is surprised later:
- pytr is an UNOFFICIAL client of TR's private mobile API (against ToS; see
  docs/TRADE_REPUBLIC.md). Logging in triggers the same 2FA as the app.
- TR's API is websocket-based. Corporate/sandboxed proxies frequently do not
  pass websockets (the environment this was developed in explicitly does
  not), so the live client is written defensively: any failure at any stage
  degrades to unavailable, never crashes the engine, and the synthetic
  fallback takes over.
- The derivative-search payload shape differs across pytr versions; the
  parsing here is deliberately tolerant (missing fields -> skip instrument).
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Settings, secret
from ..finance.knockouts import MAX_LEVERAGE, MIN_LEVERAGE
from ..logging_setup import get_logger

log = get_logger("lmtrade.tr")

# TR returns the full knockout catalog for an underlying in a single
# websocket frame — for a liquid name (AAPL etc.) that's well over the
# websockets library's default 1 MiB max_size, which pytr never raises, so
# the frame is rejected with a 1009 "message too big". 32 MiB is generous
# headroom for even the largest catalog while still bounding memory.
WS_MAX_SIZE = 32 * 1024 * 1024


# ---------------------------------------------------------------------------
# Shared pytr session. TR allows ONE active websocket per session cookie: when
# the live broker and the derivatives client each opened their own
# TradeRepublicApi, the second connection was rejected with HTTP 401 (observed
# live: "Connected." followed immediately by a 401 on the next connection).
# Everything in this process must share a single session per phone number.
_SHARED_APIS: dict[str, Any] = {}


def _new_pytr_api(phone: str, pin: str) -> Any:
    from pytr import api as pytr_api  # type: ignore

    _patch_ws_max_size(pytr_api.websockets)
    return pytr_api.TradeRepublicApi(phone_no=phone, pin=pin, save_cookies=True)


def get_shared_api(phone: str, pin: str) -> Any:
    api = _SHARED_APIS.get(phone)
    if api is None:
        api = _new_pytr_api(phone, pin)
        _SHARED_APIS[phone] = api
    return api


def drop_shared_api(phone: str) -> None:
    """Forget the shared session (stale cookie / websocket error) so the next
    get re-creates and re-resumes it."""
    _SHARED_APIS.pop(phone, None)


async def _recv_for(api: Any, sub_id: Any, max_frames: int = 10,
                    timeout: float = 20.0) -> Any:
    """Receive until the frame for OUR subscription arrives. With the shared
    websocket, recv() can hand back another consumer's frame first — matching
    on subscription id prevents e.g. an order path consuming a ticker frame
    (or vice versa) and misreading it as its own response.

    Bounded by `timeout` seconds overall: a subscription TR never answers
    once froze the whole engine (no decisions, no exit checks)."""
    import asyncio

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    for _ in range(max_frames):
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            rid, _, payload = await asyncio.wait_for(api.recv(), remaining)
        except asyncio.TimeoutError:
            break
        if str(rid) == str(sub_id):
            return payload
    raise RuntimeError(f"no response for subscription {sub_id} "
                       f"within {timeout:g}s / {max_frames} frames")


def _resume_once(api: Any) -> bool:
    """resume_websession() exactly once per shared session — a second resume
    on an already-live session can rotate tokens under the open websocket."""
    if getattr(api, "_lmtrade_resumed", False):
        return True
    ok = bool(api.resume_websession())
    if ok:
        api._lmtrade_resumed = True  # noqa: SLF001
    return ok


def _patch_ws_max_size(ws_module: Any) -> None:
    """Wrap the `connect` attribute of pytr's imported `websockets` module so
    every socket pytr opens defaults to WS_MAX_SIZE instead of the 1 MiB
    library default. Necessary because pytr calls websockets.connect() with
    no max_size and reconnects on each search, so we cannot just pre-open one
    socket ourselves. Idempotent, and leaves an explicit max_size untouched."""
    connect = getattr(ws_module, "connect", None)
    if connect is None or getattr(connect, "_lmtrade_patched", False):
        return

    def patched_connect(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("max_size", WS_MAX_SIZE)
        return connect(*args, **kwargs)

    patched_connect._lmtrade_patched = True  # type: ignore[attr-defined]
    ws_module.connect = patched_connect


@dataclass
class TRDerivativeQuote:
    isin: str
    underlying: str
    kind: str            # ko_call | ko_put
    strike: float
    barrier: float
    ratio: float
    price: float         # ask per certificate (0 until quoted — search has no prices)
    leverage: float
    issuer: str = ""
    currency: str = "EUR"   # currency of strike / barrier (the underlying's)
    exchange: str | None = None   # venue the certificate trades on (issuer's, not LSX)


def _parse_cash(payload: Any) -> float | None:
    """TR's `cash` subscription returns per-currency balances,
    e.g. [{"currencyId": "EUR", "amount": 123.45}]. Prefer EUR; fall back to
    the first parseable amount; None if the shape isn't recognized."""
    entries = payload if isinstance(payload, list) else None
    if entries is None:
        return None
    best: float | None = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            amount = float(e.get("amount"))
        except (TypeError, ValueError):
            continue
        if str(e.get("currencyId", "")).upper() == "EUR":
            return amount
        if best is None:
            best = amount
    return best


class TRDerivativesBase:
    """Interface: find the best-fitting real KO instrument for a signal."""

    # How many leverage-ranked candidates to request a live quote for, and how
    # long to wait for each quote before treating it as unavailable.
    QUOTE_TRIES = 5
    QUOTE_TIMEOUT_S = 5.0

    def available(self) -> bool:
        raise NotImplementedError

    def quote_ask(self, isin: str) -> float | None:
        """Live ask for a certificate. None when unavailable — the knockout
        search itself carries no prices."""
        return None

    def exchange_for(self, isin: str) -> str | None:
        """Venue a certificate trades on. None when unknown."""
        return None

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        raise NotImplementedError

    def account_cash(self) -> float | None:
        """Live TR account cash balance (account currency). None unless a real
        authenticated TR client overrides this — paper/fake clients have no
        real account."""
        return None

    def portfolio(self) -> list[dict] | None:
        """Real TR portfolio positions [{isin, size, avg_price}] — None when
        unavailable (no real account / session down), NEVER an empty list on
        failure: [] means 'the account truly holds nothing' and callers may
        reconcile (delete local rows) against it."""
        return None

    def _log_filter_rejections(self, underlying: str, kind: str,
                               results: list[TRDerivativeQuote], no_quote: int = 0) -> None:
        """Say WHY no instrument qualified (once per symbol per hour), with one
        raw TR result, so the response field mapping can be fixed against the
        real payload instead of silently finding nothing."""
        logged: dict = self.__dict__.setdefault("_filter_diag_ts", {})
        if time.time() - logged.get(underlying, 0.0) < 3600:
            return
        logged[underlying] = time.time()
        wrong_kind = sum(q.kind != kind for q in results)
        bad_lev = sum(not MIN_LEVERAGE <= q.leverage <= MAX_LEVERAGE for q in results)
        raw = (self.__dict__.get("_raw_sample") or {}).get(underlying)
        log.warning(
            "TR %s: no knockout qualified out of %d — wrong kind: %d, leverage outside "
            "%.0f-%.0f: %d, no ask quote: %d (of the top %d tried). Raw first result: %s",
            underlying, len(results), wrong_kind, MIN_LEVERAGE, MAX_LEVERAGE, bad_lev,
            no_quote, self.QUOTE_TRIES,
            None if raw is None else {k: raw[k] for k in list(raw)[:30]})

    def find_knockout(self, underlying: str, direction: str, spot: float,
                      target_leverage: float) -> TRDerivativeQuote | None:
        """Best instrument = tradeable leverage closest to target, within the
        sane retail band."""
        kind = "ko_call" if direction == "buy" else "ko_put"
        results = self.search(underlying, direction)
        candidates = sorted((q for q in results
                             if q.kind == kind and MIN_LEVERAGE <= q.leverage <= MAX_LEVERAGE),
                            key=lambda q: abs(q.leverage - target_leverage))
        no_quote = 0
        for q in candidates[:self.QUOTE_TRIES]:
            if q.price > 0:
                return q
            ask = self.quote_ask(q.isin)
            if ask is not None and ask > 0:
                return dataclasses.replace(q, price=ask,
                                           exchange=q.exchange or self.exchange_for(q.isin))
            no_quote += 1
        if results:
            self._log_filter_rejections(underlying, kind, results, no_quote)
        return None
        return min(candidates, key=lambda q: abs(q.leverage - target_leverage))


class FakeTRDerivatives(TRDerivativesBase):
    """Offline stand-in for tests and dry runs: a hand-fed catalog."""

    def __init__(self, catalog: dict[str, list[TRDerivativeQuote]]):
        self._catalog = catalog

    def available(self) -> bool:
        return True

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        return list(self._catalog.get(underlying, []))


class PytrDerivatives(TRDerivativesBase):
    """Live client over pytr. Lazily logs in on first use; every failure mode
    (missing pytr, login/2FA failure, websocket blocked, payload drift)
    degrades to unavailable rather than raising into the engine.

    `api_factory` is an injection point for tests (see FakeAsyncTRApi in
    test_tr_derivatives.py) so this glue code can be verified without the
    optional pytr dependency installed and without ever touching TR's real
    (websocket) API. Left unset, it lazily constructs the real
    pytr.api.TradeRepublicApi.
    """

    def __init__(self, phone: str, pin: str,
                 api_factory: Callable[[], Any] | None = None,
                 now: Callable[[], float] = time.time,
                 retry_backoff_s: float = 60.0):
        self._phone = phone
        self._pin = pin
        self._api_factory = api_factory
        self._api = None
        self._now = now
        self._retry_backoff_s = retry_backoff_s
        self._next_retry = 0.0   # earliest time to re-attempt a failed login

    def _invalidate(self) -> None:
        """Drop the session and schedule a re-login after a backoff. Used both
        when a login fails and when a live call hits a stale-session error
        (e.g. HTTP 401 on the websocket) — the session is NOT abandoned
        permanently, so once `pytr login` refreshes the cookie the next
        attempt re-resumes and the bot recovers without a restart."""
        self._api = None
        self._next_retry = self._now() + self._retry_backoff_s
        drop_shared_api(self._phone)   # stale for every consumer of the session

    def _make_api(self) -> Any:
        if self._api_factory is not None:
            return self._api_factory()
        return get_shared_api(self._phone, self._pin)

    def _login(self):
        if self._api is not None:
            return self._api
        if self._now() < self._next_retry:
            return None   # in backoff after a recent failure — don't hammer TR
        try:
            api = self._make_api()
            # pytr>=0.4 dropped the old `.login()` method. The correct
            # non-interactive flow (mirroring pytr's own `account.login()`
            # reference implementation) is: try to resume the session cookie
            # saved by a prior interactive `pytr login`. If that fails there
            # is no cached session to resume, and the fresh-pairing flow needs
            # interactive 2FA we must never block on here — so we back off and
            # retry, picking up a refreshed cookie automatically once the user
            # re-runs `pytr login`. Resumed at most once per shared session —
            # re-resuming under an open websocket rotates tokens and 401s it.
            if not _resume_once(api):
                log.warning(
                    "TR session not resumable (expired or no cookie). Re-run "
                    "`pytr login -n \"<TR_PHONE>\" -p \"<TR_PIN>\" "
                    "--store_credentials`; the bot will pick up the refreshed "
                    "session on its next attempt. Using synthetic instruments "
                    "meanwhile.")
                self._invalidate()
                return None
            self._api = api
        except Exception as exc:  # noqa: BLE001
            log.warning("TR derivatives unavailable (%s) — retrying later, "
                        "synthetic instruments meanwhile.", exc)
            self._invalidate()
        return self._api

    def available(self) -> bool:
        return self._login() is not None

    async def _resolve_isin(self, api: Any, underlying: str) -> str | None:
        """TR's derivative search takes an ISIN, not a ticker — look the
        underlying up first via TR's own instrument search."""
        sub_id = await api.search(underlying, asset_type="stock")
        payload = await _recv_for(api, sub_id)
        await api.unsubscribe(sub_id)
        for result in (payload or {}).get("results", []):
            isin = result.get("isin")
            if isin:
                return isin
        return None

    # Best-effort productCategory value — TR's backend rejected the plain
    # "knockout" as a schema-invalid payload against a live account (seen as
    # a JSON_PARSE_ERROR/"validation failed" from TR's own MAPPER service).
    # This environment has no live TR access to verify the exact value, so
    # this is a single educated guess: TR's other subscription type names are
    # camelCase compounds (portfolioAggregateHistory, instrumentSuitability,
    # timelineDetailV2), matching this value's shape better than the flat
    # lowercase one did.
    PRODUCT_CATEGORY = "knockOutProduct"

    async def _fetch_derivatives(self, api: Any, isin: str) -> list[dict]:
        sub_id = await api.search_derivative(isin, self.PRODUCT_CATEGORY)
        payload = await _recv_for(api, sub_id)
        await api.unsubscribe(sub_id)
        log.debug("TR search_derivative(%s, %s) -> %r", isin, self.PRODUCT_CATEGORY, payload)
        return (payload or {}).get("results", [])

    def search(self, underlying: str, direction: str) -> list[TRDerivativeQuote]:
        api = self._login()
        if api is None:
            return []
        try:
            import asyncio

            async def _query() -> list[dict]:
                isin = await self._resolve_isin(api, underlying)
                if isin is None:
                    return []
                return await self._fetch_derivatives(api, isin)

            items = asyncio.get_event_loop().run_until_complete(_query())
            if items and isinstance(items[0], dict):
                self.__dict__.setdefault("_raw_sample", {})[underlying] = items[0]
            out: list[TRDerivativeQuote] = []
            first_error: Exception | None = None
            for item in items:
                try:
                    # Direction from TR's own optionType (long/short) — never
                    # from the requested direction: labelling every result by
                    # the request could buy a short turbo on a buy signal.
                    kind = {"long": "ko_call", "short": "ko_put"}[
                        str(item["optionType"]).lower()]
                    size = item.get("size")
                    ratio = 1.0 / float(size) if size else float(item["ratio"])
                    out.append(TRDerivativeQuote(
                        isin=item["isin"], underlying=underlying, kind=kind,
                        strike=float(item["strike"]),
                        barrier=float(item.get("barrier") or item["strike"]),
                        ratio=ratio,
                        price=float(item.get("ask", 0) or 0),   # search has no prices
                        leverage=float(item.get("leverage", 0) or 0),
                        issuer=str(item.get("issuerDisplayName", "")),
                        currency=str(item.get("currency") or "EUR")))
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                    if first_error is None:
                        first_error = exc
                    continue  # tolerate payload drift per-instrument
            if items and not out:
                # TR returned instruments but our field mapping matched none —
                # this is exactly why positions silently become synthetic
                # options (no ISIN). Dump the real field names so the mapping
                # above can be corrected against a live account; without TR
                # access from the dev environment this is unverifiable in code.
                log.warning(
                    "TR knockout search for %s: %d instrument(s) returned but "
                    "NONE parsed (first error: %r). Actual fields on the first "
                    "item: %s. The response field mapping in tr_derivatives.py "
                    "needs updating for your pytr/TR version — falling back to "
                    "synthetic options (no ISIN) meanwhile.",
                    underlying, len(items), first_error, sorted(items[0].keys()))
            elif out:
                log.info("TR knockout search for %s: %d raw -> %d usable instrument(s).",
                         underlying, len(items), len(out))
            else:
                log.debug("TR knockout search for %s: no instruments returned.", underlying)
            return out
        except Exception as exc:  # noqa: BLE001
            # A stale-session error (e.g. HTTP 401 on the websocket) surfaces
            # here — drop the session so the next call re-resumes with a
            # (possibly refreshed) cookie instead of failing forever.
            log.warning("TR derivative search failed (%s) — dropping session, "
                        "will re-login.", exc)
            self._invalidate()
            return []

    def exchange_for(self, isin: str) -> str | None:
        """The certificate's venue from TR's instrument details (issuer
        venues for knockouts — TR never answers LSX tickers for them). LSX
        preferred when listed. Cached; None when unknown."""
        cache = self.__dict__.setdefault("_exchange_cache", {})
        if isin in cache:
            return cache[isin]
        api = self._login()
        if api is None:
            return None
        try:
            import asyncio

            async def _query() -> Any:
                sub_id = await api.instrument_details(isin)
                try:
                    return await _recv_for(api, sub_id, timeout=self.QUOTE_TIMEOUT_S)
                finally:
                    await api.unsubscribe(sub_id)

            payload = asyncio.get_event_loop().run_until_complete(_query())
        except Exception as exc:  # noqa: BLE001
            log.warning("TR instrument details %s failed: %s", isin, exc)
            return None
        ids: list[str] = []
        if isinstance(payload, dict):
            raw_ids = payload.get("exchangeIds")
            if isinstance(raw_ids, list):
                ids = [str(x) for x in raw_ids if x]
            if not ids and isinstance(payload.get("exchanges"), list):
                for x in payload["exchanges"]:
                    if isinstance(x, dict):
                        v = x.get("slug") or x.get("exchangeId") or x.get("id")
                        if v:
                            ids.append(str(v))
        if not ids:
            log.warning("TR instrument details %s: no exchange found; fields: %s", isin,
                        sorted(payload.keys()) if isinstance(payload, dict) else type(payload))
            return None
        exchange = "LSX" if "LSX" in ids else ids[0]
        cache[isin] = exchange
        return exchange

    def quote_ask(self, isin: str) -> float | None:
        api = self._login()
        if api is None:
            return None
        exchange = self.exchange_for(isin) or "LSX"
        try:
            import asyncio

            async def _query() -> Any:
                sub_id = await api.ticker(isin, exchange)
                try:
                    return await _recv_for(api, sub_id, timeout=self.QUOTE_TIMEOUT_S)
                finally:
                    await api.unsubscribe(sub_id)

            payload = asyncio.get_event_loop().run_until_complete(_query())
        except Exception as exc:  # noqa: BLE001
            log.warning("TR ticker %s failed: %s", isin, exc)
            return None
        ask = payload.get("ask") if isinstance(payload, dict) else None
        try:
            price = float(ask.get("price")) if isinstance(ask, dict) else None
        except (TypeError, ValueError):
            return None
        return price if price and price > 0 else None

    def heartbeat(self) -> bool:
        """True when TR answers a (bounded) cash read right now."""
        return self.account_cash() is not None

    def relogin(self) -> bool:
        """Drop the session and log in again immediately (no backoff)."""
        self._invalidate()
        self._next_retry = 0.0
        return self._login() is not None

    def account_cash(self) -> float | None:
        api = self._login()
        if api is None:
            return None
        try:
            import asyncio

            async def _query() -> Any:
                sub_id = await api.cash()
                payload = await _recv_for(api, sub_id)
                await api.unsubscribe(sub_id)
                return payload

            return _parse_cash(asyncio.get_event_loop().run_until_complete(_query()))
        except Exception as exc:  # noqa: BLE001
            # Drop the (stale) session so the next attempt re-resumes from the
            # cookie file — otherwise a websocket 401 recurs every cycle
            # against the same dead session and never picks up a refreshed
            # `pytr login`.
            log.warning("TR account cash fetch failed (%s) — dropping session, "
                        "will re-login.", exc)
            self._invalidate()
            return None

    # TR has served the holdings list under different subscription topics
    # across backend/account versions. Newer accounts REMOVED the classic
    # "compactPortfolio" and "portfolio" topics (both rejected live with
    # BAD_SUBSCRIPTION_TYPE "Unknown topic type: compactPortfolio") and now
    # deliver holdings via "compactPortfolioByType". We subscribe by raw topic
    # (not pytr's named methods) so topics pytr has no wrapper for are still
    # reachable, and try each in order — a rejected TOPIC is not a dead
    # SESSION.
    # Ordered most-current first: newer accounts serve holdings ONLY via
    # compactPortfolioByType and reject the classic topics with
    # BAD_SUBSCRIPTION_TYPE, so trying the modern one first means those
    # accounts never emit the (noisy but harmless) "Unknown topic type:
    # compactPortfolio" error. Older accounts fall through to the classic ones.
    _PORTFOLIO_TOPICS = (
        "compactPortfolioByType",   # current TR app; newer accounts accept only this
        "compactPortfolio",         # classic; still present on older accounts
        "portfolio",                # legacy
        "portfolioStatus",          # last-resort fallback
    )

    @staticmethod
    def _is_unknown_topic(exc: Exception) -> bool:
        """True when TR rejected the SUBSCRIPTION TYPE (wrong/removed topic
        name), which is recoverable by trying another topic — as opposed to a
        session failure (401/websocket drop) that must invalidate the session.
        pytr's TradeRepublicError carries the error payload on `.error` and
        also in its args, so check both."""
        blob = f"{getattr(exc, 'error', '')} {exc}"
        return "BAD_SUBSCRIPTION_TYPE" in blob or "Unknown topic type" in blob

    @staticmethod
    def _iter_positions(payload: Any) -> "list[dict]":
        """Flatten a holdings payload to a list of position dicts. Flat topics
        put them under "positions"; compactPortfolioByType groups them under
        "categories":[{"positions":[...]}]. Handle both."""
        items: list[dict] = []
        if not isinstance(payload, dict):
            return items
        if isinstance(payload.get("positions"), list):
            items.extend(p for p in payload["positions"] if isinstance(p, dict))
        for cat in payload.get("categories", []) or []:
            if isinstance(cat, dict) and isinstance(cat.get("positions"), list):
                items.extend(p for p in cat["positions"] if isinstance(p, dict))
        return items

    @classmethod
    def _parse_portfolio(cls, payload: Any) -> list[dict]:
        out: list[dict] = []
        for item in cls._iter_positions(payload):
            isin = item.get("instrumentId") or item.get("isin")
            try:
                size = float(item.get("netSize") or item.get("size")
                             or item.get("amount") or 0)
                avg = float(item.get("averageBuyIn") or item.get("avg_price") or 0)
            except (TypeError, ValueError):
                continue
            if isin and size > 0:
                out.append({"isin": str(isin), "size": size, "avg_price": avg})
        return out

    def portfolio(self) -> list[dict] | None:
        """Real TR portfolio. Tries each known holdings subscription topic
        until one is accepted; field names are parsed tolerantly across
        account versions and unparseable entries are skipped. None (not []) on
        any failure, so callers never reconcile against phantom-empty data."""
        api = self._login()
        if api is None:
            return None
        import asyncio

        loop = asyncio.get_event_loop()
        last_topic_error: Exception | None = None
        for topic in self._PORTFOLIO_TOPICS:
            try:
                async def _query(_t: str = topic) -> Any:
                    sub_id = await api.subscribe({"type": _t})
                    payload = await _recv_for(api, sub_id)
                    await api.unsubscribe(sub_id)
                    return payload

                payload = loop.run_until_complete(_query())
            except Exception as exc:  # noqa: BLE001
                if self._is_unknown_topic(exc):
                    # Wrong/removed topic for this backend — keep the session,
                    # try the next candidate.
                    last_topic_error = exc
                    log.info("TR portfolio topic %r not supported — trying next.",
                             topic)
                    continue
                log.warning("TR portfolio fetch failed (%s) — dropping session.", exc)
                self._invalidate()
                return None
            return self._parse_portfolio(payload)
        log.warning("TR portfolio: no supported holdings topic (tried %s; last "
                    "error: %r). Position reconciliation skipped — the bot "
                    "continues, but TR positions won't auto-import.",
                    ", ".join(self._PORTFOLIO_TOPICS), last_topic_error)
        return None


def build_tr_derivatives(settings: Settings) -> TRDerivativesBase | None:
    """Factory honoring the opt-in contract: credentials present AND
    tr.use_derivatives enabled, else None (synthetic fallback)."""
    if not settings.tr.use_derivatives:
        return None
    phone, pin = secret("TR_PHONE"), secret("TR_PIN")
    if not (phone and pin):
        return None
    return PytrDerivatives(phone, pin)
