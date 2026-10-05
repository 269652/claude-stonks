"""Trade Republic LIVE execution adapter (real money).

⚠️  Trade Republic has NO official trading API. This adapter drives the
unofficial, reverse-engineered `pytr` client over TR's private mobile API.
Using it may violate Trade Republic's Terms of Service and can get your account
locked. It requires phone + PIN login and app-based 2FA. Fills are REAL and
IRREVERSIBLE. Enable only if you accept that risk.

Defense in depth: even when this broker is constructed, it refuses to place an
order unless `armed=True` was passed (the engine passes `control.live_armed`,
which is only ever true when the dashboard's paper/live toggle is on *live*
AND the arm+confirm guard has been satisfied). So a real order requires the
control plane, the engine routing, AND this flag to all agree.
"""
from __future__ import annotations

from typing import Any, Callable

from ..config import Settings, secret
from ..core.state import Store
from ..logging_setup import get_logger
from .base import Broker, OrderResult
from .tr_derivatives import _parse_cash

log = get_logger("lmtrade.tr")

# TR's default trading venue for retail; LS Exchange (Lang & Schwarz).
DEFAULT_EXCHANGE = "LSX"


class TradeRepublicBroker(Broker):
    mode = "live"

    def __init__(self, store: Store, settings: Settings, *, armed: bool = False,
                 api_factory: Callable[[], Any] | None = None):
        self.store = store
        self.settings = settings
        self.armed = bool(armed)
        self._api_factory = api_factory
        self.phone = secret("TR_PHONE")
        self.pin = secret("TR_PIN")
        if not (self.phone and self.pin):
            raise RuntimeError(
                "Live mode needs TR_PHONE and TR_PIN in the environment. "
                "Refusing to start live trading without credentials."
            )
        self._api = None
        self._next_retry = 0.0
        self.last_cash_ok = False   # did the most recent cash() read succeed?

    def _invalidate(self) -> None:
        """Drop the session and back off, so a refreshed cookie (after the
        user re-runs `pytr login`) is picked up on a later attempt instead of
        disabling live execution permanently."""
        import time

        from .tr_derivatives import drop_shared_api
        self._api = None
        self._next_retry = time.time() + 60.0
        drop_shared_api(self.phone)

    # -- session -------------------------------------------------------------
    def _make_api(self) -> Any:
        if self._api_factory is not None:
            return self._api_factory()
        # SHARED with the derivatives client: TR allows one active websocket
        # per session cookie — a second concurrent connection gets HTTP 401.
        from .tr_derivatives import get_shared_api
        return get_shared_api(self.phone, self.pin)

    def _login(self):
        import time

        from .tr_derivatives import _resume_once
        if self._api is not None:
            return self._api
        if time.time() < self._next_retry:
            return None   # backoff after a recent failure
        try:
            api = self._make_api()
            if not _resume_once(api):
                log.warning("TR live session not resumable — re-run `pytr login "
                            "--store_credentials`; the bot will pick it up on "
                            "its next attempt.")
                self._invalidate()
                return None
            self._api = api
        except Exception as exc:  # noqa: BLE001
            log.warning("TR live session unavailable (%s) — will retry.", exc)
            self._invalidate()
        return self._api

    # -- balances ------------------------------------------------------------
    def cash(self) -> float:
        """Real TR account cash (account currency). 0.0 if unreachable — check
        `last_cash_ok` to tell a real zero from a failed read."""
        self.last_cash_ok = False
        api = self._login()
        if api is None:
            return 0.0
        try:
            import asyncio

            from .tr_derivatives import _recv_for

            async def _q() -> Any:
                # Match OUR subscription (the shared socket carries other
                # frames too) and never wait forever.
                sub_id = await api.cash()
                try:
                    return await _recv_for(api, sub_id, timeout=10.0)
                finally:
                    await api.unsubscribe(sub_id)

            value = _parse_cash(asyncio.get_event_loop().run_until_complete(_q()))
        except Exception as exc:  # noqa: BLE001
            log.warning("TR cash fetch failed (%s) — dropping session.", exc)
            self._invalidate()
            return 0.0
        if value is None:
            return 0.0
        self.last_cash_ok = True
        return value

    def price(self, symbol: str) -> float:
        return 0.0  # engine passes live prices from the data layer

    # -- execution -----------------------------------------------------------
    @staticmethod
    def _order_id(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        oid = payload.get("orderId") or payload.get("id")
        return str(oid) if oid else None

    @staticmethod
    def _warning_types(payload: Any) -> list[str]:
        out = []
        for w in (payload or {}).get("warnings", []) or []:
            if isinstance(w, dict):
                t = w.get("type") or w.get("name")
                if t:
                    out.append(str(t))
        return out

    def place_order(self, isin: str, side: str, size: float,
                    exchange: str = DEFAULT_EXCHANGE) -> OrderResult:
        """Place a REAL market order for `isin` (a knockout/warrant/equity ISIN).
        Refuses unless armed. `size` is the number of certificates/shares.

        An order only counts as placed when TR POSITIVELY confirms it with an
        order id. Live incident: TR answered a submission with a warnings-only
        acknowledgment (no error, NO order created), the old no-error check
        declared success, and the dashboard recorded fills the TR app never
        made. Warnings (e.g. cost warnings) are acknowledged ONCE by
        resubmitting with warningsShown — the double-arm consent covers that —
        and anything still unconfirmed is a failure with the raw payload
        logged for diagnosis."""
        # good-for-day market order, no fractional certificates.
        return self._submit_confirmed(
            isin, side, size, exchange,
            lambda api, warnings: api.market_order(isin, exchange, side, size, "gfd", False,
                                                   warnings_shown=warnings))

    def place_limit_order(self, isin: str, side: str, size: float, limit: float,
                          exchange: str = DEFAULT_EXCHANGE) -> OrderResult:
        """Place a REAL good-for-day LIMIT order (exchange-side entry). Same
        confirmation rules as place_order: armed only, and only counts as
        placed when TR returns an order id (OrderResult.order_id)."""
        if size <= 0 or limit <= 0:
            return OrderResult(False, isin, side, size, limit, 1.0,
                               f"invalid limit order (size {size}, limit {limit})")
        price = round(float(limit), 2)
        return self._submit_confirmed(
            isin, side, size, exchange,
            lambda api, warnings: api.limit_order(isin, exchange, side, size, price, "gfd",
                                                  warnings_shown=warnings),
            price=price)

    def _submit_confirmed(self, isin: str, side: str, size: float, exchange: str,
                          make_sub, price: float = 0.0) -> OrderResult:
        if not self.armed:
            return OrderResult(
                False, isin, side, size, price, 1.0,
                "LIVE order blocked: broker not armed. Arm + confirm live "
                "execution in the dashboard first.")
        api = self._login()
        if api is None:
            return OrderResult(False, isin, side, size, price, 1.0,
                               "LIVE order blocked: TR session unavailable.")
        try:
            import asyncio

            from .tr_derivatives import _recv_for

            async def _submit(warnings_shown: list[str] | None) -> Any:
                sub_id = await make_sub(api, warnings_shown)
                payload = await _recv_for(api, sub_id)
                await api.unsubscribe(sub_id)
                return payload

            loop = asyncio.get_event_loop()
            payload = loop.run_until_complete(_submit(None))
            errors = (payload or {}).get("errors")
            if errors:
                log.warning("TR rejected %s %s x%s: %s", side, isin, size, errors)
                return OrderResult(False, isin, side, size, 0.0, 1.0,
                                   f"TR rejected order: {errors}")
            oid = self._order_id(payload)
            warnings = self._warning_types(payload)
            if oid is None and warnings:
                log.info("TR order needs warning ack (%s) — resubmitting once "
                         "with warningsShown.", warnings)
                payload = loop.run_until_complete(_submit(warnings))
                errors = (payload or {}).get("errors")
                if errors:
                    log.warning("TR rejected %s %s x%s after warning ack: %s",
                                side, isin, size, errors)
                    return OrderResult(False, isin, side, size, 0.0, 1.0,
                                       f"TR rejected order: {errors}")
                oid = self._order_id(payload)
            if oid is None:
                log.warning("TR order UNCONFIRMED (%s %s x%s) — no order id in "
                            "response; treating as NOT placed. Raw payload: %r",
                            side, isin, size, payload)
                return OrderResult(False, isin, side, size, 0.0, 1.0,
                                   f"TR order unconfirmed (no order id): {payload!r}")
            log.info("LIVE order CONFIRMED %s: %s %s x%s on %s",
                     oid, side, isin, size, exchange)
            return OrderResult(True, isin, side, size, price, 1.0,
                               f"live order placed ({oid})", order_id=oid)
        except Exception as exc:  # noqa: BLE001
            log.warning("TR order error (%s %s x%s): %s", side, isin, size, exc)
            return OrderResult(False, isin, side, size, 0.0, 1.0,
                               f"TR order error: {exc}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order. False if TR reports an error (e.g. it was
        already filled or is unknown) or the session is down."""
        api = self._login()
        if api is None:
            return False
        try:
            import asyncio

            from .tr_derivatives import _recv_for

            async def _cancel() -> Any:
                sub_id = await api.cancel_order(order_id)
                payload = await _recv_for(api, sub_id)
                await api.unsubscribe(sub_id)
                return payload

            payload = asyncio.get_event_loop().run_until_complete(_cancel())
            if isinstance(payload, dict) and payload.get("errors"):
                log.warning("TR cancel %s rejected: %s", order_id, payload["errors"])
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("TR cancel %s error: %s", order_id, exc)
            return False

    def order_status(self, order_id: str) -> dict | None:
        """{"state": "open"|"filled"|"gone", "fill_price": float|None} for an
        order, from TR's orders overview. None when unknown (session down,
        order not listed, unrecognised payload) — callers must then NOT
        assume a fill. The overview's exact shape is unverified against the
        real API, so parsing is defensive."""
        api = self._login()
        if api is None:
            return None
        try:
            import asyncio

            from .tr_derivatives import _recv_for

            async def _query() -> Any:
                sub_id = await api.order_overview()
                payload = await _recv_for(api, sub_id)
                await api.unsubscribe(sub_id)
                return payload

            payload = asyncio.get_event_loop().run_until_complete(_query())
        except Exception as exc:  # noqa: BLE001
            log.warning("TR order overview error: %s", exc)
            return None
        return self._parse_order_status(payload, order_id)

    @staticmethod
    def _parse_order_status(payload: Any, order_id: str) -> dict | None:
        orders = payload.get("orders") if isinstance(payload, dict) else payload
        if not isinstance(orders, list):
            return None
        for o in orders:
            if not isinstance(o, dict):
                continue
            if str(o.get("id") or o.get("orderId") or "") != str(order_id):
                continue
            raw = str(o.get("status") or o.get("state") or "").lower()
            fill = None
            for key in ("averagePrice", "avgPrice", "executionPrice", "price"):
                try:
                    if o.get(key) is not None:
                        fill = float(o[key])
                        break
                except (TypeError, ValueError):
                    continue
            if any(w in raw for w in ("execut", "fill", "done", "complete")):
                return {"state": "filled", "fill_price": fill}
            if any(w in raw for w in ("cancel", "expir", "reject", "delet")):
                return {"state": "gone", "fill_price": None}
            if any(w in raw for w in ("open", "pending", "active", "new", "work")):
                return {"state": "open", "fill_price": None}
            log.warning("TR order %s has unrecognised status %r", order_id, raw)
            return None
        return None

    def buy(self, symbol: str, qty: float, price: float) -> OrderResult:
        # Equity-by-ticker isn't the bot's live path (it trades knockouts by
        # ISIN via place_order); resolve-then-place could be added if needed.
        return self.place_order(symbol, "buy", qty)

    def sell(self, symbol: str, qty: float, price: float) -> OrderResult:
        return self.place_order(symbol, "sell", qty)

    def adjust_cash(self, delta: float) -> bool:
       """Stub for real TR broker. Cash adjustments happen server-side via
       actual orders, not locally. Always returns True since we assume the
       real account will handle it."""
       return True
