"""FastAPI dashboard.

Read-only view over the Store: portfolio, economics/self-sustaining status,
trades, positions, activity feed, logs and the equity curve. The engine writes;
the web only reads, so it can run in the same or a separate process.

Run with:  lmtrade web
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import DEFAULT_TOML_PATH, Settings, load_settings
from ..core.control import LOW_BALANCE_EUR, ControlState
from ..core.engine import OPTION_FEE
from ..core.exits import exit_plan, pnl_for_premium, validate_override
from ..core.limit_orders import entry_priority
from ..core.budget import split_cash
from ..core.market_hours import market_status
from ..core.state import Store
from . import settings_editor

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"


def _json_safe(obj: Any) -> Any:
    """Replace any inf/-inf/nan float anywhere in a JSON-able structure with
    None. Store round-trips those fine (plain json.dumps/loads allow them —
    e.g. a historical activity/meta row written before a producer started
    sanitizing its own floats, such as an infinite runway_hours when
    gpu_usd_per_hour=0), but strict JSON (RFC 8259, no Infinity/NaN) does
    not, and that's what every response here must produce."""
    if isinstance(obj, float):
        return None if math.isinf(obj) or math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(_json_safe(content))


def _schedule_restart() -> None:
    """Replace the running process with a fresh copy 0.5 s after being called.

    The short delay lets FastAPI finish sending the HTTP response before the
    process image is replaced. Works for both ``lmtrade run`` (engine + web in
    one process) and ``lmtrade web`` (web-only); in both cases ``os.execv``
    re-executes the same command with the same arguments, picking up the newly
    saved ``config.toml``.
    """
    import os
    import sys
    import threading

    def _exec() -> None:
        import time
        time.sleep(0.5)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_exec, daemon=True).start()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    # Two books, one connection each. Which one every read serves is chosen
    # per request by the control state's mode, so flipping the dashboard
    # paper/live toggle swaps the whole view to the matching ledger/history.
    _books = {
        "paper": Store(settings.db_path),
        "live": Store(settings.live_db_path),
    }
    app = FastAPI(title="LMTrade", version="0.1.0")

    def control() -> ControlState:
        return ControlState.load(settings.control_path)

    def store_for(mode: str) -> Store:
        return _books["live"] if mode == "live" else _books["paper"]

    def store() -> Store:
        c = control()
        # When in live view but NOT armed, show paper store (engine writes there).
        # Only when genuinely armed for live should we show the live store.
        if c.mode == "live" and not c.armed:
            return _books["paper"]
        return store_for(c.mode)

    def view_book() -> Store:
        """The book behind the money figures the user sees: in LIVE view the
        live book (real TR account) even while unarmed — never the paper
        book the engine trades meanwhile."""
        return store_for("live") if control().mode == "live" else store()

    def research_books() -> list[Store]:
        """Where to read market research (news, analysis, signals): the active
        book first, then the paper book. Research isn't money — right after
        arming live the fresh live book has none yet, and showing nothing
        until the engine rebuilds it hides perfectly current information."""
        active = store()
        return [active] if active is _books["paper"] else [active, _books["paper"]]

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    def _asset_version(name: str = "app.js") -> str:
        """Short content hash of a static script, appended as ?v= to its URL
        so a changed file bypasses the browser cache (otherwise a pulled
        app.js is rendered against a freshly-served HTML shell — column
        mismatch)."""
        js = STATIC / name
        if not js.exists():
            return "0"
        return hashlib.sha1(js.read_bytes()).hexdigest()[:8]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Explicit utf-8: Path.read_text() uses the platform default encoding,
        # which on Windows is cp1252 and mangles the template's ⚡/· glyphs
        # into mojibake (âš¡ / Â·) before they're ever served.
        html = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")
        for name in ("chart.js", "exits.js", "app.js"):
            html = html.replace(f"/static/{name}",
                                f"/static/{name}?v={_asset_version(name)}")
        return html

    def _tr_meta(key: str):
        """TR account meta (cash/baseline), read from the live book first,
        falling back to the paper book — the engine writes it into whichever
        book it's currently trading (unarmed live trades the paper book)."""
        v = store_for("live").get_meta(key)
        return v if v is not None else store_for("paper").get_meta(key)

    def _market() -> dict:
        """Venue open/closed for real (armed live) closes; the paper book
        and an unarmed view can always close (locally)."""
        ctrl = control()
        if not (ctrl.mode == "live" and ctrl.armed):
            return {"open": True, "next_open": None, "closes": None}
        until = store_for("live").get_meta("market_closed_until")
        return market_status(time.time(), settings.tr.market_hours,
                             float(until) if until else None)

    def _buckets(book: Store, cash: float | None) -> dict | None:
        """Cash split entry / dips / reserve, as the engine applies it."""
        if cash is None:
            return None
        adds = book.get_meta("dip_adds") or {}
        open_ids = {str(o["id"]) for o in book.open_options()}
        spent = sum(float(a["size"]) * float(a["price"])
                    for k, lst in adds.items() if k in open_ids for a in lst)
        b = settings.budget_split
        return {k: round(v, 2) for k, v in
                split_cash(float(cash), b.reserve_eur, b.dips_eur, spent).items()}

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        ctrl = control()
        is_live = ctrl.mode == "live"
        # LIVE view = real money only: positions always come from the live book
        # (which mirrors the TR account), never the paper book the engine
        # trades while live is unarmed. Unarmed, the engine isn't driving that
        # book, so its rows get no close / exit-edit handles.
        book = view_book()
        can_act = not is_live or ctrl.armed
        market = _market()
        equity_positions = book.positions()
        open_opts = book.open_options()
        tr_cash = _tr_meta("tr_account_cash")
        # LIVE view: cash is the REAL TR balance (None until fetched) — never
        # default an empty live book to the paper budget, which fabricated a
        # "100.00 EUR" live net worth. Paper view: the simulated book's cash.
        if is_live:
            cash = tr_cash
        else:
            cash = float(store().get_meta("cash", settings.budget))
        econ = store().get_meta("economics", {})
        
        # Calculate equity: for live mode, use real cash + position values;
        # for paper mode, use the recorded equity curve (which includes cash + all holdings).
        if is_live and cash is not None:
            # Real live: equity = TR cash + unrealized P&L on open positions
            # Liquidation value: each marked position net of its exit fee —
            # equals cash + sum(cost + unrealized P&L) over the rows.
            open_ids = {str(o.get("id")) for o in open_opts}
            marked = [m for k, m in (book.get_meta("open_option_marks", {}) or {}).items()
                      if k in open_ids and m.get("value") is not None]
            positions_value = sum(m["value"] - OPTION_FEE for m in marked)
            equity = cash + positions_value
        else:
            curve = store().equity_curve(limit=300)
            equity = curve[-1]["equity"] if curve else (cash or 0.0)
        # Show the FULL book the engine counts toward max_positions: equity
        # positions AND open options/knockouts. Options were previously
        # omitted, so an options-only book (the common case) showed "0
        # positions" even when full. For an option, qty=contracts and
        # avg_price=entry premium; the real TR ISIN is surfaced when present.
        marks = book.get_meta("open_option_marks", {}) or {}
        pending = book.get_meta("close_requests") or []
        rows = [
            {"symbol": p.symbol, "qty": round(p.qty, 6),
             "avg_price": round(p.avg_price, 4), "kind": "equity", "isin": None,
             "value": None, "unrealized_pnl": None, "price": None, "spot": None,
             "liquidation_value": None,
             "id": None, "close_kind": "equity" if can_act else None,
             "close_symbol": p.symbol,
             "close_pending": {"kind": "equity", "symbol": p.symbol} in pending}
            for p in equity_positions
        ]
        overrides = book.get_meta("exit_overrides") or {}
        for o in open_opts:
            kind = o.get("instrument_type") or "option"
            entry, n = o.get("entry_premium", 0.0), o.get("contracts", 0.0)
            # Same effective exits the engine uses (core/exits.py).
            plan = exit_plan(o, settings.options, overrides.get(str(o.get("id"))), OPTION_FEE)
            arm_pnl = (round(plan.trail_min / (1 - plan.trail_pct), 4)
                       if 0 <= plan.trail_pct < 1 else None)

            def net_if(price: float | None) -> float | None:
                # what would be booked if closed at `price`: both 1 EUR fees out
                return None if price is None else round(pnl_for_premium(entry, n, price,
                                                                        OPTION_FEE), 4)

            def rnd(v: float | None) -> float | None:
                return None if v is None else round(v, 4)
            label = f"{o['underlying']} {o.get('kind', '')}".strip()
            # Live mark persisted by the engine each cycle; None until the
            # first cycle marks this option (don't fabricate a P&L).
            mark = marks.get(str(o.get("id")))
            rows.append({
                "symbol": label,
                "qty": round(o.get("contracts", 0.0), 6),
                "avg_price": round(o.get("entry_premium", 0.0), 4),
                "kind": kind,
                "isin": o.get("isin"),
                "value": mark.get("value") if mark else None,
                "unrealized_pnl": mark.get("unrealized_pnl") if mark else None,
                "liquidation_value": (round(mark["value"] - OPTION_FEE, 4)
                                      if mark and mark.get("value") is not None else None),
                # Current premium per unit (what the P&L is marked at) and the
                # underlying's price, from the engine's latest mark.
                "price": mark.get("mark_premium") if mark else None,
                "spot": mark.get("spot") if mark else None,
                "tp_price": rnd(plan.tp), "tp_pnl": net_if(plan.tp),
                "sl_price": rnd(plan.sl), "sl_pnl": net_if(plan.sl),
                "trail": {
                    "enabled": plan.trail_enabled,
                    "min_pnl": plan.trail_min, "pct": plan.trail_pct,
                    "arm_pnl": arm_pnl,
                    "active": bool(plan.trail_enabled and mark and mark.get("trail_active")),
                    "stop_pnl": mark.get("trail_stop_pnl") if mark and plan.trail_enabled else None,
                    "peak_pnl": mark.get("peak_pnl") if mark and plan.trail_enabled else None,
                },
                "exits_custom": plan.custom,
                "id": o.get("id"), "close_kind": "option" if can_act else None,
                "close_symbol": None,
                "close_pending": {"kind": "option", "id": o.get("id")} in pending,
                "close_blocked": not market["open"],
            })
        return SafeJSONResponse({
            "mode": store().get_meta("mode", settings.mode),
            "currency": settings.currency,
            "universe": store().get_meta("universe", settings.universe),
            "cash": round(cash, 4) if cash is not None else None,
            "equity": round(equity, 4),
            "reserve": round(store().reserve_balance(), 4),
            "tr_account_cash": tr_cash,
            "tr_baseline_net_worth": _tr_meta("tr_baseline_net_worth"),
            "last_realized_net_worth": store().get_meta("last_realized_net_worth"),
            "starting_cash": store().get_meta("starting_cash", settings.budget),
            "economics": econ,
            "positions": rows,
            "num_positions": len(rows),
            "provider_warnings": store().get_meta("provider_warnings"),
            "tr_session": _tr_meta("tr_session"),
            "market": market,
            "cash_buckets": _buckets(book, cash),
            "position_commentary": book.get_meta("position_commentary"),
            "alpha": store().get_meta("alpha"),
        })

    @app.get("/api/trades")
    def trades(limit: int = 100) -> JSONResponse:
        return SafeJSONResponse(store().recent_trades(limit))

    @app.get("/api/activity")
    def activity(limit: int = 100) -> JSONResponse:
        return SafeJSONResponse(store().recent_activity(limit))

    @app.get("/api/logs")
    def logs(limit: int = 200) -> JSONResponse:
        return SafeJSONResponse(store().recent_logs(limit))

    @app.get("/api/equity")
    def equity() -> JSONResponse:
        # LIVE view: the real account's curve (even while unarmed), without
        # the 0-cash spikes old failed TR reads recorded.
        if control().mode == "live":
            curve = [p for p in view_book().equity_curve(limit=2000)
                     if (p.get("cash") or 0) > 0]
            return SafeJSONResponse(curve)
        return SafeJSONResponse(store().equity_curve(limit=2000))

    @app.get("/api/costs")
    def costs() -> JSONResponse:
        return SafeJSONResponse(store().total_costs())

    @app.get("/api/realized")
    def realized(limit: int = 100) -> JSONResponse:
        """Closed options/knockouts with realized P&L, plus running totals —
        the counterpart to the open positions' unrealized P&L."""
        closed = view_book().closed_options(limit)   # real money only in LIVE view
        rows = []
        total = wins = losses = 0.0
        for o in closed:
            pnl = float(o.get("pnl") or 0.0)
            total += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            rows.append({
                "symbol": f"{o['underlying']} {o.get('kind', '')}".strip(),
                "kind": o.get("instrument_type") or "option",
                "isin": o.get("isin"),
                "contracts": round(o.get("contracts", 0.0), 6),
                "entry_premium": round(o.get("entry_premium", 0.0), 4),
                "exit_premium": round((o.get("exit_premium") or 0.0), 4),
                "pnl": round(pnl, 4),
                "closed_ts": o.get("closed_ts"),
            })
        return SafeJSONResponse({
            "total_pnl": round(total, 4),
            "wins": int(wins),
            "losses": int(losses),
            "rows": rows,
        })

    @app.get("/api/options")
    def options() -> JSONResponse:
        return SafeJSONResponse({
            "open": store().open_options(),
            "closed": store().closed_options(50),
        })

    @app.get("/api/leaderboard")
    def leaderboard() -> JSONResponse:
        genomes = store().get_meta("genomes", []) or []
        rows = [
            {"id": g["id"], "strategy": g["strategy"], "params": g["params"],
             "trades": g["trades"], "pnl": round(g["pnl"], 4),
             # Same shrunk fitness as the optimizer (see FITNESS_SHRINK_K).
             "fitness": round(g["pnl"] / (g["trades"] + 2.0), 5) if g["trades"] else 0.01}
            for g in genomes
        ]
        rows.sort(key=lambda r: r["fitness"], reverse=True)
        return SafeJSONResponse(rows)

    @app.get("/api/benchmark")
    def benchmark() -> JSONResponse:
        return SafeJSONResponse({
            "curve": store().benchmark_curve(500),
            "alpha": store().get_meta("alpha"),
            "entry": store().get_meta("benchmark_entry"),
        })

    @app.get("/api/news")
    def news() -> JSONResponse:
        for book in research_books():
            rows = book.recent_news(50)
            if rows:
                return SafeJSONResponse(rows)
        return SafeJSONResponse([])

    @app.get("/api/analysis")
    def analysis() -> JSONResponse:
        """The latest compiled daily market analysis (per-symbol bias)."""
        for book in research_books():
            data = book.get_meta("market_analysis", {}) or {}
            if data:
                return SafeJSONResponse(data)
        return SafeJSONResponse({})

    @app.get("/api/watchlist")
    def watchlist() -> JSONResponse:
        """Signals waiting for a better entry (core/watchlist.py): current
        stretch from the intraday SMA in sigma, the trigger, how far is left
        to go, and when the entry expires."""
        cfg = settings.entry
        rows = []
        pend = store().get_meta("pending_entries") or {}
        for sym, w in (store().get_meta("watchlist") or {}).items():
            direction = w.get("direction")
            trigger = -cfg.watch_k if direction == "buy" else cfg.watch_k
            z = w.get("z")
            to_go = None
            if w.get("instant"):
                to_go = 0.0              # high confidence: entered regardless of timing
            elif z is not None:
                to_go = max(0.0, z - trigger) if direction == "buy" else max(0.0, trigger - z)
            added = float(w.get("added_ts") or 0.0)
            # % the underlying still has to move to reach the entry target.
            spot, target = w.get("spot"), w.get("target")
            distance = None
            if w.get("instant"):
                distance = 0.0
            elif spot and target:
                gap = (spot - target) if direction == "buy" else (target - spot)
                distance = round(max(0.0, gap) / spot * 100, 4)
            rows.append({
                "symbol": sym, "direction": direction, "instant": bool(w.get("instant")),
                "spot": spot, "target": target, "distance_pct": distance,
                "sigma_pct": w.get("sigma_pct"), "high_vol": bool(w.get("high_vol")),
                "instrument": "call" if direction == "buy" else "put",
                "confidence": w.get("confidence"), "z": z, "k": cfg.watch_k,
                "trigger_z": trigger,
                "sigma_to_go": None if to_go is None else round(to_go, 4),
                "priority": round(entry_priority(w.get("confidence"), to_go), 4),
                "added_ts": added, "expires_ts": added + cfg.watch_max_hours * 3600,
                "checked_ts": w.get("checked_ts"),
                "order": ({k: pend[sym].get(k) for k in
                           ("limit", "size", "target_spot", "order_id", "placed_ts")}
                          | {"kind": pend[sym]["instrument"]["kind"],
                             "current_price": pend[sym].get("last_price")}
                          if sym in pend else None),
            })
        # Same ranking the engine uses for order slots: confidence x2 + closeness.
        rows.sort(key=lambda r: -r["priority"])
        return SafeJSONResponse(rows)

    @app.get("/api/signals")
    def signals(min_confidence: float = 0.6, limit: int = 200) -> JSONResponse:
        """Strongest recent decisions and their cause — the per-provider signal
        breakdown recorded in each 'decision' activity. Deduped to the most
        recent decision per symbol so the tab reads as 'current strong signals'."""
        seen: set[str] = set()
        out = []
        activity: list[dict] = []
        for book in research_books():
            activity = [a for a in book.recent_activity(limit) if a.get("kind") == "decision"]
            if activity:
                break
        for a in activity:
            if a.get("kind") != "decision":
                continue
            d = a.get("detail") or {}
            symbol = a.get("symbol")
            if symbol in seen:
                continue
            seen.add(symbol)
            if d.get("direction") in (None, "hold"):
                continue
            if float(d.get("confidence", 0.0)) < min_confidence:
                continue
            out.append({
                "ts": a.get("ts"),
                "symbol": symbol,
                "direction": d.get("direction"),
                "confidence": d.get("confidence"),
                "rationale": d.get("rationale", ""),
                "signals": d.get("signals", []),
            })
        out.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return SafeJSONResponse(out)

    # ------------------------------------------------------------- control plane
    def _live_net_worth() -> float | None:
        econ = store_for("live").get_meta("economics", {}) or {}
        # Prefer the REAL TR account balance (what the guard actually gates
        # on), from whichever book the engine wrote it into; fall back to the
        # live book's computed net worth.
        real = (store_for("live").get_meta("tr_account_cash")
                or store_for("paper").get_meta("tr_account_cash"))
        if real is not None:
            # Net worth = real cash + live positions after their exit fee
            # (the guard is about net worth, not cash alone).
            book = store_for("live")
            open_ids = {str(o.get("id")) for o in book.open_options()}
            marks = book.get_meta("open_option_marks", {}) or {}
            held = sum(m["value"] - OPTION_FEE for k, m in marks.items()
                       if k in open_ids and m.get("value") is not None)
            return float(real) + held
        nw = econ.get("net_worth_eur")
        return float(nw) if nw is not None else None

    def _control_payload(c, nw) -> dict:
        return {
            "mode": c.mode,
            "armed": c.armed,
            "double_armed": c.double_armed,
            "live_armed": c.live_armed,
            "net_worth": nw,
            "low_balance": c.is_low_balance(nw),
            "low_balance_threshold": LOW_BALANCE_EUR,
            # Single source of truth for the IS/IS-NOT-executing banner.
            "executing": c.is_executing(nw),
        }

    @app.get("/api/control")
    def get_control() -> JSONResponse:
        return SafeJSONResponse(_control_payload(control(), _live_net_worth()))

    @app.post("/api/control/mode")
    def set_mode(payload: dict) -> JSONResponse:
        mode = (payload or {}).get("mode")
        c = control()
        try:
            c.set_mode(mode, disarm=bool((payload or {}).get("disarm")))
        except ValueError:
            return SafeJSONResponse({"error": f"invalid mode {mode!r}"}, status_code=400)
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/arm")
    def arm(payload: dict) -> JSONResponse:
        c = control()
        c.arm(confirm=bool((payload or {}).get("confirm")))
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/disarm")
    def disarm() -> JSONResponse:
        c = control()
        c.disarm()
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/double-arm")
    def double_arm(payload: dict) -> JSONResponse:
        c = control()
        c.set_double_armed(confirm=bool((payload or {}).get("confirm")))
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    @app.post("/api/control/double-disarm")
    def double_disarm() -> JSONResponse:
        c = control()
        c.double_disarm()
        return SafeJSONResponse(_control_payload(c, _live_net_worth()))

    # ---------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings() -> JSONResponse:
        return SafeJSONResponse({
            "fields": settings_editor.schema(settings),
            "config_path": str(DEFAULT_TOML_PATH),
            "note": "Saved to config.toml. Restart lmtrade run to apply.",
        })

    @app.post("/api/settings")
    def post_settings(payload: dict) -> JSONResponse:
        changes = (payload or {}).get("changes", {})
        result = settings_editor.apply_changes(DEFAULT_TOML_PATH, changes)
        restarting = bool(result.get("applied"))
        if restarting:
            _schedule_restart()
        return SafeJSONResponse({**result, "restarting": restarting})

    @app.post("/api/positions/close")
    def close_position(payload: dict) -> JSONResponse:
        """Queue a manual close; the engine executes it at the current price
        within seconds (Engine.process_close_requests)."""
        payload = payload or {}
        kind = payload.get("kind")
        if kind == "option" and isinstance(payload.get("id"), int):
            req: dict = {"kind": "option", "id": payload["id"]}
        elif kind == "equity" and isinstance(payload.get("symbol"), str) and payload["symbol"]:
            req = {"kind": "equity", "symbol": payload["symbol"]}
        else:
            return SafeJSONResponse(
                {"queued": False,
                 "error": "expected {kind:'option', id:int} or {kind:'equity', symbol:str}"},
                status_code=400)
        if not _market()["open"]:
            return SafeJSONResponse(
                {"queued": False, "error": "market closed — closing is possible once it opens"},
                status_code=409)
        reqs = store().get_meta("close_requests") or []
        if req not in reqs:
            store().set_meta("close_requests", reqs + [req])
        return SafeJSONResponse({"queued": True, "request": req})

    @app.post("/api/positions/exits")
    def set_position_exits(payload: dict) -> JSONResponse:
        """Save (or reset) a position's own TP / SL / trailing settings, in net
        EUR. The engine applies them from its next exit check."""
        payload = payload or {}
        pid = payload.get("id")
        o = next((x for x in store().open_options() if x["id"] == pid), None)
        if o is None:
            return SafeJSONResponse({"ok": False, "error": "no open position with that id"},
                                    status_code=404)
        overrides = store().get_meta("exit_overrides") or {}
        if payload.get("reset"):
            overrides.pop(str(pid), None)
            store().set_meta("exit_overrides", overrides)
            return SafeJSONResponse({"ok": True, "reset": True})
        ov, err = validate_override(o, payload, OPTION_FEE)
        if err:
            return SafeJSONResponse({"ok": False, "error": err}, status_code=400)
        overrides[str(pid)] = ov
        store().set_meta("exit_overrides", overrides)
        return SafeJSONResponse({"ok": True, "override": ov})

    @app.post("/api/dismiss-provider-warning")
    def dismiss_provider_warning() -> JSONResponse:
        """Clear the persistent provider warnings from the store so they don't
        reappear until the next limit event."""
        store().set_meta("provider_warnings", None)
        return SafeJSONResponse({"ok": True})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


# For `uvicorn lmtrade.web.app:app`
app = create_app()
