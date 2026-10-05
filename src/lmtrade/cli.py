"""LMTrade command-line interface.

    lmtrade run            # start the trading engine (paper by default)
    lmtrade web            # launch the dashboard
    lmtrade status         # print current portfolio + economics
    lmtrade reset          # wipe runtime state (fresh account)
    lmtrade config         # show the resolved configuration
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .brokers.factory import build_broker
from .config import load_settings
from .core.engine import Engine
from .core.state import Store
from .economics.cost_accounting import CostAccountant

app = typer.Typer(add_completion=False, help="Self-sustaining LLM/SLM trading bot.")
console = Console()


def _banner(settings) -> None:
    mode = settings.mode.upper()
    color = "red" if mode == "LIVE" else "cyan"
    console.print(Panel.fit(
        f"[bold]LMTrade v{__version__}[/bold]\n"
        f"mode=[{color}]{mode}[/{color}]  budget={settings.budget} {settings.currency}  "
        f"universe={','.join(settings.universe)}\n"
        f"model stack: {', '.join(settings.model.stack)}\n"
        f"GPU: ${settings.economics.gpu_usd_per_hour}/hr  "
        f"runway floor: {settings.economics.min_runway_hours}h",
        title="⚡ engine", border_style=color,
    ))


def _launch_dashboard(settings, host: str, port: int) -> "threading.Thread":
    """Start the web dashboard in a background daemon thread so a single
    `lmtrade run` gives you a live local dashboard without a separate
    `lmtrade web` process. Reads the same on-disk state the engine writes."""
    import threading

    import uvicorn

    def _serve() -> None:
        uvicorn.run("lmtrade.web.app:app", host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return thread


def _print_diagnostics(settings, engine) -> None:
    """Surface research-provider/TR-client availability at startup — checked
    once, eagerly, so a misconfiguration (claude CLI not on PATH, TR session
    not resumable) is visible immediately instead of only inferable from the
    absence of [signal] news / knockout lines deep in the scrolling log."""
    np_ = settings.research.news_provider
    ap_ = settings.research.analysis_provider
    if "claude_cli" in (np_, ap_):
        from .models.providers import ClaudeCLIProvider

        if ClaudeCLIProvider(settings).available():
            console.print(f"[green]claude CLI found[/green] — news={np_} analysis={ap_}")
        else:
            console.print(
                f"[bold red]claude CLI NOT FOUND on PATH[/bold red] — "
                f"news_provider={np_} analysis_provider={ap_} will silently "
                f"produce no news/analysis. Check `claude` is installed and on PATH."
            )
    if engine.tr_derivatives is not None:
        if engine.tr_derivatives.available():
            console.print(
                "[green]TR live derivatives: connected[/green] — new positions "
                "use real TR knockout certificates."
            )
        else:
            console.print(
                "[bold red]TR live derivatives: unavailable[/bold red] — see "
                "warning above; falling back to synthetic options."
            )


@app.command()
def run(
    cycles: int = typer.Option(0, help="Stop after N cycles (0 = run forever)."),
    interval: int = typer.Option(0, help="Override loop interval seconds (0 = config)."),
    web: bool = typer.Option(
        True, "--web/--no-web", help="Also launch the web dashboard in the background."
    ),
    web_host: str = typer.Option("", help="Dashboard bind host (default from config)."),
    web_port: int = typer.Option(0, help="Dashboard bind port (default from config)."),
):
    """Start the trading engine (and, by default, the web dashboard alongside it)."""
    settings = load_settings()
    if interval > 0:
        settings.loop.interval_seconds = interval
    _banner(settings)
    if settings.mode == "live":
        console.print("[bold red]⚠ LIVE mode — real Trade Republic execution. "
                      "This uses an unofficial API against TR's ToS.[/bold red]")
    if web:
        h = web_host or settings.web.host
        p = web_port or settings.web.port
        _launch_dashboard(settings, h, p)
        console.print(f"[cyan]Dashboard →[/cyan] http://{h}:{p}")
    try:
        _run_with_control(settings, cycles or None)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — stopping engine.[/yellow]")


def _control_key(control) -> tuple:
    """What the running engine depends on: real trading on/off and which book
    the paper simulation feeds. Switching only the dashboard VIEW while live
    stays armed must not rebuild (and so not interrupt) the live engine."""
    return (control.live_armed, control.live_armed or control.mode)


def _build_engine_for_control(settings, control):
    """Construct engine + store + broker for the active book. The LIVE book
    receives ONLY real fills: it is used exclusively when live AND armed
    (real TR broker). Unarmed live is a real-account VIEW — the simulation
    keeps trading the paper book, so the live ledger is never polluted with
    paper fills and the paper run isn't interrupted by flipping the toggle.
    
    In all cases, fallback_store is opened to the other book so that if one
    runs out of analysis (e.g. Claude reached limits), it can use the other's."""
    from .brokers.paper import PaperBroker
    from .brokers.tr_derivatives import build_tr_derivatives

    if control.live_armed:
        from .brokers.trade_republic import TradeRepublicBroker
        try:
            # Try to set up live broker; if TR credentials are missing/invalid,
            # fall back to paper to prevent engine crash
            book = "live"
            store = Store(settings.book_db_path(book))
            broker = TradeRepublicBroker(store, settings, armed=True)
            console.print("[bold red]⚠ LIVE + ARMED — real Trade Republic orders "
                          "will be placed. Against TR ToS.[/bold red]")
        except RuntimeError as e:
            # TR credentials missing or invalid; fall back to paper
            store.close()
            book = "paper"
            store = Store(settings.book_db_path(book))
            broker = PaperBroker(store, starting_cash=settings.budget)
            console.print(f"[bold red]⚠ LIVE mode requested but TR setup failed: {e}[/bold red]")
            console.print("[yellow]Falling back to paper trading.[/yellow]")
        # Live+armed: fallback to paper for analysis
        fallback_store = Store(settings.book_db_path("paper"))
    elif control.mode == "live":
        # Unarmed live: paper engine runs; fallback to live store for analysis
        book = "paper"
        store = Store(settings.book_db_path("paper"))
        broker = PaperBroker(store, starting_cash=settings.budget)
        fallback_store = Store(settings.book_db_path("live"))
    else:
        # Paper mode: fallback to live store for analysis (in case live run succeeded)
        book = "paper"
        store = Store(settings.book_db_path("paper"))
        broker = PaperBroker(store, starting_cash=settings.budget)
        fallback_store = Store(settings.book_db_path("live"))
    tr = build_tr_derivatives(settings)
    engine = Engine(settings, store, broker, tr_derivatives=tr, fallback_store=fallback_store)
    if (control.live_armed or control.mode == "live") and tr is not None:
        # Reconcile the LIVE book against the real TR account at startup —
        # refresh real cash, import untracked TR positions (e.g. after a db
        # reset), delete phantom rows TR doesn't hold. The live view is then
        # exactly real cash + real positions.
        from .core.tr_sync import sync_tr_portfolio

        live_store = store if book == "live" else Store(settings.live_db_path)
        try:
            s = sync_tr_portfolio(live_store, tr)
            cash_txt = "—" if s["cash"] is None else f"{s['cash']:.2f} EUR"
            removed = s["removed_options"] + s["removed_positions"]
            console.print(
                f"[cyan]TR sync:[/cyan] cash={cash_txt} | "
                f"imported {s['imported']} | removed {removed} phantom row(s)")
        except Exception as exc:  # noqa: BLE001 — sync must never block startup
            console.print(f"[yellow]TR sync skipped ({exc})[/yellow]")
        finally:
            if live_store is not store:
                live_store.close()
    return engine, store



def _run_with_control(settings, max_cycles):
    """Drive the engine, rebuilding it whenever the dashboard flips the
    paper/live toggle or arms/disarms live execution."""
    from .core.control import ControlState

    remaining = max_cycles
    while True:
        control = ControlState.load(settings.control_path)
        snapshot = _control_key(control)
        engine, store = _build_engine_for_control(settings, control)
        _print_diagnostics(settings, engine)
        try:
            engine.run_forever(
                max_cycles=remaining,
                rebuild_when=lambda: _control_key(
                    ControlState.load(settings.control_path)) != snapshot,
            )
        finally:
            store.close()
            if engine.fallback_store is not None:
                engine.fallback_store.close()
        # If control is unchanged, run_forever returned because it finished
        # (max_cycles) or was stopped — don't loop forever rebuilding.
        if _control_key(ControlState.load(settings.control_path)) == snapshot:
            break


@app.command()
def web(
    host: str = typer.Option("", help="Bind host (default from config)."),
    port: int = typer.Option(0, help="Bind port (default from config)."),
):
    """Launch the web dashboard."""
    import uvicorn

    settings = load_settings()
    h = host or settings.web.host
    p = port or settings.web.port
    console.print(f"[cyan]Dashboard →[/cyan] http://{h}:{p}")
    uvicorn.run("lmtrade.web.app:app", host=h, port=p, log_level="info")


@app.command()
def status():
    """Print current portfolio and self-sustaining economics."""
    settings = load_settings()
    store = Store(settings.db_path)
    broker = build_broker(settings, store) if settings.mode == "paper" else None
    cash = float(store.get_meta("cash", settings.budget))
    positions = store.positions()
    pos_value = sum(p.qty * p.avg_price for p in positions)
    # Value open options at cost basis (entry premium) — conservative but keeps
    # premiums-at-risk in net worth without needing a live market snapshot.
    pos_value += sum(o["contracts"] * o["entry_premium"] for o in store.open_options())
    accountant = CostAccountant(settings, store)
    econ = accountant.snapshot(cash, pos_value, store.reserve_balance())

    t = Table(title="Portfolio", show_header=True, header_style="bold")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Mode", store.get_meta("mode", settings.mode))
    t.add_row("Cash (tradeable)", f"{cash:.2f} {settings.currency}")
    t.add_row("Reserve (stashed)", f"{econ.reserve_eur:.2f} {settings.currency}")
    t.add_row("Net worth", f"{econ.net_worth_eur:.2f} {settings.currency}")
    t.add_row("P&L", f"{econ.pnl_eur:+.2f} {settings.currency}")
    t.add_row("Open positions", str(len(positions)))
    t.add_row("GPU accrued", f"${econ.gpu_cost_accrued_usd:.4f}")
    t.add_row("Inference cost", f"${econ.inference_cost_usd:.4f}")
    t.add_row("Runway", f"{econ.runway_hours:.1f} h")
    t.add_row("Self-sustaining", "✅ yes" if econ.self_sustaining else "⏳ not yet")
    alpha = store.get_meta("alpha")
    t.add_row("Alpha vs retail",
              f"{alpha:+.3f} {settings.currency}" if alpha is not None else "—")
    console.print(t)

    if positions:
        pt = Table(title="Positions")
        pt.add_column("Symbol"); pt.add_column("Qty", justify="right")
        pt.add_column("Avg", justify="right")
        for p in positions:
            pt.add_row(p.symbol, f"{p.qty:.4f}", f"{p.avg_price:.2f}")
        console.print(pt)

    open_opts = store.open_options()
    ot = Table(title="Open Options")
    ot.add_column("Underlying"); ot.add_column("Type"); ot.add_column("Strike", justify="right")
    ot.add_column("Contracts", justify="right"); ot.add_column("Entry", justify="right")
    for o in open_opts:
        ot.add_row(o["underlying"], o["kind"], f"{o['strike']:.2f}",
                   f"{o['contracts']:.3f}", f"{o['entry_premium']:.3f}")
    if not open_opts:
        ot.add_row("—", "—", "—", "—", "—")
    console.print(ot)

    genomes = store.get_meta("genomes", []) or []
    lt = Table(title="Strategy Leaderboard (learning)")
    lt.add_column("Strategy"); lt.add_column("Genome"); lt.add_column("Trades", justify="right")
    lt.add_column("P&L", justify="right"); lt.add_column("Fitness", justify="right")
    ranked = sorted(genomes,
                    key=lambda g: (g["pnl"] / g["trades"]) if g["trades"] else 0.01,
                    reverse=True)
    for g in ranked[:8]:
        fit = (g["pnl"] / g["trades"]) if g["trades"] else 0.01
        lt.add_row(g["strategy"], g["id"], str(g["trades"]),
                   f"{g['pnl']:+.3f}", f"{fit:+.4f}")
    if not genomes:
        lt.add_row("—", "—", "—", "—", "—")
    console.print(lt)
    store.close()


@app.command("import-news")
def import_news(file: str = typer.Argument(..., help="JSON: [{symbol, text, sentiment?}]")):
    """Import market news items into the bot's news store (used as trading
    signals for up to 24h). Sentiment is parsed from the text if omitted."""
    from .research.news import _parse_sentiment

    path = Path(file)
    try:
        items = json.loads(path.read_text())
        assert isinstance(items, list)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Invalid news file:[/red] {exc}")
        raise typer.Exit(1)

    settings = load_settings()
    store = Store(settings.db_path)
    count = 0
    for item in items:
        symbol = str(item.get("symbol", "")).strip()
        text = str(item.get("text", "")).strip()
        if not symbol or not text:
            continue
        sentiment = item.get("sentiment") or _parse_sentiment(text)
        if sentiment not in ("bullish", "bearish", "neutral"):
            sentiment = "neutral"
        store.add_news(symbol, text[:1000], sentiment)
        count += 1
    store.add_activity("signal", f"imported {count} news items", detail={"file": file})
    store.close()
    console.print(f"[green]Imported {count} news items.[/green]")


@app.command("import-analysis")
def import_analysis(
    file: str = typer.Argument(..., help='JSON: {"valid_hours": 24, "symbols": '
                                         '{"AAPL": {"bias", "confidence", "notes"}}}')
):
    """Import the daily market analysis (compiled by Claude) — becomes a
    weighted directional signal per symbol for the validity window."""
    import time as _time

    path = Path(file)
    try:
        data = json.loads(path.read_text())
        symbols = data["symbols"]
        assert isinstance(symbols, dict) and symbols
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Invalid analysis file:[/red] {exc}")
        raise typer.Exit(1)

    cleaned: dict = {}
    for sym, entry in symbols.items():
        bias = str(entry.get("bias", "")).lower()
        if bias not in ("bullish", "bearish", "neutral"):
            console.print(f"[red]Invalid bias for {sym}:[/red] {bias!r} "
                          "(must be bullish|bearish|neutral)")
            raise typer.Exit(1)
        conf = max(0.0, min(1.0, float(entry.get("confidence", 0.5))))
        cleaned[sym] = {"bias": bias, "confidence": conf,
                        "notes": str(entry.get("notes", ""))[:500]}

    settings = load_settings()
    store = Store(settings.db_path)
    store.set_meta("market_analysis", {
        "ts": _time.time(),
        "valid_hours": float(data.get("valid_hours", 24)),
        "symbols": cleaned,
    })
    store.add_activity("insight", f"daily analysis imported for {len(cleaned)} symbols",
                       detail=cleaned)
    store.close()
    console.print(f"[green]Analysis imported for {len(cleaned)} symbols "
                  f"(valid {data.get('valid_hours', 24)}h).[/green]")


@app.command("export-ledger")
def export_ledger(
    file: str = typer.Argument(..., help="CSV path to append to (created if missing)."),
):
    """Incrementally export new trades to a diffable CSV — meant to be
    committed alongside the DB (e.g. on the bot-state branch) so trade
    history is reviewable via git, not just by querying the opaque SQLite
    file. Safe to call every run: only appends rows not yet exported."""
    import csv as csv_module

    settings = load_settings()
    store = Store(settings.db_path)
    marker_key = "ledger_last_exported_id"
    last_id = int(store.get_meta(marker_key, 0))
    rows = store.trades_since(last_id)

    if not rows:
        store.close()
        console.print("[dim]No new trades to export.[/dim]")
        return

    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "ts", "symbol", "side", "qty", "price", "fee",
                  "mode", "reason", "confidence"]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    store.set_meta(marker_key, rows[-1]["id"])
    store.close()
    console.print(f"[green]Exported {len(rows)} new trade(s) to {path}.[/green]")


@app.command()
def backtest(
    bars: int = typer.Option(500, help="Total daily bars of history."),
    train: int = typer.Option(150, "--train", help="Train window (bars) per fold."),
    test: int = typer.Option(50, "--test", help="Out-of-sample test window (bars)."),
):
    """Walk-forward backtest: pre-train the strategy genome population on
    historical data. The trained population persists, so `lmtrade run` starts
    with the learned fitness."""
    from .backtest.walk_forward import WalkForward

    settings = load_settings()
    store = Store(settings.db_path)
    console.print(f"[cyan]Walk-forward:[/cyan] {bars} bars, "
                  f"{train} train / {test} test per fold, "
                  f"universe {settings.universe}")
    wf = WalkForward(settings, store)
    result = wf.run(settings.universe, bars=bars, train_bars=train, test_bars=test)

    ft = Table(title="Walk-forward folds")
    ft.add_column("Fold", justify="right"); ft.add_column("Best strategy")
    ft.add_column("Train P&L", justify="right")
    ft.add_column("Test P&L (OOS)", justify="right")
    for f in result["folds"]:
        color = "green" if f["test_pnl"] >= 0 else "red"
        ft.add_row(str(f["fold"]), f["best_strategy"],
                   f"{f['train_pnl']:+.2f}",
                   f"[{color}]{f['test_pnl']:+.2f}[/{color}]")
    console.print(ft)
    total = result["total_test_pnl"]
    console.print(f"Total out-of-sample P&L: "
                  f"[{'green' if total >= 0 else 'red'}]{total:+.2f} "
                  f"{settings.currency}[/]")

    lt = Table(title="Trained strategy leaderboard")
    lt.add_column("Strategy"); lt.add_column("Genome")
    lt.add_column("Trades", justify="right"); lt.add_column("Fitness", justify="right")
    for g in result["leaderboard"][:8]:
        lt.add_row(g["strategy"], g["id"], str(g["trades"]), f"{g['fitness']:+.4f}")
    console.print(lt)
    console.print("[dim]Population saved — `lmtrade run` now starts pre-trained.[/dim]")
    store.close()


@app.command()
def analyze():
    """Run the daily Claude strategy review on demand (needs ANTHROPIC_API_KEY)."""
    from .research.daily import DailyAnalyst

    settings = load_settings()
    store = Store(settings.db_path)
    analyst = DailyAnalyst(store, settings)
    result = analyst.run()
    if result is None:
        console.print("[yellow]Analysis unavailable — set ANTHROPIC_API_KEY "
                      "(or the response was unusable).[/yellow]")
    else:
        console.print_json(json.dumps(result))
    store.close()


@app.command()
def viz(
    out: str = typer.Option("data/dashboard.png", help="Output PNG path."),
):
    """Render the dashboard to a PNG (headless). In notebooks use `lmtrade.viz.show()`."""
    import matplotlib
    matplotlib.use("Agg")
    from .viz.panels import dashboard_figure

    settings = load_settings()
    fig = dashboard_figure(settings.db_path)
    out_path = settings.data_dir / "dashboard.png" if out == "data/dashboard.png" else out
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    console.print(f"[green]Dashboard written to[/green] {out_path}")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes", help="Skip confirmation.")):
    """Wipe runtime state (database) and start a fresh account."""
    settings = load_settings()
    if not yes:
        typer.confirm(f"Delete {settings.db_path}? This resets all history.", abort=True)
    if settings.db_path.exists():
        settings.db_path.unlink()
    console.print("[green]State reset.[/green]")


@app.command()
def config():
    """Show the resolved configuration as JSON."""
    settings = load_settings()
    data = settings.model_dump()
    data["data_dir"] = str(data["data_dir"])
    console.print_json(json.dumps(data))


@app.command()
def version():
    """Print the version."""
    console.print(f"LMTrade v{__version__}")


if __name__ == "__main__":
    app()
