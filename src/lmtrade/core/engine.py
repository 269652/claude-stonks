"""The trading engine — high-cadence evaluation loop.

Per cycle:
  1. Scheduled research: hourly Perplexity news per instrument, daily Claude
     strategy review (both no-ops without keys; both injectable for tests).
  2. Quotes for the universe + the retail benchmark (buy-and-hold SPY).
  3. Economics gate: below the GPU-runway floor, only exits are managed.
  4. Manage open positions: option marks vs premium take-profit/stop/expiry,
     equity stop-loss/take-profit. Closed-trade P&L feeds the strategy
     optimizer (evolutionary learning).
  5. New entries: the fusion engine (financial models + SLM + LLM + news
     sentiment) is combined with the active strategy genome's signal; with
     options enabled a buy becomes a long call and a sell a long put —
     defined-risk leverage, which a tiny account needs.
  6. Persist equity, benchmark and alpha snapshots for the dashboard.
"""
from __future__ import annotations

import time
from typing import Callable

from ..agents.fusion import Decision, FusionEngine
from ..brokers.base import Broker
from ..config import Settings
from ..data.market import MarketData, Quote
from ..economics.cost_accounting import CostAccountant
from ..finance.knockouts import (
    is_knocked_out,
    knockout_price,
    strike_after_financing,
)
from ..finance.options import mark_option, synth_option
from ..finance.risk import (
    fee_aware_budget, fee_ratio_ok, min_notional_for_fee, should_exit, size_position,
    trailing_exit,
)
from ..finance.sizing import kelly_fraction, regime, vol_scale
from ..models.base import Signal
from ..models.providers import build_providers
from ..research.daily import DailyAnalyst
from ..research.news import NewsService
from ..strategies.optimizer import StrategyOptimizer
from .control import LOW_BALANCE_EUR
from .events import EventBus
from .scheduler import Scheduler
from .state import Store, Trade
from .exits import exit_plan
from .limit_orders import needs_reprice, target_spot
from .watchlist import entry_z, should_enter

OPTION_FEE = 1.0   # per option/knockout order: TR's flat 1 EUR fee, charged on
                   # entry AND exit and included in every P&L figure
NEWS_MAX_AGE_S = 24 * 3600   # news older than this no longer influences decisions
LIVE_EXIT_RETRY_S = 5 * 60   # after a rejected live sell, retry automatic exits this often
ANALYSIS_RETRY_S = 15 * 60   # when analysis is missing/unusable, retry this often
                             # (not every cycle) so a provider outage doesn't hammer


class Engine:
    def __init__(
        self, settings: Settings, store: Store, broker: Broker,
        news_fetcher=None, analysis_caller=None, market=None,
        tr_derivatives="auto",
        now: Callable[[], float] = time.time,
        fallback_store: Store | None = None,
    ):
        self.settings = settings
        self.store = store
        self.broker = broker
        self.fallback_store = fallback_store  # paper store for fallback analysis in live mode
        self.now = now
        self.bus = EventBus(store)
        self.market = market or MarketData(settings.data.provider,
                                           intraday=settings.data.intraday)
        if tr_derivatives == "auto":
            # Auto-build from settings + env; returns None without TR creds,
            # keeping TR login strictly optional. Pass tr_derivatives=None
            # explicitly to force the synthetic-options path.
            from ..brokers.tr_derivatives import build_tr_derivatives

            tr_derivatives = build_tr_derivatives(settings)
        self.tr_derivatives = tr_derivatives
        self._last_good_price: dict[str, float] = {}
        self.fusion = FusionEngine(settings, build_providers(settings))
        self.accountant = CostAccountant(settings, store)
        self.scheduler = Scheduler(store, now=now)
        self.news = NewsService(store, settings, fetcher=news_fetcher, now=now)
        self.analyst = DailyAnalyst(store, settings, caller=analysis_caller)
        self.optimizer = (
            StrategyOptimizer(store, population=settings.learning.population,
                              epsilon=settings.learning.epsilon,
                              mutation_scale=settings.learning.mutation_scale)
            if settings.learning.enabled else None
        )
        self._closed_since_evolve = 0
        self._stop = False
        self._close_wait_logged: set[tuple[str, str]] = set()   # manual-close retry notices
        self._exit_retry_at: dict[int, float] = {}   # option id -> next live-exit attempt
        self._last_exit_check = 0.0     # wall clock of the last fast exit check
        self._fast_exit_checks = 0
        store.set_meta("mode", broker.mode)
        store.set_meta("universe", settings.universe)

    # ------------------------------------------------------------------ helpers
    def _is_trustworthy(self, quote: Quote) -> bool:
        """A quote is trustworthy if its source matches what the configured
        provider promises. When live data is expected (provider != synthetic)
        but MarketData silently fell back to synthetic (network hiccup, rate
        limiting, ...), the quote is a different price regime entirely and
        must never be used to mark or trade an existing/new position — mixing
        a real strike with a synthetic mark (or vice versa) produces
        nonsensical P&L. See test_data_source_safety.py for the incident this
        guards against."""
        if self.settings.data.provider == "synthetic":
            return True
        return quote.source != "synthetic"

    def _positions_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for pos in self.store.positions():
            total += pos.qty * prices.get(pos.symbol, pos.avg_price)
        return total

    def _mark_position(self, o: dict, spot: float) -> float:
        """Current per-unit mark for an open position, by instrument type."""
        if (o.get("instrument_type") or "option") == "knockout":
            if is_knocked_out(spot, o["barrier"], o["kind"]):
                return 0.0
            days_held = max(0.0, (self.now() - o["opened_ts"]) / 86400.0)
            eff_strike = strike_after_financing(o["strike"], o["kind"], days_held)
            return knockout_price(spot, eff_strike, o["ratio"] or 1.0, o["kind"])
        return mark_option(spot, o["strike"], o["expiry_ts"], o["iv"], o["kind"])

    def _options_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for o in self.store.open_options():
            spot = prices.get(o["underlying"])
            if spot is None:
                continue
            total += o["contracts"] * self._mark_position(o, spot)
        return total

    def _persist_option_marks(self, prices: dict[str, float]) -> None:
        """Store the current mark + unrealized P&L for each open option so the
        dashboard can show live profit/loss (it's read-only over the Store and
        has no market feed of its own). Keyed by option id as a string."""
        marks: dict[str, dict] = {}
        cfg = self.settings.options
        peaks: dict = self.store.get_meta("trail_peaks") or {}
        overrides: dict = self.store.get_meta("exit_overrides") or {}
        for o in self.store.open_options():
            plan = exit_plan(o, cfg, overrides.get(str(o["id"])), OPTION_FEE)
            spot = prices.get(o["underlying"])
            # Fall back to last-known price if no fresh quote this cycle
            if spot is None:
                spot = self._last_good_price.get(o["underlying"])
            if spot is None:
                continue
            mark = self._mark_position(o, spot)
            value = o["contracts"] * mark
            cost = o["contracts"] * o["entry_premium"]
            peak = peaks.get(str(o["id"])) if plan.trail_enabled else None
            active, stop = False, None
            if peak is not None:
                net = value - cost - 2 * OPTION_FEE
                active, stop, _ = trailing_exit(float(peak), net, plan.trail_min,
                                                plan.trail_pct)
            marks[str(o["id"])] = {
                "mark_premium": round(mark, 4),
                "value": round(value, 4),
                "unrealized_pnl": round(value - cost - OPTION_FEE, 4),   # net of entry fee
                "spot": round(spot, 4),
                "peak_pnl": None if peak is None else round(float(peak), 4),
                "trail_active": active,
                "trail_stop_pnl": None if stop is None else round(stop, 4),
            }
        self.store.set_meta("open_option_marks", marks)

    def _get_analysis_with_fallback(self) -> dict | None:
        """Get daily market analysis from store, falling back to paper store if live analysis
        is missing (useful when live mode doesn't have analysis but paper run does)."""
        analysis = self.store.get_meta("market_analysis")
        if analysis is None and self.fallback_store is not None:
            analysis = self.fallback_store.get_meta("market_analysis")
        return analysis

    def _analysis_current(self, analysis: dict | None) -> bool:
        """True when a stored market analysis exists and is still inside its own
        validity window (ts + valid_hours). Used to reuse a fresh analysis
        across restarts instead of rebuilding it every launch. A malformed or
        timestamp-less analysis counts as not current so it gets refreshed."""
        if not isinstance(analysis, dict):
            return False
        ts = analysis.get("ts")
        if ts is None:
            return False
        valid_h = analysis.get(
            "valid_hours", self.settings.research.daily_analysis_interval_hours)
        try:
            return (self.now() - float(ts)) < float(valid_h) * 3600
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------ research jobs
    def _run_scheduled_jobs(self) -> None:
        # The news pass fires on the (shorter) retry cadence; get()/get_many
        # only actually refetch a symbol whose cache is missing or older than
        # news_interval_minutes, so healthy symbols aren't re-fetched — only
        # failed/corrupted ones retry sooner (default every 10 min).
        news_iv = min(self.settings.research.news_interval_minutes,
                      self.settings.research.news_retry_minutes) * 60
        if self.scheduler.due("news", news_iv):
            universe = self.settings.universe
            provider = self.settings.research.news_provider
            self.bus.info(
                f"[research] fetching news for {len(universe)} symbols "
                f"(provider={provider}) — this can take a few minutes with a "
                f"live web-search provider...", source="research")

            def _news_progress(symbol: str, item: dict | None, done: int, total: int) -> None:
                # Fired as each symbol completes (in the calling thread), so
                # the run shows live progress instead of appearing hung.
                if item:
                    self.bus.activity(
                        "signal", f"news {done}/{total} {symbol}: {item['sentiment']}",
                        symbol, {"text": str(item.get("text", ""))[:300]})
                else:
                    self.bus.info(f"[research] news {done}/{total} {symbol}: no data",
                                  source="research")

            self.news.get_many(universe, max_workers=min(8, len(universe)),
                               on_progress=_news_progress)
            self.bus.info("[research] news fetch complete", source="research")
        daily_iv = self.settings.research.daily_analysis_interval_hours * 3600
        # Reuse an up-to-date analysis already in the DB instead of rebuilding
        # it on every (re)start: a stored analysis is current while it's inside
        # its own validity window (ts + valid_hours). Only when it's missing or
        # stale do we (re)compile — gated by the scheduler so a provider outage
        # retries periodically rather than every cycle. When it IS current we
        # still advance the daily stamp so a fresh scheduler (post-restart)
        # doesn't consider it overdue and thrash.
        existing = self.store.get_meta("market_analysis")
        if self._analysis_current(existing):
            self.scheduler.due("daily_analysis", daily_iv)
        elif self.scheduler.due("daily_analysis", ANALYSIS_RETRY_S):
            provider = self.settings.research.analysis_provider
            self.bus.info(
                f"[research] compiling daily market analysis (provider={provider}) "
                f"for {len(self.settings.universe)} symbols...", source="research")
            analysis = self.analyst.compile_analysis(self.settings.universe, self.now())
            if analysis:
                self.bus.info(
                    f"[research] daily analysis compiled: {len(analysis['symbols'])} "
                    f"symbols", source="research")
            else:
                self.bus.warn("[research] daily analysis could not be compiled "
                              "(provider unavailable or empty) — will retry.",
                              source="research")
            # The risk-parameter review is a separate, best-effort step.
            result = self.analyst.run()
            if result:
                self.bus.info(f"[research] risk review applied: {result['applied']}",
                              source="research")

    def _update_realized_mark(self, net_worth: float) -> None:
        """Maintain `last_realized_net_worth`: the net worth locked in at the
        most recent REALIZED (closed) trade. The dashboard colors net worth
        red while the live mark-to-market value sits below this, green at or
        above — a realized high-water mark that resets each time a position
        closes (win or loss)."""
        if self.store.get_meta("last_realized_net_worth") is None:
            self.store.set_meta(
                "last_realized_net_worth",
                float(self.store.get_meta("starting_cash", self.settings.budget)))
        closed = self.store.closed_options_count()
        if closed > int(self.store.get_meta("realized_close_count", 0)):
            # A trade just realized — reset the mark to the current net worth.
            self.store.set_meta("last_realized_net_worth", round(net_worth, 6))
            self.store.set_meta("realized_close_count", closed)

    # ---------------------------------------------------------------- benchmark
    def _update_benchmark(self, prices: dict[str, float]) -> None:
        sym = self.settings.benchmark.symbol
        price = prices.get(sym)
        if price is None or price <= 0:
            return
        entry = self.store.get_meta("benchmark_entry")
        if entry is None:
            entry = {"symbol": sym, "price": price}
            self.store.set_meta("benchmark_entry", entry)
        bench_equity = self.settings.budget * (price / entry["price"])
        self.store.record_benchmark(bench_equity)
        cash = self.broker.cash()
        bot_equity = cash + self._positions_value(prices) + self._options_value(prices)
        self.store.set_meta("alpha", round(bot_equity - bench_equity, 6))

    # ------------------------------------------------------------ option exits
    def _manage_options(self, prices: dict[str, float], tradeable: set[str]) -> None:
        cfg = self.settings.options
        peaks: dict = self.store.get_meta("trail_peaks") or {}
        overrides: dict = self.store.get_meta("exit_overrides") or {}
        for o in self.store.open_options():
            underlying = o["underlying"]
            if underlying not in tradeable:
                continue  # no fresh trustworthy quote this cycle — leave untouched
            spot = prices.get(underlying)
            if spot is None:
                continue
            is_ko = (o.get("instrument_type") or "option") == "knockout"
            mark = self._mark_position(o, spot)
            entry = max(1e-9, o["entry_premium"])
            change = (mark - entry) / entry
            hours_left = (o["expiry_ts"] - self.now()) / 3600.0
            hours_held = (self.now() - o["opened_ts"]) / 3600.0
            # Effective exits: the position's own stops (set at open), or the
            # per-position override saved from the dashboard (core/exits.py).
            plan = exit_plan(o, cfg, overrides.get(str(o["id"])), OPTION_FEE)
            tp, sl = plan.tp, plan.sl

            expiry_hit = hours_left <= cfg.min_hours_to_expiry
            held_long_enough = hours_held >= cfg.min_hold_hours
            reason = None
            if is_ko and is_knocked_out(spot, o["barrier"], o["kind"]):
                # A knockout is an involuntary event: the certificate IS dead
                # the moment the barrier is touched — overrides every gate.
                mark = 0.0
                change = -1.0
                reason = (f"KNOCKED OUT (spot {spot:.2f} touched barrier "
                          f"{o['barrier']:.2f}) — total loss of premium")
            elif expiry_hit:
                # A hard constraint of the option itself — overrides min_hold.
                reason = f"expiry window ({hours_left:.1f}h left)"
            elif cfg.max_hold_hours > 0 and hours_held >= cfg.max_hold_hours:
                reason = f"max hold time reached ({hours_held:.1f}h)"
            elif held_long_enough and tp is not None and mark >= tp:
                reason = f"take-profit hit (mark {mark:.3f} >= TP {tp:.3f}, {change:+.0%})"
            elif held_long_enough and sl is not None and mark <= sl:
                reason = f"stop-loss hit (mark {mark:.3f} <= SL {sl:.3f}, {change:+.0%})"
            if plan.trail_enabled and not (is_ko and mark == 0.0):
                net = (mark - entry) * o["contracts"] - 2 * OPTION_FEE   # if closed now
                key = str(o["id"])
                peak = max(float(peaks.get(key, net)), net)
                peaks[key] = peak
                _, stop, hit = trailing_exit(peak, net, plan.trail_min, plan.trail_pct)
                if reason is None and hit and held_long_enough:
                    reason = (f"trailing take-profit (peak {peak:+.2f} €, now {net:+.2f} €, "
                              f"stop {stop:+.2f} €)")
            if reason is None:
                continue
            knocked = reason.startswith("KNOCKED OUT")
            self._exit_option(o, mark, reason, live_order=not knocked)

        open_ids = {str(o["id"]) for o in self.store.open_options()}
        self.store.set_meta("trail_peaks", {k: v for k, v in peaks.items() if k in open_ids})

        # Stale eviction: close positions that have been sideways for too long
        # to free a slot for a stronger incoming signal.  Runs after the normal
        # TP/SL/expiry loop so already-closed options are never double-processed
        # (close_option removes them from open_options()).  Only fires on
        # positions older than stale_hours with pnl below stale_max_profit_pct —
        # winners are always held; the normal SL cuts big losers; only dead-
        # weight "stuck" positions are cleaned up here.
        if self.settings.loop.stale_evict_enabled:
            stale_h = self.settings.loop.stale_hours
            max_profit = self.settings.loop.stale_max_profit_pct
            now = self.now()
            for o in list(self.store.open_options()):
                spot = prices.get(o["underlying"])
                if spot is None:
                    continue
                age_h = (now - o["opened_ts"]) / 3600.0
                if age_h < stale_h:
                    continue
                entry = max(1e-9, float(o["entry_premium"]))
                mark = self._mark_position(o, spot)
                pnl_pct = (mark - entry) / entry
                if pnl_pct >= max_profit:
                    continue   # winning position — don't evict
                self._exit_option(
                    o, mark,
                    f"stale ({age_h:.0f}h, {pnl_pct:+.1%}): freeing slot for stronger signal",
                    label="STALE CLOSE")

    def _live_armed(self) -> bool:
        return bool(getattr(self.broker, "armed", False)) and hasattr(self.broker, "place_order")

    def _exit_option(self, o: dict, mark: float, reason: str, label: str = "CLOSE",
                     live_order: bool = True, manual: bool = False) -> bool:
        """Close a position. In ARMED live mode a real TR sell order for the
        certificate's ISIN goes out first, and the close is only booked once
        TR confirms it; on rejection the position stays open and automatic
        retries back off for LIVE_EXIT_RETRY_S (a manual close always tries).
        `live_order=False` for a knock-out, which TR settles itself. Returns
        True when the close was booked."""
        if self._live_armed() and live_order:
            isin = o.get("isin")
            if not isin:
                self.bus.warn(
                    f"LIVE exit {o['underlying']} {o['kind']}: no ISIN on this position "
                    f"(not a real TR instrument) — closing the local record only.",
                    source="engine")
            else:
                now = self.now()
                if not manual and self._exit_retry_at.get(o["id"], 0.0) > now:
                    return False
                size = float(int(o["contracts"]))   # whole certificates
                res = self.broker.place_order(isin, "sell", size)
                if not res.ok:
                    self._exit_retry_at[o["id"]] = now + LIVE_EXIT_RETRY_S
                    self.bus.warn(
                        f"LIVE exit {o['underlying']} [{isin}] rejected/unconfirmed — "
                        f"position stays open ({reason}): {res.message}", source="engine")
                    return False
                self._exit_retry_at.pop(o["id"], None)
                reason = f"{reason} [live sell confirmed: {res.message}]"
        self._book_option_close(o, mark, reason, label=label)
        return True

    def _book_option_close(self, o: dict, mark: float, reason: str,
                           label: str = "CLOSE") -> float:
        """Close an option/knockout at `mark` (paper bookkeeping): credit
        proceeds minus the exit fee, P&L net of entry + exit fee, profit
        stash, trade record, activity and strategy learning. Returns P&L."""
        entry = max(1e-9, o["entry_premium"])
        proceeds = mark * o["contracts"] - OPTION_FEE
        pnl = (mark - entry) * o["contracts"] - 2 * OPTION_FEE   # entry + exit fee
        self.broker.adjust_cash(max(0.0, proceeds))
        self.store.record_cost("fee", OPTION_FEE, "options")
        self.store.close_option(o["id"], mark, pnl)
        for key in ("trail_peaks", "exit_overrides"):
            state = self.store.get_meta(key) or {}
            if state.pop(str(o["id"]), None) is not None:
                self.store.set_meta(key, state)

        # Profit stash: a configured fraction of realized PROFIT (never
        # losses) is moved out of tradeable cash into a reserve. Still
        # counted in net worth/alpha — just protected from being re-risked.
        stash_pct = self.settings.economics.profit_stash_pct
        if pnl > 0 and stash_pct > 0:
            stash = pnl * stash_pct
            if self.broker.adjust_cash(-stash):
                self.store.add_reserve(stash)

        self.store.record_trade(Trade(
            o["underlying"], "sell", o["contracts"], mark, OPTION_FEE,
            self.broker.mode, f"close {o['kind']} — {reason}", None))
        self.bus.activity(
            "trade",
            f"{label} {o['kind'].upper()} {o['underlying']} pnl {pnl:+.3f} — {reason}",
            o["underlying"], {"pnl": pnl})
        self._record_learning(o.get("genome_id"), pnl)
        return pnl

    def process_close_requests(self) -> int:
        """Execute manual close requests queued by the dashboard
        (meta "close_requests") at the current price. Returns how many
        positions were closed. A request without a trustworthy quote stays
        queued for a retry; requests for positions that no longer exist are
        dropped. Refused in armed live mode: this exit path only books
        locally and does not send a real TR sell order."""
        reqs = self.store.get_meta("close_requests") or []
        if not reqs:
            return 0
        armed_live = self._live_armed()
        keep: list[dict] = []
        closed = 0
        for r in reqs:
            kind = r.get("kind")
            if kind == "option":
                o = next((x for x in self.store.open_options()
                          if x["id"] == r.get("id")), None)
                if o is None:
                    continue
                symbol, what = o["underlying"], f"{o['underlying']} {o['kind']}"
            elif kind == "equity":
                pos = self.store.position(r.get("symbol") or "")
                if pos is None:
                    continue
                symbol, what = pos.symbol, pos.symbol
            else:
                continue
            if armed_live and kind == "equity":
                self.bus.warn(f"Manual close of {what} refused: the live bot can only sell "
                              f"TR certificates by ISIN — close this one in the TR app.",
                              source="engine")
                continue
            quote = self.market.quote(symbol)
            if not self._is_trustworthy(quote):
                keep.append(r)
                if (kind, symbol) not in self._close_wait_logged:
                    self._close_wait_logged.add((kind, symbol))
                    self.bus.warn(f"Manual close of {what}: no trustworthy live price "
                                  f"right now — will retry.", source="engine")
                continue
            self._close_wait_logged.discard((kind, symbol))
            self._last_good_price[symbol] = quote.price
            if kind == "option":
                is_ko = (o.get("instrument_type") or "option") == "knockout"
                knocked = is_ko and is_knocked_out(quote.price, o["barrier"], o["kind"])
                mark = 0.0 if knocked else self._mark_position(o, quote.price)
                if not self._exit_option(o, mark, "manual close from dashboard",
                                         label="MANUAL CLOSE", live_order=not knocked,
                                         manual=True):
                    continue   # TR rejected: request dropped, warning logged
            else:
                res = self.broker.sell(symbol, pos.qty, quote.price)
                if not res.ok:
                    self.bus.warn(f"Manual close of {symbol} failed: {res.message}",
                                  source="engine")
                    continue
                self.store.record_trade(Trade(
                    symbol, "sell", res.qty, res.price, res.fee, self.broker.mode,
                    "manual close from dashboard", None))
                self.bus.activity("trade", f"MANUAL CLOSE {symbol} @ {res.price:.2f}",
                                  symbol)
            closed += 1
        # Re-read: the dashboard may have queued more while we were working.
        handled = [r for r in reqs if r not in keep]
        current = self.store.get_meta("close_requests") or []
        self.store.set_meta("close_requests", [r for r in current if r not in handled])
        return closed

    def _check_exits_fast(self) -> None:
        """Exit check between full cycles: fresh quotes for the symbols we
        hold, then the normal exit ladder (TP/SL/trailing/knock-out). A full
        cycle takes minutes; without this a trailing stop could be overshot
        by whatever the price did in between."""
        now = time.time()
        if now - self._last_exit_check < self.settings.loop.exit_check_seconds:
            return
        self._last_exit_check = now
        held = {o["underlying"] for o in self.store.open_options()}
        if not held:
            return
        self._fast_exit_checks += 1
        prices: dict[str, float] = {}
        tradeable: set[str] = set()
        for symbol in held:
            try:
                q = self.market.quote(symbol)
            except Exception:  # noqa: BLE001 — a failed quote just skips this check
                continue
            if self._is_trustworthy(q):
                prices[symbol] = q.price
                self._last_good_price[symbol] = q.price
                tradeable.add(symbol)
        if tradeable:
            self._manage_options(prices, tradeable)
            self._persist_option_marks(prices)

    def _poll_close_requests(self) -> None:
        if self.store.get_meta("close_requests"):
            self.process_close_requests()

    def _idle(self, seconds: float) -> None:
        """Sleep between cycles, executing manual close requests within ~1s."""
        deadline = time.time() + max(0.0, seconds)
        while not self._stop:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
            self._poll_close_requests()
            self._check_exits_fast()

    def _record_learning(self, genome_id: str | None, pnl: float) -> None:
        if not (self.optimizer and genome_id):
            return
        self.optimizer.record_result(genome_id, pnl)
        self._closed_since_evolve += 1
        if self._closed_since_evolve >= self.settings.learning.evolve_every_trades:
            self._closed_since_evolve = 0
            mutant = self.optimizer.evolve()
            if mutant:
                self.bus.activity(
                    "learning",
                    f"evolved: new {mutant.strategy} genome {mutant.id}",
                    detail={"params": mutant.params})

    # ---------------------------------------------------------------- entries
    def _decide(self, quote: Quote) -> tuple[Decision, str | None]:
        extra: list[Signal] = []
        genome_id = None
        if self.optimizer:
            genome = self.optimizer.select()
            # Market context for cross-sectional families (xsmom, rel_value):
            # the whole cycle's histories, set by run_cycle. Kept on self (not
            # a parameter) so tests stubbing _decide(quote) stay valid.
            market_ctx = {
                "symbol": quote.symbol,
                "histories": getattr(self, "_cycle_histories", {}),
                "benchmark": self.settings.benchmark.symbol,
            }
            d, s = genome.signal(quote.history, market_ctx)
            extra.append(Signal("strategy", d, s,
                                f"{genome.strategy} {genome.params}", 0.0))
            genome_id = genome.id
        ctx: dict = {}
        latest = self.store.latest_news(quote.symbol)
        if latest and (self.now() - latest["ts"]) < NEWS_MAX_AGE_S:
            ctx["research"] = latest["text"]
            sent = latest.get("sentiment")
            if sent in ("bullish", "bearish"):
                extra.append(Signal("news", "buy" if sent == "bullish" else "sell",
                                    0.6, f"news sentiment {sent}", 0.0))
        # Daily market analysis (compiled by the Claude Routine, imported via
        # `lmtrade import-analysis`) — a directional bias per symbol, valid for
        # the window it declares. Falls back to paper store if live analysis
        # is missing (e.g. Claude reached limits but paper run succeeded).
        analysis = self._get_analysis_with_fallback()
        if analysis:
            age = self.now() - float(analysis.get("ts", 0))
            valid_s = float(analysis.get("valid_hours", 24)) * 3600
            entry = (analysis.get("symbols") or {}).get(quote.symbol)
            if age < valid_s and entry and entry.get("bias") in ("bullish", "bearish"):
                direction = "buy" if entry["bias"] == "bullish" else "sell"
                conf = max(0.0, min(1.0, float(entry.get("confidence", 0.5))))
                extra.append(Signal("analysis", direction, conf,
                                    str(entry.get("notes", ""))[:200], 0.0))
        decision = self.fusion.decide(quote, extra_signals=extra, context_extra=ctx)
        self.accountant.record_inference("fusion", decision.inference_cost)
        return decision, genome_id

    def _genome_stats(self, genome_id: str | None):
        if not (self.optimizer and genome_id):
            return None
        for g in self.optimizer.genomes():
            if g.id == genome_id:
                return g
        return None

    def _position_budget(self, quote: Quote, decision: Decision,
                         genome_id: str | None) -> float:
        """Signal-scaled budget with the fee floor baked in (see
        finance.risk.fee_aware_budget): raised to the fee minimum when the
        hard caps allow it, 0.0 when no fee-compliant order fits."""
        sized, ceiling = self._budget_bounds(quote, decision, genome_id)
        return fee_aware_budget(sized, ceiling, OPTION_FEE,
                                self.settings.risk.max_fee_pct)

    def _budget_bounds(self, quote: Quote, decision: Decision,
                       genome_id: str | None) -> tuple[float, float]:
        """Premium budget for a new position, layering the quant sizing rules:
        fractional Kelly from the genome's empirical edge (when proven),
        volatility targeting, and the storm-regime haircut. Falls back to the
        plain confidence-scaled fraction when there isn't enough data.

        The cash reserve cap (options.cash_reserve_pct) further limits the
        budget: at most (1 - reserve_pct) of current equity may be deployed in
        open positions at any time. As the portfolio grows the deployable bucket
        grows proportionally, so the engine automatically uses larger positions
        after profitable runs without any manual parameter changes."""
        cfg = self.settings.options
        s = self.settings.sizing
        cash = self.broker.cash()
        # Use all last-known prices for a portfolio-wide equity estimate,
        # not just the current quote — options on OTHER symbols have value too.
        all_prices = {**self._last_good_price, quote.symbol: quote.price}
        positions_value = self._positions_value(all_prices)
        options_value = self._options_value(all_prices)
        equity = cash + positions_value + options_value

        fraction = cfg.max_option_fraction * decision.confidence
        g = self._genome_stats(genome_id)
        if (s.kelly_enabled and g is not None
                and (g.wins + g.losses) >= s.kelly_min_trades):
            kf = kelly_fraction(g.win_rate, g.avg_win, g.avg_loss) \
                * s.kelly_fraction_of_full
            fraction = min(cfg.max_option_fraction, kf)
        fraction *= vol_scale(quote.history, s.vol_target_annual)
        if s.regime_filter_enabled and regime(quote.history) == "storm":
            fraction *= s.storm_size_factor

        # Cash reserve: cap the budget so total deployed capital never exceeds
        # (1 - cash_reserve_pct) * equity.  Already-deployed capital is
        # subtracted first; the remainder is available for this new position.
        deployed = positions_value + options_value
        # Resting limit entries reserve capital (TR blocks the cash too) —
        # except this symbol's own order, which is what's being (re)sized.
        reserved = sum(float(p.get("notional", 0.0)) + OPTION_FEE
                       for s, p in (self.store.get_meta("pending_entries") or {}).items()
                       if s != quote.symbol)
        deployed += reserved
        cash -= reserved
        max_deployable = equity * (1.0 - cfg.cash_reserve_pct)
        available = max(0.0, max_deployable - deployed)

        ceiling = min(cfg.max_option_fraction * equity, available, cash - OPTION_FEE)
        return min(fraction * equity, ceiling), ceiling

    def _watchlist_triggers(self, candidates: list[tuple[Decision, str | None]],
                            evaluated: set[str], quotes: dict[str, Quote],
                            ) -> list[tuple[Decision, str | None]]:
        """Entry timing (core/watchlist.py). Every live signal sits on the
        watch list; only those whose price is stretched >= watch_k sigma from
        the intraday SMA in their favour are returned for execution. Entries
        expire after watch_max_hours, and a signal that expired is not
        re-watched until it disappears or flips (else the timeout would just
        restart forever)."""
        cfg = self.settings.entry
        now = self.now()
        wl: dict = self.store.get_meta("watchlist") or {}
        expired: dict = self.store.get_meta("watchlist_expired") or {}
        current = {d.symbol: (d, g) for d, g in candidates}

        for sym in list(wl):
            if now - float(wl[sym]["added_ts"]) > cfg.watch_max_hours * 3600:
                expired[sym] = wl.pop(sym)["direction"]
                self.bus.info(f"[watch] {sym} expired after {cfg.watch_max_hours:g}h "
                              f"without a >={cfg.watch_k:g}σ entry — dropped",
                              source="engine")
            elif sym in evaluated and sym not in current:
                wl.pop(sym)
                self.bus.info(f"[watch] {sym} removed — signal gone", source="engine")
        for sym in list(expired):
            if sym in evaluated and (sym not in current
                                     or current[sym][0].direction != expired[sym]):
                expired.pop(sym)

        triggered: list[tuple[Decision, str | None]] = []
        for sym, (decision, genome_id) in current.items():
            if expired.get(sym) == decision.direction:
                continue
            quote = quotes[sym]
            z = entry_z(quote.price, quote.history, cfg.watch_window)
            z_txt = "n/a" if z is None else f"{z:+.2f}σ"
            side = "below" if decision.direction == "buy" else "above"
            entry = wl.get(sym)
            if entry is None or entry["direction"] != decision.direction:
                wl[sym] = {"direction": decision.direction,
                           "confidence": round(decision.confidence, 3),
                           "added_ts": now, "genome_id": genome_id}
                self.bus.info(
                    f"[watch] {sym}: {decision.direction.upper()} conf "
                    f"{decision.confidence:.2f} watching — enter at >={cfg.watch_k:g}σ "
                    f"{side} intraday SMA (now {z_txt})", source="engine")
            else:
                entry["confidence"] = round(decision.confidence, 3)
            # For the dashboard's Watchlist tab: where price is right now.
            wl[sym]["z"] = None if z is None else round(z, 4)
            wl[sym]["checked_ts"] = now
            if should_enter(decision.direction, z, cfg.watch_k):
                self.bus.info(f"[watch] {sym}: {decision.direction.upper()} triggered "
                              f"at {z_txt}", source="engine")
                triggered.append((decision, genome_id))

        self.store.set_meta("watchlist", wl)
        self.store.set_meta("watchlist_expired", expired)
        return triggered

    # ------------------------------------------------- exchange-side entries
    def _instrument_price(self, inst: dict, spot: float) -> float:
        """Model price of an instrument (option / knockout) at `spot`."""
        pseudo = {**inst, "opened_ts": self.now(), "entry_premium": 0.0, "contracts": 0.0}
        return self._mark_position(pseudo, spot)

    def _select_instrument(self, quote: Quote, direction: str, armed: bool) -> dict | None:
        """Real TR knockout when the TR client is up; synthetic option only
        when NOT armed live (a synthetic order can't rest on an exchange)."""
        tr = self.tr_derivatives
        if tr is not None and tr.available():
            ko = tr.find_knockout(quote.symbol, direction, quote.price,
                                  self.settings.tr.target_leverage)
            if ko is not None and ko.price > 0:
                return {"instrument_type": "knockout", "kind": ko.kind, "strike": ko.strike,
                        "barrier": ko.barrier, "ratio": ko.ratio, "isin": ko.isin,
                        "expiry_ts": self.now() + 365 * 86400.0, "iv": 0.0}
        if armed:
            return None
        kind = "call" if direction == "buy" else "put"
        oq = synth_option(quote.symbol, quote.price, quote.history, kind,
                          expiry_days=self.settings.options.expiry_days)
        return {"instrument_type": "option", "kind": kind, "strike": oq.strike,
                "barrier": None, "ratio": None, "isin": None,
                "expiry_ts": oq.expiry_ts, "iv": oq.iv}

    def _place_limit_entry(self, sym: str, w: dict, quote: Quote, armed: bool) -> dict | None:
        cfg = self.settings.entry
        direction = w["direction"]
        tgt = target_spot(quote.history, cfg.watch_window, cfg.watch_k, direction)
        if tgt is None:
            return None
        inst = self._select_instrument(quote, direction, armed)
        if inst is None:
            self.bus.info(f"[limit] {sym}: no tradeable instrument for a resting order",
                          source="engine")
            return None
        limit = self._instrument_price(inst, tgt)
        if limit <= 0:
            return None
        decision = Decision(sym, direction, float(w.get("confidence") or 0.0),
                            "watch-list limit entry")
        budget = self._position_budget(quote, decision, w.get("genome_id"))
        if budget <= 0.05:
            return None
        size = budget / limit
        order_id = None
        if armed:
            size = float(int(size))   # whole certificates
            if size < 1 or not fee_ratio_ok(size * limit, OPTION_FEE,
                                            self.settings.risk.max_fee_pct):
                self.bus.info(f"[limit] {sym}: whole-certificate size misses the fee "
                              f"minimum — no order", source="engine")
                return None
            from .control import ControlState
            if ControlState.load(self.settings.control_path).low_balance_blocks(
                    self.broker.cash()):
                self.bus.warn(f"LIVE limit order blocked [{inst['isin']}]: balance under "
                              f"€{LOW_BALANCE_EUR:.0f} and the low-balance guard is not armed.",
                              source="engine")
                return None
            res = self.broker.place_limit_order(inst["isin"], "buy", size, limit)
            if not res.ok:
                self.bus.warn(f"LIVE limit order rejected [{inst['isin']}]: {res.message}",
                              source="engine")
                return None
            order_id, limit = res.order_id, (res.price or limit)
        side = "below" if direction == "buy" else "above"
        self.bus.info(
            f"[limit] {sym}: {direction.upper()} {inst['kind']} resting limit "
            f"×{size:.3f} @ {limit:.3f} — fills when {sym} reaches {tgt:.2f} "
            f"({cfg.watch_k:g}σ {side} intraday SMA)", source="engine")
        return {"direction": direction, "confidence": w.get("confidence"),
                "genome_id": w.get("genome_id"), "instrument": inst, "limit": limit,
                "size": size, "notional": size * limit, "target_spot": tgt,
                "placed_ts": self.now(), "order_id": order_id}

    def _book_limit_fill(self, sym: str, p: dict, price: float) -> None:
        inst, contracts = p["instrument"], float(p["size"])
        if not self._live_armed():
            if not self.broker.adjust_cash(-(contracts * price + OPTION_FEE)):
                self.bus.warn(f"[limit] {sym}: fill needs more cash than available — "
                              f"not booked", source="engine")
                return
        self.store.record_cost("fee", OPTION_FEE, "options")
        cfg = self.settings.options
        self.store.open_option(
            sym, inst["kind"], inst["strike"], inst["expiry_ts"], inst.get("iv") or 0.0,
            contracts, price, p.get("genome_id"),
            tp_premium=price * (1 + cfg.take_profit_pct),
            sl_premium=price * (1 - cfg.stop_loss_pct),
            instrument_type=inst["instrument_type"], barrier=inst.get("barrier"),
            ratio=inst.get("ratio"), isin=inst.get("isin"))
        self.store.record_trade(Trade(
            sym, "buy", contracts, price, OPTION_FEE, self.broker.mode,
            f"limit fill {inst['kind']} {inst.get('isin') or 'synthetic'} — watch-list entry",
            p.get("confidence")))
        self.bus.activity("trade", f"LIMIT FILL {inst['kind'].upper()} {sym} "
                                   f"×{contracts:.3f} @ {price:.3f}", sym)

    def _drop_pending(self, sym: str, p: dict, why: str) -> bool:
        """Cancel a resting order. Returns False when it must stay tracked
        (live cancel failed and TR doesn't confirm it's gone)."""
        oid = p.get("order_id")
        if self._live_armed() and oid:
            if not self.broker.cancel_order(oid):
                st = self.broker.order_status(oid)
                if st and st["state"] == "filled":
                    self._book_limit_fill(sym, p, st.get("fill_price") or p["limit"])
                    return True
                if st is None or st["state"] == "open":
                    self.bus.warn(f"[limit] {sym}: could not cancel order {oid} — "
                                  f"still tracking it", source="engine")
                    return False
        self.bus.info(f"[limit] {sym}: order cancelled — {why}", source="engine")
        return True

    def _manage_limit_entries(self, quotes: dict[str, Quote], tradeable: set[str],
                              econ) -> None:
        """Lifecycle of resting limit entries: fills -> positions, cancels when
        the watch entry is gone, re-pricing on drift, and new orders for the
        watched signals closest to their entry while capital and slots last."""
        cfg = self.settings.entry
        armed = self._live_armed()
        wl: dict = self.store.get_meta("watchlist") or {}
        pend: dict = self.store.get_meta("pending_entries") or {}

        for sym, p in list(pend.items()):
            w = wl.get(sym)
            if w is None or w.get("direction") != p["direction"]:
                if self._drop_pending(sym, p, "signal gone or watch expired"):
                    pend.pop(sym)
                continue
            quote = quotes.get(sym)
            if armed:
                st = self.broker.order_status(p["order_id"]) if p.get("order_id") else None
                if st is None:
                    continue                      # unknown: never assume a fill
                if st["state"] == "filled":
                    self._book_limit_fill(sym, p, st.get("fill_price") or p["limit"])
                    pend.pop(sym)
                    continue
                if st["state"] == "gone":
                    pend.pop(sym)                 # expired at TR: may be re-placed
                    continue
            elif quote is not None and sym in tradeable:
                now_price = self._instrument_price(p["instrument"], quote.price)
                if 0 < now_price <= p["limit"]:   # buy limit: fills at or below
                    self._book_limit_fill(sym, p, now_price)
                    pend.pop(sym)
                    continue
            if quote is None or sym not in tradeable:
                continue
            tgt = target_spot(quote.history, cfg.watch_window, cfg.watch_k, p["direction"])
            if tgt is None:
                continue
            new_limit = self._instrument_price(p["instrument"], tgt)
            if new_limit <= 0 or not needs_reprice(p["limit"], new_limit, cfg.reprice_drift):
                continue
            if armed:
                if not self._drop_pending(sym, p, "re-pricing"):
                    continue
                if not self.store.open_options() or all(
                        o["underlying"] != sym for o in self.store.open_options()):
                    res = self.broker.place_limit_order(p["instrument"]["isin"], "buy",
                                                        p["size"], new_limit)
                    if not res.ok:
                        self.bus.warn(f"[limit] {sym}: re-placing failed — {res.message}",
                                      source="engine")
                        pend.pop(sym)
                        continue
                    p["order_id"], new_limit = res.order_id, (res.price or new_limit)
                else:
                    pend.pop(sym)                 # filled while re-pricing
                    continue
            self.bus.info(f"[limit] {sym}: re-priced {p['limit']:.3f} → {new_limit:.3f} "
                          f"(target {tgt:.2f})", source="engine")
            p.update(limit=new_limit, notional=p["size"] * new_limit, target_spot=tgt)
        self.store.set_meta("pending_entries", pend)

        held = {o["underlying"] for o in self.store.open_options()}
        held |= {pos.symbol for pos in self.store.positions()}
        free = self.settings.loop.max_positions - len(held) - len(pend)
        k = cfg.watch_k

        def to_go(w: dict) -> float:
            z = w.get("z")
            if z is None:
                return float("inf")
            return max(0.0, z + k) if w["direction"] == "buy" else max(0.0, k - z)

        waiting = sorted(((s, w) for s, w in wl.items()
                          if s not in pend and s not in held and s in tradeable and s in quotes),
                         key=lambda sw: to_go(sw[1]))
        for sym, w in waiting:
            if free <= 0:
                break
            order = self._place_limit_entry(sym, w, quotes[sym], armed)
            if order is not None:
                pend[sym] = order
                self.store.set_meta("pending_entries", pend)   # reserve before sizing the next
                free -= 1
        self.store.set_meta("pending_entries", pend)

    def _prune_watchlist_entered(self) -> None:
        """Drop watch entries for symbols that now hold a position."""
        wl: dict = self.store.get_meta("watchlist") or {}
        held = {o["underlying"] for o in self.store.open_options()}
        held |= {p.symbol for p in self.store.positions()}
        kept = {s: w for s, w in wl.items() if s not in held}
        if len(kept) != len(wl):
            self.store.set_meta("watchlist", kept)

    def _fee_feasible(self, quote: Quote, decision: Decision,
                      genome_id: str | None, econ) -> bool:
        """Decision-time fee check: a signal whose largest allowed order can't
        reach the fee-implied minimum is dropped here, so it never takes a
        slot or reaches execution."""
        pct = self.settings.risk.max_fee_pct
        if self.settings.options.enabled:
            fee = OPTION_FEE
            _, ceiling = self._budget_bounds(quote, decision, genome_id)
        else:
            fee = getattr(self.broker, "fee", 0.0)
            ceiling = min(self.broker.cash(),
                          econ.net_worth_eur * self.settings.risk.max_position_fraction)
        floor = min_notional_for_fee(fee, pct)
        if ceiling >= floor:
            return True
        self.bus.info(
            f"[fee-guard] {decision.symbol}: {decision.direction.upper()} "
            f"conf {decision.confidence:.2f} dropped — largest allowed order "
            f"€{max(ceiling, 0.0):.2f} < €{floor:.2f} minimum (fee {fee:.2f} "
            f"≤ {pct:.1%})", source="engine")
        return False

    def _fee_skip(self, symbol: str, notional: float, fee: float) -> None:
        pct = self.settings.risk.max_fee_pct
        self.bus.info(
            f"[fee-guard] {symbol} skipped: fee {fee:.2f} is "
            f"{fee / max(notional, 1e-9):.1%} of {notional:.2f} (max {pct:.1%})",
            source="engine")

    def _enter_knockout(self, quote: Quote, decision: Decision,
                        genome_id: str | None) -> bool:
        """Open a real-ISIN TR knockout (paper fill). Returns False when no
        TR client / no suitable instrument — caller falls back to options."""
        tr = self.tr_derivatives
        if tr is None or not tr.available():
            return False
        ko = tr.find_knockout(quote.symbol, decision.direction, quote.price,
                              self.settings.tr.target_leverage)
        if ko is None or ko.price <= 0:
            return False
        cfg = self.settings.options
        budget = self._position_budget(quote, decision, genome_id)
        if budget <= 0.05:
            return True   # handled (deliberately no trade), don't fall back
        if not fee_ratio_ok(budget, OPTION_FEE, self.settings.risk.max_fee_pct):
            self._fee_skip(quote.symbol, budget, OPTION_FEE)
            return True
        contracts = budget / ko.price

        # Real execution: when the broker is an ARMED live broker, place a
        # REAL market order for the knockout certificate. Opening a long
        # knockout (call or put variant) is always a BUY of the certificate.
        # Real fills happen at TR; we don't touch local simulated cash (the
        # live view's balances come from the real TR account).
        from .control import ControlState
        
        control = ControlState.load(self.settings.control_path)
        broker_armed = getattr(self.broker, "armed", False)
        has_place_order = hasattr(self.broker, "place_order")
        # The broker is only ever constructed armed when the control plane is
        # live+armed (see cli._build_engine_for_control), so an armed broker IS
        # the routing signal for real execution. The runtime low-balance guard
        # below is re-checked against the live balance on every order.
        armed_real = broker_armed and has_place_order

        if armed_real:
            # Runtime low-balance guard (second arming switch): below
            # LOW_BALANCE_EUR the flat ~1 EUR fee is a >1% drag, so real orders
            # are blocked unless the user has explicitly armed the second guard.
            # Checked live against the REAL account balance every order, so it
            # engages the moment net worth drops under the threshold.
            net_worth = self.broker.cash()
            if control.low_balance_blocks(net_worth):
                self.bus.warn(
                    f"LIVE order blocked [{ko.isin}]: balance "
                    f"{'unknown' if net_worth is None else f'€{net_worth:.2f}'} "
                    f"is under €{LOW_BALANCE_EUR:.0f} and the low-balance guard "
                    f"is not armed.", source="engine")
                return True
            size = int(contracts)   # whole certificates (sellFractions off)
            if size < 1:
                self.bus.warn(
                    f"LIVE knockout {ko.isin}: budget too small for one "
                    f"certificate at {ko.price:.2f} — skipping.", source="engine")
                return True
            res = self.broker.place_order(ko.isin, "buy", float(size))
            if not res.ok:
                self.bus.warn(f"LIVE knockout order rejected [{ko.isin}]: "
                              f"{res.message}", source="engine")
                return True   # never fall back to synthetic once live-armed
            # Real fill: cash is the REAL TR account (no local ledger to debit).
            # The live view's balances come from TR; we only record the fee for
            # cost accounting and then book the position locally.
            contracts = float(size)
            self.store.record_cost("fee", OPTION_FEE, "options")
        else:
            cost = contracts * ko.price + OPTION_FEE
            if not self.broker.adjust_cash(-cost):
                return True
            self.store.record_cost("fee", OPTION_FEE, "options")
        tp_premium = ko.price * (1 + cfg.take_profit_pct)
        sl_premium = ko.price * (1 - cfg.stop_loss_pct)
        # KOs are open-ended: expiry far out; the barrier is the real risk.
        expiry_ts = self.now() + 365 * 86400.0
        self.store.open_option(
            quote.symbol, ko.kind, ko.strike, expiry_ts, 0.0, contracts,
            ko.price, genome_id, tp_premium=tp_premium, sl_premium=sl_premium,
            instrument_type="knockout", barrier=ko.barrier, ratio=ko.ratio,
            isin=ko.isin)
        self.store.record_trade(Trade(
            quote.symbol, "buy", contracts, ko.price, OPTION_FEE,
            self.broker.mode,
            f"open {ko.kind} {ko.isin or 'synthetic'} K={ko.strike} "
            f"barrier={ko.barrier} {ko.leverage:.1f}x — {decision.rationale}",
            decision.confidence))
        self.bus.activity(
            "trade",
            f"OPEN {ko.kind.upper()} {quote.symbol} [{ko.isin or 'synthetic'}] "
            f"K={ko.strike} barrier={ko.barrier} {ko.leverage:.1f}x "
            f"×{contracts:.3f} @ {ko.price:.3f}",
            quote.symbol, {"genome": genome_id, "isin": ko.isin,
                           "leverage": ko.leverage})
        return True

    def _enter_option(self, quote: Quote, decision: Decision, genome_id: str | None) -> None:
        cfg = self.settings.options
        kind = "call" if decision.direction == "buy" else "put"
        oq = synth_option(quote.symbol, quote.price, quote.history, kind,
                          expiry_days=cfg.expiry_days)
        budget = self._position_budget(quote, decision, genome_id)
        if budget <= 0.05:
            return
        if not fee_ratio_ok(budget, OPTION_FEE, self.settings.risk.max_fee_pct):
            self._fee_skip(quote.symbol, budget, OPTION_FEE)
            return
        contracts = budget / oq.premium
        cost = contracts * oq.premium + OPTION_FEE
        if not self.broker.adjust_cash(-cost):
            return
        self.store.record_cost("fee", OPTION_FEE, "options")
        # Explicit stops placed with the order: every position always carries
        # its own TP/SL, immune to later config changes.
        tp_premium = oq.premium * (1 + cfg.take_profit_pct)
        sl_premium = oq.premium * (1 - cfg.stop_loss_pct)
        self.store.open_option(quote.symbol, kind, oq.strike, oq.expiry_ts,
                               oq.iv, contracts, oq.premium, genome_id,
                               tp_premium=tp_premium, sl_premium=sl_premium)
        self.store.record_trade(Trade(
            quote.symbol, "buy", contracts, oq.premium, OPTION_FEE,
            self.broker.mode, f"open {kind} K={oq.strike} — {decision.rationale}",
            decision.confidence))
        self.bus.activity(
            "trade",
            f"OPEN {kind.upper()} {quote.symbol} K={oq.strike} ×{contracts:.3f} "
            f"@ {oq.premium:.3f} (IV {oq.iv:.0%})",
            quote.symbol, {"genome": genome_id, "delta": round(oq.delta, 3)})

    def _enter_equity(self, quote: Quote, decision: Decision, econ) -> None:
        pos = self.store.position(quote.symbol)
        if decision.direction == "buy" and not pos:
            sizing = size_position(
                price=quote.price, cash=self.broker.cash(),
                equity=econ.net_worth_eur, confidence=decision.confidence,
                cfg=self.settings.risk,
                fee=getattr(self.broker, "fee", 0.0))
            if sizing.qty <= 0:
                if "fee" in sizing.reason:
                    self.bus.info(f"[fee-guard] {quote.symbol} skipped: {sizing.reason}",
                                  source="engine")
                return
            res = self.broker.buy(quote.symbol, sizing.qty, quote.price)
            if res.ok:
                self.store.record_trade(Trade(
                    quote.symbol, "buy", res.qty, res.price, res.fee,
                    self.broker.mode, decision.rationale, decision.confidence))
                self.bus.activity("trade",
                                  f"BUY {quote.symbol} ×{res.qty:.4f} @ {res.price:.2f}",
                                  quote.symbol)
        elif decision.direction == "sell" and pos:
            res = self.broker.sell(quote.symbol, pos.qty, quote.price)
            if res.ok:
                self.store.record_trade(Trade(
                    quote.symbol, "sell", res.qty, res.price, res.fee,
                    self.broker.mode, decision.rationale, decision.confidence))
                self.bus.activity("trade", f"SELL {quote.symbol} — signal", quote.symbol)

    # ------------------------------------------------------------------- cycle
    def run_cycle(self) -> None:
        self.process_close_requests()
        self._run_scheduled_jobs()

        symbols = list(dict.fromkeys(
            self.settings.universe + [self.settings.benchmark.symbol]))
        # Concurrent, bounded fetch — a large universe fetched sequentially
        # would make cycle time grow linearly with universe size. The cap
        # also keeps request bursts against Yahoo modest (see
        # data/market.py: this is what previously triggered rate limiting).
        if hasattr(self.market, "quotes_concurrent"):
            quotes: dict[str, Quote] = self.market.quotes_concurrent(
                symbols, max_workers=min(8, len(symbols)))
        else:
            quotes = {s: self.market.quote(s) for s in symbols}
        # Cycle-wide histories for cross-sectional strategies (see _decide).
        self._cycle_histories = {s: q.history for s, q in quotes.items()}
        prices: dict[str, float] = {}     # valuation prices: fresh, or last-known-good
        tradeable: set[str] = set()        # symbols with a FRESH trustworthy quote
        degraded: list[str] = []
        for symbol, q in quotes.items():
            if self._is_trustworthy(q):
                prices[symbol] = q.price
                self._last_good_price[symbol] = q.price
                tradeable.add(symbol)
            else:
                degraded.append(symbol)
                if symbol in self._last_good_price:
                    prices[symbol] = self._last_good_price[symbol]
        if degraded:
            self.bus.warn(
                f"Live data unavailable this cycle for {degraded} (fell back to "
                "synthetic) — using last known price for valuation only; no "
                "new trades or exits on these symbols this cycle.",
                source="data")

        # exits always run (even when halted), but only for symbols with a
        # fresh trustworthy quote — never mark/close against a stale or
        # mismatched-source price.
        self._manage_options(prices, tradeable)
        for symbol in self.settings.universe:
            if symbol not in tradeable:
                continue
            pos = self.store.position(symbol)
            if pos:
                exit_now, why = should_exit(avg_price=pos.avg_price,
                                            last_price=prices[symbol],
                                            cfg=self.settings.risk)
                if exit_now:
                    res = self.broker.sell(symbol, pos.qty, prices[symbol])
                    if res.ok:
                        self.store.record_trade(Trade(
                            symbol, "sell", res.qty, res.price, res.fee,
                            self.broker.mode, why, None))
                        self.bus.activity("trade", f"SELL {symbol} — {why}", symbol)

        cash = self.broker.cash()
        total_value = self._positions_value(prices) + self._options_value(prices)
        self._persist_option_marks(prices)
        # Live TR account cash for the dashboard, refreshed at most every few
        # minutes rather than every cycle — each fetch opens a websocket, so
        # doing it per-cycle hammers TR (and multiplies any 401). None-safe:
        # no-op without a real authenticated TR client.
        if self.tr_derivatives is not None and self.scheduler.due("tr_cash", 300):
            tr_cash = self.tr_derivatives.account_cash()
            if tr_cash is not None:
                self.store.set_meta("tr_account_cash", tr_cash)
                # First real balance we ever see becomes the live P&L baseline
                # (positions are ~empty at that point), so live P&L reflects the
                # change since going live rather than a meaningless delta vs the
                # paper starting budget.
                if self.store.get_meta("tr_baseline_net_worth") is None:
                    self.store.set_meta("tr_baseline_net_worth", tr_cash)
        econ = self.accountant.snapshot(cash, total_value, self.store.reserve_balance())
        self.store.set_meta("economics", econ.as_dict())
        self._update_realized_mark(econ.net_worth_eur)
        self._update_benchmark(prices)
        self.bus.activity(
            "economics",
            f"net €{econ.net_worth_eur:.2f} | runway {econ.runway_hours:.1f}h | "
            f"alpha {self.store.get_meta('alpha', 0):+0.3f} | "
            f"{'SELF-SUSTAINING' if econ.self_sustaining else 'subsidised'}",
            detail=econ.as_dict())

        if econ.sanity_breached:
            self.bus.error(
                f"SANITY BREACH: net worth €{econ.net_worth_eur:.2f} exceeds "
                f"{self.settings.economics.sanity_max_multiple}x starting budget — "
                "trading halted unconditionally. This indicates a valuation bug, "
                "not a real gain. Investigate before resuming.",
                source="economics")
        elif econ.halt_trading:
            self.bus.warn(
                f"Runway {econ.runway_hours:.1f}h < floor — exits only.",
                source="economics")
        else:
            open_count = len(self.store.open_options()) + len(self.store.positions())
            slots = self.settings.loop.max_positions - open_count
            if slots <= 0:
                # Book full — the decision/entry step is skipped this cycle.
                # Say so, otherwise the run looks stuck (only economics lines)
                # when it's actually just holding a full position book.
                self.bus.info(
                    f"position book full ({open_count}/{self.settings.loop.max_positions}) "
                    f"— no free slots, holding existing positions this cycle",
                    source="engine")
            candidates: list[tuple[Decision, str | None]] = []
            evaluated: set[str] = set()   # symbols whose signal was re-read this cycle
            for symbol in self.settings.universe:
                self._check_exits_fast()      # keep exits responsive mid-cycle
                if slots <= 0:
                    break
                if symbol not in tradeable:
                    continue
                if self.store.position(symbol) or any(
                        o["underlying"] == symbol for o in self.store.open_options()):
                    continue
                if not self.accountant.can_afford_inference(econ, est_usd=0.01):
                    break
                self._poll_close_requests()   # manual closes stay snappy mid-cycle
                decision, genome_id = self._decide(quotes[symbol])
                evaluated.add(symbol)
                self.bus.activity(
                    "decision",
                    f"{symbol}: {decision.direction.upper()} conf {decision.confidence:.2f}",
                    symbol, decision.as_dict())
                # Storm regime: demand extra conviction — measured edges are
                # the first casualty when the volatility regime flips.
                min_conf = self.settings.risk.min_confidence
                if (self.settings.sizing.regime_filter_enabled
                        and regime(quotes[symbol].history) == "storm"):
                    min_conf += self.settings.sizing.storm_extra_confidence
                if decision.direction == "hold" or decision.confidence < min_conf:
                    continue
                if not self._fee_feasible(quotes[symbol], decision, genome_id, econ):
                    continue
                candidates.append((decision, genome_id))

            if self.settings.entry.watch_enabled:
                candidates = self._watchlist_triggers(candidates, evaluated, quotes)
                if self.settings.entry.limit_orders:
                    # Exchange-side entries: resting limit orders instead of
                    # market orders on trigger (TP / SL exits stay market).
                    self._manage_limit_entries(quotes, tradeable, econ)
                    candidates = []

            # Rank by confidence so limited slots go to the strongest signals
            # across the whole universe, not just whichever symbols happened
            # to come first in the list.
            candidates.sort(key=lambda c: c[0].confidence, reverse=True)
            take = max(0, slots)
            cap = self.settings.loop.max_new_positions_per_cycle
            if cap > 0:
                take = min(take, cap)
            for decision, genome_id in candidates[:take]:
                quote = quotes[decision.symbol]
                # Re-check against CURRENT capital: earlier entries this cycle
                # may have used up the deployable bucket since selection.
                if not self._fee_feasible(quote, decision, genome_id, econ):
                    continue
                if self.settings.options.enabled:
                    # Prefer real TR knockout instruments when the (optional)
                    # TR client is authenticated; fall back to synthetic
                    # Black-Scholes options otherwise.
                    self.bus.info(
                        f"[execution] entering {decision.symbol} {decision.direction} "
                        f"in {self.broker.mode} mode (broker.armed={getattr(self.broker, 'armed', False)})",
                        source="engine")
                    if not self._enter_knockout(quote, decision, genome_id):
                        # In ARMED live mode a synthetic option is a PHANTOM: it
                        # books a local position that was NEVER sent to TR, so
                        # the dashboard shows "trades" the TR app doesn't have.
                        # Only simulate (synthetic options) in paper/unarmed
                        # mode; in armed live, skip when no real instrument is
                        # tradeable.
                        if getattr(self.broker, "armed", False):
                            self.bus.warn(
                                f"LIVE {decision.symbol}: no real TR knockout "
                                f"tradeable this cycle — skipping (no synthetic "
                                f"fallback in armed live mode).", source="engine")
                        else:
                            self._enter_option(quote, decision, genome_id)
                else:
                    self._enter_equity(quote, decision, econ)
            if self.settings.entry.watch_enabled:
                self._prune_watchlist_entered()

        cash = self.broker.cash()
        equity = cash + self._positions_value(prices) + self._options_value(prices)
        fees = self.store.total_costs().get("fee", 0.0)
        self.store.record_equity(cash, equity, fees)

    # -------------------------------------------------------------- long loop
    def stop(self) -> None:
        self._stop = True

    def run_forever(self, max_cycles: int | None = None,
                    rebuild_when=None) -> None:
        """Run cycles until stopped or max_cycles. `rebuild_when()` is checked
        after each cycle; when it returns True the loop returns so the caller
        can rebuild the engine for a new book/broker (the paper/live toggle
        or arm/disarm changed). Returning — not raising — keeps `finally`
        cleanup in the CLI simple."""
        self.bus.info(
            f"Engine start | mode={self.broker.mode} | universe={self.settings.universe} | "
            f"interval={self.settings.loop.interval_seconds}s | "
            f"options={'on' if self.settings.options.enabled else 'off'} | "
            f"learning={'on' if self.optimizer else 'off'}")
        n = 0
        while not self._stop:
            started = time.time()
            try:
                self.run_cycle()
            except Exception as exc:  # noqa: BLE001 — loop must survive one bad cycle
                self.bus.error(f"cycle error: {exc}")
            n += 1
            if max_cycles and n >= max_cycles:
                self.bus.info(f"Reached max_cycles={max_cycles}, stopping.")
                break
            if rebuild_when is not None and rebuild_when():
                self.bus.info("Control changed (paper/live/armed) — reconfiguring.")
                break
            elapsed = time.time() - started
            self._idle(self.settings.loop.interval_seconds - elapsed)
