"""The self-sustaining loop.

If the bot runs on a rented GPU, that costs money by the hour. To "pay for its own GPU costs" it
must earn more than it burns. This module:

  1. Accrues the GPU rental cost continuously (USD/hr).
  2. Tracks inference + trading fees as they happen (via the Store).
  3. Computes runway: how many more hours the current net worth can fund the GPU.
  4. Exposes a guardrail the engine checks BEFORE spending on inference or
     placing trades — if runway drops below the floor, non-essential inference
     and new positions are halted so the account isn't bled dry by its own
     compute bill.

Net worth and P&L are in the account currency (EUR); GPU/inference costs are in
USD. A single FX rate keeps the comparison honest without a market data call.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..config import Settings
from ..core.state import Store

EUR_USD = 1.08  # coarse fixed rate; refine via data layer if desired


@dataclass
class EconomicsSnapshot:
    net_worth_eur: float
    net_worth_usd: float
    gpu_cost_accrued_usd: float
    inference_cost_usd: float
    fees_eur: float
    gpu_usd_per_hour: float
    runway_hours: float
    self_sustaining: bool
    halt_trading: bool
    pnl_eur: float
    sanity_breached: bool = False
    reserve_eur: float = 0.0

    def as_dict(self) -> dict:
        # gpu_usd_per_hour=0 (no GPU rented) makes runway_hours literally
        # float('inf') — not valid JSON (Starlette's JSONResponse uses
        # allow_nan=False per RFC 8259), so it's represented as None
        # ("unlimited") here. The dataclass field itself stays the real
        # float — CLI status formatting (f"{runway:.1f} h") handles inf fine.
        runway = None if math.isinf(self.runway_hours) else round(self.runway_hours, 2)
        return {
            "net_worth_eur": round(self.net_worth_eur, 4),
            "net_worth_usd": round(self.net_worth_usd, 4),
            "gpu_cost_accrued_usd": round(self.gpu_cost_accrued_usd, 4),
            "inference_cost_usd": round(self.inference_cost_usd, 6),
            "fees_eur": round(self.fees_eur, 4),
            "gpu_usd_per_hour": self.gpu_usd_per_hour,
            "runway_hours": runway,
            "self_sustaining": self.self_sustaining,
            "halt_trading": self.halt_trading,
            "pnl_eur": round(self.pnl_eur, 4),
            "sanity_breached": self.sanity_breached,
            "reserve_eur": round(self.reserve_eur, 4),
        }


class CostAccountant:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store
        self.gpu_rate = settings.economics.gpu_usd_per_hour
        self.min_runway = settings.economics.min_runway_hours
        if store.get_meta("gpu_started_ts") is None:
            store.set_meta("gpu_started_ts", time.time())

    # -- GPU accrual ----------------------------------------------------------
    def gpu_hours_elapsed(self) -> float:
        started = float(self.store.get_meta("gpu_started_ts", time.time()))
        return max(0.0, (time.time() - started) / 3600.0)

    def gpu_cost_accrued_usd(self) -> float:
        return self.gpu_hours_elapsed() * self.gpu_rate

    def record_inference(self, provider: str, usd: float) -> None:
        if usd > 0:
            self.store.record_cost("inference", usd, provider)

    # -- valuation ------------------------------------------------------------
    def net_worth_eur(
        self, cash_eur: float, positions_value_eur: float, reserve_eur: float = 0.0
    ) -> float:
        # Stashed profit is still the account's money — net worth, P&L and
        # alpha all count it — it's just excluded from tradeable cash so it
        # can't be re-risked. See economics.profit_stash_pct.
        return cash_eur + positions_value_eur + reserve_eur

    def snapshot(
        self, cash_eur: float, positions_value_eur: float, reserve_eur: float = 0.0
    ) -> EconomicsSnapshot:
        costs = self.store.total_costs()
        inference_usd = costs.get("inference", 0.0)
        fees_eur = costs.get("fee", 0.0)
        gpu_usd = self.gpu_cost_accrued_usd()

        net_eur = self.net_worth_eur(cash_eur, positions_value_eur, reserve_eur)
        net_usd = net_eur * EUR_USD
        starting = float(self.store.get_meta("starting_cash", self.settings.budget))
        pnl_eur = net_eur - starting

        # Runway = how many more hours the *spare* net worth funds the GPU,
        # after setting aside what's already been spent on compute.
        spare_usd = net_usd - gpu_usd - inference_usd
        runway = spare_usd / self.gpu_rate if self.gpu_rate > 0 else float("inf")

        # Self-sustaining means realized+unrealized gains cover all compute spend.
        total_compute_usd = gpu_usd + inference_usd
        self_sustaining = pnl_eur * EUR_USD >= total_compute_usd

        # Circuit breaker: an implausible net-worth multiple means something is
        # corrupting valuation (a bad mark-to-market, a mispriced fill, a
        # source mismatch) — halt unconditionally rather than let it compound.
        sanity_ceiling = starting * self.settings.economics.sanity_max_multiple
        sanity_breached = net_eur > sanity_ceiling or net_eur < 0

        halt = (runway < self.min_runway) or sanity_breached
        return EconomicsSnapshot(
            net_worth_eur=net_eur,
            net_worth_usd=net_usd,
            gpu_cost_accrued_usd=gpu_usd,
            inference_cost_usd=inference_usd,
            fees_eur=fees_eur,
            gpu_usd_per_hour=self.gpu_rate,
            runway_hours=runway,
            self_sustaining=self_sustaining,
            halt_trading=halt,
            pnl_eur=pnl_eur,
            sanity_breached=sanity_breached,
            reserve_eur=reserve_eur,
        )

    def can_afford_inference(self, snapshot: EconomicsSnapshot, est_usd: float) -> bool:
        """Refuse an inference spend that would by itself push runway under the
        floor — don't let the bot think itself into bankruptcy."""
        if self.gpu_rate <= 0:
            return True
        projected_runway = (snapshot.net_worth_usd - snapshot.gpu_cost_accrued_usd
                            - snapshot.inference_cost_usd - est_usd) / self.gpu_rate
        return projected_runway >= self.min_runway
