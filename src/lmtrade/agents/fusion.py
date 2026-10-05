"""Hybrid decision fusion.

Gathers a Signal from every provider in the stack, weights them per config, and
produces a single Decision. This is where "LLMs + SLMs + financial models" are
actually combined into one action.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..data.market import Quote
from ..finance.indicators import indicator_snapshot
from ..models.base import ModelProvider, Signal
from ..models.providers import PerplexityProvider


@dataclass
class Decision:
    symbol: str
    direction: str                 # buy | sell | hold
    confidence: float              # 0..1
    rationale: str
    signals: list[Signal] = field(default_factory=list)
    inference_cost: float = 0.0    # total USD spent producing this decision

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "inference_cost": round(self.inference_cost, 6),
            "signals": [
                {
                    "provider": s.provider,
                    "direction": s.direction,
                    "confidence": round(s.confidence, 3),
                    "rationale": s.rationale,
                    "cost_usd": round(s.cost_usd, 6),
                }
                for s in self.signals
            ],
        }


class FusionEngine:
    def __init__(self, settings: Settings, providers: dict[str, ModelProvider]):
        self.settings = settings
        self.providers = providers
        self.weights = settings.model.weights
        self._llm_cache: dict[tuple[str, str], tuple[float, Signal]] = {}
        # Background LLM calls (the engine turns this on): a decision never
        # waits for the LLM; its answer counts from the next decision on.
        self.llm_async = False
        self._inflight: dict[tuple[str, str], object] = {}
        import threading
        self._llm_slots = threading.BoundedSemaphore(2)

    def decide(self, quote: Quote, extra_signals: list[Signal] | None = None,
               context_extra: dict | None = None) -> Decision:
        context: dict = {"indicators": indicator_snapshot(quote.history)}
        if context_extra:
            context.update(context_extra)

        # Research first (Perplexity) so its notes feed the SLM/cloud prompts.
        cost = 0.0
        perp = self.providers.get("perplexity")
        if isinstance(perp, PerplexityProvider) and perp.available():
            research, rcost = perp.research(quote.symbol)
            context["research"] = research
            cost += rcost

        signals: list[Signal] = list(extra_signals or [])
        screened = set(self.settings.model.screened_providers)
        # Cheap providers first; their votes form the pre-screen and the
        # context the expensive (LLM) providers see.
        for name, provider in self.providers.items():
            if name in screened:
                continue
            try:
                sig = provider.analyze(quote.symbol, context)
            except Exception as exc:  # noqa: BLE001
                sig = Signal(name, "hold", 0.5, f"error: {exc}", 0.0)
            signals.append(sig)
            cost += sig.cost_usd
        context["votes"] = [(s.provider, s.direction, s.confidence, s.rationale)
                            for s in signals if s.direction != "hold"]
        pre = abs(self._net(signals))
        for name, provider in self.providers.items():
            if name not in screened:
                continue
            sig = self._screened(name, provider, quote.symbol, context, pre)
            if sig is not None:
                signals.append(sig)
                cost += sig.cost_usd

        net = self._net(signals)
        # Weighted vote. A `hold` is an ABSTENTION, not a vote — it must not
        # dilute the conviction of the providers that did take a side, otherwise
        # every unavailable provider (no key/host) would drag the decision toward
        # inaction. Only directional signals contribute to the denominator.
        #
        # Normalize by Σw (weights only), NOT Σ w·confidence: dividing by the
        # confidence-weighted sum cancelled each voter's conviction — a lone
        # directional voter always netted ±1.0 and every decision printed
        # conf 1.00 (observed live), saturating every downstream confidence
        # gate (min_confidence, storm extra-conviction, confidence sizing,
        # slot ranking). With Σw, net is the weighted MEAN signed conviction:
        # a lone 0.6-confident voter nets ±0.6; agreement averages; conflict
        # cancels.
        direction = "buy" if net > 0.1 else "sell" if net < -0.1 else "hold"
        confidence = min(1.0, abs(net))
        agree = [s.provider for s in signals if s.direction == direction]
        rationale = (
            f"net={net:+.2f} via {len(signals)} providers; "
            f"agree: {', '.join(agree) or 'none'}"
        )
        return Decision(quote.symbol, direction, confidence, rationale, signals, cost)

    def _net(self, signals: list[Signal]) -> float:
        """Weighted mean signed conviction of the directional votes (-1..1)."""
        num = den = 0.0
        skip = set(self.settings.model.non_voting)
        for s in signals:
            if s.direction == "hold" or s.provider in skip:
                continue
            w = self.weights.get(s.provider, 1.0)
            num += w * s.signed()
            den += w
        return num / den if den else 0.0

    def _screened(self, name: str, provider: ModelProvider, symbol: str,
                  context: dict, pre: float) -> Signal | None:
        """Expensive provider, rationed: cached answer if fresh; otherwise
        only consulted when the pre-screen is strong enough or the symbol is
        held (exits depend on it)."""
        import time

        ttl = self.settings.model.decision_cache_minutes * 60
        hit = self._llm_cache.get((name, symbol))
        if ttl > 0 and hit and time.time() - hit[0] < ttl:
            return hit[1]
        if pre < self.settings.model.prescreen_confidence and not context.get("held"):
            return None
        if self.llm_async:
            self._ask_in_background(name, provider, symbol, dict(context))
            return hit[1] if hit else None          # last answer (even if stale), or none yet
        try:
            sig = provider.analyze(symbol, context)
        except Exception as exc:  # noqa: BLE001
            return Signal(name, "hold", 0.5, f"error: {exc}", 0.0)
        self._llm_cache[(name, symbol)] = (time.time(), sig)
        return sig

    def _ask_in_background(self, name: str, provider: ModelProvider, symbol: str,
                           context: dict) -> None:
        import threading
        import time

        key = (name, symbol)
        t = self._inflight.get(key)
        if t is not None and t.is_alive():
            return

        def run() -> None:
            with self._llm_slots:                   # at most 2 LLM calls at once
                try:
                    sig = provider.analyze(symbol, context)
                except Exception as exc:  # noqa: BLE001
                    sig = Signal(name, "hold", 0.5, f"error: {exc}", 0.0)
            self._llm_cache[key] = (time.time(), sig)

        t = threading.Thread(target=run, daemon=True, name=f"llm-{symbol}")
        self._inflight[key] = t
        t.start()

    def wait_llm(self, timeout: float = 30.0) -> None:
        """Wait for background LLM calls (tests / shutdown)."""
        for t in list(self._inflight.values()):
            t.join(timeout)
