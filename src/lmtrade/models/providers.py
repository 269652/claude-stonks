"""Concrete model providers.

- HeuristicProvider: pure financial-model signal (no external calls, zero cost).
- LocalSLMProvider:  small model via Ollama (local or on a GPU host).
- CloudLLMProvider:  hosted LLM (Anthropic) for harder judgment calls.
- PerplexityProvider: web-grounded research/news sentiment.

The SLM/cloud/perplexity providers degrade gracefully to a neutral `hold`
signal when their key/host is missing, so the stack never blocks the loop.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
from typing import Callable

import httpx

from ..config import Settings, secret
from ..logging_setup import get_logger
from .base import ModelProvider, Signal

log = get_logger("lmtrade.models")

CLAUDE_CLI_TIMEOUT = 180.0   # web-search research calls routinely exceed 60s
CLAUDE_CLI_BIN = "claude"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Keywords in Claude CLI stderr that indicate a subscription/token limit rather
# than a transient error.  Checked case-insensitively.
_LIMIT_KEYWORDS = ("usage limit", "rate limit", "overloaded", "exceeded", "quota",
                   "credit", "billing", "subscription", "too many requests")


class ProviderLimitError(RuntimeError):
    """Raised when a provider reports a usage/token limit rather than a
    transient error. NewsService catches this and stores a persistent warning
    in the Store so the dashboard can surface a notification banner."""


def _parse_decision(text: str, provider: str, cost: float) -> Signal:
    """Coerce a model's free-text answer into a Signal. Accepts JSON or falls
    back to keyword scanning."""
    m = _JSON_RE.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            direction = str(obj.get("direction", "hold")).lower()
            if direction not in ("buy", "sell", "hold"):
                direction = "hold"
            conf = float(obj.get("confidence", 0.5))
            return Signal(provider, direction, max(0.0, min(1.0, conf)),
                          str(obj.get("rationale", ""))[:400], cost)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    low = (text or "").lower()
    if "buy" in low and "sell" not in low:
        return Signal(provider, "buy", 0.6, text[:200], cost)
    if "sell" in low and "buy" not in low:
        return Signal(provider, "sell", 0.6, text[:200], cost)
    return Signal(provider, "hold", 0.5, text[:200], cost)


def _prompt(symbol: str, context: dict) -> str:
    ind = context.get("indicators", {})
    news = context.get("research", "")
    votes = context.get("votes") or []
    vote_lines = "\n".join(f"- {p}: {d} {c:.2f} ({r[:120]})" for p, d, c, r in votes) or "- none"
    held = context.get("held")
    held_line = (f"Currently held: {held} position (a strong opposite call closes it).\n"
                 if held else "Not currently held.\n")
    return (
        "You are a disciplined trading analyst for a small account that trades "
        "leveraged knock-out certificates (flat 1 EUR fee per order).\n"
        f"Instrument: {symbol}\n"
        f"{held_line}"
        f"Indicators (intraday): {json.dumps(ind, default=str)}\n"
        f"Other signals (daily evidence rules, strategy, news, daily analysis):\n{vote_lines}\n"
        f"News: {news[:800] or 'none'}\n\n"
        "Weigh these and decide the action for the NEXT step. Use confidence 1.0 only "
        "when the case is overwhelming. Respond ONLY with JSON: "
        '{"direction":"buy|sell|hold","confidence":0..1,"rationale":"one sentence"}'
    )


class HeuristicProvider(ModelProvider):
    """Deterministic financial-model vote from the indicator snapshot."""

    name = "heuristic"

    def analyze(self, symbol: str, context: dict) -> Signal:
        ind = context.get("indicators", {})
        last = ind.get("last")
        sma_fast, sma_slow, rsi = ind.get("sma_fast"), ind.get("sma_slow"), ind.get("rsi")
        if last is None or sma_fast is None or sma_slow is None:
            return Signal(self.name, "hold", 0.5, "insufficient history", 0.0)

        score = 0.0
        reasons = []
        if sma_fast > sma_slow:
            score += 0.4
            reasons.append("fast SMA above slow (uptrend)")
        else:
            score -= 0.4
            reasons.append("fast SMA below slow (downtrend)")
        if rsi is not None:
            if rsi < 30:
                score += 0.3
                reasons.append(f"RSI {rsi:.0f} oversold")
            elif rsi > 70:
                score -= 0.3
                reasons.append(f"RSI {rsi:.0f} overbought")

        # Multi-timeframe trend confirmation (time-series momentum:
        # Moskowitz-Ooi-Pedersen 2012; Faber 2007): halve conviction when the
        # short-term signal fights the medium-term trend. A haircut, not a
        # veto — a strong mean-reversion setup can still act, but marginal
        # counter-trend trades (the ones whose edge can't clear the flat
        # ~1 EUR fee) are suppressed. Deliberately no boost when aligned:
        # asymmetric dampening is robust, symmetric boosting overfits.
        mtrend = ind.get("trend")
        if mtrend == "down" and score > 0:
            score *= 0.5
            reasons.append("counter-trend (medium trend down): conviction halved")
        elif mtrend == "up" and score < 0:
            score *= 0.5
            reasons.append("counter-trend (medium trend up): conviction halved")

        direction = "buy" if score > 0.15 else "sell" if score < -0.15 else "hold"
        return Signal(self.name, direction, min(1.0, abs(score) + 0.4),
                      "; ".join(reasons), 0.0)


class LocalSLMProvider(ModelProvider):
    """Small language model served by Ollama (locally or on a GPU host)."""

    name = "slm"

    def __init__(self, settings: Settings):
        self.host = secret("OLLAMA_HOST") or "http://localhost:11434"
        self.model = settings.model.slm_model
        self.cost = settings.economics.inference_cost.get("slm", 0.0002)

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def analyze(self, symbol: str, context: dict) -> Signal:
        try:
            r = httpx.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": _prompt(symbol, context), "stream": False},
                timeout=30.0,
            )
            r.raise_for_status()
            text = r.json().get("response", "")
            return _parse_decision(text, self.name, self.cost)
        except Exception as exc:  # noqa: BLE001
            return Signal(self.name, "hold", 0.5, f"slm unavailable: {exc}", 0.0)


class CloudLLMProvider(ModelProvider):
    """Hosted Anthropic model for higher-quality judgment on close calls."""

    name = "cloud"

    def __init__(self, settings: Settings):
        self.key = secret("ANTHROPIC_API_KEY")
        self.model = settings.model.cloud_model
        self.cost = settings.economics.inference_cost.get("cloud", 0.004)

    def available(self) -> bool:
        return bool(self.key)

    def analyze(self, symbol: str, context: dict) -> Signal:
        if not self.key:
            return Signal(self.name, "hold", 0.5, "no ANTHROPIC_API_KEY", 0.0)
        try:
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": _prompt(symbol, context)}],
                },
                timeout=30.0,
            )
            r.raise_for_status()
            blocks = r.json().get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return _parse_decision(text, self.name, self.cost)
        except Exception as exc:  # noqa: BLE001
            return Signal(self.name, "hold", 0.5, f"cloud error: {exc}", 0.0)


def _default_claude_cli_runner(prompt: str, timeout: float = CLAUDE_CLI_TIMEOUT,
                               model: str | None = None) -> str:
    """Run a prompt through the locally-installed Claude Code CLI in
    non-interactive print mode and return its stdout.

    Two things this gets right that the naive `subprocess.run(["claude", "-p",
    prompt, ...])` did not, both of which broke it on Windows (where it failed
    instantly and the empty/error output was silently stored as "neutral"
    news):

    - The prompt goes in via **stdin**, never as an argv element. It contains
      shell metacharacters — the sentiment scale is literally
      `bullish|bearish|neutral` — which a shell would mangle. `claude -p`
      reads the prompt from stdin when none is given positionally.
    - On Windows the `claude` on PATH is a `.cmd` shim, which CreateProcess
      cannot execute directly; it must be run through `cmd /c`. We resolve the
      real path with shutil.which (honors PATHEXT) and wrap it.

    WebSearch is pre-authorized because headless mode can't answer interactive
    permission prompts, so research/analysis prompts get live web results, not
    just the model's static training-cutoff knowledge."""
    exe = shutil.which(CLAUDE_CLI_BIN) or CLAUDE_CLI_BIN
    args = ["-p", "--output-format", "text", "--allowedTools", "WebSearch"]
    if model:
        args += ["--model", model]
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", exe, *args]
    else:
        cmd = [exe, *args]
    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or f"claude CLI exited {result.returncode}"
        low = stderr.lower()
        if any(kw in low for kw in _LIMIT_KEYWORDS):
            raise ProviderLimitError(stderr)
        raise RuntimeError(stderr)
    return result.stdout


class ClaudeCLIProvider(ModelProvider):
    """Uses a locally-installed `claude` CLI (Claude Code) instead of the
    Anthropic HTTP API: no ANTHROPIC_API_KEY, billed through the user's own
    Claude subscription/login, and everything runs on the local machine.
    Serves both as a model-stack decision provider (`analyze`) and as a
    research source (`research`) for the news/daily-analysis services."""

    name = "claude_cli"

    def __init__(self, settings: Settings | None = None,
                 runner: Callable[[str, float], str] | None = None,
                 model: str | None = None):
        self.timeout = (settings.research.claude_cli_timeout_seconds
                        if settings is not None else CLAUDE_CLI_TIMEOUT)
        self.model = model
        self._runner = runner or (
            functools.partial(_default_claude_cli_runner, model=model)
            if model else _default_claude_cli_runner)

    def available(self) -> bool:
        return shutil.which(CLAUDE_CLI_BIN) is not None

    def ask(self, prompt: str) -> tuple[str, float]:
        """Run a raw prompt through the CLI. Returns (text, cost=0.0) — no
        per-token API cost is billed here."""
        if not self.available():
            return "", 0.0
        try:
            return self._runner(prompt, self.timeout), 0.0
        except Exception as exc:  # noqa: BLE001
            return f"(claude cli error: {exc})", 0.0

    def analyze(self, symbol: str, context: dict) -> Signal:
        if not self.available():
            return Signal(self.name, "hold", 0.5, "claude CLI not found on PATH", 0.0)
        text, _ = self.ask(_prompt(symbol, context))
        return _parse_decision(text, self.name, 0.0)

    def research(self, symbol: str) -> tuple[str, float]:
        if not self.available():
            return "", 0.0
        prompt = (
            f"Use web search to find today's actual market-moving news for "
            f"{symbol} — do not rely on prior/training knowledge, the market "
            f"has moved since then. In 3 sentences: summarize what you find "
            f"and its sentiment. End with SENTIMENT: bullish|bearish|neutral."
        )
        try:
            text = self._runner(prompt, self.timeout)
        except ProviderLimitError:
            # Re-raise so NewsService can persist a persistent warning banner
            # in the store — the caller must distinguish "limit hit" from
            # "transient error" to avoid crying wolf on every network hiccup.
            raise
        except Exception as exc:  # noqa: BLE001
            # Transient errors (network, timeout, interrupted) are logged and
            # discarded — the symbol just gets no news this cycle.
            log.warning("claude CLI research failed for %s: %s", symbol, exc)
            return "", 0.0
        # A valid response MUST contain the SENTIMENT marker the prompt demands.
        if "sentiment" not in (text or "").lower():
            log.warning("claude CLI research for %s returned no SENTIMENT marker "
                        "(likely interrupted/error output) — skipping.", symbol)
            return "", 0.0
        return text, 0.0


class PerplexityProvider(ModelProvider):
    """Web-grounded research: fetches recent news/sentiment for the instrument."""

    name = "perplexity"

    def __init__(self, settings: Settings):
        self.key = secret("PERPLEXITY_API_KEY")
        self.model = settings.model.perplexity_model
        self.cost = settings.economics.inference_cost.get("perplexity", 0.005)

    def available(self) -> bool:
        return bool(self.key)

    def research(self, symbol: str) -> tuple[str, float]:
        """Return (research_text, cost). Empty text if unavailable."""
        if not self.key:
            return "", 0.0
        try:
            r = httpx.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.key}",
                         "content-type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"In 3 sentences: latest market-moving news and sentiment for "
                            f"{symbol}. End with SENTIMENT: bullish|bearish|neutral."
                        ),
                    }],
                },
                timeout=30.0,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            return text, self.cost
        except Exception as exc:  # noqa: BLE001
            return f"(research unavailable: {exc})", 0.0

    def analyze(self, symbol: str, context: dict) -> Signal:
        text = context.get("research", "")
        low = text.lower()
        if "bullish" in low:
            return Signal(self.name, "buy", 0.65, text[:300], 0.0)
        if "bearish" in low:
            return Signal(self.name, "sell", 0.65, text[:300], 0.0)
        return Signal(self.name, "hold", 0.5, text[:300] or "no research", 0.0)


def build_providers(settings: Settings) -> dict[str, ModelProvider]:
    """Instantiate the providers named in the configured model stack."""
    registry = {
        "heuristic": lambda: HeuristicProvider(),
        "slm": lambda: LocalSLMProvider(settings),
        "cloud": lambda: CloudLLMProvider(settings),
        "perplexity": lambda: PerplexityProvider(settings),
        "claude_cli": lambda: ClaudeCLIProvider(settings, model=settings.model.decision_model),
    }
    out: dict[str, ModelProvider] = {}
    for name in settings.model.stack:
        if name in registry:
            out[name] = registry[name]()
    return out
