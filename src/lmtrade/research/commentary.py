"""Live commentary on the open positions (~300 characters, every few minutes).

The engine builds one row per open position (direction, entry, current
price, P&L, exits, time held, underlying move), asks Claude for a short
plain-language read of how they are developing, and stores the cleaned text
for the dashboard. Uses the same provider as the daily analysis
(research.analysis_provider) with its own, cheaper model
(research.commentary_model)."""
from __future__ import annotations

import re
from typing import Callable

import httpx

from ..config import Settings, secret


def _fmt(v, spec: str = "+.2f") -> str:
    return "n/a" if v is None else format(v, spec)


def build_prompt(rows: list[dict], max_chars: int = 300) -> str:
    lines = []
    for r in rows:
        lines.append(
            f"- {r['symbol']} {r['direction']} knock-out/option: entry {_fmt(r.get('entry'), '.3f')}, "
            f"now {_fmt(r.get('price'), '.3f')}, P&L {_fmt(r.get('pnl_eur'))} EUR "
            f"({_fmt(r.get('pnl_pct'), '+.1f')}%), TP {_fmt(r.get('tp'), '.3f')}, "
            f"SL {_fmt(r.get('sl'), '.3f')}, trailing {r.get('trail') or 'off'}, "
            f"held {r.get('held_min', 'n/a')} min, underlying {_fmt(r.get('spot'), '.2f')} "
            f"({_fmt(r.get('spot_move_pct'), '+.2f')}% since entry)")
    return (
        "You are monitoring a small trading account's open positions. Start with exactly "
        "one verdict — Fine, Watch or Going badly — then a colon, then at most "
        f"{max_chars} characters in total of plain text (no markdown, no lists): how the "
        "positions are developing right now and what to watch (closest exit, momentum). "
        "Be factual; do not give financial advice.\n"
        "Field notes: P&L is after fees. 'trailing arms at +X EUR' means the trailing "
        "stop is not active yet — it only activates once profit reaches +X EUR; "
        "'trailing active, stop +Y EUR' means it closes if profit falls to +Y EUR.\n\n"
        "Positions:\n" + "\n".join(lines))


def parse_verdict(text: str | None) -> str | None:
    """fine | watch | bad from the commentary's opening verdict, else None."""
    head = (text or "").strip().lower()[:20]
    if head.startswith("going badly") or head.startswith("bad"):
        return "bad"
    if head.startswith("watch"):
        return "watch"
    if head.startswith("fine"):
        return "fine"
    return None


def clean_commentary(text: str | None, limit: int = 300) -> str | None:
    if not text:
        return None
    t = re.sub(r"\s+", " ", str(text)).strip().strip('"')
    if not t:
        return None
    if len(t) > limit:
        t = t[:limit - 1].rstrip() + "…"
    return t


def make_caller(settings: Settings) -> Callable[[str], str] | None:
    """prompt -> text. Raises on failure (the engine keeps the last text).
    None when the configured provider isn't available."""
    model = settings.research.commentary_model
    if settings.research.analysis_provider == "claude_cli":
        from ..models.providers import ClaudeCLIProvider, _default_claude_cli_runner
        provider = ClaudeCLIProvider(settings)
        if not provider.available():
            return None
        timeout = settings.research.claude_cli_timeout_seconds
        return lambda prompt: _default_claude_cli_runner(prompt, timeout, model=model)
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None

    def call(prompt: str) -> str:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": 200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60.0)
        r.raise_for_status()
        blocks = r.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    return call
