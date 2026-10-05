"""Claude sign-off for automatic LIVE orders.

Before the engine sends a real order to TR it asks a reviewer model
(signoff.model, Opus by default) whether the order makes sense given why the
bot wants it, the position and the latest price bars. Reject -> the order is
not sent. If the reviewer can't answer (CLI missing, timeout, unparseable
reply) entries are blocked, exits go through: no new risk without a review,
but never trap a position behind an outage. Manual dashboard closes skip
this (the user signed them off)."""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from ..config import Settings


def build_prompt(order: dict) -> str:
    bars = ", ".join(f"{float(x):.2f}" for x in (order.get("bars") or [])[-30:]) or "n/a"
    fields = {k: v for k, v in order.items() if k != "bars"}
    side = str(order.get("side", "")).upper()
    return (
        "You are the final risk check before a trading bot sends a REAL-MONEY order "
        "to Trade Republic (leveraged knock-out certificates, flat 1 EUR fee).\n"
        f"Proposed order: {side} {order.get('size')} x {order.get('symbol')} "
        f"[{order.get('isin')}]\n"
        f"Details: {json.dumps(fields, default=str, ensure_ascii=False)}\n"
        f"Underlying's latest 1-minute closes (oldest -> newest): {bars}\n\n"
        "Reject orders that look bogus: triggered by a one-off price spike or a bad "
        "print that has already reverted, prices inconsistent with the bars, sizes or "
        "values that don't add up, or a reason that contradicts the data. Approve "
        "legitimate orders, including real stop-losses on genuine moves. This is an "
        "order sanity check, not investment advice.\n"
        'Respond ONLY with JSON: {"approve": true|false, "reason": "one short sentence"}')


def parse_reply(text: str | None) -> tuple[bool | None, str]:
    """(approve, reason); approve is None when the reply can't be read."""
    if not text:
        return None, "empty reply"
    m = re.search(r"\{.*?\}", str(text), re.DOTALL)
    if not m:
        return None, "no JSON in reply"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "invalid JSON in reply"
    approve = data.get("approve")
    if not isinstance(approve, bool):
        return None, "reply without approve flag"
    return approve, str(data.get("reason") or "")[:200]


def default_caller(settings: Settings) -> Callable[[str], str] | None:
    from ..models.providers import ClaudeCLIProvider, _default_claude_cli_runner
    if not ClaudeCLIProvider(settings).available():
        return None
    cfg = settings.signoff
    return lambda prompt: _default_claude_cli_runner(prompt, cfg.timeout_seconds,
                                                      model=cfg.model)


class SignOff:
    def __init__(self, settings: Settings, caller: Callable[[str], str] | None = None):
        self.settings = settings
        self._caller = caller if caller is not None else "auto"

    def _get_caller(self) -> Callable[[str], str] | None:
        if self._caller == "auto":
            self._caller = default_caller(self.settings)
        return self._caller

    def review(self, order: dict) -> tuple[bool, str]:
        """(send it?, reason)."""
        if not self.settings.signoff.enabled:
            return True, "sign-off disabled"
        is_exit = order.get("side") == "sell"
        caller = self._get_caller()
        if caller is None:
            approve, why = None, "reviewer unavailable"
        else:
            try:
                approve, why = parse_reply(caller(build_prompt(order)))
            except Exception as exc:  # noqa: BLE001 — any reviewer failure
                approve, why = None, f"reviewer failed: {exc}"
        if approve is None:
            return (True, f"{why} — exit sent unreviewed") if is_exit else \
                (False, f"{why} — entry blocked")
        return approve, why
