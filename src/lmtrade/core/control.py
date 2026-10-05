"""Runtime control plane for the paper/live toggle and the live-execution
arming guard.

Persisted to a small JSON file so the engine (which executes trades) and the
web dashboard (which flips the toggle) agree even when they run in separate
processes. This module only holds the GATE STATE — it never places an order.
Real execution reads `mode == "live" and armed` before doing anything with
real money; see brokers/trade_republic.py and the engine's execution routing.

Two deliberate frictions, both here so they can't be bypassed by a stray UI
bug:
  1. Arming is only possible in live mode and requires an explicit confirm.
  2. When net worth is below LOW_BALANCE_EUR — where Trade Republic's flat
     ~1 EUR fee is more than 1% of the account, a severe drag — arming needs
     a SECOND ("double") confirmation.
Switching back to paper always disarms.
"""
from __future__ import annotations

import json
from pathlib import Path

# Below this net worth the flat ~1 EUR TR fee exceeds 1% of the account, so
# live execution demands the extra double-confirm.
LOW_BALANCE_EUR = 100.0

_VALID_MODES = ("paper", "live")


class ControlState:
    def __init__(self, path: Path, mode: str = "paper", armed: bool = False,
                 double_armed: bool = False):
        self.path = path
        self.mode = mode if mode in _VALID_MODES else "paper"
        # `mode` is the dashboard VIEW; `armed` alone decides real trading, so
        # it survives switching the view to paper (arming itself still
        # requires the live view, see arm()).
        self.armed = bool(armed)
        # Second guard is meaningless unless the first guard is armed.
        self.double_armed = bool(double_armed) and self.armed

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "ControlState":
        try:
            data = json.loads(Path(path).read_text())
            return cls(path, mode=data.get("mode", "paper"),
                       armed=bool(data.get("armed", False)),
                       double_armed=bool(data.get("double_armed", False)))
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            return cls(path)   # safe default: paper, disarmed

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "mode": self.mode, "armed": self.armed,
            "double_armed": self.double_armed}))

    def as_dict(self) -> dict:
        return {"mode": self.mode, "armed": self.armed,
                "double_armed": self.double_armed}

    # -- transitions ---------------------------------------------------------
    def set_mode(self, mode: str, disarm: bool = False) -> None:
        """Switch the dashboard view. Live trading keeps running unless
        `disarm` (the dashboard asks when switching to paper)."""
        if mode not in _VALID_MODES:
            raise ValueError(f"invalid mode {mode!r}; expected one of {_VALID_MODES}")
        self.mode = mode
        if disarm:
            self.armed = False
            self.double_armed = False
        self._save()

    def is_low_balance(self, net_worth: float | None) -> bool:
        """Below the fee-drag threshold. An UNKNOWN balance is treated as low —
        the guard fails safe (blocks) rather than executing on a balance it
        couldn't verify."""
        return net_worth is None or net_worth < LOW_BALANCE_EUR

    def arm(self, confirm: bool) -> bool:
        """Arm the first guard (live execution). Requires live mode + confirm.
        Below LOW_BALANCE_EUR this alone does NOT enable execution — the second
        guard (set_double_armed) is enforced separately at execution time."""
        if self.mode != "live" or not confirm:
            return False
        self.armed = True
        self._save()
        return True

    def disarm(self) -> None:
        self.armed = False
        self.double_armed = False
        self._save()

    def set_double_armed(self, confirm: bool) -> bool:
        """Arm the second (low-balance) guard. Requires the first guard armed
        and its own explicit confirm."""
        if not self.armed or not confirm:
            return False
        self.double_armed = True
        self._save()
        return True

    def double_disarm(self) -> None:
        self.double_armed = False
        self._save()

    @property
    def live_armed(self) -> bool:
        """First guard: whether the LIVE (real) broker is used at all. The
        low-balance second guard is enforced per-order via is_executing()."""
        return self.armed

    def low_balance_blocks(self, net_worth: float | None) -> bool:
        """True when a real order must be blocked purely because the balance is
        low and the second guard isn't armed."""
        return self.is_low_balance(net_worth) and not self.double_armed

    def is_executing(self, net_worth: float | None) -> bool:
        """Whether REAL orders actually fire right now — the single source of
        truth for both the engine's runtime block and the dashboard banner."""
        return self.live_armed and not self.low_balance_blocks(net_worth)
