"""Control plane for the paper/live toggle and the live-execution arming
guard. Persisted so the engine (which executes) and the web dashboard (which
toggles) agree across processes. Written before implementation per strict
TDD. Nothing here places a real order — this is only the gate state."""
from __future__ import annotations

from pathlib import Path

import pytest

from lmtrade.core.control import LOW_BALANCE_EUR, ControlState


@pytest.fixture()
def path(tmp_path: Path) -> Path:
    return tmp_path / "control.json"


class TestDefaultsAndPersistence:
    def test_defaults_to_paper_disarmed(self, path):
        c = ControlState.load(path)
        assert c.mode == "paper"
        assert c.armed is False

    def test_roundtrips_via_disk(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True)
        again = ControlState.load(path)
        assert again.mode == "live"
        assert again.armed is True

    def test_missing_file_is_safe_default(self, path):
        assert not path.exists()
        assert ControlState.load(path).mode == "paper"

    def test_corrupt_file_falls_back_to_paper(self, path):
        path.write_text("{ not json")
        c = ControlState.load(path)
        assert c.mode == "paper" and c.armed is False


class TestModeSwitch:
    def test_switch_to_live_does_not_auto_arm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.mode == "live"
        assert c.armed is False        # live view, still simulated until armed

    def test_switch_to_paper_keeps_live_armed_by_default(self, path):
        # The badge only switches the VIEW; live trading keeps running unless
        # the user chooses to disarm (dashboard prompts and remembers).
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True)
        c.set_double_armed(confirm=True)
        c.set_mode("paper")
        assert c.armed is True and c.double_armed is True and c.live_armed is True
        again = ControlState.load(path)
        assert again.mode == "paper" and again.armed is True and again.live_armed is True

    def test_switch_to_paper_with_disarm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True)
        c.set_mode("paper", disarm=True)
        assert c.armed is False and c.double_armed is False and c.live_armed is False

    def test_invalid_mode_rejected(self, path):
        c = ControlState.load(path)
        with pytest.raises(ValueError):
            c.set_mode("bogus")


class TestArmingGuard:
    def test_cannot_arm_in_paper_mode(self, path):
        c = ControlState.load(path)
        assert c.arm(confirm=True) is False
        assert c.armed is False

    def test_arm_requires_confirm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.arm(confirm=False) is False
        assert c.armed is False

    def test_arm_with_confirm(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.arm(confirm=True) is True
        assert c.armed is True

    def test_disarm_clears_both_guards(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True)
        c.set_double_armed(confirm=True)
        c.disarm()
        assert c.armed is False
        assert c.double_armed is False

    def test_low_balance_flag(self, path):
        c = ControlState.load(path)
        assert c.is_low_balance(50.0) is True
        assert c.is_low_balance(150.0) is False
        assert c.is_low_balance(None) is True        # unknown -> treat as low


class TestSecondGuard:
    """The second (low-balance) guard is a SEPARATE persistent switch,
    enforced at execution time — not a one-shot confirm at arm time. Real
    orders below LOW_BALANCE_EUR require it in addition to the first guard."""

    def _armed_live(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True)
        return c

    def test_double_arm_requires_first_guard(self, path):
        c = ControlState.load(path)          # paper, not armed
        assert c.set_double_armed(confirm=True) is False
        assert c.double_armed is False

    def test_double_arm_requires_confirm(self, path):
        c = self._armed_live(path)
        assert c.set_double_armed(confirm=False) is False
        assert c.double_armed is False

    def test_double_arm_sets_flag(self, path):
        c = self._armed_live(path)
        assert c.set_double_armed(confirm=True) is True
        assert c.double_armed is True

    def test_double_disarm(self, path):
        c = self._armed_live(path)
        c.set_double_armed(confirm=True)
        c.double_disarm()
        assert c.double_armed is False

    def test_disarming_first_guard_clears_second(self, path):
        c = self._armed_live(path)
        c.set_double_armed(confirm=True)
        c.disarm()
        assert c.double_armed is False

    def test_switching_to_paper_with_disarm_clears_both(self, path):
        # Plain view switches keep both guards (see TestModeSwitch); the
        # explicit disarm choice clears both.
        c = self._armed_live(path)
        c.set_double_armed(confirm=True)
        c.set_mode("paper", disarm=True)
        assert c.armed is False and c.double_armed is False


class TestExecutionGating:
    """is_executing(net_worth) — whether REAL orders fire right now — drives
    both the engine's runtime block and the dashboard's IS/IS-NOT banner."""

    def _armed(self, path, double=False):
        c = ControlState.load(path)
        c.set_mode("live")
        c.arm(confirm=True)
        if double:
            c.set_double_armed(confirm=True)
        return c

    def test_not_executing_in_paper(self, path):
        assert ControlState.load(path).is_executing(500.0) is False

    def test_not_executing_when_unarmed(self, path):
        c = ControlState.load(path)
        c.set_mode("live")
        assert c.is_executing(500.0) is False

    def test_executing_when_armed_and_healthy_balance(self, path):
        assert self._armed(path).is_executing(500.0) is True

    def test_blocked_below_threshold_without_second_guard(self, path):
        c = self._armed(path)
        assert c.is_executing(LOW_BALANCE_EUR - 1) is False
        assert c.is_executing(None) is False          # unknown balance blocks too

    def test_allowed_below_threshold_with_second_guard(self, path):
        c = self._armed(path, double=True)
        assert c.is_executing(LOW_BALANCE_EUR - 1) is True

    def test_at_threshold_no_second_guard_needed(self, path):
        assert self._armed(path).is_executing(LOW_BALANCE_EUR) is True

    def test_second_guard_irrelevant_above_threshold(self, path):
        # double-armed but healthy balance: still executing (guard is a no-op).
        assert self._armed(path, double=True).is_executing(500.0) is True
