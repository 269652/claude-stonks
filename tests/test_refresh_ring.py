"""The header ring shows the time until the next dashboard refresh. It used to
be a free-running 5s CSS loop next to an independent setInterval, so the two
drifted apart. Now each refresh schedules the next one REFRESH_MS after it
finishes and restarts the ring at that moment. Written before implementation."""
from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).parents[1] / "src" / "lmtrade" / "web"


def test_ring_is_driven_by_the_refresh_loop():
    js = (WEB / "static" / "app.js").read_text(encoding="utf-8")
    html = (WEB / "templates" / "dashboard.html").read_text(encoding="utf-8")
    assert "setInterval(refresh" not in js            # no independent clock
    assert "restartRing" in js and "REFRESH_MS" in js
    ring_css = html[html.index(".ring-progress"):][:120]
    assert "infinite" not in ring_css                  # no free-running loop
