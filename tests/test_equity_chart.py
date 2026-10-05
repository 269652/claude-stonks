"""Equity chart model (web/static/chart.js): time-proportional x axis, a
baseline at the starting value, colour relative to that baseline, and a
nearest-point lookup for hover tooltips. Pure JS, exercised through Node;
skipped when Node isn't installed. Written before implementation (TDD)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CHART_JS = Path(__file__).parents[1] / "src" / "lmtrade" / "web" / "static" / "chart.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def run(expr: str):
    script = (f"const C = require({json.dumps(str(CHART_JS))});"
              f"process.stdout.write(JSON.stringify({expr}));")
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


CURVE = [
    {"ts": 0, "equity": 300.0, "cash": 300.0},
    {"ts": 10, "equity": 310.0, "cash": 140.0},
    {"ts": 100, "equity": 290.0, "cash": 140.0},   # long gap before this point
]


@needs_node
class TestChartModel:
    def test_x_is_proportional_to_time_not_index(self):
        m = run(f"C.chartModel({json.dumps(CURVE)}, 300, 600, 120)")
        xs = [p["x"] for p in m["points"]]
        assert xs[0] == pytest.approx(0)
        assert xs[1] == pytest.approx(60)      # 10% of the time span
        assert xs[2] == pytest.approx(600)

    def test_baseline_inside_range_and_between_extremes(self):
        m = run(f"C.chartModel({json.dumps(CURVE)}, 300, 600, 120)")
        ys = {p["equity"]: p["y"] for p in m["points"]}
        assert ys[310.0] < m["baselineY"] < ys[290.0]   # svg y grows downward
        assert ys[300.0] == pytest.approx(m["baselineY"])

    def test_colour_relative_to_baseline(self):
        down = run(f"C.chartModel({json.dumps(CURVE)}, 300, 600, 120).up")
        assert down is False
        up = run(f"C.chartModel({json.dumps(CURVE)}, 280, 600, 120).up")
        assert up is True

    def test_range_includes_baseline_even_if_far_away(self):
        m = run(f"C.chartModel({json.dumps(CURVE)}, 400, 600, 120)")
        assert m["max"] >= 400

    def test_tick_labels_cover_min_and_max(self):
        m = run(f"C.chartModel({json.dumps(CURVE)}, 300, 600, 120)")
        values = [t["value"] for t in m["ticks"]]
        assert min(values) <= 290.0 and max(values) >= 310.0

    def test_missing_baseline_falls_back_to_first_point(self):
        m = run(f"C.chartModel({json.dumps(CURVE)}, null, 600, 120)")
        assert m["baseline"] == pytest.approx(300.0)

    def test_identical_timestamps_fall_back_to_index_spacing(self):
        same = [{"ts": 5, "equity": 1.0}, {"ts": 5, "equity": 2.0}]
        xs = [p["x"] for p in run(f"C.chartModel({json.dumps(same)}, 1, 600, 120)")["points"]]
        assert xs == [pytest.approx(0), pytest.approx(600)]

    def test_too_few_points_is_null(self):
        assert run("C.chartModel([], 300, 600, 120)") is None
        assert run('C.chartModel([{"ts":1,"equity":1}], 300, 600, 120)') is None

    def test_nearest_index(self):
        pts = [{"x": 0}, {"x": 60}, {"x": 600}]
        assert run(f"C.nearestIndex({json.dumps(pts)}, 50)") == 1
        assert run(f"C.nearestIndex({json.dumps(pts)}, 400)") == 2
        assert run("C.nearestIndex([], 5)") == -1


class TestChartWiring:
    def test_dashboard_loads_chart_js_cache_busted(self, tmp_path):
        from fastapi.testclient import TestClient

        from lmtrade.config import Settings
        from lmtrade.web.app import create_app

        s = Settings(mode="paper", budget=100.0, universe=["AAPL"])
        s.data_dir = tmp_path
        html = TestClient(create_app(s)).get("/").text
        assert "/static/chart.js?v=" in html
        assert html.index("chart.js") < html.index("app.js")

    def test_tooltip_element_exists(self):
        html = (Path(__file__).parents[1] / "src" / "lmtrade" / "web" / "templates"
                / "dashboard.html").read_text(encoding="utf-8")
        assert 'id="chart-tip"' in html


@needs_node
class TestLivePoint:
    """The curve is only recorded once per engine cycle (minutes apart), so
    the chart lagged the 5-second net-worth tile. The tile's current value is
    appended as the last point."""

    def test_appends_live_point(self):
        out = run(f'C.withLivePoint({json.dumps(CURVE)}, {{"equity": 297.5, "cash": 139}}, 200)')
        assert len(out) == 4
        assert out[-1] == {"ts": 200, "equity": 297.5, "cash": 139}

    def test_does_not_mutate_input(self):
        out = run(f'(() => {{ const c = {json.dumps(CURVE)}; '
                  f'C.withLivePoint(c, {{"equity": 1}}, 200); return c.length; }})()')
        assert out == 3

    def test_no_live_value_returns_curve_unchanged(self):
        assert len(run(f"C.withLivePoint({json.dumps(CURVE)}, null, 200)")) == 3
        assert len(run(f'C.withLivePoint({json.dumps(CURVE)}, {{"equity": null}}, 200)')) == 3

    def test_live_point_never_goes_back_in_time(self):
        out = run(f'C.withLivePoint({json.dumps(CURVE)}, {{"equity": 1}}, 50)')
        assert len(out) == 3      # older than the last recorded point: skip

    def test_empty_curve_gets_single_point(self):
        assert run('C.withLivePoint([], {"equity": 5}, 1)') == [{"ts": 1, "equity": 5, "cash": None}]


@needs_node
class TestTickSpacing:
    def test_ticks_too_close_to_baseline_are_dropped(self):
        # baseline 300 sits right next to the middle tick (~300): no overlap
        curve = [{"ts": 0, "equity": 250.0}, {"ts": 1, "equity": 350.0}]
        m = run(f"C.chartModel({json.dumps(curve)}, 300, 600, 120)")
        assert all(abs(t["y"] - m["baselineY"]) >= 12 for t in m["ticks"])
        assert len(m["ticks"]) == 2      # max and min stay
