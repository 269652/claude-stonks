// Equity chart model — pure functions, no DOM. Loaded before app.js in the
// browser (window.LMChart) and require()-able from Node for tests.
(function (root) {
  "use strict";

  // Map an equity curve onto an SVG of W x H. x is proportional to TIME (so
  // overnight gaps aren't squashed to one step), the y range always contains
  // the baseline (starting value) so "above/below start" is visible, and the
  // colour says whether the last point is above or below that baseline.
  function chartModel(curve, baseline, W, H, pad) {
    if (!curve || curve.length < 2) return null;
    pad = pad == null ? 6 : pad;
    const vals = curve.map(p => Number(p.equity));
    const base = baseline != null && isFinite(baseline) ? Number(baseline) : vals[0];
    let lo = Math.min(base, ...vals), hi = Math.max(base, ...vals);
    const margin = (hi - lo) * 0.05 || Math.max(1, Math.abs(hi) * 0.01);
    lo -= margin; hi += margin;
    const t0 = Number(curve[0].ts), t1 = Number(curve[curve.length - 1].ts);
    const byTime = t1 > t0;
    const x = (p, i) => byTime ? ((Number(p.ts) - t0) / (t1 - t0)) * W
                               : (i / (curve.length - 1)) * W;
    const y = v => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
    const points = curve.map((p, i) => ({
      x: x(p, i), y: y(Number(p.equity)),
      ts: Number(p.ts), equity: Number(p.equity),
      cash: p.cash != null ? Number(p.cash) : null,
    }));
    // Axis ticks; any too close to the labelled baseline would overlap it.
    const baseY = y(base);
    const ticks = [hi, (hi + lo) / 2, lo].map(v => ({ value: v, y: y(v) }))
      .filter(t => Math.abs(t.y - baseY) >= 12);
    return {
      points, ticks, baseline: base, baselineY: baseY, min: lo, max: hi,
      up: vals[vals.length - 1] >= base,
    };
  }

  // Index of the point whose x is closest to `x` (for hover tooltips).
  function nearestIndex(points, x) {
    let best = -1, dist = Infinity;
    for (let i = 0; i < points.length; i++) {
      const d = Math.abs(points[i].x - x);
      if (d < dist) { dist = d; best = i; }
    }
    return best;
  }

  // The curve is recorded once per engine cycle (minutes apart); append the
  // current net worth (same value as the headline tile) as the live last
  // point so the chart never lags the tile. Returns a new array.
  function withLivePoint(curve, live, nowTs) {
    const out = (curve || []).slice();
    if (!live || live.equity == null || !isFinite(live.equity)) return out;
    const last = out[out.length - 1];
    if (last && Number(last.ts) >= nowTs) return out;
    out.push({ ts: nowTs, equity: Number(live.equity),
               cash: live.cash != null ? Number(live.cash) : null });
    return out;
  }

  const api = { chartModel, nearestIndex, withLivePoint };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LMChart = api;
})(typeof window !== "undefined" ? window : this);
