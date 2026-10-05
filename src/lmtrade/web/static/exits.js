// Position exit editing helpers — pure functions, no DOM. Loaded before app.js
// in the browser (window.LMExits) and require()-able from Node for tests.
(function (root) {
  "use strict";

  const INLINE_FIELDS = ["tp_pnl", "sl_pnl"];

  // Request body for /api/positions/exits when one TP/SL value is edited
  // inline: the position's current exit settings with just `field` replaced.
  // Empty input turns that exit off (null); anything non-numeric -> null body.
  function inlineExitBody(row, field, raw) {
    if (!INLINE_FIELDS.includes(field)) return null;
    const text = String(raw == null ? "" : raw).trim();
    let value = null;
    if (text !== "") {
      value = Number(text);
      if (!isFinite(value)) return null;
    }
    const t = row.trail || {};
    const body = {
      id: row.id,
      tp_pnl: row.tp_pnl != null ? Number(row.tp_pnl) : null,
      sl_pnl: row.sl_pnl != null ? Number(row.sl_pnl) : null,
      trail_enabled: !!t.enabled,
      trail_min_profit_eur: t.min_pnl != null ? Number(t.min_pnl) : null,
      trail_pct: t.pct != null ? Number(t.pct) : null,
    };
    body[field] = value;
    return body;
  }

  const api = { inlineExitBody };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LMExits = api;
})(typeof window !== "undefined" ? window : this);
