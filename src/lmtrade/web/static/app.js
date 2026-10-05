// LMTrade dashboard — polls the read-only JSON API and repaints every 5s.
const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
const time = (ts) => new Date(ts * 1000).toLocaleTimeString();

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}

function sideClass(s) { return s === "buy" ? "buy" : s === "sell" ? "sell" : "hold"; }

async function refresh() {
  try {
    const [s, trades, activity, logs, equity, strategies, realized, signals, news, analysis, ctrl,
           watch] =
      await Promise.all([
        getJSON("/api/summary"),
        getJSON("/api/trades?limit=100"),
        getJSON("/api/activity?limit=100"),
        getJSON("/api/logs?limit=200"),
        getJSON("/api/equity"),
        getJSON("/api/leaderboard"),
        getJSON("/api/realized?limit=200"),
        getJSON("/api/signals?min_confidence=0.6"),
        getJSON("/api/news"),
        getJSON("/api/analysis"),
        getJSON("/api/control"),
        getJSON("/api/watchlist"),
      ]);
    window._ctrl = ctrl;
    paintControl(ctrl, s);
    paintSummary(s);
    paintPositions(s.positions);
    paintTrades(trades);
    paintActivity(activity);
    paintLogs(logs);
    // Same baseline as the P&L tile in paper mode; live falls back to the first point.
    paintChart(equity, ctrl.mode === "live" ? null : s.starting_cash);
    paintStrategies(strategies);
    paintRealized(realized);
    paintWatchlist(watch);   // before signals: they show its entry distance
    paintSignals(signals);
    paintNews(news);
    paintAnalysis(analysis);
  } catch (e) {
    console.error(e);
  }
}

function paintControl(ctrl, s) {
  const isLive = ctrl.mode === "live";
  const badge = $("mode");
  badge.textContent = ctrl.mode;
  badge.className = "badge " + (isLive ? "live" : "paper");
  $("armwrap").classList.toggle("show", isLive);
  $("arm-toggle").checked = !!ctrl.armed;
  $("armed-flag").classList.toggle("show", !!ctrl.armed);

  // Second guard: only relevant when armed AND the balance is low.
  const showGuard2 = isLive && ctrl.armed && ctrl.low_balance;
  $("guard2").classList.toggle("show", showGuard2);
  $("arm2-toggle").checked = !!ctrl.double_armed;

  // IS / IS-NOT executing real orders banner — the single most important
  // status in live mode. Hidden entirely in paper mode.
  const b = $("exec-banner");
  b.classList.toggle("show", isLive);
  if (isLive) {
    if (ctrl.executing) {
      b.className = "exec-banner show exec-live";
      $("exec-ic").textContent = "🔴";
      $("exec-text").innerHTML = "<b>EXECUTING REAL ORDERS</b> — live Trade Republic "
        + "fills with real money. Disarm to stop.";
    } else {
      b.className = "exec-banner show exec-sim";
      $("exec-ic").textContent = "🛡️";
      const why = !ctrl.armed
        ? "not armed — simulating on paper"
        : ctrl.low_balance
          ? `net worth under €${ctrl.low_balance_threshold} — low-balance guard not armed`
          : "simulating";
      $("exec-text").innerHTML = `<b>NOT executing real orders</b> (${why}).`;
    }
  }
}

function paintSummary(s) {
  const cur = s.currency || "EUR";
  $("curr").textContent = cur;
  const mode = $("mode");
  // mode text/class is owned by paintControl; leave it here as a fallback.
  if (!window._ctrl) { mode.textContent = s.mode; mode.className = "badge " + (s.mode === "live" ? "live" : "paper"); }

  const econ = s.economics || {};
  const ctrl = window._ctrl || {};
  const isLive = ctrl.mode === "live";
  // Liquidation value (value minus the exit fee): net worth = cash + this.
  const posValue = (s.positions || []).reduce(
    (a, p) => a + (p.liquidation_value != null ? p.liquidation_value : (p.value || 0)), 0);

  // In LIVE mode the headline figures are the REAL Trade Republic account —
  // cash from the live balance, net worth = real cash + open position value,
  // P&L vs the baseline captured when live began. Shown as "—" (never the
  // simulated paper 100) until the real balance is actually fetched. In paper
  // mode they're the simulated book's economics.
  let net, cash, pnl, netKnown;
  if (isLive) {
    cash = s.tr_account_cash;
    net = cash != null ? cash + posValue : null;
    netKnown = net != null;
    const base = s.tr_baseline_net_worth != null ? s.tr_baseline_net_worth : cash;
    pnl = netKnown ? net - base : null;
  } else {
    cash = s.cash;
    net = econ.net_worth_eur != null ? econ.net_worth_eur : s.equity;
    netKnown = true;
    pnl = econ.pnl_eur != null ? econ.pnl_eur : net - (s.starting_cash || 0);
  }
  // Live last point for the equity chart, so it ends exactly at this tile.
  window._liveNow = netKnown ? {equity: net, cash: cash} : null;

  $("net").textContent = netKnown ? fmt(net) + " " + cur : "—";
  const pnlEl = $("pnl");
  pnlEl.className = "sub";
  if (pnl != null) {
    pnlEl.textContent = (pnl >= 0 ? "▲ " : "▼ ") + fmt(pnl) + " " + cur + " P&L";
  } else {
    pnlEl.textContent = isLive ? "awaiting live TR balance…" : "—";
  }
  // Net worth is red while mark-to-market sits below the last REALIZED net
  // worth (locked in at the most recent closed trade), green at or above.
  const mark = s.last_realized_net_worth != null ? s.last_realized_net_worth : (s.starting_cash || 0);
  $("net").className = "v " + (!netKnown ? "" : net < mark - 1e-9 ? "neg" : "pos");

  const cashEl = $("cash");
  cashEl.textContent = cash != null ? fmt(cash) + " " + cur : "—";
  cashEl.className = "v" + (isLive && ctrl.low_balance ? " warn" : "");
  $("reserve").textContent = fmt(s.reserve != null ? s.reserve : (econ.reserve_eur || 0)) + " " + cur;
  $("npos").textContent = s.num_positions;
  const banner = $("econ");
  banner.style.display = "";
  if (econ.self_sustaining) {
    banner.className = "econ-banner econ-ok";
    const alpha = s.alpha;
    const alphaText = alpha != null ? (alpha >= 0 ? '+' : '') + fmt(alpha) + ' EUR' : '—';
    banner.innerHTML = `✅ <b>Outperforming SP 500</b> — alpha ${alphaText}. Runway ${fmt(econ.runway_hours,1)}h.`;
  } else if (econ.halt_trading) {
    banner.className = "econ-banner econ-warn";
    banner.innerHTML = `⛔ <b>Runway below floor</b> (${fmt(econ.runway_hours,1)}h) — new entries halted, managing exits only.`;
  } else {
    // Normal state (not yet covering compute): no banner.
    banner.style.display = "none";
  }

  // Provider warning banner (e.g. Claude usage limit hit)
  paintHeartbeat(s.tr_session);
  const pwBanner = $("provider-warning-banner");
  const pw = s.provider_warnings;
  const trs = s.tr_session;
  if (trs && trs.ok === false) {
    // TR heartbeat + automatic re-login failed: needs a manual pytr login.
    pwBanner.innerHTML = `⛔ <b>TR connection lost</b> since ${time(trs.since || trs.ts)} — ${trs.msg}`;
    pwBanner.classList.add("show");
  } else if (pw && pw.msg) {
    pwBanner.innerHTML = `⚠️ <b>Research provider warning:</b> ${pw.msg}
      <button onclick="dismissProviderWarning()" style="margin-left:12px;padding:2px 8px;cursor:pointer;border-radius:4px;border:none;background:#555;color:#fff;font-size:12px">Dismiss</button>`;
    pwBanner.classList.add("show");
  } else {
    pwBanner.classList.remove("show");
  }
}

// Header dot = TR heartbeat: colour by state, a blink whenever a new answer
// from TR has arrived since the last refresh.
let lastBeatTs = null;
function paintHeartbeat(trs) {
  const dot = $("hb-dot");
  const state = LMChart.heartbeatState(trs, Date.now() / 1000);
  dot.className = "dot hb-" + state;
  const t = v => v ? time(v) : "—";
  dot.title = trs
    ? `TR heartbeat: ${state} · sent ${t(trs.sent_ts)} · answered ${t(trs.ts)}`
    : "TR heartbeat: no data (no TR connection)";
  if (trs && trs.ts && lastBeatTs !== null && trs.ts !== lastBeatTs && state === "ok") {
    void dot.offsetWidth;            // restart the animation
    dot.classList.add("hb-blink");
  }
  if (trs && trs.ts) lastBeatTs = trs.ts;
}

function pnlCell(v) {
  if (v == null) return `<td class="muted">—</td>`;
  const cls = v >= 0 ? "pos" : "neg";
  const sign = v >= 0 ? "+" : "";
  return `<td class="num ${cls}">${sign}${fmt(v)}</td>`;
}

// TP / SL as premium level plus the net EUR result if hit (both fees out),
// and the trailing take-profit state.
function exitsCell(p) {
  const eur = v => v == null ? "—" : (v >= 0 ? "+" : "") + fmt(v) + " €";
  const parts = [];
  // TP / SL are click-to-edit in place (net EUR); empty input turns them off.
  const editable = p.close_kind === "option" && p.id != null;
  const tag = (field, html) => editable
    ? `<span class="inline-exit" data-id="${p.id}" data-field="${field}" title="Click to edit (€ net)">${html}</span>`
    : html;
  parts.push(p.tp_price != null
    ? tag("tp_pnl", `<span class="pos">TP</span> ${fmt(p.tp_price)} <span class="muted">(${eur(p.tp_pnl)})</span>`)
    : tag("tp_pnl", `<span class="muted">TP off</span>`));
  parts.push(p.sl_price != null
    ? tag("sl_pnl", `<span class="neg">SL</span> ${fmt(p.sl_price)} <span class="muted">(${eur(p.sl_pnl)})</span>`)
    : tag("sl_pnl", `<span class="muted">SL off</span>`));
  const t = p.trail;
  if (t && t.enabled) {
    parts.push(t.active
      ? `<span class="lock">🔒 Trail ≥ ${eur(t.stop_pnl)}</span> <span class="muted">(peak ${eur(t.peak_pnl)})</span>`
      : `<span class="muted">Trail arms at ${eur(t.arm_pnl)}</span>`);
  }
  // Lines on the left, edit pencil beside them (highlighted when custom).
  const btn = p.close_kind === "option" && p.id != null
    ? `<button class="exits-btn${p.exits_custom ? " custom" : ""}" data-id="${p.id}"`
      + ` title="${p.exits_custom ? "Custom exits — edit" : "Configure exits"}">✎</button>`
    : "";
  return parts.length
    ? `<td class="exits"><div class="exits-wrap"><div class="exits-lines">${parts.join("<br>")}</div>${btn}</div></td>`
    : `<td class="muted">—</td>`;
}

function closeButton(p) {
  if (!p.close_kind) return "<td></td>";
  if (p.close_pending) return `<td><button class="close-btn" disabled>closing…</button></td>`;
  const attrs = p.close_kind === "option" ? `data-id="${p.id}"` : `data-symbol="${p.close_symbol}"`;
  return `<td><button class="close-btn" data-kind="${p.close_kind}" ${attrs}`
    + ` data-label="${p.symbol}">Close</button></td>`;
}

function paintPositions(rows) {
  if (window._inlineEditing) return;   // don't wipe an inline TP/SL edit
  window._posRows = {};
  rows.forEach(p => { if (p.id != null) window._posRows[p.id] = p; });
  $("positions").innerHTML = rows.length
    ? rows.map(p => `<tr><td>${p.symbol}</td><td><span class="kind">${p.kind || "equity"}</span></td>`
        + `<td>${p.isin || "—"}</td><td>${fmt(p.qty,4)}</td><td>${fmt(p.avg_price)}</td>`
        + `<td>${p.price == null ? "—" : fmt(p.price)}`
        + `${p.spot == null ? "" : `<div class="sub">${p.symbol.split(" ")[0]} ${fmt(p.spot)}</div>`}</td>`
        + `<td>${p.value == null ? "—" : fmt(p.value)}</td>${pnlCell(p.unrealized_pnl)}`
        + `${exitsCell(p)}${closeButton(p)}</tr>`).join("")
    : `<tr><td colspan="10" class="muted">No open positions.</td></tr>`;
}

// Inline TP/SL edit: click the value, type net EUR, Enter/blur saves, Esc cancels.
$("positions").addEventListener("click", ev => {
  const span = ev.target.closest(".inline-exit");
  if (!span || window._inlineEditing) return;
  const row = window._posRows[span.dataset.id];
  if (!row) return;
  const field = span.dataset.field;
  window._inlineEditing = true;
  const input = document.createElement("input");
  input.type = "text"; input.className = "inline-exit-input";
  input.placeholder = "off";
  input.value = row[field] == null ? "" : Math.round(row[field] * 100) / 100;
  span.replaceWith(input);
  input.focus(); input.select();
  let done = false;
  const finish = async save => {
    if (done) return;
    if (save) {
      const body = LMExits.inlineExitBody(row, field, input.value);
      if (!body) { input.classList.add("bad"); input.focus(); return; }
      done = true;
      const res = await postJSON("/api/positions/exits", body);
      if (!res.ok) alert("Not saved: " + (res.error || "unknown error"));
    }
    done = true;
    window._inlineEditing = false;
    refresh();
  };
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => {
    if (!done && !input.classList.contains("bad")) finish(true);
    else if (!done) finish(false);
  });
});

// Per-position exits modal (+ / ✎ in the Exits column). Values in net EUR.
let exitsId = null;
const num = v => (v === "" || v == null) ? null : Number(v);
function paintTrailArm() {
  const min = num($("ex-trail-min").value), pct = num($("ex-trail-pct").value);
  $("ex-trail-arm").textContent = ($("ex-trail-on").value === "1" && min != null && pct != null
      && pct >= 0 && pct < 100) ? `+${fmt(min / (1 - pct / 100))} € profit` : "—";
}
function openExits(p) {
  exitsId = p.id;
  const r2 = v => v == null ? "" : Math.round(v * 100) / 100;
  $("exits-title").textContent = `Exits · ${p.symbol}`;
  $("ex-tp").value = r2(p.tp_pnl);
  $("ex-sl").value = r2(p.sl_pnl);
  $("ex-trail-on").value = p.trail && p.trail.enabled ? "1" : "0";
  $("ex-trail-min").value = p.trail ? r2(p.trail.min_pnl) : "";
  $("ex-trail-pct").value = p.trail ? r2(p.trail.pct * 100) : "";
  $("exits-status").textContent = p.exits_custom ? "custom settings for this position" : "using defaults";
  paintTrailArm();
  $("exits-overlay").classList.add("show");
}
["ex-trail-on", "ex-trail-min", "ex-trail-pct"].forEach(id =>
  $(id).addEventListener("input", paintTrailArm));
$("positions").addEventListener("click", ev => {
  const btn = ev.target.closest(".exits-btn");
  if (btn && window._posRows[btn.dataset.id]) openExits(window._posRows[btn.dataset.id]);
});
async function saveExits(body) {
  const res = await postJSON("/api/positions/exits", {id: exitsId, ...body});
  if (!res.ok) { $("exits-status").textContent = "⚠ " + (res.error || "could not save"); return; }
  $("exits-overlay").classList.remove("show");
  refresh();
}
$("exits-save").addEventListener("click", () => {
  const pct = num($("ex-trail-pct").value);
  saveExits({
    tp_pnl: num($("ex-tp").value), sl_pnl: num($("ex-sl").value),
    trail_enabled: $("ex-trail-on").value === "1",
    trail_min_profit_eur: num($("ex-trail-min").value),
    trail_pct: pct == null ? null : pct / 100,
  });
});
$("exits-reset").addEventListener("click", () => saveExits({reset: true}));
$("exits-cancel").addEventListener("click", () => $("exits-overlay").classList.remove("show"));
$("exits-overlay").addEventListener("click", e => {
  if (e.target.id === "exits-overlay") $("exits-overlay").classList.remove("show");
});

// Manual close: queue a request; the engine sells at the current price within
// seconds (fee + P&L booked like any other exit).
$("positions").addEventListener("click", async ev => {
  const btn = ev.target.closest(".close-btn");
  if (!btn || btn.disabled) return;
  if (!confirm(`Close ${btn.dataset.label} now at the current price?\n\n`
               + "The 1 EUR exit fee applies.")) return;
  btn.disabled = true; btn.textContent = "closing…";
  const body = btn.dataset.kind === "option"
    ? {kind: "option", id: Number(btn.dataset.id)}
    : {kind: "equity", symbol: btn.dataset.symbol};
  const res = await postJSON("/api/positions/close", body);
  if (!res.queued) { alert("Close failed: " + (res.error || "unknown error")); }
  refresh();
});

function paintStrategies(rows) {
  $("strategies").innerHTML = rows && rows.length
    ? rows.map(g => {
        const params = Object.entries(g.params || {})
          .map(([k, v]) => `${k}=${typeof v === "number" ? Number(v).toFixed(2) : v}`).join(", ");
        return `<tr><td>${g.strategy}</td><td class="muted">${params}</td>`
          + `<td>${g.trades}</td>${pnlCell(g.pnl)}<td>${fmt(g.fitness, 4)}</td></tr>`;
      }).join("")
    : `<tr><td colspan="5" class="muted">No strategies evolved yet — they appear once trades close and the optimizer runs.</td></tr>`;
}

function paintRealized(r) {
  const total = (r && r.total_pnl) || 0;
  const card = $("realized");
  card.textContent = (total >= 0 ? "+" : "") + fmt(total);
  card.className = "v " + (total > 0 ? "pos" : total < 0 ? "neg" : "");
  $("realized-sub").textContent = r ? `${r.wins}W / ${r.losses}L closed` : "closed trades";
  const rows = (r && r.rows) || [];
  $("realized-rows").innerHTML = rows.length
    ? rows.map(o => `<tr><td>${o.closed_ts ? time(o.closed_ts) : "—"}</td><td>${o.symbol}</td>`
        + `<td><span class="kind">${o.kind}</span></td><td>${o.isin || "—"}</td>`
        + `<td>${fmt(o.entry_premium)}</td><td>${fmt(o.exit_premium)}</td>${pnlCell(o.pnl)}</tr>`).join("")
    : `<tr><td colspan="7" class="muted">No closed trades yet.</td></tr>`;

  // Unread badge: number of realized closes the user hasn't looked at. Total
  // closed count is wins+losses; we persist the last-seen count and show the
  // delta until the Realized tab is opened.
  const closedCount = r ? (r.wins || 0) + (r.losses || 0) : 0;
  const seen = Number(localStorage.getItem("realizedSeen") || 0);
  const badge = $("realized-unread");
  const active = document.querySelector('.tab[data-tab="realized"]').classList.contains("active");
  if (active) {
    localStorage.setItem("realizedSeen", String(closedCount));
    badge.classList.add("hidden");
  } else {
    const unread = Math.max(0, closedCount - seen);
    badge.textContent = unread > 99 ? "99+" : String(unread);
    badge.classList.toggle("hidden", unread === 0);
  }
}

function paintTrades(rows) {
  $("trades").innerHTML = rows.length
    ? rows.map(t => `<tr><td>${time(t.ts)}</td><td>${t.symbol}</td>
        <td><span class="pill ${sideClass(t.side)}">${t.side}</span></td>
        <td>${fmt(t.qty,4)}</td><td>${fmt(t.price)}</td><td>${fmt(t.fee)}</td>
        <td>${t.confidence != null ? fmt(t.confidence) : "—"}</td></tr>`).join("")
    : `<tr><td colspan="7" class="muted">No trades yet.</td></tr>`;
}

function paintActivity(rows) {
  $("activity").innerHTML = rows.length
    ? rows.map(a => `<tr><td>${time(a.ts)}</td><td><span class="kind">${a.kind}</span></td>
        <td>${a.symbol || "—"}</td><td>${a.summary}</td></tr>`).join("")
    : `<tr><td colspan="4" class="muted">No activity yet.</td></tr>`;
}

function paintLogs(rows) {
  $("logs").innerHTML = rows.map(l =>
    `<div class="log-${l.level}">${time(l.ts)} [${l.level}] ${l.source || ""} ${l.message}</div>`
  ).join("") || `<div class="muted">No logs.</div>`;
}

function sentClass(s) {
  return s === "bullish" ? "buy" : s === "bearish" ? "sell" : "hold";
}

// Watchlist: signals waiting for a >= k-sigma stretch from the intraday SMA
// (below for buys -> call, above for sells -> put) before they are entered.
function paintWatchlist(rows) {
  rows = rows || [];
  window._watchBySym = {};
  rows.forEach(r => { window._watchBySym[r.symbol] = r; });
  $("watch-count").textContent = rows.length ? ` (${rows.length})` : "";
  const k = rows.length ? rows[0].k : null;
  $("watch-note").textContent = rows.length
    ? `Entered once price is ${k}σ below (buy → call) or above (sell → put) its intraday average; `
      + "dropped when the signal disappears or the entry expires. "
      + "Ranked by priority = (2 × confidence + closeness) / 3."
    : "";
  const sig = v => v == null ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(2) + "σ";
  const left = ts => {
    const m = Math.max(0, Math.round((ts - Date.now() / 1000) / 60));
    return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`;
  };
  $("watchlist").innerHTML = rows.length
    ? rows.map(r => `<tr><td>${r.symbol}</td>`
        + `<td><span class="${r.direction === "buy" ? "pos" : "neg"}">${r.direction.toUpperCase()}</span>`
        + ` <span class="kind">${r.instrument}</span></td>`
        + `<td>${fmt(r.confidence)}</td>`
        + `<td>${r.spot == null ? "—" : fmt(r.spot)}</td>`
        + `<td>${r.target == null ? "—" : fmt(r.target)}</td>`
        + `<td>${r.distance_pct == null ? "—" : r.distance_pct === 0 ? "<b>reached</b>"
              : (r.direction === "buy" ? "▼ " : "▲ ") + Number(r.distance_pct).toFixed(2) + "%"}</td>`
        + `<td>${sig(r.z)}</td><td>${sig(r.trigger_z)}</td>`
        + `<td>${r.sigma_to_go == null ? "—" : r.sigma_to_go === 0 ? "ready" : Number(r.sigma_to_go).toFixed(2) + "σ"}</td>`
        + `<td>${r.order
              ? `limit ${fmt(r.order.limit)} × ${fmt(r.order.size, 2)}`
                + `<div class="sub">now ${r.order.current_price == null ? "—" : fmt(r.order.current_price)}`
                + `${r.order.order_id ? " · " + r.order.order_id.slice(0, 8) : ""}</div>`
              : r.high_vol
                ? `<span class="muted">market on trigger</span><div class="sub">σ ${Number(r.sigma_pct).toFixed(2)}%</div>`
                : `<span class="muted">—</span>`}</td>`
        + `<td>${left(r.expires_ts)}</td></tr>`).join("")
    : `<tr><td colspan="11" class="muted">Nothing on the watchlist — new signals appear here while they wait for a good entry.</td></tr>`;
}

// Watch-list status for a signal: current sigma and how far to the entry.
function entryDistance(symbol) {
  const w = (window._watchBySym || {})[symbol];
  if (!w) return `<span class="muted">not watched</span>`;
  const z = w.z == null ? "—" : (w.z >= 0 ? "+" : "") + Number(w.z).toFixed(2) + "σ";
  if (w.sigma_to_go == null) return `<span class="muted">${z}</span>`;
  if (w.sigma_to_go === 0) return `${z} · <b>ready</b>`;
  return `${z} · <span class="muted">${Number(w.sigma_to_go).toFixed(2)}σ to go</span>`;
}

function paintSignals(rows) {
  $("signals").innerHTML = rows && rows.length
    ? rows.map(r => {
        const cause = (r.signals || [])
          .filter(s => s.direction && s.direction !== "hold")
          .map(s => `<span class="pill ${sideClass(s.direction)}">${s.provider} ${fmt(s.confidence)}</span> ${s.rationale || ""}`)
          .join("<br>") || `<span class="muted">${r.rationale || "—"}</span>`;
        return `<tr><td>${r.symbol}</td><td><span class="pill ${sideClass(r.direction)}">${r.direction}</span></td>`
          + `<td>${fmt(r.confidence)}</td><td>${entryDistance(r.symbol)}</td><td>${cause}</td></tr>`;
      }).join("")
    : `<tr><td colspan="5" class="muted">No strong signals right now (needs confidence ≥ 0.6).</td></tr>`;
}

function paintNews(rows) {
  $("news").innerHTML = rows && rows.length
    ? rows.map(n => `<tr><td>${n.ts ? time(n.ts) : "—"}</td><td>${n.symbol}</td>`
        + `<td><span class="pill ${sentClass(n.sentiment)}">${n.sentiment || "—"}</span></td>`
        + `<td>${(n.text || "").slice(0, 240)}</td></tr>`).join("")
    : `<tr><td colspan="4" class="muted">No news yet.</td></tr>`;
}

function paintAnalysis(a) {
  const syms = (a && a.symbols) || {};
  const keys = Object.keys(syms);
  const meta = $("analysis-meta");
  if (a && a.ts) {
    const age = ((Date.now() / 1000 - a.ts) / 3600).toFixed(1);
    meta.textContent = `Compiled ${age}h ago · valid ${a.valid_hours || 24}h · ${keys.length} symbols`;
  } else {
    meta.textContent = "";
  }
  $("analysis").innerHTML = keys.length
    ? keys.map(k => {
        const v = syms[k] || {};
        return `<tr><td>${k}</td><td><span class="pill ${sentClass(v.bias)}">${v.bias || "—"}</span></td>`
          + `<td>${fmt(v.confidence)}</td><td class="muted">${v.notes || ""}</td></tr>`;
      }).join("")
    : `<tr><td colspan="4" class="muted">No daily analysis compiled yet.</td></tr>`;
}

const CHART_W = 600, CHART_H = 120;

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
}

function paintChart(curve, baseline) {
  const svg = $("chart");
  curve = LMChart.withLivePoint(curve, window._liveNow, Date.now() / 1000);
  const m = LMChart.chartModel(curve, baseline, CHART_W, CHART_H);
  window._chart = m;
  if (!m) {
    svg.innerHTML = ""; $("chart-yaxis").innerHTML = "";
    $("chart-x0").textContent = ""; $("chart-x1").textContent = "";
    return;
  }
  const color = m.up ? "#3fb950" : "#f85149";
  const pts = m.points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  svg.innerHTML = `
    <polygon fill="${color}22" points="0,${CHART_H} ${pts} ${CHART_W},${CHART_H}" />
    <line x1="0" x2="${CHART_W}" y1="${m.baselineY.toFixed(1)}" y2="${m.baselineY.toFixed(1)}"
          stroke="#8b98a5" stroke-width="1" stroke-dasharray="4 4" vector-effect="non-scaling-stroke" />
    <polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"
              vector-effect="non-scaling-stroke" />
    <line class="chart-cursor" x1="0" x2="0" y1="0" y2="${CHART_H}" stroke="#8b98a5"
          stroke-width="1" vector-effect="non-scaling-stroke" visibility="hidden" />`;
  const labels = m.ticks.map(t => `<span style="top:${t.y}px">${fmt(t.value)}</span>`);
  labels.push(`<span style="top:${m.baselineY}px">start ${fmt(m.baseline)}</span>`);
  $("chart-yaxis").innerHTML = labels.join("");
  $("chart-x0").textContent = fmtTime(m.points[0].ts);
  $("chart-x1").textContent = fmtTime(m.points[m.points.length - 1].ts);
  // The 5s refresh redraws the SVG; keep an active hover in place.
  if (window._hoverX != null) showChartTip(window._hoverX);
}

// Hover tooltip: nearest point by time, with net worth, P&L vs start, cash.
function showChartTip(clientX) {
  const m = window._chart, svg = $("chart"), tip = $("chart-tip");
  if (!m) return;
  const rect = svg.getBoundingClientRect();
  if (!rect.width) return;
  const i = LMChart.nearestIndex(m.points, (clientX - rect.left) / rect.width * CHART_W);
  if (i < 0) return;
  const p = m.points[i];
  const cursor = $("chart").querySelector(".chart-cursor");
  cursor.setAttribute("x1", p.x); cursor.setAttribute("x2", p.x);
  cursor.setAttribute("visibility", "visible");
  const diff = p.equity - m.baseline;
  tip.innerHTML = `<div class="t">${fmtTime(p.ts)}</div>` +
    `<div><b>${fmt(p.equity)} EUR</b></div>` +
    `<div class="${diff >= 0 ? "pos" : "neg"}">${diff >= 0 ? "+" : ""}${fmt(diff)} vs start</div>` +
    (p.cash != null ? `<div class="t">cash ${fmt(p.cash)}</div>` : "");
  tip.style.display = "block";
  const wrap = $("chart-wrap").getBoundingClientRect();
  const px = rect.left - wrap.left + p.x / CHART_W * rect.width;
  tip.style.left = Math.min(Math.max(px + 12, 0), wrap.width - tip.offsetWidth) + "px";
}
$("chart").addEventListener("mousemove", ev => {
  window._hoverX = ev.clientX;
  showChartTip(ev.clientX);
});
$("chart").addEventListener("mouseleave", () => {
  window._hoverX = null;
  $("chart-tip").style.display = "none";
  const cursor = $("chart").querySelector(".chart-cursor");
  if (cursor) cursor.setAttribute("visibility", "hidden");
});

// Tabs
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    ["positions", "realized", "watchlist", "signals", "news", "analysis", "strategies", "trades", "activity", "logs"].forEach(name =>
      $("tab-" + name).classList.toggle("hidden", name !== tab.dataset.tab));
  });
});

async function postJSON(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

// Paper/live toggle — click the badge.
$("mode").addEventListener("click", async () => {
  const ctrl = window._ctrl || {mode: "paper"};
  const next = ctrl.mode === "live" ? "paper" : "live";
  if (next === "live" &&
      !confirm("Switch to LIVE mode?\n\nThis shows your real Trade Republic " +
               "account. Execution stays SIMULATED until you separately arm it.")) return;
  await postJSON("/api/control/mode", {mode: next});
  refresh();
});

// Arm / disarm real live execution.
$("arm-toggle").addEventListener("change", async (e) => {
  const ctrl = window._ctrl || {};
  if (!e.target.checked) {
    await postJSON("/api/control/disarm", {});
    refresh();
    return;
  }
  // First guard: hard confirmation. Below the fee-drag threshold this arms
  // but does NOT execute yet — the second guard (below) is required.
  const nw = ctrl.net_worth != null ? ctrl.net_worth.toFixed(2) + " EUR" : "unknown";
  if (!confirm("⚠ ARM REAL LIVE EXECUTION ⚠\n\nThe bot will place REAL Trade " +
               "Republic orders with REAL money, against TR's ToS. Fills are " +
               "irreversible.\n\nLive net worth: " + nw + "\n\nProceed?")) {
    e.target.checked = false; return;
  }
  const res = await postJSON("/api/control/arm", {confirm: true});
  if (!res.armed) e.target.checked = false;
  refresh();
});

// Second guard: allow real orders while net worth is under the threshold.
$("arm2-toggle").addEventListener("change", async (e) => {
  const ctrl = window._ctrl || {};
  if (!e.target.checked) {
    await postJSON("/api/control/double-disarm", {});
    refresh();
    return;
  }
  const thr = ctrl.low_balance_threshold || 100;
  const nw = ctrl.net_worth != null ? ctrl.net_worth.toFixed(2) + " EUR" : "unknown";
  if (!confirm("⚠ LOW-BALANCE EXECUTION ⚠\n\nNet worth (" + nw + ") is under €" +
               thr + ". At this size the flat ~1 EUR Trade Republic fee is more " +
               "than 10% of a typical order — a severe drag that is STRONGLY " +
               "ADVISED AGAINST.\n\nArm the second guard to execute real orders " +
               "anyway?")) {
    e.target.checked = false; return;
  }
  const res = await postJSON("/api/control/double-arm", {confirm: true});
  if (!res.double_armed) e.target.checked = false;
  refresh();
});

// ------------------------------------------------------------------ settings
const SECTION_LABELS = {
  general: "General", loop: "Loop", options: "Options", risk: "Risk",
  economics: "Economics", research: "Research", tr: "Trade Republic",
  data: "Data", model: "Model", entry: "Watchlist entry",
};

function fieldInput(f) {
  const id = "set_" + f.key.replace(/\./g, "_");
  if (f.kind === "bool") {
    return `<input type="checkbox" id="${id}" data-key="${f.key}" data-kind="bool" ${f.value ? "checked" : ""}>`;
  }
  if (f.kind === "select") {
    const opts = (f.options || []).map(o =>
      `<option value="${o}" ${String(f.value) === o ? "selected" : ""}>${o}</option>`).join("");
    return `<select id="${id}" data-key="${f.key}" data-kind="select">${opts}</select>`;
  }
  const type = (f.kind === "number" || f.kind === "int") ? "number" : "text";
  const step = f.kind === "int" ? "1" : "any";
  return `<input type="${type}" step="${step}" id="${id}" data-key="${f.key}" data-kind="${f.kind}" value="${f.value}">`;
}

async function openSettings() {
  const data = await getJSON("/api/settings");
  $("settings-note").textContent = data.note || "";
  const bySection = {};
  data.fields.forEach(f => { (bySection[f.section] = bySection[f.section] || []).push(f); });
  let html = "";
  Object.keys(bySection).forEach(sec => {
    html += `<div class="setsection">${SECTION_LABELS[sec] || sec}</div>`;
    bySection[sec].forEach(f => {
      html += `<div class="setrow"><label>${f.field}</label>${fieldInput(f)}</div>`;
    });
  });
  $("settings-fields").innerHTML = html;
  $("settings-status").textContent = "";
  $("settings-overlay").classList.add("show");
}

async function saveSettings() {
  const changes = {};
  document.querySelectorAll("#settings-fields [data-key]").forEach(el => {
    changes[el.dataset.key] = el.dataset.kind === "bool" ? el.checked : el.value;
  });
  $("settings-status").textContent = "Saving…";
  const res = await postJSON("/api/settings", {changes});
  const n = Object.keys(res.applied || {}).length;
  const bad = (res.rejected || []).length;
  if (res.restarting) {
    $("settings-overlay").classList.remove("show");
    showRestartOverlay();
  } else {
    $("settings-status").textContent =
      `Saved ${n} setting${n === 1 ? "" : "s"}${bad ? `, ${bad} rejected` : ""}` +
      (n > 0 ? " — restart `lmtrade run` to apply." : ".");
  }
}

function showRestartOverlay() {
  const el = document.createElement("div");
  el.id = "restart-overlay";
  el.className = "restart-overlay";
  el.innerHTML =
    `<div class="r-title">⟳ Restarting engine…</div>` +
    `<div class="r-sub">Applying new settings, back in a moment.</div>`;
  document.body.appendChild(el);
  pollUntilBack(el);
}

async function pollUntilBack(el) {
  // Brief pause so the server process has time to begin shutting down.
  await new Promise(r => setTimeout(r, 1500));
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch("/healthz");
      if (r.ok) {
        el.remove();
        refresh();
        return;
      }
    } catch (_) { /* server temporarily down — keep polling */ }
    await new Promise(r => setTimeout(r, 1000));
  }
  el.querySelector(".r-title").textContent = "⚠ Restart timed out";
  el.querySelector(".r-sub").textContent = "Please reload the page manually.";
}

async function dismissProviderWarning() {
  // Clear the persistent warning from the store so it won't reappear until
  // the next limit event. Uses a lightweight PATCH-style endpoint.
  try { await fetch("/api/dismiss-provider-warning", {method: "POST"}); } catch (_) {}
  const el = $("provider-warning-banner");
  if (el) el.classList.remove("show");
}

$("settings-btn").addEventListener("click", openSettings);
$("settings-cancel").addEventListener("click", () => $("settings-overlay").classList.remove("show"));
$("settings-save").addEventListener("click", saveSettings);
$("settings-overlay").addEventListener("click", (e) => {
  if (e.target.id === "settings-overlay") $("settings-overlay").classList.remove("show");
});

// Refresh loop: the next refresh is scheduled REFRESH_MS after the previous
// one FINISHED, and the header ring restarts at that moment — so a full ring
// always coincides with fresh data (no second, drifting clock).
const REFRESH_MS = 5000;
let refreshTimer = null;
function restartRing() {
  const ring = document.querySelector(".ring-progress");
  if (!ring) return;
  ring.style.animation = "none";
  void ring.getBoundingClientRect();          // restart the CSS animation
  ring.style.animation = `ringfill ${REFRESH_MS}ms linear forwards`;
}
async function refreshLoop() {
  clearTimeout(refreshTimer);
  await refresh();
  restartRing();
  refreshTimer = setTimeout(refreshLoop, REFRESH_MS);
}
refreshLoop();
