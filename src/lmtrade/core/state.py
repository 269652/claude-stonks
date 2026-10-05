"""SQLite-backed state store shared by the engine and the web dashboard.

Everything the dashboard shows (trades, positions, logs, activity feed, equity
and cost history) is persisted here so the web process can read a consistent
snapshot without touching the engine's memory.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL,          -- buy | sell
    qty        REAL NOT NULL,
    price      REAL NOT NULL,
    fee        REAL NOT NULL DEFAULT 0,
    mode       TEXT NOT NULL,          -- paper | live
    reason     TEXT,                   -- fused rationale
    confidence REAL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol    TEXT PRIMARY KEY,
    qty       REAL NOT NULL,
    avg_price REAL NOT NULL,
    opened_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    source  TEXT,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activity (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    kind    TEXT NOT NULL,             -- decision | signal | trade | risk | economics | system
    symbol  TEXT,
    summary TEXT NOT NULL,
    detail  TEXT                       -- JSON blob
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts     REAL PRIMARY KEY,
    cash   REAL NOT NULL,
    equity REAL NOT NULL,             -- cash + mark-to-market positions
    fees   REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS costs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,           -- gpu | inference | fee
    provider TEXT,
    amount   REAL NOT NULL            -- USD
);
CREATE TABLE IF NOT EXISTS option_positions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    underlying    TEXT NOT NULL,
    kind          TEXT NOT NULL,      -- call | put
    strike        REAL NOT NULL,
    expiry_ts     REAL NOT NULL,
    iv            REAL NOT NULL,
    contracts     REAL NOT NULL,
    entry_premium REAL NOT NULL,
    opened_ts     REAL NOT NULL,
    genome_id     TEXT,               -- learning attribution
    status        TEXT NOT NULL DEFAULT 'open',   -- open | closed
    exit_premium  REAL,
    closed_ts     REAL,
    pnl           REAL,
    tp_premium    REAL,               -- explicit take-profit level, set at open
    sl_premium    REAL,               -- explicit stop-loss level, set at open
    instrument_type TEXT DEFAULT 'option',  -- option | knockout
    barrier       REAL,               -- KO barrier (knockouts only)
    ratio         REAL,               -- KO subscription ratio (knockouts only)
    isin          TEXT                -- real TR instrument ISIN when available
);
CREATE TABLE IF NOT EXISTS news (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    symbol    TEXT NOT NULL,
    text      TEXT NOT NULL,
    sentiment TEXT                    -- bullish | bearish | neutral
);
CREATE TABLE IF NOT EXISTS benchmark_curve (
    ts     REAL PRIMARY KEY,
    equity REAL NOT NULL             -- buy-and-hold benchmark equity
);
"""


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    opened_ts: float


@dataclass
class Trade:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0
    mode: str = "paper"
    reason: str | None = None
    confidence: float | None = None
    ts: float = field(default_factory=time.time)


class Store:
    """Thread-safe SQLite wrapper. One connection guarded by a lock — fine for
    the modest write volume of a small bot, and keeps the web read path simple."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """In-place migrations for DBs created before schema additions
        (CREATE TABLE IF NOT EXISTS won't add columns to existing tables —
        the persisted bot-state DB outlives code changes)."""
        cols = {r["name"] for r in
                self._conn.execute("PRAGMA table_info(option_positions)")}
        for col, decl in (("tp_premium", "REAL"), ("sl_premium", "REAL"),
                          ("instrument_type", "TEXT DEFAULT 'option'"),
                          ("barrier", "REAL"), ("ratio", "REAL"),
                          ("isin", "TEXT")):
            if col not in cols:
                self._conn.execute(
                    f"ALTER TABLE option_positions ADD COLUMN {col} {decl}")

    # -- meta -----------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # -- logs -----------------------------------------------------------------
    def add_log(self, level: str, message: str, source: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO logs(ts,level,source,message) VALUES(?,?,?,?)",
                (time.time(), level, source, message),
            )
            self._conn.commit()

    def recent_logs(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- activity -------------------------------------------------------------
    def add_activity(
        self, kind: str, summary: str, symbol: str | None = None, detail: dict | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO activity(ts,kind,symbol,summary,detail) VALUES(?,?,?,?,?)",
                (time.time(), kind, symbol, summary, json.dumps(detail or {})),
            )
            self._conn.commit()

    def recent_activity(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d["detail"]) if d["detail"] else {}
            except json.JSONDecodeError:
                d["detail"] = {}
            out.append(d)
        return out

    # -- trades ---------------------------------------------------------------
    def record_trade(self, trade: Trade) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO trades(ts,symbol,side,qty,price,fee,mode,reason,confidence) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    trade.ts, trade.symbol, trade.side, trade.qty, trade.price,
                    trade.fee, trade.mode, trade.reason, trade.confidence,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_trades(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def trades_since(self, last_id: int) -> list[dict]:
        """Trades with id > last_id, ascending — for incremental ledger export."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE id > ? ORDER BY id ASC", (last_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- reserve (profit stash) ------------------------------------------------
    def add_reserve(self, amount: float) -> None:
        self.set_meta("reserve_eur", self.reserve_balance() + amount)

    def reserve_balance(self) -> float:
        return float(self.get_meta("reserve_eur", 0.0))

    # -- positions ------------------------------------------------------------
    def upsert_position(self, pos: Position) -> None:
        with self._lock:
            if pos.qty <= 1e-9:
                self._conn.execute("DELETE FROM positions WHERE symbol=?", (pos.symbol,))
            else:
                self._conn.execute(
                    "INSERT INTO positions(symbol,qty,avg_price,opened_ts) VALUES(?,?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, "
                    "avg_price=excluded.avg_price",
                    (pos.symbol, pos.qty, pos.avg_price, pos.opened_ts),
                )
            self._conn.commit()

    def positions(self) -> list[Position]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM positions").fetchall()
        return [Position(**dict(r)) for r in rows]

    def position(self, symbol: str) -> Position | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM positions WHERE symbol=?", (symbol,)
            ).fetchone()
        return Position(**dict(row)) if row else None

    # -- equity & costs -------------------------------------------------------
    def record_equity(self, cash: float, equity: float, fees: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO equity_curve(ts,cash,equity,fees) VALUES(?,?,?,?) "
                "ON CONFLICT(ts) DO NOTHING",
                (time.time(), cash, equity, fees),
            )
            self._conn.commit()

    def equity_curve(self, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM equity_curve ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def record_cost(self, kind: str, amount: float, provider: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO costs(ts,kind,provider,amount) VALUES(?,?,?,?)",
                (time.time(), kind, provider, amount),
            )
            self._conn.commit()

    def total_costs(self) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, SUM(amount) AS total FROM costs GROUP BY kind"
            ).fetchall()
        return {r["kind"]: r["total"] for r in rows}

    # -- option positions -------------------------------------------------------
    def open_option(
        self, underlying: str, kind: str, strike: float, expiry_ts: float,
        iv: float, contracts: float, entry_premium: float, genome_id: str | None,
        tp_premium: float | None = None, sl_premium: float | None = None,
        instrument_type: str = "option", barrier: float | None = None,
        ratio: float | None = None, isin: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO option_positions(underlying,kind,strike,expiry_ts,iv,"
                "contracts,entry_premium,opened_ts,genome_id,tp_premium,sl_premium,"
                "instrument_type,barrier,ratio,isin) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (underlying, kind, strike, expiry_ts, iv, contracts, entry_premium,
                 time.time(), genome_id, tp_premium, sl_premium,
                 instrument_type, barrier, ratio, isin),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_to_option(self, opt_id: int, contracts: float, price: float) -> None:
        """Average into an open position: more contracts at `price`, entry
        premium becomes the size-weighted average."""
        with self._lock:
            self._conn.execute(
                "UPDATE option_positions SET entry_premium="
                "(entry_premium*contracts + ?*?)/(contracts + ?), contracts=contracts + ? "
                "WHERE id=? AND status='open'",
                (price, contracts, contracts, contracts, opt_id),
            )
            self._conn.commit()

    def close_option(self, opt_id: int, exit_premium: float, pnl: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE option_positions SET status='closed', exit_premium=?, "
                "closed_ts=?, pnl=? WHERE id=?",
                (exit_premium, time.time(), pnl, opt_id),
            )
            self._conn.commit()

    def open_options(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM option_positions WHERE status='open'"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_option(self, opt_id: int) -> None:
        """Remove an option row entirely — for phantom positions found during
        live-book reconciliation (never a real fill, so closing it would
        pollute realized P&L)."""
        with self._lock:
            self._conn.execute("DELETE FROM option_positions WHERE id=?", (opt_id,))
            self._conn.commit()

    def closed_options_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM option_positions WHERE status='closed'"
            ).fetchone()
        return int(row["c"])

    def closed_options(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM option_positions WHERE status='closed' "
                "ORDER BY closed_ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- news cache -------------------------------------------------------------
    def add_news(
        self, symbol: str, text: str, sentiment: str | None, ts: float | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO news(ts,symbol,text,sentiment) VALUES(?,?,?,?)",
                (ts if ts is not None else time.time(), symbol, text, sentiment),
            )
            self._conn.commit()

    def latest_news(self, symbol: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM news WHERE symbol=? ORDER BY ts DESC LIMIT 1", (symbol,)
            ).fetchone()
        return dict(row) if row else None

    def recent_news(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM news ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- benchmark ---------------------------------------------------------------
    def record_benchmark(self, equity: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO benchmark_curve(ts,equity) VALUES(?,?) "
                "ON CONFLICT(ts) DO NOTHING", (time.time(), equity),
            )
            self._conn.commit()

    def benchmark_curve(self, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM benchmark_curve ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
