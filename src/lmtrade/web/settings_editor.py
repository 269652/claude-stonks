"""Read/write the small whitelist of user-tunable settings the dashboard
settings overlay exposes, syncing them to the gitignored config.toml (the
per-fork override file). Only whitelisted keys are ever written, and a
malformed value for one field never blocks the others.

config.toml is a plain override file (comments aren't preserved on write);
config.example.toml stays the annotated reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

# (dotted path, kind, options). kind: number|int|bool|csv|select
EDITABLE: list[tuple[str, str, list[str] | None]] = [
    ("budget", "number", None),
    ("universe", "csv", None),
    ("model.stack", "csv", None),
    ("model.decision_model", "select",
     ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"]),
    ("model.prescreen_confidence", "number", None),
    ("model.decision_cache_minutes", "int", None),
    ("loop.interval_seconds", "int", None),
    ("loop.max_positions", "int", None),
    ("loop.max_new_positions_per_cycle", "int", None),
    ("loop.stale_evict_enabled", "bool", None),
    ("loop.stale_hours", "number", None),
    ("loop.stale_max_profit_pct", "number", None),
    ("options.take_profit_pct", "number", None),
    ("options.stop_loss_pct", "number", None),
    ("options.flip_exit_confidence", "number", None),
    ("options.trail_enabled", "bool", None),
    ("options.trail_min_profit_eur", "number", None),
    ("options.trail_pct", "number", None),
    ("options.max_option_fraction", "number", None),
    ("options.cash_reserve_pct", "number", None),
    ("options.min_hold_hours", "number", None),
    ("options.max_hold_hours", "number", None),
    ("options.expiry_days", "number", None),
    ("risk.min_confidence", "number", None),
    ("entry.watch_enabled", "bool", None),
    ("entry.watch_k", "number", None),
    ("entry.watch_window", "int", None),
    ("entry.watch_max_hours", "number", None),
    ("entry.limit_orders", "bool", None),
    ("entry.reprice_drift", "number", None),
    ("entry.instant_confidence", "number", None),
    ("entry.instant_slippage", "number", None),
    ("entry.market_above_sigma_pct", "number", None),
    ("economics.profit_stash_pct", "number", None),
    ("economics.gpu_usd_per_hour", "number", None),
    ("economics.min_runway_hours", "number", None),
    ("economics.sanity_max_multiple", "number", None),
    ("research.news_provider", "select", ["perplexity", "claude_cli"]),
    ("research.analysis_provider", "select", ["anthropic", "claude_cli"]),
    ("research.news_interval_minutes", "int", None),
    ("research.commentary_minutes", "int", None),
    ("research.commentary_model", "select",
     ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"]),
    ("research.daily_analysis_interval_hours", "int", None),
    ("research.news_retry_minutes", "int", None),
    ("research.claude_cli_timeout_seconds", "number", None),
    ("tr.use_derivatives", "bool", None),
    ("tr.heartbeat_seconds", "number", None),
    ("tr.market_hours", "text", None),
    ("budget_split.reserve_eur", "number", None),
    ("budget_split.dips_eur", "number", None),
    ("dips.enabled", "bool", None),
    ("dips.k_sigma", "number", None),
    ("dips.min_confidence", "number", None),
    ("dips.max_fee_pct", "number", None),
    ("dips.max_adds", "int", None),
    ("dips.min_barrier_distance", "number", None),
    ("signoff.enabled", "bool", None),
    ("signoff.model", "select",
     ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"]),
    ("signoff.timeout_seconds", "number", None),
    ("tr.target_leverage", "number", None),
    ("data.provider", "select", ["auto", "yahoo", "synthetic"]),
]
_KINDS = {path: (kind, opts) for path, kind, opts in EDITABLE}


def _get(settings: Any, path: str) -> Any:
    obj = settings
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def schema(settings: Any) -> list[dict]:
    """Current value + type metadata for each editable field."""
    out = []
    for path, kind, opts in EDITABLE:
        section, _, key = path.rpartition(".")
        val = _get(settings, path)
        if kind == "csv" and isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        out.append({"key": path, "section": section or "general", "field": key,
                    "kind": kind, "options": opts, "value": val})
    return out


def _coerce(kind: str, raw: Any) -> Any:
    if kind == "bool":
        return raw is True or str(raw).lower() in ("true", "1", "yes", "on")
    if kind == "int":
        return int(float(raw))
    if kind == "number":
        f = float(raw)
        return int(f) if f.is_integer() else f
    if kind == "csv":
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        return [s.strip() for s in str(raw).split(",") if s.strip()]
    return str(raw)


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _to_toml(data: dict, prefix: str = "") -> str:
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines = [f"{k} = {_toml_scalar(v)}" for k, v in scalars.items()]
    for k, v in tables.items():
        name = f"{prefix}{k}"
        lines.append("")
        lines.append(f"[{name}]")
        body = _to_toml(v, prefix=name + ".")
        if body:
            lines.append(body)
    return "\n".join(lines)


def apply_changes(toml_path: Path, changes: dict) -> dict:
    """Merge whitelisted changes into config.toml. Returns {applied, rejected}.
    Never raises on a single bad value — it's rejected and the rest proceed."""
    data: dict = {}
    if toml_path.exists():
        try:
            with toml_path.open("rb") as f:
                data = tomllib.load(f)
        except Exception:  # noqa: BLE001 — start clean rather than fail
            data = {}

    applied: dict[str, Any] = {}
    rejected: list[str] = []
    for path, raw in (changes or {}).items():
        if path not in _KINDS:
            rejected.append(path)
            continue
        kind, opts = _KINDS[path]
        try:
            val = _coerce(kind, raw)
        except (TypeError, ValueError):
            rejected.append(path)
            continue
        if opts is not None and str(val) not in opts:
            rejected.append(path)
            continue
        section, _, key = path.rpartition(".")
        node = data
        for part in section.split(".") if section else []:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # a scalar sat where a table must go
                rejected.append(path)
                node = None
                break
        if node is None:
            continue
        node[key] = val
        applied[path] = val

    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(_to_toml(data) + "\n")
    return {"applied": applied, "rejected": rejected}
