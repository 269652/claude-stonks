"""Configuration loading.

Precedence (highest wins): environment variables (LMTRADE_* / provider keys) >
config.toml (gitignored, per-fork user overrides) > config/default.yaml >
built-in defaults. Loading is deliberately dependency light so the bot boots
even with a bare install.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

try:
    import tomllib  # stdlib, Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - only hit on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
DEFAULT_TOML_PATH = REPO_ROOT / "config.toml"


_INLINE_COMMENT_RE = re.compile(r"\s#.*$")


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard python-dotenv dependency).
    Strips trailing ` # comment` from values (only when '#' is preceded by
    whitespace, so values legitimately containing '#' are left alone) —
    .env.example documents several keys as `KEY=            # hint`, and
    filling in the value in place must not glue the hint onto it."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = _INLINE_COMMENT_RE.sub("", value)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class ModelConfig(BaseModel):
    stack: list[str] = ["heuristic"]
    slm_model: str = "qwen2.5:1.5b"
    cloud_model: str = "claude-haiku-4-5-20251001"
    perplexity_model: str = "sonar"
    weights: dict[str, float] = Field(default_factory=lambda: {"heuristic": 1.0})
    # LLM decision model (claude_cli provider) and how it is rationed: only
    # consulted when the cheap votes' pre-screen reaches prescreen_confidence
    # (or the symbol is held), answers cached decision_cache_minutes.
    decision_model: str = "claude-opus-5-5"
    prescreen_confidence: float = 0.5
    decision_cache_minutes: int = 30
    screened_providers: list[str] = Field(default_factory=lambda: ["claude_cli", "cloud"])


class RiskConfig(BaseModel):
    max_position_fraction: float = 0.5
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.08
    min_confidence: float = 0.55
    max_fee_pct: float = 0.01   # skip entries whose fee > this fraction of notional (0 = off)


class EconomicsConfig(BaseModel):
    gpu_usd_per_hour: float = 0.20
    min_runway_hours: float = 6.0
    inference_cost: dict[str, float] = Field(
        default_factory=lambda: {"slm": 0.0002, "cloud": 0.004, "perplexity": 0.005}
    )
    # Circuit breaker: halt trading outright if net worth ever implies this
    # many multiples of the starting budget. Catches runaway valuations from
    # any bug (e.g. a corrupted mark-to-market) long before they compound,
    # independent of the root cause. 20x is generous for legitimate options
    # leverage on a small account over any realistic short timeframe.
    sanity_max_multiple: float = 20.0
    # Fraction of realized PROFIT (never losses, never principal) on each
    # winning close that's moved into a reserve excluded from future trading
    # capital. 0.5 = 50/50 reinvest/stash, 0.7 = stash 70%. Net worth and
    # alpha still count the reserve — it's not lost, just protected from
    # being re-risked.
    profit_stash_pct: float = 0.0

    @field_validator("profit_stash_pct")
    @classmethod
    def normalise_stash_pct(cls, v: float) -> float:
        """Accept either fraction (0.30) or integer percent (30) for 30%.
        Values > 1 are treated as percentages and divided by 100."""
        return v / 100.0 if v > 1.0 else v


class LoopConfig(BaseModel):
    interval_seconds: int = 60
    # Re-check exits (TP/SL/trailing) of open positions this often, between
    # symbol decisions and during the idle wait — quotes for held symbols only.
    exit_check_seconds: float = 10.0
    max_positions: int = 2
    # Cap new positions opened in a single cycle, independent of how many
    # slots are free — controls "how many trades per hour" alongside the
    # hourly cycle count. 0 = no extra cap (fill up to remaining slots, the
    # historical behavior).
    max_new_positions_per_cycle: int = 0
    # Stale eviction: close positions that have been sideways for too long to
    # free a slot for a stronger incoming signal.  A position is "stale" when
    # it has been held for at least stale_hours AND its unrealised P&L is still
    # below stale_max_profit_pct (e.g. flat or slightly negative).  Big winners
    # are never evicted — only dead weight is cleaned up.
    stale_evict_enabled: bool = True
    stale_hours: float = 48.0              # minimum hold time before stale check
    stale_max_profit_pct: float = 0.10     # evict if pnl% < this (< 10% gain)


class ResearchConfig(BaseModel):
    # Live ~300-char Claude commentary on the open positions (0 = off), using
    # the analysis provider with this cheaper model.
    commentary_minutes: int = 5
    commentary_model: str = "claude-opus-5-5"
    news_interval_minutes: int = 60          # hourly Perplexity news
    daily_analysis_interval_hours: int = 24  # daily Claude strategy review
    # "perplexity" (needs PERPLEXITY_API_KEY) or "claude_cli" (local `claude`
    # CLI, no API key — runs entirely on the local machine).
    news_provider: str = "perplexity"
    # "anthropic" (needs ANTHROPIC_API_KEY) or "claude_cli" (local `claude`
    # CLI, no API key).
    analysis_provider: str = "anthropic"
    # Model for the DAILY strategy review (independent of the cheap per-symbol
    # `model.cloud_model`); used by both the anthropic and claude_cli providers.
    analysis_model: str = "claude-opus-5-5"
    # Per-call timeout for the local `claude` CLI. A single web-search-backed
    # research/analysis call routinely takes well over a minute, so the
    # default is generous; raise it further on a slow connection.
    claude_cli_timeout_seconds: float = 180.0
    # How often the news pass runs. Successful symbols are still only refetched
    # once news_interval_minutes elapses (served from cache meanwhile), so this
    # shorter cadence just lets FAILED/corrupted symbols retry sooner instead of
    # waiting the full hour.
    news_retry_minutes: int = 10


class EntryConfig(BaseModel):
    """Watch-list entry timing (core/watchlist.py). Off in code so callers that
    build Settings() directly keep immediate entries; config/default.yaml
    turns it on for the running bot."""
    watch_enabled: bool = False
    watch_k: float = 1.0           # enter at >= k sigma from the intraday SMA
    watch_window: int = 60         # bars of history for the SMA / std
    watch_max_hours: float = 4.0   # drop the watch entry after this long
    # Exchange-side entries: a resting limit order at the trigger price instead
    # of a market order once triggered (core/limit_orders.py). Re-priced when
    # the target drifts more than reprice_drift (fraction).
    limit_orders: bool = False
    reprice_drift: float = 0.005
    # Signals at or above this confidence skip the timing and are entered at
    # the current price (limit mode: a marketable limit + instant_slippage).
    instant_confidence: float = 1.0
    instant_slippage: float = 0.005
    # Volatile symbols (intraday sigma >= this % of price) get no resting limit
    # order — they are entered with a market order once triggered. 0 = off.
    market_above_sigma_pct: float = 0.0


class OptionsConfig(BaseModel):
    enabled: bool = True
    expiry_days: float = 7.0
    max_option_fraction: float = 0.3   # max fraction of equity in one premium
    take_profit_pct: float = 0.5       # +50% premium -> take profit
    stop_loss_pct: float = 0.4         # -40% premium -> cut
    # Trailing take-profit on net EUR profit (after entry + exit fee): arms at
    # trail_min_profit_eur / (1 - trail_pct), then closes when profit falls
    # trail_pct below its peak — never below trail_min_profit_eur.
    # Close a held position when the fused signal for its underlying points
    # against it with at least this confidence (0 = off).
    flip_exit_confidence: float = 0.8
    trail_enabled: bool = False
    trail_min_profit_eur: float = 20.0
    trail_pct: float = 0.15
    min_hours_to_expiry: float = 24.0  # force-close inside this window
    min_hold_hours: float = 0.0        # block TP/SL exits before this many
                                       # hours have passed (0 = no minimum).
                                       # The expiry force-close is a hard
                                       # constraint of the option itself and
                                       # always overrides this.
    max_hold_hours: float = 0.0        # force-close after this many hours
                                       # regardless of TP/SL (0 = disabled,
                                       # relies on expiry_days instead)
    # Cash reserve: keep at least this fraction of net worth in cash rather
    # than deploying everything into positions.  As the portfolio grows the
    # deployable bucket grows proportionally, so larger positions are
    # automatically available without any manual configuration change.
    # Accepts either a fraction (0.40) or an integer percent (40 → 0.40).
    cash_reserve_pct: float = 0.40

    @field_validator("cash_reserve_pct")
    @classmethod
    def normalise_cash_reserve_pct(cls, v: float) -> float:
        """Accept either fraction (0.40) or integer percent (40) for 40%."""
        return v / 100.0 if v > 1.0 else v


class LearningConfig(BaseModel):
    enabled: bool = True
    population: int = 14   # ~2 genomes per strategy family
    epsilon: float = 0.2
    mutation_scale: float = 0.3
    evolve_every_trades: int = 10      # run evolution after N closed trades


class BudgetConfig(BaseModel):
    """Cash buckets (core/budget.py), EUR. reserve_eur is never touched by the
    bot (manual actions); dips_eur only funds dip buys; entries get the rest.
    Zero in code; config/default.yaml sets the running bot's amounts."""
    reserve_eur: float = 0.0
    dips_eur: float = 0.0


class DipConfig(BaseModel):
    """Dip buys: add to a held, losing position when its underlying is
    >= k_sigma below the intraday SMA (above, for shorts) and the fused
    signal still backs it with >= min_confidence."""
    enabled: bool = False
    k_sigma: float = 1.5
    min_confidence: float = 0.6
    # Dip buys are small (<= dips_eur): the 1 EUR fee is a bigger share.
    max_fee_pct: float = 0.03
    max_adds: int = 1                  # per position
    min_barrier_distance: float = 0.10  # knock-outs: spot at least 10% from the barrier


class SignoffConfig(BaseModel):
    """Claude sign-off for automatic live orders (core/signoff.py). Off in
    code; config/default.yaml turns it on for the running bot."""
    enabled: bool = False
    model: str = "claude-opus-5-5"
    timeout_seconds: float = 120.0


class TRConfig(BaseModel):
    # Use real Trade Republic knockout certificates (real ISINs) for paper
    # trading when TR credentials are configured. Strictly optional: without
    # TR_PHONE/TR_PIN in the environment this is inert and the synthetic
    # options layer is used. Login is via pytr (unofficial, against TR ToS —
    # see docs/TRADE_REPUBLIC.md).
    use_derivatives: bool = True
    # Check every N seconds that TR still answers; re-login immediately when
    # it doesn't. 0 = off.
    heartbeat_seconds: float = 30.0
    # Trading hours of the certificates' venue (Mon-Fri, Europe/Berlin).
    # Outside them armed live exits wait and the dashboard disables closing.
    # "" = always open.
    market_hours: str = "08:00-22:00"
    target_leverage: float = 5.0    # preferred KO leverage when selecting


class SizingConfig(BaseModel):
    # Fractional Kelly from each genome's empirical win/loss record. Full
    # Kelly is notoriously too aggressive for estimated edges — 0.5 ("half
    # Kelly") is the standard practitioner compromise.
    kelly_enabled: bool = True
    kelly_fraction_of_full: float = 0.5
    kelly_min_trades: int = 5       # fall back to confidence sizing below this
    # Volatility targeting: scale size down when realized vol exceeds target.
    vol_target_annual: float = 0.20
    # Regime filter: in a vol-spike ("storm") regime demand extra conviction
    # and halve size — measured edges break first when volatility regimes flip.
    regime_filter_enabled: bool = True
    storm_extra_confidence: float = 0.10
    storm_size_factor: float = 0.5


class BenchmarkConfig(BaseModel):
    symbol: str = "SPY"                # "retail baseline": buy-and-hold SPY


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class DataConfig(BaseModel):
    provider: str = "auto"
    intraday: bool = True     # 1m bars (Yahoo, via httpx) / 10s synthetic ticks


class Settings(BaseModel):
    mode: str = "paper"
    budget: float = 100.0
    currency: str = "EUR"
    universe: list[str] = ["AAPL", "MSFT", "SPY"]
    loop: LoopConfig = LoopConfig()
    model: ModelConfig = ModelConfig()
    risk: RiskConfig = RiskConfig()
    economics: EconomicsConfig = EconomicsConfig()
    research: ResearchConfig = ResearchConfig()
    options: OptionsConfig = OptionsConfig()
    entry: EntryConfig = EntryConfig()
    learning: LearningConfig = LearningConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    tr: TRConfig = TRConfig()
    signoff: SignoffConfig = SignoffConfig()
    budget_split: BudgetConfig = BudgetConfig()
    dips: DipConfig = DipConfig()
    sizing: SizingConfig = SizingConfig()
    web: WebConfig = WebConfig()
    data: DataConfig = DataConfig()

    # runtime data directory (state db, logs)
    data_dir: Path = REPO_ROOT / "data"

    @property
    def db_path(self) -> Path:
        """Paper-trading book (default). Fully separate from the live book."""
        return self.data_dir / "lmtrade.db"

    @property
    def live_db_path(self) -> Path:
        """Live-trading book — real fills only. Kept in a separate database so
        paper and live ledgers, trade history and activity never mix."""
        return self.data_dir / "lmtrade-live.db"

    def book_db_path(self, book: str) -> Path:
        return self.live_db_path if book == "live" else self.db_path

    @property
    def control_path(self) -> Path:
        """Persisted paper/live + armed control state shared by engine & web."""
        return self.data_dir / "control.json"


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    def env(name: str) -> str | None:
        v = os.environ.get(name)
        return v if v not in (None, "") else None

    if v := env("LMTRADE_MODE"):
        cfg["mode"] = v
    if v := env("LMTRADE_BUDGET"):
        cfg["budget"] = float(v)
    if v := env("LMTRADE_UNIVERSE"):
        cfg["universe"] = [s.strip() for s in v.split(",") if s.strip()]
    if v := env("LMTRADE_MODEL_STACK"):
        cfg.setdefault("model", {})["stack"] = [s.strip() for s in v.split(",") if s.strip()]
    if v := env("LMTRADE_SLM_MODEL"):
        cfg.setdefault("model", {})["slm_model"] = v
    if v := env("LMTRADE_CLOUD_MODEL"):
        cfg.setdefault("model", {})["cloud_model"] = v
    if v := env("LMTRADE_PERPLEXITY_MODEL"):
        cfg.setdefault("model", {})["perplexity_model"] = v
    if v := env("LMTRADE_NEWS_PROVIDER"):
        cfg.setdefault("research", {})["news_provider"] = v
    if v := env("LMTRADE_ANALYSIS_PROVIDER"):
        cfg.setdefault("research", {})["analysis_provider"] = v
    if v := env("LMTRADE_ANALYSIS_MODEL"):
        cfg.setdefault("research", {})["analysis_model"] = v
    if v := env("LMTRADE_GPU_USD_PER_HOUR"):
        cfg.setdefault("economics", {})["gpu_usd_per_hour"] = float(v)
    if v := env("LMTRADE_MIN_RUNWAY_HOURS"):
        cfg.setdefault("economics", {})["min_runway_hours"] = float(v)
    if v := env("LMTRADE_WEB_HOST"):
        cfg.setdefault("web", {})["host"] = v
    if v := env("LMTRADE_WEB_PORT"):
        cfg.setdefault("web", {})["port"] = int(v)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` onto `base`, recursing into nested dicts so e.g. a
    toml [options] section overrides only the keys it specifies, not the
    whole options block."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml_overrides(path: Path) -> dict[str, Any]:
    """Load the optional, gitignored config.toml override file. Never
    raises: a malformed file is logged and ignored rather than crashing the
    bot, matching the project's degrade-gracefully convention."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("lmtrade").warning(
            "Ignoring malformed %s: %s", path, exc)
        return {}


def load_settings(
    config_path: Path | None = None, load_env: bool = True,
    toml_path: Path | None = None,
) -> Settings:
    """Build a Settings object from yaml + config.toml + environment overrides."""
    if load_env:
        _load_dotenv(REPO_ROOT / ".env")

    path = config_path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    toml_overrides = _load_toml_overrides(toml_path or DEFAULT_TOML_PATH)
    raw = _deep_merge(raw, toml_overrides)

    raw = _apply_env_overrides(raw)
    settings = Settings(**raw)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


# Provider secrets pulled straight from the environment (never persisted).
def secret(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None
