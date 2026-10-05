# ⚡ LMTrade

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/269652/LMTrade/blob/claude/trading-bot-hybrid-agent-tvs90n/notebooks/LMTrade_Colab.ipynb)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

![Paper trading dashboard](https://raw.githubusercontent.com/269652/LMTrade/bot-state/state/dashboard.png)

A trading bot that fuses **LLMs, SLMs and classical financial models** into one
decision, built around the goal of **paying for its own GPU costs**. Whether it
reaches that goal is measured by the bot itself — it has not been demonstrated.

It runs **locally**. News, the daily market analysis and LLM trading votes go
through the **Claude CLI** (your Claude subscription, no API key) or, optionally,
the Anthropic / Perplexity APIs; a small local model can be added via Ollama. A
cost-accounting core measures compute spend (an hourly GPU rate if you rent
one, plus inference) against realised P&L and **halts new entries when the
runway runs out**.

> **Safety first.** LMTrade runs in **paper mode** by default (simulated fills,
> no real money). Live Trade Republic execution is an explicit, guarded opt-in
> through the dashboard — see [`docs/TRADE_REPUBLIC.md`](docs/TRADE_REPUBLIC.md)
> for the risks and for what has (and hasn't) been verified against a real account.

## Fork this and run your own

1. **Fork the repo** on GitHub.
2. **Open a Claude Code project** and point it at your fork.
3. Say **"Setup the project."** Claude follows [`SETUP.md`](SETUP.md): asks
   your configuration (budget, universe, position limits, hold times, TP/SL,
   profit-stash ratio, GPU rate, API keys), writes secrets to a gitignored
   `.env` and everything else to a gitignored `config.toml`, installs, runs
   the test suite, and optionally sets up the permanent hourly Routine.

Nothing you configure is committed — `.env` and `config.toml` are both
gitignored. `.env.example` and [`config.example.toml`](config.example.toml)
document every available setting.

---

## What's in the box

| Layer | Module | What it does |
|-------|--------|--------------|
| **CLI** | `lmtrade.cli` | `run`, `backtest`, `web`, `viz`, `status`, `analyze`, `reset`, `config`, `import-news`, `import-analysis`, `export-ledger` |
| **Backtest** | `backtest.walk_forward` | Walk-forward folds over historical bars; pre-trains the genome population |
| **Web dashboard** | `lmtrade.web` | Positions (price, P&L, exits, close), watchlist, signals, news, analysis, equity curve, trades, logs, settings |
| **Inline viz** | `lmtrade.viz` | Notebook-native matplotlib dashboard (Colab/Jupyter) + `lmtrade viz` PNG |
| **Engine** | `core.engine` | High-cadence loop: research jobs → data → economics gate → fusion+strategy → options/equity execution |
| **Options** | `finance.options` | Black-Scholes pricing/greeks, synthetic near-ATM chain, mark-to-market |
| **Learning** | `strategies.*` | Momentum/mean-reversion/breakout genomes; evolutionary optimizer driven by realized paper P&L |
| **Research** | `research.*` | Hourly Perplexity news (cached) + daily Claude strategy review with clamped parameter updates |
| **Benchmark** | `core.engine` | Live alpha vs buy-and-hold SPY — the "retail baseline" |
| **Fusion** | `agents.fusion` | Weighted vote: financial models + SLM + LLM + news sentiment + learned strategy |
| **Models** | `models.providers` | Heuristic (financial), local SLM (Ollama), cloud LLM, Perplexity research |
| **Finance** | `finance.*` | SMA/EMA/RSI/MACD indicators, position sizing, stop-loss/take-profit |
| **Brokers** | `brokers.*` | `PaperBroker` (default) + Trade Republic adapter (knockouts, limit/market orders, armed from the dashboard) |
| **Economics** | `economics.cost_accounting` | GPU burn accrual, runway, self-sustaining check, spend guardrail |

## Trading mechanics

What the engine does each cycle, all configurable in `config/default.yaml`
(overridable in `config.toml` or the dashboard's ⚙ settings):

- **Fees in every number.** Trade Republic's flat €1 per order is charged on
  entry and exit. Position P&L is net of the entry fee; realized P&L of both;
  net worth is cash + positions after the exit fee each would still cost.
- **Fee guard.** Entries whose fee would exceed `risk.max_fee_pct` (default 1%)
  of the order are skipped — i.e. no order below €100 at the defaults. Small
  signals are raised to that minimum when the position and cash caps allow it.
- **Watch-list timing.** A signal isn't bought at once: it waits until the
  price is `entry.watch_k` standard deviations from the intraday average in its
  favour (below for buys, above for sells), up to `watch_max_hours`. Only
  symbols with a matching Trade Republic knockout are watched when TR is
  connected. Signals at or above `instant_confidence` skip the wait.
- **Limit entries** (`entry.limit_orders`). A watched signal gets a resting
  limit order at the trigger price, re-priced when the target drifts, cancelled
  when the signal goes away. Volatile symbols (`market_above_sigma_pct`) use a
  market order on trigger instead. Paper mode simulates the fills.
- **Exits.** Take-profit / stop-loss per position, a trailing take-profit with a
  minimum-profit floor (`options.trail_*`: arms at min ÷ (1 − pct), closes pct
  below the peak profit — not guaranteed, see below), stale-position eviction, and a manual Close button.
  TP / SL / trailing can be edited per position. Open positions are re-checked
  every `loop.exit_check_seconds` (default 10s); a fast move between two checks
  can still close below a level.
- **Profit stash.** `economics.profit_stash_pct` of each realized profit is
  moved out of tradeable cash (still counted in net worth).

These rules are covered by the offline test suite. None of them has been shown
to make money.

## The self-sustaining loop

```
 compute ($/hr, optional) ──► Claude CLI / LLM APIs / SLM ──────► decisions ──► trades
        ▲                                                                            │
        │                                                                            ▼
   runway guardrail ◄──── net worth vs (GPU + inference + fees) ◄──── realised P&L
```

Every cycle the engine computes **runway** = spare net worth ÷ GPU $/hr. If it
drops below the configured floor (`min_runway_hours`), new entries are halted and
only exits are managed, so the bot stops opening positions it would have to fund
from a shrinking account. It reports itself **self-sustaining** once cumulative P&L covers all
GPU + inference spend.

## Quick start

```bash
# 1. Install (paper mode needs no keys, no GPU, no network)
pip install -e .

# 2. Run the engine for a few cycles (synthetic data fallback works offline)
lmtrade run --cycles 5 --interval 1

# 3. See where you stand
lmtrade status

# 4. Launch the dashboard
lmtrade web            # → http://localhost:8000
```

Live market data (real prices) works out of the box — no extra install, no API
key. `data.provider: auto` fetches closes directly from Yahoo Finance's public
chart endpoint via `httpx` and falls back to synthetic data on any network
hiccup. (Deliberately not the `yfinance` PyPI package: its `curl_cffi` backend
does browser TLS-fingerprint impersonation, which breaks under some sandboxed
/ proxied network setups — plain `httpx` against the same Yahoo endpoint does
not have that problem.)

Optional model providers:

```bash
cp .env.example .env              # add OLLAMA_HOST, ANTHROPIC_API_KEY, PERPLEXITY_API_KEY…
```

## Local installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | 3.11+ recommended (ships `tomllib`) |
| **Git** | clone + the `bot-state` branch |
| **pip** | any recent version |

### 1 — Clone and install

```bash
git clone https://github.com/269652/LMTrade.git
cd LMTrade
pip install -e '.[dev]'    # installs lmtrade + pytest/ruff for local dev
```

Paper mode with synthetic data needs nothing else — run it right away:

```bash
lmtrade run --cycles 5 --interval 1   # fully offline
lmtrade web                            # → http://localhost:8000
```

### 2 — Copy and edit the config files

```bash
cp .env.example .env              # secrets — gitignored, never committed
cp config.example.toml config.toml    # non-secret settings — gitignored
```

Edit `.env` with the providers you want to enable (all optional, see below).
`lmtrade config` prints the fully resolved settings so you can confirm what took effect.

### 3 — Optional providers

#### Claude subscription (recommended — fully local, no API key)

Install the [Claude CLI](https://claude.ai/code) and log in once:

```bash
# macOS / Linux
curl -fsSL https://claude.ai/install.sh | sh
claude auth login          # one-time browser login

# Windows — use the installer at https://claude.ai/code
```

Then tell LMTrade to use it in `config.toml`:

```toml
[model]
stack = ["heuristic", "slm", "claude_cli"]
[research]
news_provider    = "claude_cli"
analysis_provider = "claude_cli"
```

All LLM calls (news, daily analysis, trading decisions) now go through your
Claude subscription — no `ANTHROPIC_API_KEY` or `PERPLEXITY_API_KEY` needed.

#### Anthropic / Perplexity API keys

For cloud LLM and web-search-grounded news without the local CLI:

```bash
# in .env
ANTHROPIC_API_KEY=sk-ant-...
PERPLEXITY_API_KEY=pplx-...
```

#### Local SLM via Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:1.5b          # T4-sized model (~1 GB)
# For an A100:  ollama pull qwen2.5:7b

# Windows — download the installer at https://ollama.ai
```

Add to `.env`:

```bash
OLLAMA_HOST=http://localhost:11434
LMTRADE_SLM_MODEL=qwen2.5:1.5b
```

#### Trade Republic (real instruments, optional live execution)

With TR credentials the bot uses **real TR knockout certificates** (real ISINs,
priced from TR's live quotes) for its trades. In paper mode nothing is sent to
TR. Real orders are only placed after you switch the dashboard to **LIVE** and
arm live execution. Without credentials you get synthetic Black-Scholes options.

1. Install the extra:
   ```bash
   pip install 'lmtrade[traderepublic]'
   ```
2. Add to `.env`:
   ```bash
   TR_PHONE=+49123456789   # your TR account phone number (exact format matters)
   TR_PIN=1234             # your TR app PIN (4 digits)
   ```
3. Run the **one-time 2FA pairing** (only needed once per machine):
   ```bash
   pytr login -n "+49123456789" -p "1234" --store_credentials
   ```
   > **Important:** The phone number here must match `TR_PHONE` in `.env`
   > **character for character** — `pytr` names the session cookie file after it.
   > A spacing or prefix mismatch silently breaks session resumption every time
   > the bot starts. `--store_credentials` is mandatory; without it `pytr`
   > completes the 2FA for that one process and writes nothing to disk.
4. Enable real instruments in `config.toml`:
   ```toml
   [tr]
   use_derivatives = true
   ```

> ⚠️ **Live execution** (real orders) is against TR's Terms of Service and only
> partly verified against a real account. Read
> [`docs/TRADE_REPUBLIC.md`](docs/TRADE_REPUBLIC.md) before arming it.

### 4 — Verify

```bash
python -m pytest            # full suite must be green
lmtrade config              # shows the fully resolved settings
lmtrade status              # portfolio + economics snapshot
lmtrade web                 # → http://localhost:8000
```

### 5 — Running permanently

> **Quickest path:** say **"Setup the project"** in a [Claude Code](https://claude.ai/code)
> session pointed at this repo — the guided flow asks all your settings, writes
> both config files, installs, runs the test suite, and offers to set up the
> permanent hourly Routine for you.

See the [**Running permanently**](#running-permanently-hourly-claude-routine) section below.

## Run on Google Colab

Click the **Open in Colab** badge above (or open
[`notebooks/LMTrade_Colab.ipynb`](notebooks/LMTrade_Colab.ipynb)). Colab gives you
a free GPU to serve the SLM; the notebook installs everything, optionally runs
Ollama, and can persist state to Google Drive.

Because Colab can't reliably expose the web server's port, the dashboard there is
rendered **inline with matplotlib** rather than served over HTTP:

```python
import lmtrade.viz as viz
viz.show()                 # one-shot dashboard figure
viz.live(interval=5)       # auto-refreshing dashboard
viz.trades_df(); viz.activity_df(); viz.positions_df()   # feeds as DataFrames
```

The same panels render anywhere matplotlib works; `lmtrade viz` writes them to a
PNG for headless use. Note Colab is for **testing** — its runtime is ephemeral and
idles out, so a long-running bot belongs on your own machine.

## Configuration

Three layers, highest wins: **environment variables** (`LMTRADE_*`, see
[`.env.example`](.env.example)) > **`config.toml`** (gitignored, per-fork
overrides — copy from [`config.example.toml`](config.example.toml)) >
**[`config/default.yaml`](config/default.yaml)** (committed baseline).

If you forked this repo, say **"Setup the project"** in a Claude Code session
pointed at it — see [`SETUP.md`](SETUP.md) for the guided flow, which asks
your settings and writes both files for you. Manually, `config.toml` takes any
key from `config/default.yaml`, e.g.:

```toml
budget = 250.0
[options]
take_profit_pct = 0.4
min_hold_hours = 2.0        # don't exit before a position has been open 2h
[economics]
profit_stash_pct = 0.5      # stash 50% of every winning trade's profit —
                            # protected from being re-risked, still counted
                            # in net worth/alpha
[loop]
max_new_positions_per_cycle = 2   # cap new trades per cycle
```

Env-var equivalents for the most common knobs:

- `LMTRADE_MODE` — `paper` (default) or `live`
- `LMTRADE_BUDGET` — starting cash (default 100)
- `LMTRADE_MODEL_STACK` — e.g. `heuristic,slm,perplexity`
- `LMTRADE_GPU_USD_PER_HOUR` / `LMTRADE_MIN_RUNWAY_HOURS` — the economics floor

`lmtrade config` prints the fully resolved settings so you can check what
actually took effect across all three layers.

## Running permanently (hourly Claude Routine)

Claude Code sessions run in ephemeral containers, so a background `lmtrade
run` process doesn't survive indefinitely. The alternative used here: a
Claude Routine fires every hour (the minimum interval Routines support),
gathers market news, compiles a daily analysis (baked in as a fusion signal
via `lmtrade import-news` / `lmtrade import-analysis`), runs a batch of
trading cycles, and persists state — see [`SETUP.md`](SETUP.md) step 4 to set
this up on a fork, or `scripts/hourly_routine.sh` for the mechanics.

State lives on a dedicated **`bot-state`** branch (not the code branch, to
keep hourly commits out of your PR history):

- `state/lmtrade.db` — the SQLite state the engine reads/writes.
- `state/ledger.csv` — every trade, incrementally appended each run. Unlike
  the SQLite file, this is diffable — review trade history directly via
  `git log`/`git diff` on the `bot-state` branch, no query needed.

The script pulls with `--rebase` before pushing, so two overlapping firings
can't silently clobber each other's state.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The suite runs the whole stack end-to-end in paper mode with **no network and no
API keys** (external services and Trade Republic are faked). Two tests are
currently flaky: `test_engine_opens_option_with_genome_attribution` (randomness
in a short simulated run) and `test_latest_news_roundtrip` (same-second
timestamps). Some dashboard logic is tested through Node when it's installed.

## Walk-forward backtesting

Pre-train the strategies on history before any live paper trading:

```bash
lmtrade backtest --bars 500 --train 150 --test 50
```

Rolling **[train | test]** folds step through the data: every genome trades the
train window (simulated with the same Black-Scholes option rules as the live
engine, P&L feeding the evolutionary optimizer), then the fittest genome is
scored on the *unseen* test window — the out-of-sample column is the honest
number. The trained population persists, so `lmtrade run` starts pre-trained.
Uses real daily bars from Yahoo Finance (via httpx) when reachable, synthetic
data offline.

## The learning loop

The bot maintains a population of strategy **genomes** (momentum, mean-reversion,
breakout — each with mutable parameters). Every cycle the epsilon-greedy
optimizer picks a genome whose signal is fused with the model votes; every
closed trade's realized P&L is attributed back to its genome; every N closed
trades the worst performer is replaced by a **mutated copy of the best**. State
persists across restarts, and `lmtrade status` / the dashboard show the
leaderboard. A **daily Claude review** additionally
adjusts risk parameters inside hard safety clamps, and **hourly Perplexity news**
feeds sentiment into every decision.

Outperformance is *measured*, not promised: the benchmark tracker holds SPY from
the same starting budget (the retail baseline) and the dashboard shows live
**alpha** against it.

## Compute costs

`economics.gpu_usd_per_hour` (env `LMTRADE_GPU_USD_PER_HOUR`) is the hourly cost
of any GPU you rent for the optional SLM. Running locally with the Claude CLI,
set it to `0` — the runway is then unlimited and only inference and trading
fees count. With a non-zero rate the runway guardrail scales with it.

## Honest limitations

- **True HFT is impossible on Trade Republic.** No official API exists; the
  unofficial mobile API has seconds-to-minutes latency and no options chains.
  LMTrade is an *intraday* bot: a full decision pass over the default 30-symbol
  universe takes a few minutes. Without TR credentials its options are
  synthetic Black-Scholes prices — read
  [`docs/TRADE_REPUBLIC.md`](docs/TRADE_REPUBLIC.md) before thinking about live mode.
- **Small accounts.** With €1 per order, the default fee guard needs orders of
  €100+, so a €300 account holds one or two positions at a time. Covering a GPU
  bill on top is genuinely hard — the economics layer reports that honestly.
- **Live execution is only partly verified** — see the list in
  [`docs/TRADE_REPUBLIC.md`](docs/TRADE_REPUBLIC.md#what-has-and-hasnt-been-verified-against-a-real-account).
- **Windows:** a settings save in the dashboard restarts the engine by re-running
  its command line; that re-launch fails when the engine was started via the
  `lmtrade` console script. Start it with a Python script path instead.
- **No performance guarantees.** "Outperform retail" and "self-sustaining in
  3–6 months" are goals the benchmark and economics layers *measure*; nothing
  here promises returns. This is a framework and research tool, not financial
  advice. Trading risks real loss. Use paper mode until you understand exactly
  what it does.

## License

[Creative Commons Attribution-NonCommercial 4.0 International](LICENSE) (CC BY-NC 4.0)

**Free for personal, educational, and research use. Commercial use is not permitted.**

You may share and adapt this work with attribution; you may not use it for
commercial purposes. See the [`LICENSE`](LICENSE) file for the full terms.
