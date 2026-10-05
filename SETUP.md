# Setup instructions for Claude

> **Trigger:** the user says something like "Setup the project", "configure
> this", or "run setup" in a Claude Code session pointed at this repo (usually
> a fork). Follow this file mechanically — it's written for you, not for a
> human reader. Do not paraphrase or skip steps.

## What this project is

LMTrade is a self-sustaining paper-trading bot (options + evolutionary
strategy learning, real market data) with an hourly Claude-Routine-driven
runner. Full description: `README.md`. Do not re-explain the whole project to
the user here — just get it configured.

## Step 0 — Sanity checks

```bash
python3 --version   # need 3.10+ (3.11+ preferred, gives stdlib tomllib)
git status
```

If `.env` or `config.toml` already exist in this repo, **stop and ask the
user** whether they want to reconfigure (overwriting) or leave the existing
setup alone — don't silently clobber a prior configuration.

## Step 1 — Ask the configuration questions

Use `AskUserQuestion` (or plain conversational Q&A if that tool isn't
available) to gather these settings. Mark the recommended option first in
each question, as instructed by the tool. Group related settings into a
single call where the AskUserQuestion tool allows multiple questions at once.

1. **Mode** — paper (recommended, safe, no real money) or live (real Trade
   Republic execution via an unofficial API, against TR's ToS — only if the
   user explicitly wants this; see `docs/TRADE_REPUBLIC.md` and make sure
   they've read the risk before proceeding).
2. **Starting budget** — default 100 (EUR). Ask for a number.
3. **Universe** — use the default 30-symbol universe in
   `config/default.yaml` (US mega-caps, EU blue chips, ETFs, crypto), or let
   the user customize it. If customizing, remind them: only symbols with
   real data on Yahoo Finance work (test with `python3 -c "from
   lmtrade.data.market import default_yahoo_fetcher as f; print(f('TICKER',
   '5d','1d'))"` for anything unusual before committing to it).
4. **Cadence** — how often the engine loop evaluates within a run
   (`loop.interval_seconds`, default 15s) and, if they're setting up the
   hourly Routine (step 4 below), how many cycles run per hourly firing
   (`LMTRADE_HOURLY_CYCLES` in `scripts/hourly_routine.sh`, default 12) —
   this is "trades per hour" alongside `loop.max_new_positions_per_cycle`.
   Note: a Claude Routine's own firing cadence has a hard 1-hour minimum —
   you cannot make the *Routine* fire more than once an hour, only the
   engine's internal loop and cycles-per-firing are tunable below that.
5. **Position limits** — `loop.max_positions` (concurrent open positions,
   default 8) and `loop.max_new_positions_per_cycle` (cap on new entries per
   cycle, default 0 = unlimited up to max_positions).
6. **Hold time** — `options.min_hold_hours` (block TP/SL exits before this
   many hours, default 0 = no minimum) and `options.max_hold_hours`
   (force-close after this many hours regardless of TP/SL, default 0 =
   disabled, relies on `expiry_days` instead).
7. **TP / SL** — `options.take_profit_pct` (default 0.5 = +50%) and
   `options.stop_loss_pct` (default 0.4 = -40%). These are option premium
   percentages, not underlying-price percentages.
8. **Profit stash** — what fraction of realized profit (never losses) on
   each winning trade gets moved into a reserve excluded from future
   trading capital? Offer presets: 0% (reinvest everything, default), 50%
   (50/50 split), 70% (stash 70%, reinvest 30%), or a custom number
   0.0–1.0. Explain: this money isn't lost — it still counts toward net
   worth and alpha — it's just protected from being re-risked.
9. **GPU rate** — `economics.gpu_usd_per_hour` (default 0.20; use 0 if not
   renting a GPU at all — the SLM step below is optional).
10. **API keys** (all optional — the bot degrades gracefully without them):
    - `ANTHROPIC_API_KEY` — cloud LLM signal + daily analysis fallback (not
      required if the user will run the hourly Routine themselves, since
      that already does research using the Routine session's own Claude
      access, not this key).
    - `PERPLEXITY_API_KEY` — in-engine hourly news fetch (separate from, and
      redundant with, the Routine's own news-gathering — most users won't
      need this if they're using the hourly Routine).
    - Ollama host / SLM model — only if running a local SLM.
11. **Trade Republic credentials** — ask in two cases. (a) The user chose
    `live` mode in question 1, or (b) the user wants **paper trading on
    real TR instruments** (`tr.use_derivatives`): with `TR_PHONE`/`TR_PIN`
    set, the bot selects real TR knockout certificates (real ISINs, barrier
    pricing) for its simulated trades instead of synthetic options — no
    orders are ever placed. Trade Republic has **no API key** — the
    unofficial adapter uses phone number + app PIN (same as the mobile
    app), first pairing needs one interactive 2FA — run
    `pytr login -n "<TR_PHONE value>" -p "<TR_PIN value>" --store_credentials`
    (the `--store_credentials` flag is required or nothing persists to disk
    and the bot fails to resume the session every time; the phone number
    must match `TR_PHONE` character-for-character since pytr names the
    cookie file after it). TR's API is websocket-based (blocked by some
    proxies), and it's against TR's ToS either way. For live mode,
    reiterate clearly: order placement is
    disabled by a deliberate code guard (`brokers/trade_republic.py`) that
    the user must remove themselves after reading and accepting the risk —
    do not offer to remove that guard as part of setup.

## Step 2 — Write the config files

**Secrets → `.env`** (gitignored). Copy `.env.example` to `.env` if it
doesn't exist, then fill in exactly the keys the user provided answers for in
step 1 questions 9–11. Leave everything else blank/commented — never invent a
placeholder value.

**Non-secret settings → `config.toml`** (gitignored). Copy
`config.example.toml` to `config.toml` if it doesn't exist, then set only the
keys the user's answers changed from the defaults — leave the rest as
comments/defaults so the file stays readable and it's clear what's custom.

Never print the contents of `.env` back to the user in chat (secrets), and
never `git add` either file — both are gitignored; if `git status` somehow
shows them as trackable, stop and tell the user their `.gitignore` may be
misconfigured rather than committing anything.

## Step 3 — Install and verify

```bash
pip install -e '.[dev]'
python -m pytest -q          # must be fully green before proceeding
lmtrade config                # print the resolved settings — sanity check
                              # against what the user actually asked for
lmtrade run --cycles 2 --interval 2   # quick smoke run, real data if reachable
lmtrade status
```

If the test suite doesn't pass cleanly, stop and report the failure — don't
proceed to offer the hourly Routine setup on a broken install.

## Step 4 — Offer the permanent hourly Routine (optional)

Ask the user if they want the bot running permanently via an hourly Claude
Routine (this is what the original LMTrade project uses — see the git log /
README for context). If yes:

1. Confirm which `mcp__Claude_Code_Remote__*` tools are available in this
   session (`create_trigger`, `list_triggers`, `delete_trigger`). If they
   aren't available, explain that this feature needs Claude Code Remote and
   skip this step.
2. Create the `bot-state` orphan branch if it doesn't already exist on the
   fork's remote (mirror the pattern: `git worktree add --orphan -b
   bot-state <tmpdir>`, add a `state/README.md`, commit, push, remove the
   worktree).
3. Ask whether to bind the Routine to this session (updates appear in this
   chat, but ties the Routine to this session's lifetime) or spawn a fresh
   session each firing (`create_new_session_on_fire: true`, more robust for
   true "permanent," but you won't see hourly updates unless you check back).
   Default recommendation: bind to this session if the user is actively
   watching it get set up; fresh-session otherwise.
4. Create the trigger with `cron_expression: "0 * * * *"` and a prompt
   modeled on the one already used in this project's history (ask the news
   for the configured universe by theme — don't search every symbol
   individually if the universe is large — compile a daily analysis once
   per ~24h, run `scripts/hourly_routine.sh` with `LMTRADE_NEWS_FILE` /
   `LMTRADE_ANALYSIS_FILE` env vars pointing at files it just wrote, report
   a short summary, and flag red flags: tracebacks, "SANITY BREACH", git
   push failures, or halt_trading persisting across several hours). Use
   `LMTRADE_HOURLY_CYCLES` / `LMTRADE_HOURLY_INTERVAL` env vars in the
   script invocation if the user chose non-default cycle/interval settings
   in step 1.
5. Report the trigger ID and next scheduled fire time.

## Step 5 — Summarize

Tell the user, concisely:
- What mode it's running in (paper/live) and starting budget.
- Where secrets live (`.env`, gitignored, never shown) and where config
  lives (`config.toml`, gitignored).
- The key tunables they set (position limits, hold times, TP/SL, profit
  stash) in one line each.
- Whether the hourly Routine was set up, and its next fire time.
- That `lmtrade status`, `lmtrade viz`, and `lmtrade web` are how to check on
  it going forward, and `config.example.toml` / `.env.example` are the
  reference for every available setting.

Don't dump the full resolved config back into chat unless the user asks —
they already saw it in `lmtrade config` during step 3 if they were watching.
