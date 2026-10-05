# CLAUDE.md — LMTrade project rules

## Development policy: STRICT TDD (non-negotiable)

Every change to this codebase must follow test-driven development:

1. **Write the test first.** Before implementing any function, class, module, or
   behavior change, write a failing test in `tests/` that specifies the expected
   behavior. Run it and confirm it fails for the right reason.
2. **Implement the minimum** to make the failing test pass.
3. **Run the full suite** (`python -m pytest`) after every implementation step —
   not just the new tests. All tests must pass before moving on.
4. **Every operation must be tested.** No module ships without tests covering:
   - the happy path,
   - at least one edge case (empty data, zero/negative values, missing keys),
   - failure/degradation behavior (no network, no API key, provider down).
5. **Bug fixes start with a regression test** that reproduces the bug, then the
   fix. The test stays in the suite permanently.
6. **No untested code in commits.** If a commit adds or changes runtime code,
   it must add or change tests in the same commit.

Tests must be runnable **offline with no API keys** — external services
(Ollama, Anthropic, Perplexity, Claude CLI, Yahoo Finance, Trade Republic) are
always optional at runtime and must be faked/stubbed or skipped in tests. The
synthetic market-data provider exists for exactly this purpose. Live market
data goes through `httpx` directly against Yahoo's chart API — not the
`yfinance` PyPI package, whose `curl_cffi` backend does browser TLS-fingerprint
impersonation that breaks under some sandboxed/proxied networks.

## Project overview

LMTrade is a trading bot that runs locally: it fuses classical financial
models, an optional local SLM (Ollama) and LLM research / analysis (the local
Claude CLI, or the Anthropic / Perplexity APIs) into trading decisions. Its
economics layer halts new entries when the runway for its compute costs
(economics.gpu_usd_per_hour, 0 when nothing is rented) runs out.

- **Paper mode is the default.** Live Trade Republic execution is unofficial
  (pytr), against TR ToS, and stays behind a deliberate guard. Never enable it
  silently.
- Package layout: `src/lmtrade/` — `core/` (engine, state, events),
  `agents/` (fusion), `models/` (providers), `finance/` (indicators, risk,
  options), `brokers/`, `economics/`, `data/`, `strategies/`, `research/`,
  `web/`, `viz/`, `infra/`.
- State is a single SQLite store (`core/state.py`) shared by engine, web and viz.

## Commands

```bash
pip install -e '.[dev]'     # install with test deps
python -m pytest            # full suite — must pass before any commit
lmtrade run --cycles 3 --interval 1   # quick engine smoke run
lmtrade status | web | viz | reset | config
```

## Conventions

- Python 3.10+, ruff line length 100.
- Providers degrade gracefully: missing key/host → neutral signal or skip,
  never a crash.
- All runtime state lives under `data/` (gitignored via anchored `/data/` —
  never use an unanchored `data/` pattern, it shadows `src/lmtrade/data/`).
- Currency: account values in EUR, compute costs in USD.
