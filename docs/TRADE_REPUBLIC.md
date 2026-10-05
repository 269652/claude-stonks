# Trade Republic integration (live mode)

> ⚠️ **Read this before enabling live trading.**

## There is no official API

Trade Republic does **not** publish a trading API. The only programmatic access
is via **reverse-engineered clients** such as [`pytr`](https://github.com/pytr-org/pytr),
which speak TR's private mobile app protocol.

**Consequences you accept by enabling live mode:**

- It likely **violates Trade Republic's Terms of Service**.
- Your account can be **flagged, frozen, or closed**.
- The private API can change without notice and break execution mid-trade.
- There is **no support** and no guarantee orders fill as intended.

LMTrade defaults to **paper mode**. Real orders are only sent when you switch the
dashboard to **LIVE** and tick **"arm live execution"** (with a confirmation).
Below €100 net worth a second checkbox ("allow low-balance orders") is required
as well. Switching the dashboard back to PAPER asks whether to disarm.

## Setup

1. Install the optional extra:
   ```bash
   pip install 'lmtrade[traderepublic]'
   ```
2. Put credentials in `.env` (gitignored):
   ```
   TR_PHONE=+49...
   TR_PIN=....
   ```
3. Complete the **one-time 2FA pairing** — and pass `--store_credentials`, or
   nothing gets saved and the bot will fail to resume the session every time:
   ```bash
   pytr login -n "+49..." -p "<PIN>" --store_credentials
   ```
   Use the **exact same phone number format** here as `TR_PHONE` in `.env` —
   pytr names the cookie file after it (`~/.pytr/cookies.<phone_no>.txt`), so
   a formatting mismatch means the bot silently finds no session to resume.
   The saved session expires after a while; the dashboard then shows
   "TR connection lost" and you repeat this step.

## What the bot does with TR

- **Instruments:** knockout certificates found via TR's derivative search
  (direction from TR's `optionType`, ratio from `size`), priced from TR's live
  ask (the search itself carries no prices), on the certificate's own exchange
  (resolved via `instrument_details`; issuer venues, not LS Exchange).
- **Entries:** good-for-day **limit** buy orders at the entry trigger
  (watch-list timing), or a marketable limit / market order for high-confidence
  or volatile signals. Whole certificates only. An order only counts as placed
  when TR returns an order id.
- **Fills:** booked when TR reports them, or when the certificate shows up in
  the TR portfolio (placed orders are remembered for 7 days for this).
- **Exits:** TP / SL / trailing / stale / manual close send a market **sell**
  for the ISIN and are only booked when TR confirms; rejected sells keep the
  position open and retry.
- **Session:** a heartbeat every 30s (`tr.heartbeat_seconds`); when TR doesn't
  answer, the session is dropped and resumed immediately.

## What has and hasn't been verified against a real account

Observed on one account on 2026-10-05 (pytr 0.4.10), small amounts:

- Session resume from stored cookies; knockout search results (fields
  `optionType`, `size`, `leverage`, `strike`, `barrier`, `currency`, no price).
- Ticker requests on `LSX` for knockouts got **no answer**; quotes and orders on
  the exchange from `instrument_details` (e.g. `SGL`) worked.
- Two limit buy orders were confirmed by TR with an order id and both filled.
  One fill was **missed by order tracking** and later booked by the portfolio
  reconciliation; a cancel request for that already-filled order was accepted
  by TR **without an error** — a successful cancel response does not prove
  the order was cancelled.

**Not yet observed against a real account** (implemented, tested only with
fakes): live exit sells (TP / SL / trailing / manual close), parsing of TR's
order-status payload, the automatic re-login after a dropped session, limit
order re-pricing (cancel + replace), and the fill price a position is anchored
to (the order's target underlying price is used as the reference, so marks can
be off by a few cents). Check each of these in the TR app the first time it
happens.

## Recommended safer alternatives

- **Stay in paper mode** with live market data to validate the strategy first.
- Use a broker that offers an **official API** (e.g. Alpaca, Interactive
  Brokers) — the `Broker` interface in `brokers/base.py` is designed so you can
  drop in a compliant adapter without touching the engine.

## Fees

Trade Republic charges a flat **€1 external fee** per order (entry and exit).
The bot models both: P&L and net worth are shown after fees, and the fee guard
(`risk.max_fee_pct`, default 1%) skips any entry smaller than fee ÷ max_fee_pct
(€100 at the defaults). A strategy must clear that hurdle *and* the compute bill
before it is self-sustaining.
