# Phase 2 — Account and Strategy Adaptation (Progress Report)

Status: **complete per Part 7.4 acceptance criteria**. All numeric paths
are pure deterministic functions; no LLM output feeds any calculation.

## What was built

### Position sizing service (Part 7.2) — `core/sizing.py`

Pure functions, fail-closed:

- `size_fixed_risk`: percent-risk sizing from stop distance; lot rounded
  **down** to the lot step (rounding up would exceed intended risk); a raw
  size below `min_lot` fails with a reason code — the 0.01-lot floor
  would exceed the intended risk.
- `size_atr`: volatility sizing, stop = `atr_mult` x ATR.
- `size_fixed_lot`: reports implied risk/margin for a fixed lot, rejects
  off-grid, below-minimum and above-maximum sizes.
- Account-currency conversion via a timestamped `CurrencyRate`
  (1 profit/margin-ccy unit = `rate` account units); risk and margin are
  reported in the ACCOUNT currency.
- Margin = `lot * contract_size * price / leverage`; pass/fail against
  equity and symbol limits, with reason codes.

**41 hand-calculated XAUUSD cases** in `tests/test_sizing.py` (t01–t41):
different equities (500 → 10M), prices (2000/2700), stops (0.5 → 15),
leverages (2 → 500), EUR/GBP accounts, caps, and fail-closed inputs.
Code output matches the hand calculations exactly (documented formulas in
the test module docstring). Two of my own hand calculations were corrected
during test writing (t14 margin, t17 failure mode) — the tests, not the
code, were fixed.

### Account model (Part 7.1) — `core/account.py`

`Account` (balance, equity, free margin, used margin, margin level,
leverage, currency, positions) with derived properties, immutable-style
position updates, and per-account `AccountLimits` (risk per trade, daily
risk, max positions, exposure lots, leverage cap, drawdown limit).
Multiple accounts = multiple instances with independent settings.

### Adaptation logic (Part 7.3) — `core/adaptation.py`

- `adapt_config`: risk per trade = min(account cap, strategy
  recommendation); exposure cap = min(account limit, margin-derived cap
  at 50% equity utilization); max positions and trades/day derived
  deterministically.
- `suitability_report`: four checks — min-lot vs intended risk, margin
  headroom (80% of free margin), historical worst DD vs account DD limit,
  cost impact vs intended risk (33% cap). Each check reports passed +
  human-readable detail.
- `suggest_adaptation`: equity-change and volatility-change detection
  returning SUGGESTIONS ONLY — never mutates an account or config;
  approval and audit logging are the caller's job (wired to the audit
  trail in Phase 3/4 integration).

`StrategyProfile` carries only measured facts from the Phase 1 stack
(worst DD, typical stop, round-trip cost) — nothing invented.

### Real-data demo — `scripts/adapt_accounts.py`

The validated ema_cross ATR cell adapted to 10k and 100k accounts on the
same real H1-2025 bars (london_ny session, ATR 1.5 stops, RR 1.5):

| Account | Suitable | Lot | Risk/trade | Margin | Result |
|---|---|---|---|---|---|
| 10k | yes | 0.08 | $96 | $265 | +61.2%, Sharpe 3.04, DD −11.7% |
| 100k | yes | 0.83 | $996 | $2,749 | +61.2%, Sharpe 3.04, DD −11.7% |

Percent-risk sizing is account-size invariant: identical percentages,
10x different absolute money — the expected behaviour. Artifacts:
`reports/adapted_accounts_202501.csv`, `reports/adapted_<acct>_202501.csv`
(equity + drawdown curves per account).

## Verification

164 tests pass. Added in this phase: `tests/test_sizing.py` (44),
`tests/test_account.py` (20: account model, 10k-vs-100k adaptation,
suitability flags, suggestions-only guarantees, adapted real-bar backtest).

## Acceptance criteria (Part 7.4)

- [x] Position sizing matches the 30+ hand-calculated cases exactly (41 cases).
- [x] Same strategy produces sensible, different configs for 10k and 100k
      accounts (same risk %, 10x lot, different binding exposure caps).
- [x] Suitability report flags unsuitable combinations (tiny account +
      large stop → min-lot risk 2.4% > intended 1%; DD over limit; cost
      impact over cap; margin headroom).
- [x] No LLM output in any numeric calculation (pure functions only).

## What remains open

1. Wire `suggest_adaptation` output to the audit log and approval flow
   (Part 9/12 order flow lands the audit trail; suggestions are ready).
2. The Risk Engine (Part 8) consumes these limits next — spread/stale-price
   filters, kill switch, correlated exposure buckets.
