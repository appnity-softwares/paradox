# Phase 1B — Tick Engine (Progress Report)

Status: **core complete, validated on real data** — one item carried to Phase 1C.

Scope per Master Prompt Part 6.2–6.4: a deterministic tick replay engine that
consumes the Phase 1A tick store, shares one strategy contract with the bar
engine, and produces reproducible, cost-complete trade logs. AI agents analyse;
this engine computes and enforces.

## What was built

| Module | Content |
|---|---|
| `core/strategy_api.py` | Shared `Strategy` contract (`on_bar` / `on_tick` / `on_fill`), `Signal`/`Fill`/`Context`/`PositionView`, array-backed `BarWindow` with lazy DataFrames, `LookAheadError` guard on window access. Signal-only: strategies never compute P&L, sizing, or costs. |
| `core/tick_engine.py` | `TickEngine.run()` — vectorized per-minute chunk scans over bid/ask; market/limit/stop orders; per-tick SL/TP in true tick order; partial closes via `SIDE_CLOSE` fractions; breakeven (RR + offset), trailing stop with high-water persistence, time exit; latency (`fill_latency_ms`, searchsorted on target ms); commission per side, swap at UTC-midnight rollovers (3× triple day), news-window slippage multiplier; risk-based sizing with `round_lot`; margin + broker stop-out level; mandatory-SL rule (`require_sl`, default on); end-of-data liquidation; final partial bar never delivered. |
| `BacktestConfig` | Frozen dataclass, `config_hash` = sha256 over the sorted JSON; result carries `dataset_hash` + `engine_version` → byte-identical reruns. |
| `_summarize()` | Metrics include `lookahead_violations`, `valid` flag, `ticks_per_second`, `costs_paid_usd`. |

## Design decisions (documented tolerances)

- **Same-tick stop-through is not simulated.** A stop is evaluated from the
  tick after entry (+1 scan offset); intra-tick paths inside one tick are
  unknowable from recorded bid/ask. Conservative slippage on stop fills
  (`stop_slippage`) partially compensates.
- **SL/TP close the full position**; partial exits exist only via explicit
  `Signal(SIDE_CLOSE, fraction=...)`.
- **Bar OHLC from mid** `(bid+ask)/2`; bar closes only when the next chunk
  exists, so no signal ever sees an unfinished bar.
- **Swap charged once per boundary crossing**, not per elapsed day; 3× on the
  configured triple-swap weekday.
- **Equity is marked side-corrected** (bid for longs, ask for shorts).
- **Close orders pay the market side too** (bid − slip / ask + slip), matching
  the bar engine's exit convention.

## Verification

62 tests pass (`python -m pytest tests/ -q`):

- `tests/test_tick_engine.py` (19): quote sides, per-tick SL/TP order (tick
  engine disagrees with the bar engine's conservative SL-first assumption when
  the tick order says otherwise — by design), limit/stop fills, partial close,
  breakeven, trailing, time exit, commission per side, news slippage, latency
  fill timing, stop-out, determinism, cheater detection, mandatory SL, lot
  rounding, partial-bar suppression.
- `tests/test_engine_consistency.py` (1): trivial EMA-cross through **both**
  engines on the same ticks. Documented tolerance: trade count ±1 (end-of-data
  handling), identical side sequence, entry prices within $1.00. PASS.
- `tests/test_golden.py` (1): full trade log pinned against
  `tests/golden/tick_engine_golden.json` (config hash, metrics, every entry/
  exit price and P&L). Regenerate deliberately with `REGEN_GOLDEN=1`.
- 41 pre-existing tests (bar engine, market data, config, routing) unchanged.

## Quality report artifact (Part 6.1)

`reports/quality_report.json` is generated from the real ingested
store and committed as evidence. Gap classification is market-calendar
aware (`core/marketdata/quality.py`): XAUUSD trades Sun 22:00 → Fri
~21:00 UTC; a gap that does not intersect that window (weekends, Good
Friday, Easter Monday, Christmas, New Year — computed per year, plus
documented early closes) is reported as a `market_closure` and does
NOT quarantine the dataset. A genuine hole during trading hours still
quarantines. On the real data: golden week **ok** (977,612 ticks, 0
data gaps); 2025-H1 **ok** (33.2M ticks, 0 data gaps, 1 closure — the
Easter weekend 2025-04-17 20:58 → 2025-04-20 22:00, with only 0.03h of
open-window time). 5 tests in `tests/test_quality_closures.py` prove
both directions: closures pass, a missing trading week still
quarantines.

## Performance on real data (Master Prompt 6.4 budget)

`python -m scripts.bench_tick_engine XAUUSD 2025 6` — 4,752,206 real stored
ticks replayed in **0.9 s ≈ 5.5M ticks/s** (Parquet load 0.2 s). Projected
one-year replay (70M ticks): **≈ 0.2 min** against the 5-minute budget →
**PASS** with a large margin; the budget also holds if the real one-year tick
count is 2–3× the estimate.

## What remains open

1. **Legacy strategy adapter** — bridge the 10 library strategies
   (`strategies/library.py`, `generate_signals(df)` contract) onto the shared
   interface so they can run under the tick engine. Carried to Phase 1C work,
   where the library is extended to 12+ strategies anyway.
2. **Monte Carlo / cost-stress / robustness verdicts** (`PAPER_TRADE_ELIGIBLE`
   / `RESEARCH_ONLY`) — Phase 1C per Part 7.
3. The bar engine's conservative same-bar SL-first rule stays as documented
   behaviour for the bar path; the tick engine is the reference for order-level
   realism.
