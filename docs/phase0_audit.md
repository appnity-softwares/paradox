# Phase 0 Audit & Cleanup — gold/prototype

Date: 2026-11-23
Scope: Parts 2–5 of the Master Development Prompt. No new features.

## Checklist status

| # | Master Prompt item | Status |
| --- | --- | --- |
| 1 | `pyarrow` in requirements, tests green | **Done** — `pyarrow>=15,<25` added (hard import in `core/vertex_store.py`); `truststore>=0.9,<1` added for OS trust store; `requirements-dev.txt` (pytest) created. **35/35 tests pass.** |
| 2 | No synthetic/random-walk data in the tick path | **Done** — see "Data honesty" below. |
| 3 | No hardcoded contract size / slippage / spread | **Done** — `core/config.py` is the single registry; engines resolve from it. |
| 4 | No secrets committed | **OK** — `.env` gitignored, `.env.example` complete. Rotate the Gemini key (returns HTTP 402). |
| 5 | Split `dashboard/app.py` into routers | **Done** — 1,969 → 47 lines; 49 endpoints across 6 routers, zero duplicates. |

## Data honesty — what was actually wrong

The prototype was presenting a **mix of three different feeds as one broker
feed**. `MacMT5Broker._stitch_latest_live_bars()` concatenated:

1. MT5 Mac terminal cache — real XAUUSD spot (old)
2. Yahoo Finance `GC=F` — CME gold **futures** (different price level)
3. Binance `PAXGUSDT` — tokenised gold **spot proxy** (24/7)

…and wrote the result into `data/xauusd_*_real.csv`, served as
`provider="MetaTrader 5 (Mac Spot)", real_data=True, spot=True, delayed=False`.

That single defect explains the "broken candles": the impossible wicks
(`low=3914.4` against a 4060 body) were **provider switches**, not bad ticks.

Two more fabrications in the same file:

- `fetch_tick()` stamped `datetime.now()` onto a bar close and invented
  `bid = close - 0.15`, `ask = close + 0.15`. `/api/quote` therefore reported
  `stale=false, age_seconds=0` for data that was hours or days old.
- `fetch_ticks()` returned one pseudo-tick per 1m bar and labelled it
  `is_real_tick=True`.
- `account_info()` returned a hardcoded 10,000 USD demo balance as if real.

### Fixes

- **Quarantined, not deleted**: the mixed CSVs and `data/cache/mt5_mac/` moved to
  `data/quarantine/` with a README explaining why they must not be used.
- **`MacMT5Broker` rewritten**: serves only genuine MT5 artifacts. Bars require a
  provenance marker written by the MT5 history extractor; ticks come from the
  terminal's raw `ticks.dat` (real bid/ask, true timestamps) or from the EA
  webhook. Missing artifacts raise a clear error. No stitching, no invented
  quotes, no fake account.
- **`PaxgBroker` added** as a first-class provider, labelled
  `Binance PAXG (spot proxy)` with a new `proxy=True` flag that flows through
  `BrokerStatus` → `MarketDataResult.metadata()` → `/api/info` → the UI.
- **Provider order**: MT5 Mac → MT5 Windows → PAXG (labelled proxy) → Yahoo
  (futures, research only). Feeds are never merged.
- **`/api/ticks` and `/api/ticks/history`**: real ticks only, from whichever
  provider actually has them, with `data_as_of`, `age_seconds`, `stale` and a
  plain-English note. They fail closed (503/404) instead of synthesising.
- **`/api/quote`**: `delayed`/`stale`/`age_seconds` computed from the tick's own
  timestamp.
- **`core/data.py`**: `_sanitize_ohlc()` clamps only physically implausible wicks
  (>6× local median range beyond the body) and reports `sanitized_bars`;
  `_bars_stale()` marks data stale when the newest bar is older than ~2 intervals.
- **`vertex_store`**: bar-expansion and random-walk fallbacks removed; raises
  `NoTickDataError`. The store now holds **2,048 real MT5 ticks**
  (`source=mt5_ticks_dat`, 4.4× Parquet compression).

## Configuration (Part 5.1)

`core/config.py` is the single registry: `SymbolSpec` (XAUUSD/EURUSD/GBPUSD),
`SlippageModel` (market/stop/limit/news), `CostModel` (ECN + standard),
`BrokerSpec` (leverage, stops/freeze levels, margin-call/stop-out), plus
`round_lot`, `pnl_per_point`, `required_margin`.

Engines wired: `core/backtest.py`, `core/tick_backtest.py`,
`core/execution_engine.py`. Defaults reproduce the previous numbers
(100 oz/lot, 1:100, 0.30 spread, 0.05 slippage) so results stay reproducible —
but they are configuration now, not literals.

## Router split (Part 4)

```
dashboard/app.py        47 lines   composition root only
dashboard/state.py      ~370       engines, agents, request models, helpers
dashboard/routers/
  data.py               6 routes   info, data, quote, ticks, ticks/history, ws
  mt5.py                7 routes   status, config, tick webhook, commands, sync
  backtest.py           2 routes   bar + tick backtest
  agents.py            17 routes   builder, copilot, council
  execution.py         11 routes   trade desk + paper trading
  research.py           5 routes   prop firm, vertex, RAG
```

The previously existing router files were **re-implementations** that had
regressed behaviour (they re-introduced OHLCV-synthesised ticks) and were
registered *before* the originals, silently shadowing 24 endpoints. Every
endpoint now exists exactly once, with the original implementation.

## Current data reality (this Mac)

| Feed | What it is | Label shown |
| --- | --- | --- |
| Chart + quote | Binance PAXG klines/bookTicker, live 24/7 | `Binance PAXG · proxy` |
| Historical ticks | 2,048 real MT5 ticks, 2026-11-12 12:12–12:29 UTC | `real broker ticks · 10.9d old` |
| MT5 bars | unavailable — XAUUSD `cache/` is empty (`2026.hcc` exists but is not decoded) | provider reports the reason |

The MT5 terminal *is* running under Wine (`2026.hcc` updated today), so real
spot bars are obtainable by decoding `.hcc` or by opening a XAUUSD chart to
populate the `.hc` cache. That is Phase 1A work.

## Verified

- `python3 -m unittest discover -s tests` → **35 passed**
- `node --check` on the extracted inline script → OK
- All 49 routes registered once; `/`, `/api/info`, `/api/data`, `/api/quote`,
  `/api/ticks`, `/api/vertex/*`, `/api/tick_backtest`, `/api/paper/*` → 200
- Browser: no console errors, no failed requests, no horizontal overflow,
  999 candles render, live ticks stream, honest labels on every surface
- Tick backtest on real ticks reports a **loss** (-2.39%) — reported as a loss.

## Next (per the document's golden rule)

Phase 1A — real tick pipeline. Decision taken: **MT5 export on Windows**.
Needs a Windows host/VPS beside a logged-in MT5 terminal, `CopyTicks` export,
the normalized tick schema in partitioned Parquet, the quality report with
quarantine thresholds, and dataset hashing.
