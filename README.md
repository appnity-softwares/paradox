# AI Quant Terminal — Professional Prototype

A deterministic XAUUSD research terminal with provenance-aware market data,
strategy backtesting, walk-forward validation, a rule-based strategy builder,
and an MT5 adapter.

> This repository is a research/paper-trading prototype. Live order execution
> is deliberately not exposed through the dashboard until account permissions,
> deterministic risk gates, audit logging and a kill switch are complete.

## Current capabilities

- 7 configurable strategies: EMA, MACD, RSI reversal, Bollinger, VWAP+EMA,
  Supertrend and session breakout
- Correct bar event order: signal at close → fill next open → same-bar SL/TP
- Spread, slippage, ATR/fixed stops, RR targets and end-of-data liquidation
- Grid optimization and leakage-safe walk-forward boundaries
- Natural-language rule parser → validated strategy spec → test → explanation
- TradingView-style chart with Candles/Bars/Line/Area, volume, EMA, VWAP,
  Supertrend, Bollinger, RSI and MACD
- Source-aware market-data cache (`real`, `delayed`, `stale`, `synthetic`)
- MT5 tick WebSocket endpoint when running beside a Windows MT5 terminal
- Yahoo GC=F delayed futures data as the cross-platform research fallback
- Automated regression tests for critical engine/data/API behavior

## Data-feed truth

| Environment | Feed | UI status | Tick stream | Execution |
|---|---|---|---|---|
| Windows + connected MT5 | Broker spot feed | `LIVE · MT5` | WebSocket | Adapter only; UI disabled |
| macOS/Linux | Yahoo GC=F futures | `DELAYED · YAHOO` | No | No |
| Offline test mode | Synthetic generator | `SYNTHETIC` | No | No |

Yahoo data is real market data, but it is gold futures—not broker spot XAUUSD
and not a per-tick live feed. The dashboard never labels it `LIVE`.

## Install

Python version support (U0, PyPI-verified 2026-11-24):

- Dev workstation / CI: **Python 3.14** (and 3.13 in the CI matrix). The
  full test suite passes on both.
- Windows MT5 bridge VPS: **Python 3.13 or older** — the `MetaTrader5`
  package (5.0.6231) publishes wheels only for CPython 3.8–3.13; there is
  no 3.14 wheel yet. Do not install the bridge runtime on 3.14.

```bash
cd prototype
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

For MT5, copy `.env.example` to `.env` or set the environment variables in the
service manager. Do not commit credentials.

## Run dashboard

```bash
python3 -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8123
```

Open <http://127.0.0.1:8123>.

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## CLI research tools

```bash
python3 run.py
python3 run.py --strategy vwap_ema --session london --risk 0.5
python3 run_optimize.py --strategy ema_cross --folds 4
python3 run_builder.py "EMA 20/50 crossover, fixed 12 stop, 1:2 RR, London session"
```

## Architecture

```text
Dashboard / CLI
      │
      ├── Strategy Builder ──> validated StrategyConfig
      ├── Backtest Engine ──> Metrics
      ├── Optimization ──> Walk-forward validation
      │
      └── Market Data Service
              ├── provider-specific cache + metadata
              ├── MT5 adapter (spot/ticks, Windows)
              └── Yahoo adapter (delayed GC=F futures)
```

Important files:

- `core/data.py` — source-aware cache and freshness policy
- `core/data_manager.py` — provider selection without silent mislabeling
- `core/backtest.py` — deterministic event ordering
- `core/optimize.py` — grid and walk-forward validation
- `core/brokers/mt5_broker.py` — MT5 data/account/order adapter
- `ai/builder.py` — natural-language rule parser and explanation
- `dashboard/app.py` — validated FastAPI contracts and MT5 tick WebSocket
- `dashboard/index.html` — chart terminal

## MT5 setup

1. Use a Windows workstation/VPS.
2. Install the broker's MT5 terminal.
3. Log in once and confirm algorithmic/API trading is allowed.
4. Install this project and the Windows-only `MetaTrader5` package.
5. Start the API beside the terminal.

The web dashboard can then be opened remotely through a secured reverse proxy;
the user does not need to interact with the MT5 UI for normal monitoring.

## Safety boundary

Before enabling live orders, the project still requires:

- account-level deterministic risk engine
- broker/prop rule engine
- idempotent order intents and `order_check`
- persistent audit log
- emergency kill switch
- authentication, RBAC and encrypted secrets
- paper-trading soak tests and Windows MT5 integration tests

No profitability or prop-challenge outcome is guaranteed.
