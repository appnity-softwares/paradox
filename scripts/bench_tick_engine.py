"""Benchmark the tick engine on real stored ticks (Master Prompt 6.4).

Usage:
    python -m scripts.bench_tick_engine [symbol] [year] [month]

Loads one month of real ticks from the partitioned Parquet store, replays a
deterministic EMA-cross strategy through the shared strategy interface and
reports ticks/second plus the projected runtime for a full year, against the
Master Prompt budget of "a one-year tick backtest in under five minutes".
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

from core.marketdata import store
from core.tick_engine import BacktestConfig, TickEngine
from core.strategy_api import Context, Signal, Strategy, SIDE_LONG, SIDE_SHORT


class FastEmaCross(Strategy):
    """O(1)-per-bar EMA cross using the closed-bar window only."""

    name = "fast_ema_cross_bench"
    version = "1"
    warmup_bars = 100

    def __init__(self, fast: int = 20, slow: int = 80):
        self.fast, self.slow = fast, slow
        self.ema_f: float | None = None
        self.ema_s: float | None = None
        self.af = 2.0 / (fast + 1)
        self.as_ = 2.0 / (slow + 1)

    def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
        c = bar["close"]
        self.ema_f = c if self.ema_f is None else self.af * c + (1 - self.af) * self.ema_f
        self.ema_s = c if self.ema_s is None else self.as_ * c + (1 - self.as_) * self.ema_s
        if len(ctx.bars) < self.warmup_bars:
            return []
        if self.ema_f > self.ema_s and ctx.position.side <= 0:
            return [Signal(SIDE_LONG, sl=bar["close"] - 3.0)]
        if self.ema_f < self.ema_s and ctx.position.side >= 0:
            return [Signal(SIDE_SHORT, sl=bar["close"] + 3.0)]
        return []


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    month = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    t0 = time.perf_counter()
    ticks = store.read_range(symbol, start, end)
    load_sec = time.perf_counter() - t0
    if ticks.empty:
        print(f"no ticks stored for {symbol} {year}-{month:02d}")
        return 1

    cfg = BacktestConfig(
        symbol=symbol,
        dataset_hash=store.dataset_hash(symbol),
        start_ms=start,
        end_ms=end,
    )
    result = TickEngine().run(FastEmaCross(), ticks, cfg)

    n = len(ticks)
    secs = result["runtime_seconds"]
    tps = result["ticks_per_second"] or 0
    year_ticks = 70_000_000
    projected = year_ticks / tps if tps else float("inf")
    print(f"month            : {year}-{month:02d}")
    print(f"ticks            : {n:,}  (load {load_sec:.1f}s)")
    print(f"runtime          : {secs:.1f}s  ({tps:,.0f} ticks/s)")
    print(f"trades           : {result['total_trades']}")
    print(f"net profit       : ${result['net_profit_usd']:,.2f}")
    print(f"max drawdown     : ${result['max_drawdown_usd']:,.2f}")
    print(f"costs paid       : {result['costs_paid_usd']}")
    print(f"valid (no peeking): {result['valid']}")
    print(f"projected 1-year replay ({year_ticks:,} ticks): {projected / 60:.1f} min "
          f"(budget: 5 min -> {'PASS' if projected < 300 else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
