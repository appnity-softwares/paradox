#!/usr/bin/env python3
"""Prototype runner: backtest all strategies on XAUUSD and print the comparison table.

Usage:
  python3 run.py                          # all strategies, default config
  python3 run.py --strategy ema_cross     # single strategy
  python3 run.py --session london_ny --risk 1.0 --rr 2 --max-trades-day 3
  python3 run.py --csv path/to/data.csv   # your own OHLCV data (time,open,high,low,close,volume)
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data import load_data_result
from core.backtest import run_backtest
from core.metrics import compute_metrics
from strategies import ALL_STRATEGIES, StrategyConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def infer_tf_label(index: pd.DatetimeIndex) -> str:
    d = index.to_series().diff().median()
    mins = int(d.total_seconds() // 60) if d and d > pd.Timedelta(0) else 15
    return f"{mins}m" if mins < 60 else f"{mins // 60}h"


def main() -> None:
    p = argparse.ArgumentParser(description="AI Quant Platform - Prototype Backtester")
    p.add_argument("--strategy", default="all", choices=["all", *ALL_STRATEGIES.keys()])
    p.add_argument("--csv", default=None, help="Your own OHLCV csv (time,open,high,low,close,volume)")
    p.add_argument("--synthetic", action="store_true", help="Force synthetic data (skip yfinance)")
    p.add_argument("--risk", type=float, default=1.0, help="Risk %% per trade (notebook: 0.5-2%%)")
    p.add_argument("--rr", type=float, default=2.0, help="Reward:Risk ratio")
    p.add_argument("--sl", default="atr", choices=["atr", "fixed"], help="Stop loss method")
    p.add_argument("--sl-atr-mult", type=float, default=1.5, help="ATR multiplier for SL")
    p.add_argument("--session", default="all", choices=["all", "tokyo", "london", "ny", "london_ny"])
    p.add_argument("--max-trades-day", type=int, default=4)
    p.add_argument("--daily-loss-limit", type=float, default=3.0, help="Daily loss limit %%")
    args = p.parse_args()

    print("Loading data...")
    data_result = load_data_result(csv_path=args.csv, force_synthetic=args.synthetic)
    df = data_result.data
    tf_label = infer_tf_label(df.index)
    flags = ["synthetic" if data_result.synthetic else "real" if data_result.real_data else "unverified"]
    if data_result.delayed: flags.append("delayed")
    if data_result.stale: flags.append("stale")
    print(f"  {len(df)} bars | {df.index[0]} -> {df.index[-1]} | TF: {tf_label}")
    print(f"  source: {data_result.provider} ({', '.join(flags)})\n")

    keys = list(ALL_STRATEGIES) if args.strategy == "all" else [args.strategy]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for key in keys:
        cfg = StrategyConfig(
            risk_pct=args.risk,
            rr=args.rr,
            sl_method=args.sl,
            sl_atr_mult=args.sl_atr_mult,
            session=args.session,
            max_trades_per_day=args.max_trades_day,
            daily_loss_limit_pct=args.daily_loss_limit,
        )
        strat = ALL_STRATEGIES[key](cfg)

        equity, trades = run_backtest(df, strat, cfg)
        metrics = compute_metrics(equity, trades, cfg, tf_label)
        rows.append(metrics)

        out = os.path.join(RESULTS_DIR, f"trades_{key}.csv")
        trades.to_csv(out, index=False)
        print(f"  [ok] {metrics['Strategy']:<16} trades: {metrics['Total Trades']:>4}  -> {os.path.basename(out)}")

    summary = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(out, index=False)

    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON  (notebook dashboard table)")
    print("=" * 100)
    cols = [
        "Strategy", "Win Rate %", "Max DD %", "Total Trades", "Sharpe", "Profit Factor",
        "RR (realized)", "Max Trades/Day", "Con. Wins", "Con. Losses",
        "Net P&L $", "Return %", "Risk/Trade %", "TF", "Session",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary[cols].to_string(index=False))
    print("=" * 100)
    print(f"Saved: results/summary.csv + per-strategy trade logs")
    print(f"Config: risk {args.risk}%/trade | RR 1:{args.rr} | SL {args.sl} ({args.sl_atr_mult}xATR)"
          f" | session {args.session} | max {args.max_trades_day} trades/day | daily loss limit {args.daily_loss_limit}%")


if __name__ == "__main__":
    main()
