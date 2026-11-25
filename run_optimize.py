#!/usr/bin/env python3
"""Optimization + walk-forward validation runner.

Usage:
  python3 run_optimize.py                     # all strategies, 4 folds
  python3 run_optimize.py --strategy ema_cross
  python3 run_optimize.py --folds 3 --min-trades 10
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data import load_data_result
from core.optimize import PARAM_GRID, grid_search, param_combos, walk_forward
from strategies import ALL_STRATEGIES, StrategyConfig
from run import infer_tf_label, RESULTS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description="AI Quant Platform - Optimizer + Walk-Forward")
    p.add_argument("--strategy", default="all", choices=["all", *ALL_STRATEGIES.keys()])
    p.add_argument("--folds", type=int, default=4, help="Walk-forward folds (train/test splits)")
    p.add_argument("--min-trades", type=int, default=15, help="Min trades for a param set to rank")
    p.add_argument("--csv", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--risk", type=float, default=1.0)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--sl", default="atr", choices=["atr", "fixed"])
    p.add_argument("--sl-atr-mult", type=float, default=1.5)
    p.add_argument("--session", default="all", choices=["all", "tokyo", "london", "ny", "london_ny"])
    p.add_argument("--max-trades-day", type=int, default=4)
    p.add_argument("--daily-loss-limit", type=float, default=3.0)
    args = p.parse_args()

    print("Loading data...")
    data_result = load_data_result(csv_path=args.csv, force_synthetic=args.synthetic)
    df = data_result.data
    tf_label = infer_tf_label(df.index)
    flags = ["synthetic" if data_result.synthetic else "real" if data_result.real_data else "unverified"]
    if data_result.delayed: flags.append("delayed")
    if data_result.stale: flags.append("stale")
    print(f"  {len(df)} bars | {df.index[0]} -> {df.index[-1]} | TF: {tf_label}")
    print(f"  source: {data_result.provider} ({', '.join(flags)})")

    base_cfg = StrategyConfig(
        risk_pct=args.risk,
        rr=args.rr,
        sl_method=args.sl,
        sl_atr_mult=args.sl_atr_mult,
        session=args.session,
        max_trades_per_day=args.max_trades_day,
        daily_loss_limit_pct=args.daily_loss_limit,
    )

    keys = list(ALL_STRATEGIES) if args.strategy == "all" else [args.strategy]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    engine_kwargs = {}

    summary_rows = []
    for key in keys:
        cls = ALL_STRATEGIES[key]
        grid = PARAM_GRID[key]
        n_combos = len(list(param_combos(grid)))

        print(f"\n{'=' * 100}")
        print(f"STRATEGY: {key}")
        print(f"{'=' * 100}")

        # 1) full-data grid search (in-sample picture)
        print(f"  [1/2] Grid search on full data ({n_combos} combos)...")
        res, best = grid_search(df, cls, grid, base_cfg, tf_label, args.min_trades, engine_kwargs)
        res.to_csv(os.path.join(RESULTS_DIR, f"opt_full_{key}.csv"), index=False)
        params_str = ", ".join(f"{k}={best[k]:g}" for k in grid)
        print(f"        best: {params_str}  |  Sharpe {best['Sharpe']:.2f}  |  "
              f"Ret {best['Return %']:.1f}%  |  {int(best['Total Trades'])} trades")

        # 2) walk-forward validation (out-of-sample truth)
        print(f"  [2/2] Walk-forward validation ({args.folds} folds)...")
        folds_df, pooled, wf = walk_forward(
            df, cls, grid, base_cfg, tf_label, args.folds, args.min_trades, engine_kwargs
        )
        folds_df.to_csv(os.path.join(RESULTS_DIR, f"wf_folds_{key}.csv"), index=False)
        if len(pooled):
            pooled.to_csv(os.path.join(RESULTS_DIR, f"wf_oos_trades_{key}.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(folds_df.to_string(index=False))
        print(f"        OOS verdict: {wf['Verdict']}  |  OOS Ret {wf['OOS Ret %']}%  |  "
              f"OOS Sharpe {wf['OOS Sharpe (med)']}  |  param sets used: {wf['Param Sets Used']}/{args.folds}")

        summary_rows.append({
            "Strategy": key,
            "Best Params": params_str,
            "Full-data Sharpe": best["Sharpe"],
            "Full-data Ret %": best["Return %"],
            **wf,
        })

    print(f"\n{'=' * 100}")
    print("OPTIMIZATION SUMMARY  (full-data = in-sample | OOS = walk-forward out-of-sample)")
    print("=" * 100)
    summary = pd.DataFrame(summary_rows)
    cols = [
        "Strategy", "Best Params", "Full-data Sharpe", "Full-data Ret %",
        "OOS Ret %", "OOS Sharpe (med)", "OOS Win %", "OOS Trades", "OOS Max DD %",
        "Profitable Folds %", "Param Sets Used", "Verdict",
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summary[cols].to_string(index=False))
    print("=" * 100)
    print("Verdict: ROBUST requires OOS profit, Sharpe >=0.5, >=60% profitable folds, >=50 trades and DD better than -25%")
    print("Rule: full-data numbers dekh ke strategy deploy mat karo - OOS verdict hi sach hai.")
    out = os.path.join(RESULTS_DIR, "optimization_summary.csv")
    summary.to_csv(out, index=False)
    print(f"Saved: {os.path.basename(out)} + full grids + fold tables + OOS trade logs")


if __name__ == "__main__":
    main()
