#!/usr/bin/env python3
"""AI Strategy Builder CLI - natural language in, tested strategy + explanation out.

Usage:
  python3 run_builder.py "EMA 20/50 crossover with ATR stop loss and 1:2 risk reward, london session, 1% risk"
  python3 run_builder.py "supertrend 10 with 2.5 multiplier, ny session, 0.5% risk, max 2 trades a day"
  python3 run_builder.py "vwap ema strategy 50 ema with slope, daily loss 2%" --tf 1h
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data import load_data_result
from ai.builder import run_pipeline
from run import infer_tf_label


def main() -> None:
    p = argparse.ArgumentParser(description="AI Quant Platform - AI Strategy Builder (prototype)")
    p.add_argument("command", nargs="+", help="Natural language strategy description")
    p.add_argument("--tf", default="15m", choices=["15m", "1h"], help="Data timeframe")
    p.add_argument("--folds", type=int, default=4, help="Walk-forward folds")
    args = p.parse_args()

    text = " ".join(args.command)

    print("=" * 100)
    print("AI STRATEGY BUILDER (prototype)")
    print("=" * 100)
    print(f'Tum likha: "{text}"\n')

    print("Loading data...")
    data_result = load_data_result(interval=args.tf)
    df = data_result.data
    tf_label = infer_tf_label(df.index)
    flags = ["synthetic" if data_result.synthetic else "real" if data_result.real_data else "unverified"]
    if data_result.delayed: flags.append("delayed")
    if data_result.stale: flags.append("stale")
    print(f"  {len(df)} bars | {df.index[0]} -> {df.index[-1]} | TF: {tf_label}")
    print(f"  source: {data_result.provider} ({', '.join(flags)})\n")

    spec, metrics, wf, explanation, trades, wf_folds, equity = run_pipeline(text, df, tf_label, folds=args.folds)

    print("-" * 100)
    print("STEP 1-2 | UNDERSTANDING + STRATEGY SPEC (LLM baad mein yahi JSON schema output karega)")
    print("-" * 100)
    print(json.dumps({k: v for k, v in spec.items() if k != "warnings"}, indent=2, ensure_ascii=False))

    if spec["warnings"]:
        print("\nSTEP 3 | VALIDATION:")
        for w in spec["warnings"]:
            print(f"  ⚠️  {w}")

    if spec["notes"]:
        print("\nNOTES:")
        for note in spec["notes"]:
            print(f"  ℹ️  {note}")

    print("\n" + "-" * 100)
    print("STEP 4-5 | BACKTEST + WALK-FORWARD VERIFICATION (fixed params, unseen folds)")
    print("-" * 100)
    if wf_folds is not None:
        print(wf_folds.to_string(index=False))
        print(f"\n  OOS pooled: ret {wf['OOS Ret %']}% | Sharpe {wf['OOS Sharpe (med)']} | "
              f"win {wf['OOS Win %']}% | trades {wf['OOS Trades']} | DD {wf['OOS Max DD %']}% | verdict: {wf['Verdict']}")

    print("\n" + "-" * 100)
    print("STEP 6-7 | AI EXPLANATION + SUGGESTIONS")
    print("-" * 100)
    for line in explanation:
        print(line)

    print("\n" + "=" * 100)
    print("Rule: ye prototype hai - live deploy se pehle aur data, aur folds, aur paper trading.")


if __name__ == "__main__":
    main()
