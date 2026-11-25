"""Bayesian (TPE) parameter search on REAL bars with an IS/OOS split (Part 6.6).

Builds on :mod:`scripts.random_search`: same wider ranges, same 70/30
IS/OOS split, same honest protocol. Optuna's TPESampler with a fixed seed
keeps the search deterministic. The ``fast < slow`` constraint is encoded
in the search space itself (slow is sampled from ``max(lo, fast + 1)``) so
recorded parameters are exactly the parameters run. Every trial is
recorded; the top-K in-sample combos are then evaluated out-of-sample.

Usage:
    python -m scripts.optuna_search [symbol] [year] [month] [months] [n_trials] [top_k]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import pandas as pd

from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from scripts.random_search import RANDOM_RANGES, STRATEGIES, run_one
from strategies.base import StrategyConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

MIN_TRADES_IS = 15


def make_objective(key, cls, is_bars, cfg):
    space = RANDOM_RANGES[key]

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}
        for name, (lo, hi) in space.items():
            if name == "fast":
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            elif name == "slow":
                # Enforce fast < slow inside the space so recorded params
                # are exactly the params run.
                params[name] = trial.suggest_int(
                    name, max(int(lo), params["fast"] + 1), int(hi)
                )
            elif name in ("length", "adx_length"):
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            else:
                params[name] = trial.suggest_float(name, float(lo), float(hi))
        try:
            _, _, m = run_one(is_bars, cls, params, cfg)
        except Exception:
            return -10.0  # invalid combo: rank last, stay recorded
        n = int(m["Total Trades"])
        if n < MIN_TRADES_IS:
            # Discourage without hiding: score scales with how close to
            # the minimum trade count the combo got.
            return -10.0 + n / 100.0
        return float(m["Sharpe"])

    return objective


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    month = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    months = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    n_trials = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    top_k = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    stamp = f"{year}{month:02d}"

    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_year = year + (month + months - 1) // 12
    end_month = (month + months - 1) % 12 + 1
    end = int(datetime(end_year, end_month, 1, tzinfo=timezone.utc).timestamp() * 1000)

    ticks = store.read_range(symbol, start, end)
    if ticks.empty:
        print(f"no ticks stored for {symbol} [{start},{end})")
        return 1
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0, risk_pct=1.0)
    bars = resample_ticks_to_bars(ticks, "15m", price_col="mid")
    split = int(len(bars) * 0.7)
    is_bars, oos_bars = bars.iloc[:split], bars.iloc[split:]
    print(f"bars total={len(bars)} IS={len(is_bars)} OOS={len(oos_bars)} "
          f"n_trials={n_trials} per strategy")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    oos_rows = []

    for key, cls in STRATEGIES.items():
        sampler = optuna.samplers.TPESampler(seed=42)  # deterministic
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(make_objective(key, cls, is_bars, cfg), n_trials=n_trials)

        trials_df = study.trials_dataframe()
        trials_df.insert(0, "strategy", key)
        trials_df.to_csv(out_dir / f"optuna_trials_{key}_{stamp}.csv", index=False)

        # Top-K completed trials by IS value -> OOS evaluation.
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        completed.sort(key=lambda t: t.value, reverse=True)
        for rank, t in enumerate(completed[:top_k], 1):
            params = dict(t.params)
            try:
                _, _, m_oos = run_one(oos_bars, cls, params, cfg)
                oos_rows.append({"strategy": key, "rank": rank, **params, **{
                    "IS_value": t.value,
                    "OOS_Sharpe": m_oos["Sharpe"], "OOS_Ret %": m_oos["Return %"],
                    "OOS_Trades": m_oos["Total Trades"],
                    "OOS_ProfitFactor": m_oos["Profit Factor"],
                }})
            except Exception as exc:
                oos_rows.append({"strategy": key, "rank": rank, **params,
                                 "error": str(exc)[:120]})
        best = completed[0] if completed else None
        print(f"{key:<13} best IS Sharpe={best.value if best else float('nan'):>7.2f} "
              f"params={best.params if best else {}}")

    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(out_dir / f"optuna_oos_{stamp}.csv", index=False)
    print("\n== OOS evaluation of top-K ==")
    print(oos_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
