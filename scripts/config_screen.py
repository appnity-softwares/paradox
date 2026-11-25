"""Config-matrix screen on REAL bars (Part 6.6 follow-up).

The random and Bayesian searches showed that parameter grinding on the
fixed-SL template finds no edge. This script screens STRUCTURAL config
changes instead: SL method (fixed vs ATR-multiple), reward:risk, and
session filter. Each cell is scored with the honest fixed-params
walk-forward OOS result on real resampled bars; every cell is recorded.

Usage:
    python -m scripts.config_screen [symbol] [year] [month] [months]
"""
from __future__ import annotations

import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from core.optimize import fixed_walk_forward
from strategies.base import StrategyConfig
from scripts.random_search import STRATEGIES

# Default params per strategy key (validation-set defaults).
DEFAULTS = {
    "ema_cross": {"fast": 20, "slow": 60},
    "rsi_reversal": {"length": 14, "oversold": 30, "overbought": 70},
    "supertrend": {"length": 10, "mult": 3.0},
    "bollinger": {"length": 20, "num_std": 2.0},
    "adx_trend": {"fast": 20, "slow": 60, "adx_threshold": 20.0},
}

SL_VARIANTS = [
    ("fixed_2.0", {"sl_method": "fixed", "sl_fixed": 2.0}),
    ("atr_1.5", {"sl_method": "atr", "sl_atr_mult": 1.5}),
    ("atr_2.5", {"sl_method": "atr", "sl_atr_mult": 2.5}),
]
RR_VARIANTS = [("rr_1.5", 1.5), ("rr_2.0", 2.0), ("rr_3.0", 3.0)]
SESSION_VARIANTS = [("all", "all"), ("london_ny", "london_ny")]


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    month = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    months = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    stamp = f"{year}{month:02d}"

    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_year = year + (month + months - 1) // 12
    end_month = (month + months - 1) % 12 + 1
    end = int(datetime(end_year, end_month, 1, tzinfo=timezone.utc).timestamp() * 1000)

    ticks = store.read_range(symbol, start, end)
    if ticks.empty:
        print(f"no ticks stored for {symbol} [{start},{end})")
        return 1
    bars = resample_ticks_to_bars(ticks, "15m", price_col="mid")
    print(f"bars={len(bars)}  cells={len(STRATEGIES) * len(SL_VARIANTS) * len(RR_VARIANTS) * len(SESSION_VARIANTS)}")

    rows = []
    for key, cls in STRATEGIES.items():
        for sl_name, sl_kwargs in SL_VARIANTS:
            for rr_name, rr in RR_VARIANTS:
                for sess_name, sess in SESSION_VARIANTS:
                    cfg = StrategyConfig(session=sess, rr=rr, **sl_kwargs)
                    label = f"{key}|{sl_name}|{rr_name}|{sess_name}"
                    try:
                        _, _, wf = fixed_walk_forward(
                            bars, cls, dict(DEFAULTS[key]), cfg, "15m")
                    except Exception as exc:
                        rows.append({"cell": label, "error": str(exc)[:120]})
                        continue
                    rows.append({
                        "cell": label,
                        "strategy": key, "sl": sl_name, "rr": rr_name,
                        "session": sess_name,
                        "OOS Ret %": wf.get("OOS Ret %"),
                        "OOS Sharpe (med)": wf.get("OOS Sharpe (med)"),
                        "OOS Trades": wf.get("OOS Trades"),
                        "OOS Max DD %": wf.get("OOS Max DD %"),
                        "Profitable Folds %": wf.get("Profitable Folds %"),
                        "Verdict": wf.get("Verdict"),
                    })

    df = pd.DataFrame(rows)
    out = Path("reports") / f"config_screen_{stamp}.csv"
    df.to_csv(out, index=False)

    # Rank: OOS return, then Sharpe. Print the whole matrix honestly.
    if "OOS Ret %" in df.columns:
        df = df.sort_values("OOS Ret %", ascending=False)
    print(df.to_string(index=False))
    n_pos = int((df.get("OOS Ret %", pd.Series(dtype=float)) > 0).sum())
    print(f"\npositive OOS cells: {n_pos} / {len(df)}  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
