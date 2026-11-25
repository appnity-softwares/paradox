"""Validation reports for real data (Master Prompt 6.6-6.8).

Runs the full validation battery (walk-forward, Monte Carlo trade
resampling, randomized start dates, cost stress, deflated Sharpe, robustness
score, readiness verdict) for a set of strategies on REAL stored ticks
resampled to 15-minute bars, and exports CSV reports under reports/.

Usage:
    python -m scripts.validate_strategies [symbol] [year] [start_month] [months]

Honesty rules: verdicts are never softened; every grid trial is recorded;
the report separates in-sample trial stats from out-of-sample walk-forward.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.config import get_symbol_spec
from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from core.optimize import PARAM_GRID
from core.validate import validate_strategy
from strategies.base import StrategyConfig
from strategies.library import (
    ADXTrend,
    BollingerReversion,
    EMACross,
    RSIReversal,
    Supertrend,
)

# Validation set: at least 3 strategies across categories (Part 6.8).
VALIDATION_SET = {
    "ema_cross": EMACross,
    "rsi_reversal": RSIReversal,
    "supertrend": Supertrend,
    "adx_trend": ADXTrend,
    "bollinger": BollingerReversion,
}

DEFAULT_PARAMS = {
    "ema_cross": {"fast": 20, "slow": 60},
    "rsi_reversal": {"length": 14, "oversold": 30, "overbought": 70},
    "supertrend": {"length": 10, "mult": 3.0},
    "adx_trend": {"fast": 20, "slow": 60, "adx_threshold": 20.0},
    "bollinger": {"length": 20, "num_std": 2.0},
}


def load_bars(symbol: str, year: int, month: int, months: int) -> pd.DataFrame:
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_year = year + (month + months - 1) // 12
    end_month = (month + months - 1) % 12 + 1
    end = int(datetime(end_year, end_month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    ticks = store.read_range(symbol, start, end)
    if ticks.empty:
        raise SystemExit(f"no ticks stored for {symbol} [{start},{end})")
    bars = resample_ticks_to_bars(ticks, "15m", price_col="mid")
    print(f"loaded {len(ticks):,} ticks -> {len(bars):,} x 15min bars "
          f"({bars.index[0]} .. {bars.index[-1]})")
    return bars


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    month = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    months = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    bars = load_bars(symbol, year, month, months)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    stamp = f"{year}{month:02d}"

    cfg = StrategyConfig(sl_method="fixed", sl_fixed=3.0, rr=2.0, risk_pct=1.0)
    dataset_hash = store.dataset_hash(symbol)

    summary_rows = []
    for key, cls in VALIDATION_SET.items():
        params = DEFAULT_PARAMS[key]
        print(f"\n=== validating {key} {params} ===")
        rep = validate_strategy(bars, key, cls, params, cfg, "15m", symbol=symbol)
        d = rep.to_dict()
        d["dataset_hash"] = dataset_hash
        d["bars"] = len(bars)
        summary_rows.append(d)
        print(f"  verdict={rep.verdict} score={rep.robustness_score:.0f} "
              f"trials={rep.n_trials} dsr={rep.deflated_sharpe_value:.3f}")
        print(f"  wf: {rep.walk_forward}")
        print(f"  mc: p5={rep.monte_carlo['p5_ret_pct']:.1f}% "
              f"prob_loss={rep.monte_carlo['prob_loss']:.2f}")
        print(f"  costs: {[ (r['multiplier'], round(r['Return %'],1)) for r in rep.cost_stress ]}")

        # Full per-strategy artifacts: trials + folds + trade log of the baseline.
        rep.trials.to_csv(out_dir / f"trials_{key}_{stamp}.csv", index=False)

    summary = pd.json_normalize(summary_rows)
    summary_path = out_dir / f"validation_summary_{stamp}.csv"
    summary.to_csv(summary_path, index=False)
    meta = {
        "symbol": symbol,
        "window": f"{year}-{month:02d} +{months}mo (15min bars)",
        "bars": len(bars),
        "dataset_hash": dataset_hash,
        "strategies": list(VALIDATION_SET),
    }
    (out_dir / f"validation_meta_{stamp}.json").write_text(json.dumps(meta, indent=1))
    print(f"\nsummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
