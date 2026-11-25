"""Random parameter search on REAL bars with an IS/OOS split (Part 6.6).

Anti-overfitting protocol:
- Random-sample parameter combos from WIDER ranges than the fixed grids
  (random search over a 3x3 grid is just a shuffle).
- Evaluate each combo on the IN-SAMPLE split only.
- Rank by IS Sharpe with a minimum trade count; evaluate the top-K on the
  OUT-OF-SAMPLE split and report both, honestly.
- Every trial (IS and OOS) is recorded to CSV under reports/.

Usage:
    python -m scripts.random_search [symbol] [year] [month] [months] [n_trials] [top_k]
"""
from __future__ import annotations

import copy
import random
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.backtest import run_backtest
from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from core.metrics import compute_metrics
from strategies.base import StrategyConfig
from strategies.library import (
    ADXTrend,
    BollingerReversion,
    EMACross,
    RSIReversal,
    Supertrend,
)

STRATEGIES = {
    "ema_cross": EMACross,
    "rsi_reversal": RSIReversal,
    "supertrend": Supertrend,
    "bollinger": BollingerReversion,
    "adx_trend": ADXTrend,
}

# Wider random ranges than the fixed PARAM_GRID grids (random search over a
# 3x3 grid is just a shuffle).
RANDOM_RANGES = {
    "ema_cross": {"fast": (5, 40), "slow": (40, 220)},
    "rsi_reversal": {"length": (5, 21), "oversold": (15, 40), "overbought": (60, 90)},
    "supertrend": {"length": (5, 30), "mult": (1.5, 4.5)},
    "bollinger": {"length": (10, 50), "num_std": (1.5, 3.5)},
    "adx_trend": {"fast": (8, 40), "slow": (40, 160), "adx_threshold": (10.0, 35.0)},
}

MIN_TRADES_IS = 15
TOP_K = 3


def sample_params(key: str, rng: random.Random) -> dict:
    ranges = RANDOM_RANGES[key]
    out = {}
    for name, (lo, hi) in ranges.items():
        if name in ("fast", "slow", "length", "adx_length"):
            out[name] = rng.randint(int(lo), int(hi))
        else:
            out[name] = round(rng.uniform(lo, hi), 2)
    if "fast" in out and out["fast"] >= out["slow"]:
        out["fast"], out["slow"] = out["slow"], out["fast"]
    return out


def run_one(df, cls, params, cfg):
    cfg_c = copy.deepcopy(cfg)
    strat = cls(cfg_c, **params)
    equity, trades = run_backtest(df, strat, cfg_c)
    m = compute_metrics(equity, trades, cfg_c, "15m")
    return equity, trades, m


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    month = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    months = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    n_trials = int(sys.argv[5]) if len(sys.argv) > 5 else 40
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
    # IS/OOS split: first ~70% in-sample, last ~30% out-of-sample.
    split = int(len(bars) * 0.7)
    is_bars, oos_bars = bars.iloc[:split], bars.iloc[split:]
    print(f"bars total={len(bars)} IS={len(is_bars)} OOS={len(oos_bars)}")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    trial_rows = []
    oos_rows = []

    for key, cls in STRATEGIES.items():
        rng = random.Random(42 + zlib.crc32(key.encode()))
        seen = set()
        is_results = []
        attempts = 0
        while len(is_results) < n_trials and attempts < n_trials * 4:
            attempts += 1
            params = sample_params(key, rng)
            frozen = tuple(sorted(params.items()))
            if frozen in seen:
                continue
            seen.add(frozen)
            try:
                _, _, m = run_one(is_bars, cls, params, cfg)
            except Exception as exc:  # invalid combo, record and continue
                trial_rows.append({"strategy": key, **params, "error": str(exc)[:120]})
                continue
            is_results.append((m, params))
            trial_rows.append({"strategy": key, **params, **{
                "IS_Sharpe": m["Sharpe"], "IS_Ret %": m["Return %"],
                "IS_Trades": m["Total Trades"], "IS_ProfitFactor": m["Profit Factor"],
            }})

        # Rank on IS Sharpe with a minimum trade count; fall back to all.
        valid = [r for r in is_results if r[0]["Total Trades"] >= MIN_TRADES_IS]
        pool = valid if valid else is_results
        pool.sort(key=lambda r: r[0]["Sharpe"], reverse=True)
        for rank, (m, params) in enumerate(pool[:top_k], 1):
            try:
                _, _, m_oos = run_one(oos_bars, cls, params, cfg)
                oos_rows.append({"strategy": key, "rank": rank, **params, **{
                    "OOS_Sharpe": m_oos["Sharpe"], "OOS_Ret %": m_oos["Return %"],
                    "OOS_Trades": m_oos["Total Trades"],
                    "OOS_ProfitFactor": m_oos["Profit Factor"],
                    "IS_Sharpe": m["Sharpe"], "IS_Ret %": m["Return %"],
                    "IS_Trades": m["Total Trades"],
                }})
            except Exception as exc:
                oos_rows.append({"strategy": key, "rank": rank, **params,
                                 "error": str(exc)[:120]})

    pd.DataFrame(trial_rows).to_csv(out_dir / f"random_search_trials_{stamp}.csv", index=False)
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(out_dir / f"random_search_oos_{stamp}.csv", index=False)

    print("\n== IS top trials ==")
    print(pd.DataFrame(trial_rows).to_string(index=False))
    print("\n== OOS evaluation of top-K ==")
    print(oos_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
