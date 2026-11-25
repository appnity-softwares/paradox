"""Verify promising config-screen cells with the FULL validation stack.

A 90-cell screen is multiple comparison; the best cells must survive:
1. the full validation stack (Monte Carlo, cost stress, deflated Sharpe
   adjusted for the number of trials actually explored), and
2. a TRUE holdout: the same fixed config on a different, later period.

Usage:
    python -m scripts.verify_cells [symbol] [year] [month] [months] \
        [holdout_year] [holdout_month] [holdout_months] [cell ...]

Cell format:  key|sl_name|rr_name|session   (names from config_screen).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from core.optimize import fixed_walk_forward
from core.validate import validate_strategy
from strategies.base import StrategyConfig
from scripts.config_screen import DEFAULTS, RR_VARIANTS, SL_VARIANTS
from scripts.random_search import STRATEGIES

SL_KW = dict(SL_VARIANTS)
RR_VAL = dict(RR_VARIANTS)


def parse_cell(cell: str):
    key, sl_name, rr_name, sess = cell.split("|")
    cfg = StrategyConfig(session=sess, rr=RR_VAL[rr_name], **SL_KW[sl_name])
    return key, cfg


def load_bars(symbol, year, month, months):
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_year = year + (month + months - 1) // 12
    end_month = (month + months - 1) % 12 + 1
    end = int(datetime(end_year, end_month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    ticks = store.read_range(symbol, start, end)
    return resample_ticks_to_bars(ticks, "15m", price_col="mid")


def main() -> int:
    args = sys.argv[1:]
    symbol = args[0] if args else "XAUUSD"
    year, month, months = int(args[1]), int(args[2]), int(args[3])
    hy, hm, hmonths = int(args[4]), int(args[5]), int(args[6])
    cells = args[7:] or [
        "ema_cross|atr_1.5|rr_1.5|london_ny",
        "ema_cross|atr_1.5|rr_3.0|all",
        "ema_cross|atr_2.5|rr_2.0|all",
    ]
    stamp = f"{year}{month:02d}"

    bars = load_bars(symbol, year, month, months)
    holdout = load_bars(symbol, hy, hm, hmonths)
    print(f"validation bars={len(bars)}  holdout bars={len(holdout)} "
          f"({hy}-{hm:02d}, unseen by the screen)")

    rows = []
    for cell in cells:
        key, cfg = parse_cell(cell)
        cls = STRATEGIES[key]
        params = dict(DEFAULTS[key])
        print(f"\n=== {cell} ===")
        rep = validate_strategy(bars, key, cls, params, cfg, "15m", symbol=symbol)
        v = rep.to_dict()
        wf = v["wf"]
        mc = v["mc"]
        cs2 = next((d for d in v["cost_stress"]
                    if d.get("multiplier") == 2.0), {})
        print(f"  verdict={v['verdict']} score={v['robustness_score']} "
              f"dsr={v['deflated_sharpe']:.3f}")
        print(f"  wf : OOS Ret {wf.get('OOS Ret %')}%  Sharpe {wf.get('OOS Sharpe (med)')}  "
              f"folds+ {wf.get('Profitable Folds %')}%")
        print(f"  mc : p5 {mc.get('p5_ret_pct')}%  prob_loss {mc.get('prob_loss')}")
        print(f"  cost: 2x -> {cs2.get('Return %')}%")
        # True holdout: same fixed config, later unseen period.
        _, _, hwf = fixed_walk_forward(holdout, cls, params, cfg, "15m")
        print(f"  holdout {hy}-{hm:02d}: OOS Ret {hwf.get('OOS Ret %')}%  "
              f"Sharpe {hwf.get('OOS Sharpe (med)')}  folds+ "
              f"{hwf.get('Profitable Folds %')}%  verdict {hwf.get('Verdict')}")
        rows.append({
            "cell": cell,
            "verdict": v["verdict"], "score": v["robustness_score"],
            "dsr": v["deflated_sharpe"],
            "wf_oos_ret": wf.get("OOS Ret %"),
            "wf_sharpe": wf.get("OOS Sharpe (med)"),
            "mc_p5": mc.get("p5_ret_pct"), "mc_prob_loss": mc.get("prob_loss"),
            "cost2x_ret": cs2.get("Return %"),
            "holdout_ret": hwf.get("OOS Ret %"),
            "holdout_sharpe": hwf.get("OOS Sharpe (med)"),
            "holdout_folds_pos": hwf.get("Profitable Folds %"),
            "holdout_verdict": hwf.get("Verdict"),
        })

    out = Path("reports") / f"verify_cells_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
