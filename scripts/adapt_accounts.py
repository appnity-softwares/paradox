"""Part 7.3 demo on REAL data: adapt the validated cell to 10k/100k accounts.

Runs the adapted configuration for a 10,000 USD and a 100,000 USD account
on the same real XAUUSD 15m bars, and exports per-account suitability
reports, adapted configs, equity curves and the account-specific
drawdown. Uses the ema_cross ATR cell measured in Phase 1C.

Usage:
    python -m scripts.adapt_accounts [symbol] [year] [month] [months]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.account import Account, AccountLimits
from core.adaptation import (
    StrategyProfile,
    adapt_config,
    suggest_adaptation,
    suitability_report,
)
from core.backtest import run_backtest
from core.config import SYMBOL_SPECS
from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from core.metrics import compute_metrics
from strategies.base import StrategyConfig
from strategies.library import EMACross

XAU = SYMBOL_SPECS["XAUUSD"]

# Profile measured by the Phase 1 stack (ema_cross ATR cell, H1-2025):
# worst DD 35%, typical ATR(14) x 1.5 stop, round-trip cost 0.50 price units.
PROFILE = StrategyProfile(
    key="ema_cross", name="EMA cross (ATR 1.5 stops, RR 1.5, London+NY)",
    max_dd_pct=35.0, typical_stop_distance=12.0,
    recommended_risk_pct=1.0, cost_per_round_trip=0.50,
)

ACCOUNTS = {
    "10k": Account(account_id="10k", balance=10_000.0, leverage=100,
                   limits=AccountLimits(risk_per_trade_pct=1.0,
                                        max_drawdown_pct=50.0)),
    "100k": Account(account_id="100k", balance=100_000.0, leverage=100,
                    limits=AccountLimits(risk_per_trade_pct=1.0,
                                         max_drawdown_pct=50.0)),
}


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

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    summary_rows = []

    for label, acct in ACCOUNTS.items():
        adapted = adapt_config(acct, PROFILE, XAU, price=float(bars["close"].iloc[-1]))
        suit = suitability_report(acct, PROFILE, XAU,
                                  price=float(bars["close"].iloc[-1]))
        strat_cfg = StrategyConfig(
            sl_method="atr", sl_atr_mult=1.5, rr=1.5,
            session="london_ny",
            risk_pct=adapted.risk_per_trade_pct,
            max_trades_per_day=adapted.max_trades_per_day,
            daily_loss_limit_pct=adapted.daily_loss_limit_pct,
        )
        equity, trades = run_backtest(
            bars, EMACross(strat_cfg, fast=20, slow=60), strat_cfg,
            start_equity=acct.balance,
        )
        metrics = compute_metrics(equity, trades, strat_cfg, "15m")

        eq_df = pd.DataFrame({"equity": equity})
        eq_df["drawdown_pct"] = eq_df["equity"] / eq_df["equity"].cummax() - 1.0
        eq_df.to_csv(out_dir / f"adapted_{label}_{stamp}.csv")

        suggestions = suggest_adaptation(
            acct, PROFILE, XAU, price=float(bars["close"].iloc[-1]),
            current_atr=PROFILE.typical_stop_distance / 1.5,
            equity_high_watermark=acct.balance,
        )
        summary_rows.append({
            "account": label,
            "balance": acct.balance,
            "suitable": suit.suitable,
            "lot": adapted.lot,
            "risk_money": round(adapted.risk_money, 2),
            "margin_required": round(adapted.margin_required, 2),
            "max_open_positions": adapted.max_open_positions,
            "max_total_exposure_lots": adapted.max_total_exposure_lots,
            "max_trades_per_day": adapted.max_trades_per_day,
            "ret_pct": metrics["Return %"],
            "sharpe": metrics["Sharpe"],
            "max_dd_pct": metrics["Max DD %"],
            "trades": metrics["Total Trades"],
            "suggestions": json.dumps([s["what"] + ":" + str(s["to"]) for s in suggestions]),
        })
        print(f"\n== {label} account ({acct.balance:,.0f} {acct.currency}) ==")
        print(f"  suitable: {suit.suitable}")
        for c in suit.checks:
            print(f"   [{'x' if c['passed'] else ' '}] {c['name']}: {c['detail']}")
        print(f"  adapted : lot {adapted.lot}  risk {adapted.risk_money:.2f}  "
              f"margin {adapted.margin_required:.2f}  "
              f"max_pos {adapted.max_open_positions}  "
              f"max_trades/day {adapted.max_trades_per_day}")
        print(f"  result  : ret {metrics['Return %']:.1f}%  sharpe {metrics['Sharpe']:.2f}  "
              f"dd {metrics['Max DD %']:.1f}%  trades {metrics['Total Trades']}")

    pd.DataFrame(summary_rows).to_csv(out_dir / f"adapted_accounts_{stamp}.csv",
                                      index=False)
    print(f"\nreports -> adapted_accounts_{stamp}.csv, adapted_<acct>_{stamp}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
