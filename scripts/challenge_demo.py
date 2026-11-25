"""Part 10.3 demo: challenge tracker on REAL data.

Replays the validated ema_cross ATR cell (real XAUUSD H1-2025 bars) as a
trader would have traded it, then runs the deterministic challenge
simulator against a VERIFIED FTMO-style rule set. Reports pass/fail,
drawdown path, days used and rule hits.

A simulation is NOT a guarantee of passing a real challenge.

Usage: python -m scripts.challenge_demo [symbol] [year] [month] [months]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.backtest import run_backtest
from core.compliance import FirmRuleSet, RuleStore, simulate_challenge
from core.compliance.rules import DailyLossRule, MaxLossRule
from core.config import SYMBOL_SPECS
from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from strategies.base import StrategyConfig
from strategies.library import EMACross


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    month = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    months = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_year = year + (month + months - 1) // 12
    end_month = (month + months - 1) % 12 + 1
    end = int(datetime(end_year, end_month, 1, tzinfo=timezone.utc).timestamp() * 1000)

    ticks = store.read_range(symbol, start, end)
    if ticks.empty:
        print(f"no ticks stored for {symbol}")
        return 1
    bars = resample_ticks_to_bars(ticks, "15m", price_col="mid")

    # The validated cell: ema_cross, ATR 1.5 stops, RR 1.5, London+NY.
    cfg = StrategyConfig(sl_method="atr", sl_atr_mult=1.5, rr=1.5,
                         session="london_ny", risk_pct=1.0)
    equity, trades = run_backtest(bars, EMACross(cfg, fast=20, slow=60), cfg,
                                  start_equity=100_000.0)
    trades = trades.copy()
    trades["volume"] = trades["units"] / 100.0  # units -> lots (contract 100)

    # A verified FTMO-style rule set (schema per Part 10.1).
    rs = FirmRuleSet(
        firm="ftmo_style", program="100k", version="demo-v1",
        status="verified", verified_by="human-demo",
        ea_policy="allowed",
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=100_000.0, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00",
                                 "America/New_York", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
        min_trading_days=4, weekend_holding="allowed",
    )
    rule_store = RuleStore("state/rules")
    try:
        rule_store.add_draft(FirmRuleSet(**{**rs.to_dict(), "status": "draft"}))
    except ValueError:
        pass  # version already stored from a previous run
    try:
        rule_store.verify(rs.firm, rs.program, rs.version, "human-demo")
    except ValueError:
        pass  # already verified in a previous run

    print(f"== Challenge replay: {rs.firm} {rs.program} on real "
          f"{symbol} {year}-{month:02d} +{months}m ==")
    for risk in (1.0, 0.5):
        cfg_r = StrategyConfig(sl_method="atr", sl_atr_mult=1.5, rr=1.5,
                               session="london_ny", risk_pct=risk)
        eq_r, tr_r = run_backtest(bars, EMACross(cfg_r, fast=20, slow=60),
                                  cfg_r, start_equity=100_000.0)
        tr_r = tr_r.copy()
        tr_r["volume"] = tr_r["units"] / 100.0
        result = simulate_challenge(rs, eq_r, tr_r)
        print(f"  risk {risk}%: trades {len(tr_r)}  "
              f"return {result.final_equity / rs.starting_balance * 100 - 100:+.1f}%  "
              f"max DD {result.max_drawdown_pct:.1f}%  "
              f"days {result.trading_days_used}  "
              f"target {result.target_hit}  breached {result.breached}  "
              f"PASSED {result.passed}")
        for hit in result.rule_hits:
            print(f"    rule hit: {hit}")
    print("  note: deterministic replay of ONE price path; not a guarantee "
          "of passing a real challenge.")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    pd.DataFrame([{k: v for k, v in result.__dict__.items()
                   if k not in ("daily_dd_path", "daily_loss_path", "notes")}]
                 ).to_csv(out / "challenge_demo_202501.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
