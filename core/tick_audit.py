"""Tick audit — hold the bar backtest accountable against real ticks.

The bar engine fills at the next bar's open ± half spread. That is a
model. The tick archive holds what ACTUALLY printed, second by second,
for every month since Jan 2024. This module runs the SAME strategy with
the SAME config through both engines and reports, trade by trade, how
wrong the bar model was:

- entry slippage (tick fill vs bar fill, signed so positive = paid worse)
- exit difference and per-trade P&L delta
- reason mismatches (bar said TP, ticks said stop — the dangerous ones)
- an honest verdict line

No smoothing, no excuses: if the bar backtest flatters itself, this says
by exactly how much.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from core.backtest import run_backtest
from core.tick_replay import DEFAULT_TICK_DIR, run_tick_replay


def _reason_counts(trades: list[dict]) -> dict:
    return dict(Counter(t["reason"] for t in trades))


def _streaks(pnls: list[float]) -> tuple[int, int]:
    """(longest win streak, longest loss streak)."""
    best_win = best_loss = cur_win = cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        best_win = max(best_win, cur_win)
        best_loss = max(best_loss, cur_loss)
    return best_win, best_loss


def _monthly_pnl(trades: list[dict], last: int = 6) -> list[dict]:
    if not trades:
        return []
    by_month: dict[str, float] = {}
    for t in trades:
        key = str(t["entry_time"])[:7]
        by_month[key] = by_month.get(key, 0.0) + float(t["pnl"])
    months = sorted(by_month)[-last:]
    return [{"month": m, "pnl": round(by_month[m], 2)} for m in months]


def audit_bar_vs_tick(df, strategy, cfg, start_equity: float = 10_000.0,
                      tick_dir=DEFAULT_TICK_DIR, slippage: float = 0.0) -> dict:
    ohlc = df[["open", "high", "low", "close", "volume"]]
    _, bar_tr = run_backtest(ohlc, strategy, cfg)
    tick = run_tick_replay(df, strategy, cfg, start_equity=start_equity,
                           tick_dir=tick_dir, slippage=slippage)
    bar_rows = bar_tr.to_dict("records") if len(bar_tr) else []
    tick_rows = tick["trades"]

    # align by (entry_time, side) — both engines enter on the same signal bar
    tick_by_key: dict[tuple, dict] = {}
    for t in tick_rows:
        key = (str(t["entry_time"]), t["side"])
        tick_by_key.setdefault(key, t)
    matched_keys = set()

    pairs = []
    slips, exit_diffs, pnl_deltas = [], [], []
    reason_mismatches = []
    for b in bar_rows:
        key = (str(b["entry_time"]), b["side"])
        tk = tick_by_key.get(key)
        if tk is None:
            continue
        matched_keys.add(key)
        direction = 1 if b["side"] == "LONG" else -1
        entry_slip = (tk["entry"] - b["entry"]) * direction   # + = paid worse
        exit_diff = (tk["exit"] - b["exit"]) * direction      # + = exited worse
        d_pnl = tk["pnl"] - b["pnl"]
        slips.append(entry_slip)
        exit_diffs.append(exit_diff)
        pnl_deltas.append(d_pnl)
        if b["reason"] != tk["reason"]:
            reason_mismatches.append({
                "entry_time": str(b["entry_time"]), "side": b["side"],
                "bar_reason": b["reason"], "tick_reason": tk["reason"],
                "pnl_delta": round(d_pnl, 2),
            })
        pairs.append({
            "entry_time": str(b["entry_time"]), "side": b["side"],
            "bar_entry": round(float(b["entry"]), 3),
            "tick_entry": round(float(tk["entry"]), 3),
            "entry_slip": round(entry_slip, 4),
            "bar_exit": round(float(b["exit"]), 3),
            "tick_exit": round(float(tk["exit"]), 3),
            "bar_pnl": round(float(b["pnl"]), 2),
            "tick_pnl": round(float(tk["pnl"]), 2),
            "pnl_delta": round(d_pnl, 2),
            "bar_reason": b["reason"], "tick_reason": tk["reason"],
        })

    bar_pnls = [float(b["pnl"]) for b in bar_rows]
    tick_pnls = [float(t["pnl"]) for t in tick_rows]
    bar_win_streak, bar_loss_streak = _streaks(bar_pnls)

    bar_only = len(bar_rows) - len(matched_keys)
    tick_only = len(tick_rows) - len(matched_keys)
    n = len(pairs)
    avg_slip = sum(slips) / n if n else 0.0
    avg_exit = sum(exit_diffs) / n if n else 0.0
    # the honest aggregate: total tick P&L minus total bar P&L (matched or not)
    pnl_delta_total = sum(tick_pnls) - sum(bar_pnls)
    bar_net = sum(bar_pnls)
    tick_net = sum(tick_pnls)

    # honest verdict
    lines = []
    if n == 0:
        lines.append("No overlapping trades to audit.")
    else:
        if abs(avg_slip) < 0.01:
            lines.append("Entries: bar fills match real ticks closely.")
        elif avg_slip > 0:
            lines.append(f"Entries: real ticks fill ${avg_slip:.3f}/oz WORSE than the bar model on average — the bar backtest flatters itself.")
        else:
            lines.append(f"Entries: real ticks fill ${-avg_slip:.3f}/oz BETTER than the bar model on average.")
        if bar_only or tick_only:
            lines.append(f"Alignment: {n} trades matched; {bar_only} bar-only, {tick_only} tick-only (skipped entries or diverging equity paths — e.g. the daily loss limit tripped at a different moment).")
        if reason_mismatches:
            lines.append(f"{len(reason_mismatches)} matched trades changed OUTCOME between bar model and real ticks — these are the trades where the bar engine lied.")
        else:
            lines.append("Every matched trade resolved the same way in both engines.")
        share = abs(pnl_delta_total) / abs(bar_net) * 100 if abs(bar_net) > 1e-9 else 0.0
        if share < 5:
            lines.append(f"Net P&L delta {pnl_delta_total:+.2f} ({share:.1f}% of bar result) — bar model trustworthy here.")
        else:
            lines.append(f"Net P&L delta {pnl_delta_total:+.2f} ({share:.1f}% of bar result) — do not trust the bar-only number.")

    return {
        "matched": n,
        "bar": {
            "trades": len(bar_rows),
            "net_pnl": round(bar_net, 2),
            "wins": sum(1 for p in bar_pnls if p > 0),
            "losses": sum(1 for p in bar_pnls if p <= 0),
            "win_streak": bar_win_streak,
            "loss_streak": bar_loss_streak,
            "best_trade": round(max(bar_pnls), 2) if bar_pnls else 0.0,
            "worst_trade": round(min(bar_pnls), 2) if bar_pnls else 0.0,
            "reasons": _reason_counts(bar_rows),
            "monthly": _monthly_pnl(bar_rows),
        },
        "tick": {
            "trades": tick["num_trades"],
            "net_pnl": round(tick_net, 2),
            "wins": sum(1 for p in tick_pnls if p > 0),
            "losses": sum(1 for p in tick_pnls if p <= 0),
            "reasons": _reason_counts(tick_rows),
            "ticks_scanned": tick["coverage"]["ticks_scanned"],
            "bars_without_ticks": tick["coverage"]["bars_without_ticks"],
        },
        "avg_entry_slip": round(avg_slip, 4),
        "worst_entry_slip": round(max(slips), 4) if slips else 0.0,
        "avg_exit_diff": round(avg_exit, 4),
        "pnl_delta_total": round(pnl_delta_total, 2),
        "bar_only": bar_only,
        "tick_only": tick_only,
        "reason_mismatches": reason_mismatches[:20],
        "pairs": pairs[:40],
        "verdict": lines,
    }
