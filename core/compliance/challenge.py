"""Challenge tracker and simulator (Part 10.3, last item).

DETERMINISTIC: replays a realized or paper equity curve plus its trade
list against a VERIFIED FirmRuleSet and reports pass/fail per rule, the
drawdown path, trading days used and every rule hit.

A simulation is NOT a guarantee of passing a real challenge — it is a
deterministic replay of one price path under one cost model.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import pandas as pd

from core.compliance.rules import STATUS_VERIFIED, FirmRuleSet


@dataclass
class ChallengeResult:
    passed: bool
    target_hit: bool
    breached: bool
    rule_hits: list[dict] = field(default_factory=list)
    final_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    trading_days_used: int = 0
    daily_loss_path: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def simulate_challenge(rs: FirmRuleSet, equity: pd.Series,
                       trades: pd.DataFrame) -> ChallengeResult:
    """Replay `equity` (tz-aware DatetimeIndex) and `trades` (columns:
    pnl, volume, entry_time, exit_time) against the rule set."""
    if rs.status != STATUS_VERIFIED:
        return ChallengeResult(
            passed=False, target_hit=False, breached=False,
            notes=["ruleset not verified: simulation refused"])

    tz = rs.daily_loss.reset_tz
    reset_h, reset_m = (int(x) for x in rs.daily_loss.reset_time.split(":"))
    daily_limit = rs.daily_loss_amount
    max_limit = rs.max_loss_amount
    starting_balance = rs.starting_balance

    def firm_day(ts: pd.Timestamp) -> str:
        t = ts.tz_convert(tz) if ts.tzinfo is not None else ts.tz_localize(tz)
        return (t - _dt.timedelta(hours=reset_h, minutes=reset_m)) \
            .date().isoformat()

    # ── pass 1: attribute closed trades to firm-days ────────────────────
    day_pnl: dict[str, float] = {}
    day_volume: dict[str, float] = {}
    rule_hits: list[dict] = []
    if not trades.empty:
        for _, tr in trades.iterrows():
            day = firm_day(pd.Timestamp(tr["exit_time"]))
            day_pnl[day] = day_pnl.get(day, 0.0) + float(tr["pnl"])
            vol = float(tr.get("volume", 0.0) or 0.0)
            day_volume[day] = day_volume.get(day, 0.0) + vol
            if rs.max_lot is not None and vol > rs.max_lot:
                rule_hits.append({"rule": "max_lot",
                                  "time": str(tr["exit_time"])})
            if rs.weekend_holding == "forbidden" and \
                    tr.get("entry_time") is not None:
                entry = pd.Timestamp(tr["entry_time"])
                entry = entry.tz_convert(tz) if entry.tzinfo is not None \
                    else entry.tz_localize(tz)
                if entry.weekday() == 4 and \
                        entry.hour >= rs.weekend_cutoff_utc_hour:
                    rule_hits.append({"rule": "weekend_entry",
                                      "time": str(tr["entry_time"])})

    # ── pass 2: replay the equity path ───────────────────────────────────
    result = ChallengeResult(passed=False, target_hit=False, breached=False)
    hwm = starting_balance
    breached_reason: str | None = None
    current_day: str | None = None
    day_min_equity = starting_balance
    day_start_equity = starting_balance
    prev_eq = starting_balance

    def close_day(day: str) -> None:
        nonlocal breached_reason
        if rs.daily_loss.includes_floating:
            loss = day_start_equity - day_min_equity
        else:
            loss = -day_pnl.get(day, 0.0)
        result.daily_loss_path.append(round(loss, 2))
        if loss >= daily_limit:
            breached_reason = breached_reason or \
                (f"daily_loss {loss:.2f} >= {daily_limit:.2f} on {day}")
        if day_volume.get(day, 0.0) >= rs.min_day_volume_lots:
            result.trading_days_used += 1

    for ts, eq in equity.items():
        eq = float(eq)
        day = firm_day(pd.Timestamp(ts))
        if current_day is None:
            current_day, day_start_equity, day_min_equity = day, eq, eq
        elif day != current_day:
            close_day(current_day)
            # The new day's reference is the last equity sample BEFORE the
            # reset (best observable estimate of equity at reset).
            current_day = day
            day_start_equity = prev_eq
            day_min_equity = min(prev_eq, eq)
        day_min_equity = min(day_min_equity, eq)
        prev_eq = eq

        if eq > hwm:
            hwm = eq
        dd_ref = starting_balance if rs.max_loss.type == "static" else hwm
        dd = dd_ref - eq
        if dd >= max_limit:
            breached_reason = breached_reason or \
                f"max_loss {dd:.2f} >= {max_limit:.2f} at {ts}"
        if eq - starting_balance >= rs.profit_target_amount:
            result.target_hit = True

    if current_day is not None:
        close_day(current_day)

    # ── verdict ──────────────────────────────────────────────────────────
    result.final_equity = float(equity.iloc[-1])
    result.max_drawdown_pct = round(
        max((1.0 - e / h) for h, e in zip(equity.cummax(), equity)), 6) * 100.0
    result.breached = breached_reason is not None
    if breached_reason:
        rule_hits.append({"rule": "breach", "detail": breached_reason})
    result.rule_hits = rule_hits
    result.notes.append(
        "Deterministic replay of ONE price path; not a guarantee of "
        "passing a real challenge.")
    result.passed = result.target_hit and not result.breached and \
        result.trading_days_used >= rs.min_trading_days
    return result
