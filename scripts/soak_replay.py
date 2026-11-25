"""Replay soak (Part 14.4 paper-soak readiness): fast-forwards the
EXACT live pipeline — tick feed -> PaperBroker fills + TradeManager
management -> 15m resample -> causal strategy signals -> ExecutionRouter
(risk + compliance + mode gate) — over REAL historical XAUUSD ticks.

This is not a backtest: every order goes through the same single order
path the live router enforces, every tick drives broker and management
exactly as the live feed would. It shakes out state bugs (position
books, pending orders, daily resets, kill-switch interaction) in
minutes instead of waiting four wall-clock weeks.

Honesty labels: this is a REPLAY of real data through live code — not
live trading and not a performance forecast. The summary JSON records
source_label "REPLAY_SOAK".

Usage: python -m scripts.soak_replay [days] [start_ms]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.account import Account
from core.adaptation import StrategyProfile
from core.compliance import ComplianceEngine, FirmRuleSet
from core.compliance.rules import DailyLossRule, MaxLossRule
from core.config import SYMBOL_SPECS
from core.execution import (AuditLog, ExecutionRouter, ModeGate, Notifier,
                            PaperBroker)
from core.execution.management import ManagementRules, TradeManager
from core.indicators import atr
from core.marketdata import store
from core.marketdata.resampler import resample_ticks_to_bars
from core.risk import AccountSnapshot, DeterministicRiskEngine, RiskConfig
from strategies.base import StrategyConfig, SESSIONS
from strategies.library import EMACross

TICK_MS = 60_000          # resample granularity for the live bar builder
FAST, SLOW = 20, 60


def build_stack(start_equity: float = 100_000.0):
    """The live stack exactly as deployment wires it."""
    broker = PaperBroker(start_balance=start_equity)
    broker.connect()
    rules = FirmRuleSet(
        firm="soak_firm", program="100k", version="soak-v1",
        status="verified", verified_by="human-soak-operator",
        ea_policy="allowed",
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=start_equity, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
        min_trading_days=0, weekend_holding="allowed")
    router = ExecutionRouter(broker,
                             DeterministicRiskEngine(RiskConfig(version="v1")),
                             ComplianceEngine(), ModeGate(),
                             AuditLog(":memory:"), Notifier())
    account = Account(account_id="soak-1", balance=start_equity, leverage=100)
    router.register_account(
        account,
        AccountSnapshot("soak-1", start_equity, start_equity, 0.0, 0, {},
                        0.0, 0.0, start_equity, {}, {}, 0),
        StrategyProfile("ema_cross", "EMA cross", 35.0, 12.0, 0.5, 0.5),
        rules, SYMBOL_SPECS["XAUUSD"], 2600.0)
    router.modes.set_mode("soak-1", "paper")
    manager = TradeManager(broker, ManagementRules(), router.audit,
                           router.notifier)
    return broker, router, manager, rules


def _snapshot(router, broker, ts_ms: int, spread: float = 0.20):
    """Refresh the router's account snapshot from live broker state,
    including the daily realized PnL since the 17:00 UTC reset — the
    compliance daily-loss rule cannot fire without it."""
    old = router.snapshots["soak-1"]
    equity = broker.equity
    day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    reset = day.replace(hour=17, minute=0, second=0, microsecond=0)
    boundary = int((reset if day >= reset
                    else reset - timedelta(days=1)).timestamp() * 1000)
    daily_realized = sum(float(c["pnl"]) for c in broker.closed
                         if c.get("ts_ms", 0) >= boundary)
    floating = 0.0
    for pos in broker.positions():
        mark = ask_of(broker, pos)
        if mark is not None:
            floating += (mark - pos["entry_price"]) * pos["side"]                 * pos["lot"] * 100.0
    router.snapshots["soak-1"] = AccountSnapshot(
        "soak-1", equity, broker.balance, 0.0, len(broker.positions()),
        {}, daily_realized, floating,
        max(old.equity_high_watermark, equity),
        {"XAUUSD": ts_ms}, {"XAUUSD": spread}, ts_ms)
    return router.snapshots["soak-1"]


def ask_of(broker, pos):
    bid, ask = broker.quotes.get(pos.get("symbol", "XAUUSD"), (None, None))
    if bid is None:
        return None
    return ask if pos["side"] == 1 else bid


def run_soak(days: int = 5, start_ms: int | None = None) -> dict:
    from core.config import COST_MODELS
    spec = SYMBOL_SPECS["XAUUSD"]
    if start_ms is None:
        start_ms = 1735776000000                     # 2025-01-02
    end_ms = start_ms + days * 86_400_000
    ticks = store.read_range("XAUUSD", start_ms, end_ms)
    if ticks.empty:
        raise SystemExit(f"no real ticks stored for the soak window")
    times = ticks["time_utc_ms"].to_numpy(dtype=np.int64)
    bid = ticks["bid"].to_numpy(dtype=float)
    ask = ticks["ask"].to_numpy(dtype=float)

    # Causal signal plan from the same strategy + config the backtests
    # validated: signal at bar i-1 acts at bar i open, ATR(14) stops.
    bars = resample_ticks_to_bars(ticks, "15m", price_col="mid")
    cfg = StrategyConfig(sl_method="atr", sl_atr_mult=1.5, rr=1.5,
                         session="london_ny", risk_pct=0.5)
    sig = EMACross(cfg, fast=FAST, slow=SLOW).generate_signals(bars) \
        .fillna(0.0).to_numpy(dtype=float)
    atr_np = atr(bars, 14).to_numpy(dtype=float)
    bar_index = {ts: i for i, ts in enumerate(bars.index)}
    bar_open_ms = np.array([int(ts.timestamp() * 1000)
                            for ts in bars.index])
    sess_lo, sess_hi = SESSIONS[cfg.session]

    broker, router, manager, _rules = build_stack()
    cm = COST_MODELS["XAUUSD_ECN"]

    stats = {"orders_submitted": 0, "filled": 0, "rejected": 0,
             "pending": 0, "signals": 0}
    rejections: dict[str, int] = {}
    equity_path: list[tuple[int, float]] = []
    key_n = 0
    sig_i = 0            # next bar whose PREVIOUS signal can act
    n = len(times)

    for k in range(n):
        t = int(times[k])
        b, a = float(bid[k]), float(ask[k])
        broker.on_tick("XAUUSD", b, a, t)
        manager.on_tick("XAUUSD", bid=b, ask=a, ts_ms=t)
        # A bar has opened: act on the PREVIOUS bar's signal (causal).
        while sig_i < len(bar_open_ms) and bar_open_ms[sig_i] <= t:
            if sig_i >= 1:
                signal = sig[sig_i - 1]
                ts_bar = bars.index[sig_i]
                hour = ts_bar.hour
                in_session = sess_lo is None or (
                    sess_lo <= hour < sess_hi)
                atr_v = atr_np[sig_i - 1]
                router.prices["soak-1"] = (a + b) / 2.0
                snap = _snapshot(router, broker, t, a - b)
                if signal != 0 and in_session and np.isfinite(atr_v) \
                        and not broker.positions():
                    direction = 1 if signal > 0 else -1
                    sl_dist = max(1.5 * float(atr_v), 1e-9)
                    ref = (a if direction == 1 else b)
                    sl = ref - direction * sl_dist
                    tp = ref + direction * sl_dist * cfg.rr
                    key_n += 1
                    from core.execution import OrderIntent
                    from core.execution.versions import make_version
                    intent = OrderIntent(
                        f"soak-{key_n}", "soak-1", "XAUUSD", direction,
                        0.5, make_version(type("S", (), {}), {}), sl,
                        tp_price=tp, source="strategy",
                        created_ms=t)
                    stats["orders_submitted"] += 1
                    res = router.submit(intent)
                    if res.status == "filled":
                        stats["filled"] += 1
                    elif res.status == "pending_accepted":
                        stats["pending"] += 1
                    else:
                        stats["rejected"] += 1
                        for r in res.reasons:
                            rejections[r] = rejections.get(r, 0) + 1
                elif signal != 0:
                    stats["signals"] += 1
            sig_i += 1
        if k % 5000 == 0 or k == n - 1:
            equity_path.append((t, round(broker.equity, 2)))

    # ── integrity invariants (fail loudly, never soften) ──────────────
    problems = []
    if broker.equity <= 0:
        problems.append("equity went non-positive")
    seen_keys = set()
    for rec in broker.order_history:
        fill_price = getattr(rec, "fill_price", None)
        key = getattr(rec, "idempotency_key", None)
        if fill_price is None or not np.isfinite(fill_price):
            problems.append(f"order {getattr(rec, 'order_ref', '?')} in "
                            "history without a fill price")
        if key is not None:
            if key in seen_keys:
                problems.append(f"duplicate idempotency key filled: {key}")
            seen_keys.add(key)
            found = broker.find_by_idempotency_key(key)
            if not found or found.get("fill_price") is None:
                problems.append(f"idempotency key does not resolve: {key}")
    open_positions = broker.positions()
    for pos in open_positions:
        if pos.get("sl_price") is None:
            problems.append(f"open position "
                            f"{pos.get('position_id')} without SL")
    audit_events = router.audit.events() if hasattr(router.audit,
                                                    "events") else []
    summary = {
        "source_label": "REPLAY_SOAK",
        "source_detail": "real historical ticks replayed through the "
                         "live pipeline — not live trading, not a "
                         "forecast",
        "window": {"start_ms": start_ms, "end_ms": end_ms,
                   "ticks": int(n)},
        "start_equity": 100_000.0,
        "end_equity": round(broker.equity, 2),
        "open_positions": len(open_positions),
        "stats": stats,
        "rejection_reasons": rejections,
        "closed_trades": [
            {"pnl": round(float(c.get("pnl", 0.0)), 2),
             "reason": c.get("reason", "unknown")}
            for c in broker.closed],
        "integrity_problems": problems,
        "equity_path_samples": equity_path,
    }
    import json
    out = Path("reports/soak_replay.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    start_ms = int(sys.argv[2]) if len(sys.argv) > 2 else None
    s = run_soak(days, start_ms)
    print(f"soak window: {s['window']['ticks']:,} real ticks")
    print(f"equity: {s['start_equity']:,.0f} -> {s['end_equity']:,.2f}")
    print(f"stats: {s['stats']}")
    print(f"open positions: {s['open_positions']}")
    if s["integrity_problems"]:
        print("INTEGRITY PROBLEMS:")
        for p in s["integrity_problems"]:
            print("  -", p)
        return 1
    print("integrity: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
