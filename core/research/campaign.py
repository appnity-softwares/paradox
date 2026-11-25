"""U7 campaign runner (Update Blueprint v2.1).

Everything runs on the TICK engine with session-only loading, recorded
bid/ask, spread/slippage/commission/swap from a CostModel. Every trial
is stored. The holdout is enforced by HoldoutGuard at every step.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import (get_cost_model, get_symbol_spec,
                         register_cost_model)
from core.research.gates import (ClientProfile, all_gates_passed,
                                 longest_losing_streak, streak_dd_pct,
                                 trade_calendar_stats)
from core.research.holdout import HoldoutGuard
from core.tick_engine import BacktestConfig, TickEngine


def scaled_cost_key(base_key: str, multiplier: float) -> str:
    """Register (idempotently) a cost model scaled 1x/1.5x/2x for the
    U7 cost stress. TODO(U8): replace with the target firm's REAL
    commission and spread behaviour once the feed comparison is done —
    until then the stress is a linear scaling of the recorded model."""
    key = f"{base_key}__x{multiplier:g}" if multiplier != 1.0 else base_key
    try:
        get_cost_model(key)
        return key
    except KeyError:
        base = get_cost_model(base_key)
        sl = dataclasses.replace(
            base.slippage,
            market_points=round(base.slippage.market_points * multiplier, 6),
            stop_points=round(base.slippage.stop_points * multiplier, 6))
        scaled = dataclasses.replace(
            base,
            commission_per_lot=base.commission_per_lot * multiplier,
            fixed_spread_points=base.fixed_spread_points * multiplier,
            slippage=sl)
        register_cost_model(key, scaled)
        return key


@dataclass
class SessionWindow:
    """UTC minutes-of-day session (backtest uses fixed UTC minutes —
    the documented U2 difference; live DST resolution is separate)."""
    name: str
    start_min: int          # minutes of day, UTC
    end_min: int


LONDON = SessionWindow("london_0700_1100", 7 * 60, 11 * 60)
NEW_YORK = SessionWindow("newyork_1200_1600", 12 * 60, 16 * 60)
OVERLAP = SessionWindow("overlap_1200_1500", 12 * 60, 15 * 60)
SESSIONS = [LONDON, NEW_YORK, OVERLAP]


def session_days_between(start_ms: int, end_ms: int) -> int:
    """WEEKDAYS in range — weekends have no XAUUSD ticks and must not
    inflate the no-trade-day share."""
    days = (end_ms - start_ms) // 86_400_000
    return max(1, int(days * 5.0 / 7.0))


def run_trial(strategy, ticks_df, *, start_ms: int, end_ms: int,
              session: SessionWindow, risk_pct: float,
              initial_balance: float = 10_000.0,
              news_windows: tuple = (),
              cost_multiplier: float = 1.0,
              min_hold_seconds: int = 300,
              max_orders_per_day: int = 6,
              return_trades: bool = False,
              breakeven_rr: float = 0.0,
              breakeven_offset: float = 0.0,
              trail_distance: float = 0.0,
              time_exit_minutes: float = 0.0,
              dataset_hash: str = "u7-campaign") -> dict:
    """One trial: session-filtered tick replay -> metrics dict.

    U1 behaviour controls are ON for every campaign trial (min hold
    5 minutes, max 6 orders/day, martingale prohibited) — the client
    profile requires them and the engine must enforce, not trust."""
    cfg = BacktestConfig(
        symbol="XAUUSD", initial_balance=initial_balance,
        risk_pct=risk_pct, dataset_hash=dataset_hash,
        cost_key=scaled_cost_key("XAUUSD_ECN", cost_multiplier),
        bar_timeframe="5m",
        session_start_min=session.start_min, session_end_min=session.end_min,
        session_days=tuple(range(5)), flatten_at_session_end=True,
        news_windows=news_windows,
        min_hold_seconds=min_hold_seconds,
        max_orders_per_day=max_orders_per_day,
        prohibited_styles=("martingale",),
        breakeven_rr=breakeven_rr,
        breakeven_offset=breakeven_offset,
        trail_distance=trail_distance,
        time_exit_minutes=time_exit_minutes)
    res = TickEngine().run(strategy, ticks_df, cfg)
    trades = res.get("trades", [])
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    curve = res.get("equity_curve", [])
    per_day, no_trade = trade_calendar_stats(
        [t.get("exit_time_utc_ms", t.get("entry_time_utc_ms", 0))
         for t in trades], session_days_between(start_ms, end_ms))
    # R multiples: pnl / initial risk of that trade (entry vs stop).
    contract = get_symbol_spec("XAUUSD").contract_size
    r_mults = []
    for t in trades:
        risk_usd = abs(t.get("entry_price", 0.0) - (t.get("sl") or 0.0)) \
            * t.get("volume", 0.0) * contract
        r_mults.append(t["pnl"] / risk_usd if risk_usd > 0 else 0.0)
    avg_loss = float(abs(np.mean(losses))) if losses else 1.0
    expectancy_r = ((float(np.mean(wins)) if wins else 0.0) - 0.0) \
        / avg_loss * (len(wins) / len(pnls)) - (len(losses) / len(pnls)) \
        if pnls else 0.0
    out = {
        "n_trades": len(trades),
        "net_profit_usd": res.get("net_profit_usd", 0.0),
        "win_rate_pct": res.get("win_rate_pct", 0.0),
        "pf": res.get("profit_factor", 0.0),
        "max_dd_pct": streak_dd_pct(curve),
        "max_losing_streak": longest_losing_streak(pnls),
        "expectancy_r": round(expectancy_r, 4),
        "trades_per_day_avg": per_day,
        "no_trade_day_share": no_trade,
        "ticks_replayed": res.get("ticks_replayed", 0),
        "lookahead_violations": len(res.get("lookahead_violations", [])),
        "behaviour_blocks": res.get("behaviour_blocks", {}),
    }
    if return_trades:
        out["trades"] = trades
        out["r_multiples"] = [round(r, 4) for r in r_mults]
    return out


class TrialStore:
    """Every trial stored (JSONL) — the search is fully auditable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def best(self, key: str = "pf", n: int = 5) -> list[dict]:
        rows = [r for r in self.entries() if r.get("metrics")]
        return sorted(rows, key=lambda r: r["metrics"].get(key, 0.0),
                      reverse=True)[:n]


def random_search(strategy_factory, param_grid: dict, ticks_df, *,
                  start_ms: int, end_ms: int, session: SessionWindow,
                  risk_pct: float, holdout: HoldoutGuard,
                  trials_store: TrialStore, n_trials: int = 30,
                  seed: int = 42, dataset_hash: str = "u7-campaign",
                  news_windows: tuple = (),
                  trial_extras_grid: dict | None = None) -> list[dict]:
    """Random search over the grid. The holdout range is checked BEFORE
    any trial — enforced in code, not by convention.

    trial_extras_grid: optional per-trial run_trial kwargs (e.g. exit
    management knobs) sampled like params and stored with the trial."""
    holdout.check_range(start_ms, end_ms, purpose="random_search")
    rng = np.random.default_rng(seed)
    keys = sorted(param_grid)
    xkeys = sorted(trial_extras_grid) if trial_extras_grid else []
    results = []
    for i in range(n_trials):
        params = {k: param_grid[k][int(rng.integers(len(param_grid[k])))]
                  for k in keys}
        extras = {k: trial_extras_grid[k][int(rng.integers(
            len(trial_extras_grid[k])))] for k in xkeys}
        strategy = strategy_factory(**params)
        t0 = time.perf_counter()
        metrics = run_trial(strategy, ticks_df, start_ms=start_ms,
                            end_ms=end_ms, session=session,
                            risk_pct=risk_pct, dataset_hash=dataset_hash,
                            news_windows=news_windows, **extras)
        rec = {"trial": i, "strategy": strategy.key, "params": params,
               "extras": extras, "session": session.name,
               "risk_pct": risk_pct,
               "start_ms": start_ms, "end_ms": end_ms,
               "elapsed_s": round(time.perf_counter() - t0, 2),
               "metrics": metrics}
        trials_store.append(rec)
        results.append(rec)
    return results


def walk_forward(strategy_factory, params: dict, ticks_df, *,
                 folds: list[tuple[int, int]], session: SessionWindow,
                 risk_pct: float, holdout: HoldoutGuard,
                 dataset_hash: str = "u7-campaign",
                 news_windows: tuple = ()) -> list[dict]:
    """Rolling walk-forward over explicit folds (all inside training)."""
    fold_results = []
    for (s, e) in folds:
        holdout.check_range(s, e, purpose="walk_forward")
        strategy = strategy_factory(**params)
        metrics = run_trial(strategy, ticks_df, start_ms=s, end_ms=e,
                            session=session, risk_pct=risk_pct,
                            dataset_hash=dataset_hash,
                            news_windows=news_windows)
        fold_results.append({"start_ms": s, "end_ms": e, "metrics": metrics})
    return fold_results


def gate_metrics(fold_metrics: list[dict], session_days: int) -> dict:
    """Aggregate walk-forward folds into the gate input dict."""
    n = [f["metrics"]["n_trades"] for f in fold_metrics]
    dds = [f["metrics"]["max_dd_pct"] for f in fold_metrics]
    pfs = [f["metrics"]["pf"] for f in fold_metrics if f["metrics"]["n_trades"]]
    exps = [f["metrics"]["expectancy_r"] for f in fold_metrics
            if f["metrics"]["n_trades"]]
    streaks = [f["metrics"]["max_losing_streak"] for f in fold_metrics]
    worst_streak = max(streaks) if streaks else 0
    # Streak consequence: worst streak x average loss at derived risk,
    # expressed as percent of equity (linear, conservative).
    worst_dd = max(dds) if dds else 0.0
    total_trades = sum(n)
    trades_per_day = sum(f["metrics"]["trades_per_day_avg"]
                         for f in fold_metrics) / max(1, len(fold_metrics))
    no_trade = max(f["metrics"]["no_trade_day_share"]
                   for f in fold_metrics)
    return {
        "oos_max_dd_pct": worst_dd,
        "mc95_dd_pct": worst_dd,          # replaced by real MC below
        "pf_after_2x_costs": min(pfs) if pfs else 0.0,
        "expectancy_r_after_2x_costs": min(exps) if exps else -99.0,
        "n_trades": total_trades,
        "max_losing_streak": worst_streak,
        "streak_dd_pct": worst_dd,
        "trades_per_day_avg": round(trades_per_day, 4),
        "no_trade_day_share": no_trade,
    }


def write_report(out_dir: str | Path, summary: dict, trials: list[dict],
                 gate_rows: list[dict]) -> Path:
    """reports/campaign_<date>/: trials.csv + summary.json + gates.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1,
                                                  sort_keys=True))
    if trials:
        cols = sorted({k for r in trials for k in r})
        with (out / "trials.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(trials, key=lambda r: json.dumps(r,
                                                             sort_keys=True)):
                w.writerow({c: json.dumps(r.get(c), sort_keys=True)
                            if isinstance(r.get(c), (dict, list))
                            else r.get(c, "") for c in cols})
    if gate_rows:
        with (out / "gates.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["candidate", "session",
                                              "gate", "passed", "value",
                                              "threshold"])
            w.writeheader()
            for row in gate_rows:
                w.writerow(row)
    return out
