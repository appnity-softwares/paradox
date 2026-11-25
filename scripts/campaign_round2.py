#!/usr/bin/env python3
"""U7 ROUND 2 campaign: new hypothesis classes + exit management.

Differences from round 1 (see docs/strategy_rules_u7.md):
- three NEW candidates (asia_range_breakout, squeeze_breakout,
  first_move_continuation) — different hypothesis classes, not re-tunes;
- exit management (breakeven / trail / time-exit) searched jointly,
  targeting round-1's drawdown failure;
- deflated-Sharpe selection count is CUMULATIVE (all trials in the
  store, round 1 + round 2);
- same frozen holdout, same gates, same derived risk.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.marketdata.store import dataset_hash, read_range  # noqa: E402
from core.research import (LONDON, NEW_YORK, OVERLAP, ClientProfile,
                           HoldoutGuard, SessionWindow, TrialStore,
                           all_gates_passed, derive_risk_pct,
                           evaluate_gates, random_search, run_trial,
                           write_report)  # noqa: E402
from core.validate import deflated_sharpe, monte_carlo_trades  # noqa: E402
from strategies.variants import (AsiaRangeBreakout, EmaCrossVolFilter,
                                 FirstMoveContinuation, LondonOpeningRange,
                                 SqueezeBreakout, TrendPullback)  # noqa: E402

DAY = 86_400_000
HOLDOUT_START = 1_778_995_200_000          # 2026-05-22 00:00 UTC
DATASET_END = 1_794_720_000_000            # 2026-11-22 00:00 UTC
TRAIN_START = 1_704_067_200_000            # 2024-01-01 00:00 UTC
TRAIN_END = HOLDOUT_START

REPORT_DIR = Path("reports/campaign_2026-11-25")
TRIALS = REPORT_DIR / "trials.jsonl"        # SHARED with round 1 (cumulative)
STATE = REPORT_DIR / "campaign_state_round2.json"
DSH = dataset_hash("XAUUSD")[:16]

EXTENDED = SessionWindow("extended_0000_1100", 0, 660)

# (factory, entry grid, session, exit grid)
CANDIDATES = {
    "asia_range_breakout": (
        AsiaRangeBreakout, {"rr": [1.0, 1.5, 2.0]}, EXTENDED,
        {"breakeven_rr": [0.0, 1.0], "breakeven_offset": [0.0, 0.05],
         "trail_distance": [0.0, 0.5, 1.0],
         "time_exit_minutes": [0.0, 120.0, 240.0]}),
    "squeeze_breakout": (
        SqueezeBreakout, {"squeeze_pct": [20.0, 30.0],
                          "rr": [1.0, 1.5, 2.0]},
        None,   # all three sessions
        {"breakeven_rr": [0.0, 1.0], "breakeven_offset": [0.0, 0.05],
         "trail_distance": [0.0, 0.5, 1.0],
         "time_exit_minutes": [0.0, 120.0, 240.0]}),
    "first_move_continuation": (
        FirstMoveContinuation, {"rr": [1.0, 1.5, 2.0]}, EXTENDED,
        {"breakeven_rr": [0.0, 1.0], "breakeven_offset": [0.0, 0.05],
         "trail_distance": [0.0, 0.5, 1.0],
         "time_exit_minutes": [0.0, 120.0, 240.0]}),
    # Round-1 survivors get the NEW exit knobs only (fresh hypothesis:
    # same entries, better exits) — this is a different trial space.
    "ema_cross_vol_filter_r2": (
        EmaCrossVolFilter, {"atr_pct": [40.0, 50.0, 60.0]}, None,
        {"breakeven_rr": [0.0, 1.0], "breakeven_offset": [0.0, 0.05],
         "trail_distance": [0.0, 0.5, 1.0],
         "time_exit_minutes": [0.0, 120.0, 240.0]}),
    "trend_pullback_r2": (
        TrendPullback, {"atr_mult": [1.0, 1.5, 2.0],
                        "rr": [1.0, 1.5, 2.0]}, None,
        {"breakeven_rr": [0.0, 1.0], "breakeven_offset": [0.0, 0.05],
         "trail_distance": [0.0, 0.5, 1.0],
         "time_exit_minutes": [0.0, 120.0, 240.0]}),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    profile = ClientProfile()
    holdout = HoldoutGuard(HOLDOUT_START, DATASET_END,
                           log_path="state/holdout_openings.jsonl")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    risk = derive_risk_pct(profile)
    store = TrialStore(TRIALS)
    prior_trials = len(store.entries())
    log(f"ROUND 2 start: risk={risk}% prior_trials={prior_trials} "
        f"holdout_opened={holdout.opened}")
    assert not holdout.opened

    s_start, s_end = TRAIN_START, TRAIN_START + 365 * DAY
    holdout.check_range(s_start, s_end, purpose="round2_screen")
    df = load_range(s_start, s_end)
    sessions = {"london": LONDON, "newyork": NEW_YORK, "overlap": OVERLAP}
    for name, (cls, grid, sess, exits) in CANDIDATES.items():
        sess_list = [sess] if sess is not None else list(sessions.values())
        for s in sess_list:
            log(f"screen {name} @ {s.name}: {n_trials} trials")
            random_search(cls, grid, df, start_ms=s_start, end_ms=s_end,
                          session=s, risk_pct=risk, holdout=holdout,
                          trials_store=store, n_trials=n_trials, seed=43,
                          dataset_hash=DSH, trial_extras_grid=exits)
    total = len(store.entries())
    log(f"screening done: {total - prior_trials} new trials, "
        f"{total} cumulative")

    # Walk-forward top-4 trials from THIS run only (round-1 candidates
    # already walked forward and failed in round 1; re-walking them adds
    # no information). Selection effect still counted cumulatively.
    all_entries = store.entries()
    rows = [r for r in all_entries[prior_trials:]
            if r["metrics"]["n_trades"] >= 20]
    rows.sort(key=lambda r: (r["metrics"]["pf"],
                             r["metrics"]["n_trades"]), reverse=True)
    seen, candidates = set(), []
    for r in rows:
        k = (r["strategy"], r["session"], json.dumps(r["params"],
                                                     sort_keys=True),
             json.dumps(r.get("extras", {}), sort_keys=True))
        if k in seen:
            continue
        seen.add(k)
        candidates.append(r)
        if len(candidates) >= 4:
            break
    log(f"walk-forward candidates: "
        f"{[(c['strategy'], c['session']) for c in candidates]}")

    folds = []
    f = TRAIN_START
    while f + 180 * DAY <= TRAIN_END:
        folds.append((f, min(f + 180 * DAY, TRAIN_END)))
        f += 180 * DAY

    KEY2CLS = {c[0]().key: c[0] for c in CANDIDATES.values()}
    wf_results = []
    for cand in candidates:
        cls = KEY2CLS[cand["strategy"]]
        sess = next(s for s in (*sessions.values(), EXTENDED)
                    if s.name == cand["session"])
        extras = cand.get("extras", {})
        fold_rows, r_mults_all = [], []
        for (fs, fe) in folds:
            fdf = load_range(fs, fe)
            m = run_trial(cls(**cand["params"]), fdf, start_ms=fs,
                          end_ms=fe, session=sess, risk_pct=risk,
                          return_trades=True, dataset_hash=DSH, **extras)
            fold_rows.append({"start_ms": fs, "end_ms": fe,
                              "metrics": {k: v for k, v in m.items()
                                          if k not in ("trades",
                                                       "r_multiples")}})
            r_mults_all.extend(m.get("r_multiples", []))
            log(f"  fold: trades={m['n_trades']} pf={m['pf']:.2f} "
                f"dd={m['max_dd_pct']:.2f}%")
        rm = np.asarray(r_mults_all, dtype=float)
        mc = monte_carlo_trades(rm * (risk / 100.0 * 10_000.0),
                                start_equity=10_000.0, n_sims=2_000,
                                seed=42)
        # 2x cost stress on the last fold.
        fdf = load_range(*folds[-1])
        m2 = run_trial(cls(**cand["params"]), fdf, start_ms=folds[-1][0],
                       end_ms=folds[-1][1], session=sess, risk_pct=risk,
                       cost_multiplier=2.0, dataset_hash=DSH, **extras)
        sr = float(np.mean(rm) / np.std(rm)) if rm.size > 1 else 0.0
        skew = float(((rm - rm.mean()) ** 3).mean()
                     / max(rm.std() ** 3, 1e-12)) if rm.size > 2 else 0.0
        kurt = float(((rm - rm.mean()) ** 4).mean()
                     / max(rm.std() ** 4, 1e-12)) if rm.size > 2 else 3.0
        dsr = deflated_sharpe(sr, n_obs=len(rm), n_trials=max(1, total),
                              skew=skew, kurtosis=kurt)
        dds = [f["metrics"]["max_dd_pct"] for f in fold_rows]
        pfs = [f["metrics"]["pf"] for f in fold_rows
               if f["metrics"]["n_trades"]]
        exps = [f["metrics"]["expectancy_r"] for f in fold_rows
                if f["metrics"]["n_trades"]]
        gate_metrics = {
            "oos_max_dd_pct": max(dds) if dds else 0.0,
            "mc95_dd_pct": round(mc["p95_max_dd_pct"], 3),
            "pf_after_2x_costs": m2["pf"],
            "expectancy_r_after_2x_costs": m2["expectancy_r"],
            "n_trades": sum(f["metrics"]["n_trades"] for f in fold_rows),
            "max_losing_streak": max(f["metrics"]["max_losing_streak"]
                                     for f in fold_rows),
            "streak_dd_pct": max(dds) if dds else 0.0,
            "trades_per_day_avg": round(float(np.mean(
                [f["metrics"]["trades_per_day_avg"]
                 for f in fold_rows])), 4),
            "no_trade_day_share": float(np.max(
                [f["metrics"]["no_trade_day_share"] for f in fold_rows])),
            "win_rate": float(np.mean(
                [f["metrics"]["win_rate_pct"] for f in fold_rows
                 if f["metrics"]["n_trades"]] or [0])),
        }
        gates = evaluate_gates(gate_metrics, profile)
        wf_results.append({
            "strategy": cand["strategy"], "session": cand["session"],
            "params": cand["params"], "extras": extras, "risk_pct": risk,
            "folds": fold_rows, "monte_carlo": mc,
            "cost_stress_x2": {"pf": m2["pf"],
                               "expectancy_r": m2["expectancy_r"]},
            "deflated_sharpe_prob": round(dsr, 4),
            "gate_metrics": gate_metrics,
            "gates": [g.to_dict() for g in gates],
            "all_gates_passed": all_gates_passed(gates),
        })
        log(f"{cand['strategy']}@{cand['session']}: "
            f"gates={'PASS' if all_gates_passed(gates) else 'FAIL'} "
            f"(dd={gate_metrics['oos_max_dd_pct']:.2f}% "
            f"pf2x={m2['pf']:.2f} dsr={dsr:.4f})")

    passing = [c for c in wf_results if c["all_gates_passed"]]
    holdout_results = []
    if passing:
        holdout.open_holdout(
            by="jatin-kurrey (authorized agent run)",
            reason="U7 round-2 final holdout after walk-forward gates")
        hdf = load_range(HOLDOUT_START, DATASET_END)
        for cand in passing:
            cls = KEY2CLS[cand["strategy"]]
            sess = next(s for s in (*sessions.values(), EXTENDED)
                        if s.name == cand["session"])
            m = run_trial(cls(**cand["params"]), hdf,
                          start_ms=HOLDOUT_START, end_ms=DATASET_END,
                          session=sess, risk_pct=risk, return_trades=True,
                          dataset_hash=DSH, **cand.get("extras", {}))
            holdout_results.append({
                "strategy": cand["strategy"], "session": cand["session"],
                "metrics": {k: v for k, v in m.items()
                            if k not in ("trades", "r_multiples")}})
            log(f"HOLDOUT {cand['strategy']}: trades={m['n_trades']} "
                f"pf={m['pf']:.2f} dd={m['max_dd_pct']:.2f}%")
    else:
        log("NO candidate passed all gates — holdout stays FROZEN")

    summary = {
        "round": 2,
        "dataset_hash_prefix": DSH,
        "prior_trials_round1": prior_trials,
        "new_trials_round2": total - prior_trials,
        "cumulative_trials": total,
        "risk_pct_derived": risk,
        "holdout_opened": holdout.opened,
        "walk_forward": wf_results,
        "holdout_results": holdout_results,
        "any_candidate_passed": bool(passing),
    }
    gate_rows = []
    for c in wf_results:
        for g in c["gates"]:
            gate_rows.append({"candidate": f"{c['strategy']}@{c['session']}",
                              "session": c["session"], "gate": g["name"],
                              "passed": g["passed"], "value": g["value"],
                              "threshold": g["threshold"]})
    write_report(REPORT_DIR, summary, store.entries(), gate_rows)
    STATE.write_text(json.dumps(summary, indent=1, sort_keys=True))
    log("round 2 done")


def load_range(start_ms: int, end_ms: int):
    t0 = time.perf_counter()
    df = read_range("XAUUSD", start_ms, end_ms)
    log(f"loaded {len(df):,} ticks in {time.perf_counter() - t0:.1f}s")
    return df


if __name__ == "__main__":
    main()
