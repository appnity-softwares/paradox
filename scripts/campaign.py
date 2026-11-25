#!/usr/bin/env python3
"""U7 research campaign — real-data orchestration (Update Blueprint v2.1).

Phases:
  screen      random search over (strategy x session x params) on the
              TRAINING window; every trial stored.
  walkforward top candidates re-run over rolling folds + Monte Carlo +
              cost stress 1x/1.5x/2x + deflated Sharpe.
  gates       acceptance gates per candidate (PASS/FAIL per gate).
  holdout     ONLY if some candidate passes all gates: the explicit
              one-time logged holdout run.
  report      reports/campaign_<date>/ + docs/campaign_summary.md.

The holdout is NEVER touched by screen/walkforward/gates — enforced by
HoldoutGuard in code. A failed campaign is a valid result and is
reported honestly.
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
                           HoldoutGuard, TrialStore, all_gates_passed,
                           derive_risk_pct, evaluate_gates, random_search,
                           run_trial, walk_forward, write_report)  # noqa: E402
from core.validate import deflated_sharpe, monte_carlo_trades  # noqa: E402
from strategies.variants import (EmaCrossVolFilter, LondonOpeningRange,
                                 TrendPullback)  # noqa: E402

DAY = 86_400_000
# Dataset: 2024-01-01 .. 2026-11-22. Holdout = LAST 6 months (frozen).
HOLDOUT_START = 1_778_995_200_000          # 2026-05-22 00:00 UTC
DATASET_END = 1_794_720_000_000            # 2026-11-22 00:00 UTC (exclusive)
TRAIN_START = 1_735_689_600_000            # 2025-01-01 00:00 UTC  (fix below)
TRAIN_START = 1_704_067_200_000            # 2024-01-01 00:00 UTC
TRAIN_END = HOLDOUT_START                  # 2026-05-22 (exclusive)

REPORT_DIR = Path("reports/campaign_2026-11-25")
TRIALS = REPORT_DIR / "trials.jsonl"
STATE = REPORT_DIR / "campaign_state.json"

DSH = dataset_hash("XAUUSD")[:16]

STRATEGIES = {
    "london_opening_range": (
        LondonOpeningRange,
        {"range_minutes": [10, 15, 30], "rr": [1.0, 1.5, 2.0]}),
    "trend_pullback": (
        TrendPullback,
        {"atr_mult": [1.0, 1.5, 2.0], "rr": [1.0, 1.5, 2.0]}),
    "ema_cross_vol_filter": (
        EmaCrossVolFilter,
        {"atr_pct": [40.0, 50.0, 60.0]}),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_range(start_ms: int, end_ms: int) -> object:
    t0 = time.perf_counter()
    df = read_range("XAUUSD", start_ms, end_ms)
    log(f"loaded {len(df):,} ticks [{start_ms}-{end_ms}] "
        f"in {time.perf_counter() - t0:.1f}s")
    return df


def phase_screen(profile: ClientProfile, holdout: HoldoutGuard,
                 n_trials: int) -> list[dict]:
    risk = derive_risk_pct(profile)
    log(f"derived risk per trade: {risk}% (deterministic from DD caps)")
    # Screening window: the FIRST 12 training months.
    s_start, s_end = TRAIN_START, TRAIN_START + 365 * DAY
    holdout.check_range(s_start, s_end, purpose="screen")
    df = load_range(s_start, s_end)
    store = TrialStore(TRIALS)
    all_res = []
    for sess in (LONDON, NEW_YORK, OVERLAP):
        for key, (cls, grid) in STRATEGIES.items():
            log(f"screen {key} @ {sess.name}: {n_trials} trials")
            res = random_search(cls, grid, df, start_ms=s_start,
                                end_ms=s_end, session=sess, risk_pct=risk,
                                holdout=holdout, trials_store=store,
                                n_trials=n_trials, seed=42,
                                dataset_hash=DSH)
            all_res.extend(res)
            best = max(res, key=lambda r: r["metrics"]["pf"])
            m = best["metrics"]
            log(f"  best pf={m['pf']:.2f} trades={m['n_trades']} "
                f"wr={m['win_rate_pct']:.0f}% dd={m['max_dd_pct']:.2f}%")
    return all_res


def top_configs(n: int = 3) -> list[dict]:
    rows = [r for r in TrialStore(TRIALS).entries()
            if r["metrics"]["n_trades"] >= 20]
    rows.sort(key=lambda r: (r["metrics"]["pf"], r["metrics"]["n_trades"]),
              reverse=True)
    seen, out = set(), []
    for r in rows:
        k = (r["strategy"], r["session"], json.dumps(r["params"],
                                                     sort_keys=True))
        if k not in seen:
            seen.add(k)
            out.append(r)
        if len(out) >= n:
            break
    return out


def phase_walkforward(profile: ClientProfile, holdout: HoldoutGuard,
                      candidates: list[dict]) -> list[dict]:
    risk = derive_risk_pct(profile)
    folds = []
    f = TRAIN_START
    while f + 180 * DAY <= TRAIN_END:
        folds.append((f, min(f + 180 * DAY, TRAIN_END)))
        f += 180 * DAY
    log(f"walk-forward folds: {len(folds)}")
    results = []
    for cand in candidates:
        cls, _ = STRATEGIES[cand["strategy"]]
        sess = next(s for s in (LONDON, NEW_YORK, OVERLAP)
                    if s.name == cand["session"])
        fold_rows = []
        r_mults_all: list[float] = []
        for (fs, fe) in folds:
            df = load_range(fs, fe)
            m = run_trial(cls(**cand["params"]), df, start_ms=fs, end_ms=fe,
                          session=sess, risk_pct=risk, return_trades=True,
                          dataset_hash=DSH)
            fold_rows.append({"start_ms": fs, "end_ms": fe,
                              "metrics": {k: v for k, v in m.items()
                                          if k not in ("trades",
                                                       "r_multiples")}})
            r_mults_all.extend(m.get("r_multiples", []))
            log(f"  fold [{fs}-{fe}]: trades={m['n_trades']} "
                f"pf={m['pf']:.2f} dd={m['max_dd_pct']:.2f}%")
        # Monte Carlo on pooled R multiples (converted to USD at risk).
        rm = np.asarray(r_mults_all, dtype=float)
        mc = monte_carlo_trades(rm * (risk / 100.0 * 10_000.0),
                                start_equity=10_000.0, n_sims=2_000,
                                seed=42)
        # Cost stress 1x/1.5x/2x on the LAST fold (worst-case check).
        stress = {}
        df_last = load_range(*folds[-1])
        for mult in (1.0, 1.5, 2.0):
            m = run_trial(cls(**cand["params"]), df_last,
                          start_ms=folds[-1][0], end_ms=folds[-1][1],
                          session=sess, risk_pct=risk,
                          cost_multiplier=mult, dataset_hash=DSH)
            stress[f"x{mult:g}"] = {"pf": m["pf"],
                                    "expectancy_r": m["expectancy_r"]}
        # Deflated Sharpe: number of screening trials for this strategy
        # is the selection effect N.
        n_trials_strategy = len([r for r in TrialStore(TRIALS).entries()
                                 if r["strategy"] == cand["strategy"]])
        sr = float(np.mean(rm) / np.std(rm)) if rm.size > 1 else 0.0
        skew = float(((rm - rm.mean()) ** 3).mean() / max(rm.std() ** 3,
                                                          1e-12)) \
            if rm.size > 2 else 0.0
        kurt = float(((rm - rm.mean()) ** 4).mean() / max(rm.std() ** 4,
                                                          1e-12)) \
            if rm.size > 2 else 3.0
        dsr = deflated_sharpe(sr, n_obs=len(rm),
                              n_trials=max(1, n_trials_strategy),
                              skew=skew, kurtosis=kurt)
        # Aggregate gate metrics from folds.
        dds = [f["metrics"]["max_dd_pct"] for f in fold_rows]
        pfs = [f["metrics"]["pf"] for f in fold_rows
               if f["metrics"]["n_trades"]]
        exps = [f["metrics"]["expectancy_r"] for f in fold_rows
                if f["metrics"]["n_trades"]]
        total_trades = sum(f["metrics"]["n_trades"] for f in fold_rows)
        per_day = float(np.mean([f["metrics"]["trades_per_day_avg"]
                                 for f in fold_rows]))
        no_trade = float(np.max([f["metrics"]["no_trade_day_share"]
                                 for f in fold_rows]))
        gate_metrics = {
            "oos_max_dd_pct": max(dds) if dds else 0.0,
            "mc95_dd_pct": round(mc["p95_max_dd_pct"], 3),
            "pf_after_2x_costs": stress["x2"]["pf"],
            "expectancy_r_after_2x_costs": stress["x2"]["expectancy_r"],
            "n_trades": total_trades,
            "max_losing_streak": max(f["metrics"]["max_losing_streak"]
                                     for f in fold_rows),
            "streak_dd_pct": max(dds) if dds else 0.0,
            "trades_per_day_avg": round(per_day, 4),
            "no_trade_day_share": no_trade,
            "win_rate": float(np.mean([f["metrics"]["win_rate_pct"]
                                       for f in fold_rows
                                       if f["metrics"]["n_trades"]] or [0])),
        }
        gates = evaluate_gates(gate_metrics, profile)
        results.append({
            "strategy": cand["strategy"], "session": cand["session"],
            "params": cand["params"], "risk_pct": risk,
            "folds": fold_rows, "monte_carlo": mc, "cost_stress": stress,
            "deflated_sharpe_prob": round(dsr, 4),
            "per_trade_sharpe": round(sr, 4),
            "gate_metrics": gate_metrics,
            "gates": [g.to_dict() for g in gates],
            "all_gates_passed": all_gates_passed(gates),
        })
        log(f"{cand['strategy']}@{cand['session']}: "
            f"gates={'PASS' if all_gates_passed(gates) else 'FAIL'} "
            f"(mc95dd={mc['p95_max_dd_pct']:.2f}% "
            f"pf2x={stress['x2']['pf']:.2f} dsr={dsr:.3f})")
    return results


def phase_holdout(profile: ClientProfile, holdout: HoldoutGuard,
                  passing: list[dict]) -> list[dict]:
    """ONE-TIME logged holdout run for gate-passing candidates only."""
    holdout.open_holdout(
        by="jatin-kurrey (authorized agent run)",
        reason="U7 final holdout evaluation after walk-forward gates")
    risk = derive_risk_pct(profile)
    out = []
    df = load_range(HOLDOUT_START, DATASET_END)
    for cand in passing:
        cls, _ = STRATEGIES[cand["strategy"]]
        sess = next(s for s in (LONDON, NEW_YORK, OVERLAP)
                    if s.name == cand["session"])
        m = run_trial(cls(**cand["params"]), df, start_ms=HOLDOUT_START,
                      end_ms=DATASET_END, session=sess, risk_pct=risk,
                      return_trades=True, dataset_hash=DSH)
        out.append({"strategy": cand["strategy"], "session": cand["session"],
                    "params": cand["params"],
                    "metrics": {k: v for k, v in m.items()
                                if k not in ("trades", "r_multiples")}})
        log(f"HOLDOUT {cand['strategy']}@{cand['session']}: "
            f"trades={m['n_trades']} pf={m['pf']:.2f} "
            f"dd={m['max_dd_pct']:.2f}%")
    return out


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    profile = ClientProfile()
    holdout = HoldoutGuard(HOLDOUT_START, DATASET_END,
                           log_path="state/holdout_openings.jsonl")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log(f"campaign start: dataset_hash={DSH} risk="
        f"{derive_risk_pct(profile)}% holdout_opened={holdout.opened}")
    assert not holdout.opened, "holdout must NOT be open before gates"

    phase_screen(profile, holdout, n_trials)
    candidates = top_configs(3)
    log(f"top candidates for walk-forward: "
        f"{[(c['strategy'], c['session']) for c in candidates]}")
    wf = phase_walkforward(profile, holdout, candidates)
    passing = [c for c in wf if c["all_gates_passed"]]
    holdout_results: list[dict] = []
    if passing:
        log(f"{len(passing)} candidate(s) passed ALL gates -> holdout")
        holdout_results = phase_holdout(profile, holdout, passing)
    else:
        log("NO candidate passed all gates — holdout stays FROZEN "
            "(a failed campaign is a valid result)")

    summary = {
        "dataset_hash_prefix": DSH,
        "train_range": [TRAIN_START, TRAIN_END],
        "holdout_range": [HOLDOUT_START, DATASET_END],
        "holdout_opened": holdout.opened,
        "screening_trials": len(TrialStore(TRIALS).entries()),
        "risk_pct_derived": derive_risk_pct(profile),
        "walk_forward": wf,
        "holdout_results": holdout_results,
        "any_candidate_passed": bool(passing),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    gate_rows = []
    for c in wf:
        for g in c["gates"]:
            gate_rows.append({"candidate": f"{c['strategy']}@{c['session']}",
                              "session": c["session"], "gate": g["name"],
                              "passed": g["passed"], "value": g["value"],
                              "threshold": g["threshold"]})
    write_report(REPORT_DIR, summary,
                 TrialStore(TRIALS).entries(), gate_rows)
    STATE.write_text(json.dumps(summary, indent=1, sort_keys=True))
    log(f"campaign done in {summary['elapsed_s']}s -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
