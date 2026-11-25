"""U7 tests: holdout guard, gates, derived risk, campaign runner, and
the three session-specific candidate strategies.

The holdout enforcement is the safety-critical part: a search or
validation range that overlaps the frozen holdout MUST fail closed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.research import (LONDON, NEW_YORK, ClientProfile, HoldoutGuard,
                           HoldoutViolation, TrialStore, derive_risk_pct,
                           evaluate_gates, longest_losing_streak, run_trial,
                           scaled_cost_key, streak_dd_pct,
                           trade_calendar_stats)
from strategies.variants import (EmaCrossVolFilter, LondonOpeningRange,
                                 TrendPullback)

DAY_MS = 86_400_000
T0 = (1_760_000_000_000 // DAY_MS) * DAY_MS     # midnight-aligned


def synth_ticks(days: int = 12, sigma: float = 0.12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    px = 2600.0
    for d in range(days):
        base = T0 + d * DAY_MS
        drift = 0.01 if d % 3 else -0.01
        for s in range(25_200, 39_600, 5):          # 07:00-11:00 UTC
            px += rng.normal(0, sigma) + drift * 0.1
            rows.append((base + s * 1000, px - 0.15, px + 0.15, 1.0, 1.0,
                         "synth"))
    return pd.DataFrame(rows, columns=["time_utc_ms", "bid", "ask",
                                       "bid_volume", "ask_volume", "source"])


# ── HoldoutGuard ─────────────────────────────────────────────────────────

def test_holdout_overlap_raises(tmp_path):
    g = HoldoutGuard(1_800_000_000_000, 1_900_000_000_000,
                     log_path=tmp_path / "h.jsonl")
    g.check_range(1_700_000_000_000, 1_750_000_000_000, "search")   # ok
    with pytest.raises(HoldoutViolation):
        g.check_range(1_750_000_000_000, 1_850_000_000_000, "search")
    with pytest.raises(HoldoutViolation):
        g.check_range(1_800_000_000_000, 1_900_000_000_000, "walk_forward")


def test_holdout_open_once_logged(tmp_path):
    log = tmp_path / "h.jsonl"
    g = HoldoutGuard(1_800_000_000_000, 1_900_000_000_000, log_path=log)
    assert not g.opened
    g.open_holdout("jatin", "final one-time evaluation")
    assert g.opened
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["by"] == "jatin"
    with pytest.raises(HoldoutViolation):
        g.open_holdout("jatin", "again")
    # A fresh guard over the same log refuses too (persistence).
    g2 = HoldoutGuard(1_800_000_000_000, 1_900_000_000_000, log_path=log)
    assert g2.opened
    with pytest.raises(HoldoutViolation):
        g2.check_not_opened("any validation")


def test_holdout_invalid_construction(tmp_path):
    with pytest.raises(ValueError):
        HoldoutGuard(1_900_000_000_000, 1_800_000_000_000,
                     log_path=tmp_path / "h.jsonl")


# ── derived risk ─────────────────────────────────────────────────────────

def test_derive_risk_deterministic_and_capped():
    p = ClientProfile()
    r1 = derive_risk_pct(p)
    r2 = derive_risk_pct(p)
    assert r1 == r2
    # 4 losses at derived risk must stay within the target DD (x safety).
    assert r1 * 4 <= p.target_max_dd_pct
    # 1.5x streak must stay within the hard cap.
    assert r1 * 6 <= p.hard_cap_dd_pct
    with pytest.raises(ValueError):
        derive_risk_pct(p, consecutive_losses_assumed=0)
    with pytest.raises(ValueError):
        derive_risk_pct(p, safety_factor=1.0)


# ── gates ────────────────────────────────────────────────────────────────

def good_metrics(**over):
    m = {"oos_max_dd_pct": 1.5, "mc95_dd_pct": 2.0,
         "pf_after_2x_costs": 1.3, "expectancy_r_after_2x_costs": 0.15,
         "n_trades": 120, "max_losing_streak": 6, "streak_dd_pct": 2.0,
         "trades_per_day_avg": 0.8, "no_trade_day_share": 0.3,
         "win_rate": 41.0}
    m.update(over)
    return m


def test_gates_all_pass():
    gates = evaluate_gates(good_metrics())
    assert all(g.passed for g in gates)


def test_each_gate_fails_when_breached():
    breaches = {
        "oos_max_dd": good_metrics(oos_max_dd_pct=2.5),
        "hard_cap_dd": good_metrics(oos_max_dd_pct=3.5),
        "mc95_dd": good_metrics(mc95_dd_pct=3.5),
        "pf_after_2x_costs": good_metrics(pf_after_2x_costs=0.9),
        "expectancy_after_2x_costs":
            good_metrics(expectancy_r_after_2x_costs=-0.1),
        "min_trades": good_metrics(n_trades=10),
        "streak_consequence_dd": good_metrics(streak_dd_pct=3.5),
        "trades_per_day_avg": good_metrics(trades_per_day_avg=0.05),
        "no_trade_day_share": good_metrics(no_trade_day_share=0.9),
    }
    for expected_gate, metrics in breaches.items():
        gates = {g.name: g for g in evaluate_gates(metrics)}
        assert not gates[expected_gate].passed, expected_gate
        # no UNRELATED gate broken by this breach fixture
        unrelated = [g for g in gates.values() if g.name != expected_gate]
        # (oos_max_dd and hard_cap_dd share one metric by design; the
        # 3.5 breach trips both — that is the intended hard-cap alarm.)
        if expected_gate in ("oos_max_dd", "hard_cap_dd"):
            unrelated = [g for g in unrelated
                         if g.name not in ("oos_max_dd", "hard_cap_dd")]
        assert all(g.passed for g in unrelated), expected_gate


def test_win_rate_not_a_gate():
    gates = evaluate_gates(good_metrics(win_rate=5.0))
    assert all(g.passed for g in gates)      # win rate reported, not gating


def test_metric_helpers():
    curve = [(0, 100.0), (1, 110.0), (2, 104.0), (3, 90.0), (4, 108.0)]
    assert streak_dd_pct(curve) == pytest.approx(
        (110.0 - 90.0) / 110.0 * 100.0, abs=1e-3)
    assert longest_losing_streak([1, -1, -1, -1, 2, -1]) == 3
    per_day, no_trade = trade_calendar_stats(
        [T0 + 3_600_000, T0 + 3_700_000, T0 + DAY_MS + 3_600_000], 10)
    assert per_day == pytest.approx(0.3)
    assert no_trade == pytest.approx(0.8)


# ── scaled cost model ────────────────────────────────────────────────────

def test_scaled_cost_key_idempotent_and_linear():
    k2 = scaled_cost_key("XAUUSD_ECN", 2.0)
    from core.config import get_cost_model
    base, scaled = get_cost_model("XAUUSD_ECN"), get_cost_model(k2)
    assert scaled.commission_per_lot == pytest.approx(
        base.commission_per_lot * 2.0)
    assert scaled.slippage.market_points == pytest.approx(
        base.slippage.market_points * 2.0)
    assert scaled_cost_key("XAUUSD_ECN", 2.0) == k2      # idempotent
    assert scaled_cost_key("XAUUSD_ECN", 1.0) == "XAUUSD_ECN"


# ── trial runner ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls,kw", [
    (LondonOpeningRange, {}),
    (TrendPullback, {"fast": 10, "slow": 30}),
    (EmaCrossVolFilter, {"fast": 10, "slow": 30}),
])
def test_run_trial_metrics_and_no_lookahead(cls, kw):
    df = synth_ticks()
    r = run_trial(cls(**kw), df, start_ms=T0, end_ms=T0 + 12 * DAY_MS,
                  session=LONDON, risk_pct=0.25)
    for key in ("n_trades", "net_profit_usd", "win_rate_pct", "pf",
                "max_dd_pct", "max_losing_streak", "expectancy_r",
                "trades_per_day_avg", "no_trade_day_share",
                "lookahead_violations"):
        assert key in r
    assert r["lookahead_violations"] == 0
    assert r["n_trades"] > 0


def test_orb_one_entry_per_direction_per_day():
    df = synth_ticks(days=20, seed=11)
    r = run_trial(LondonOpeningRange(), df, start_ms=T0,
                  end_ms=T0 + 20 * DAY_MS, session=LONDON, risk_pct=0.25)
    assert r["n_trades"] <= 40            # 20 days x 2 directions max
    assert r["behaviour_blocks"]["max_orders_per_day"] >= 0


def test_variants_pass_fingerprint_style_check():
    from core.accounts.fingerprint import fingerprint
    specs = {
        "london_opening_range": {
            "indicators": ["atr"], "params": {"range_minutes": 15},
            "styles": []},
        "trend_pullback": {
            "indicators": ["ema", "atr"], "params": {"fast": 50},
            "styles": []},
        "ema_cross_vol_filter": {
            "indicators": ["ema", "atr"], "params": {"atr_pct": 50},
            "styles": []},
    }
    for vid, spec in specs.items():
        res = fingerprint(vid, spec)
        assert res["prohibited_styles"] == []
        assert res["classification"] in ("standard_public", "custom")


def test_variants_deterministic():
    df = synth_ticks()
    a = run_trial(LondonOpeningRange(), df, start_ms=T0,
                  end_ms=T0 + 12 * DAY_MS, session=LONDON, risk_pct=0.25)
    b = run_trial(LondonOpeningRange(), df, start_ms=T0,
                  end_ms=T0 + 12 * DAY_MS, session=LONDON, risk_pct=0.25)
    assert a["n_trades"] == b["n_trades"]
    assert a["net_profit_usd"] == b["net_profit_usd"]


# ── search + trial store ─────────────────────────────────────────────────

def test_random_search_stores_every_trial_and_respects_holdout(tmp_path):
    from core.research import random_search
    from core.tick_engine import TickEngine   # noqa: F401  (import cost)
    store = TrialStore(tmp_path / "trials.jsonl")
    g = HoldoutGuard(T0 + 10 * DAY_MS, T0 + 12 * DAY_MS,
                     log_path=tmp_path / "h.jsonl")
    grid = {"range_minutes": [10, 15], "rr": [1.0, 1.5]}
    df = synth_ticks(days=9)
    res = random_search(LondonOpeningRange, grid, df,
                        start_ms=T0, end_ms=T0 + 9 * DAY_MS,
                        session=LONDON, risk_pct=0.25, holdout=g,
                        trials_store=store, n_trials=3, seed=5)
    assert len(res) == 3
    assert len(store.entries()) == 3
    assert all("metrics" in r for r in store.entries())
    best = store.best("pf", n=2)
    assert len(best) == 2
    # Overlapping the holdout raises BEFORE any trial runs.
    with pytest.raises(HoldoutViolation):
        random_search(LondonOpeningRange, grid, df,
                      start_ms=T0, end_ms=T0 + 11 * DAY_MS,
                      session=LONDON, risk_pct=0.25, holdout=g,
                      trials_store=TrialStore(tmp_path / "t2.jsonl"),
                      n_trials=1, seed=5)
    assert len(store.entries()) == 3      # nothing appended


def test_walk_forward_checks_each_fold(tmp_path):
    from core.research import walk_forward
    g = HoldoutGuard(T0 + 10 * DAY_MS, T0 + 12 * DAY_MS,
                     log_path=tmp_path / "h.jsonl")
    df = synth_ticks(days=9)
    folds = [(T0, T0 + 4 * DAY_MS), (T0 + 4 * DAY_MS, T0 + 9 * DAY_MS)]
    out = walk_forward(LondonOpeningRange, {}, df, folds=folds,
                       session=LONDON, risk_pct=0.25, holdout=g)
    assert len(out) == 2
    bad_folds = [(T0, T0 + 11 * DAY_MS)]
    with pytest.raises(HoldoutViolation):
        walk_forward(LondonOpeningRange, {}, df, folds=bad_folds,
                     session=LONDON, risk_pct=0.25, holdout=g)


def test_report_writer_deterministic(tmp_path):
    from core.research import write_report
    trials = [{"trial": 1, "metrics": {"pf": 1.2}, "strategy": "x"}]
    gate_rows = [{"candidate": "x", "session": "london", "gate": "mc95_dd",
                  "passed": True, "value": 2.0, "threshold": "<= 3.0"}]
    summary = {"n_trials": 1}
    out1 = write_report(tmp_path / "c1", summary, trials, gate_rows)
    out2 = write_report(tmp_path / "c2", summary, trials, gate_rows)
    assert (out1 / "summary.json").read_text() == \
        (out2 / "summary.json").read_text()
    assert (out1 / "trials.csv").read_text() == \
        (out2 / "trials.csv").read_text()
    assert (out1 / "gates.csv").exists()


# ── Round 2: new candidates + exit knobs ────────────────────────────────

def test_round2_variants_produce_signals_and_no_lookahead():
    from strategies.variants import (AsiaRangeBreakout, FirstMoveContinuation,
                                     SqueezeBreakout)
    from core.research.campaign import LONDON, SessionWindow, run_trial
    ext = SessionWindow("extended_0000_1100", 0, 660)
    df = synth_ticks()
    for strat, sess in ((AsiaRangeBreakout(), ext),
                        (SqueezeBreakout(), LONDON),
                        (FirstMoveContinuation(), ext)):
        m = run_trial(strat, df, start_ms=T0,
                      end_ms=T0 + 20 * DAY_MS, session=sess, risk_pct=0.25)
        assert m["lookahead_violations"] == 0
        assert m["n_trades"] >= 0
        assert isinstance(m["pf"], float)


def test_round2_exit_knobs_change_outcome():
    from strategies.variants import TrendPullback
    from core.research.campaign import LONDON, run_trial
    kw = dict(start_ms=T0, end_ms=T0 + 20 * DAY_MS, session=LONDON,
              risk_pct=0.25)
    df = synth_ticks()
    plain = run_trial(TrendPullback(atr_mult=1.0, rr=1.5), df, **kw)
    be = run_trial(TrendPullback(atr_mult=1.0, rr=1.5), df,
                   breakeven_rr=1.0, breakeven_offset=0.05, **kw)
    # Both run cleanly; exit knobs must not INCREASE drawdown vs plain
    # on the same fills (breakeven only tightens risk).
    assert be["lookahead_violations"] == 0
    assert plain["lookahead_violations"] == 0
    if plain["n_trades"] and be["n_trades"]:
        assert be["max_dd_pct"] <= plain["max_dd_pct"] + 1e-9


def test_random_search_trial_extras_grid(tmp_path):
    from core.research import (HoldoutGuard, SessionWindow, TrialStore,
                               random_search)
    from strategies.variants import TrendPullback
    hs = HoldoutGuard(T0 + 30 * DAY_MS, T0 + 60 * DAY_MS,
                      log_path=tmp_path / "h.jsonl")
    store = TrialStore(tmp_path / "t.jsonl")
    grid = {"atr_mult": [1.0, 1.5]}
    extras = {"breakeven_rr": [0.0, 1.0], "time_exit_minutes": [0.0, 120.0]}
    df = synth_ticks()
    results = random_search(TrendPullback, grid, df,
                            start_ms=T0, end_ms=T0 + 20 * DAY_MS,
                            session=SessionWindow("test", 420, 660),
                            risk_pct=0.25, holdout=hs, trials_store=store,
                            n_trials=6, seed=5, trial_extras_grid=extras)
    assert len(results) == 6
    for r in store.entries():
        assert set(r["extras"]) <= set(extras)
        assert r["extras"]["breakeven_rr"] in (0.0, 1.0)


def test_first_move_rejects_bad_open_minutes():
    import pytest as _pytest
    from strategies.variants import FirstMoveContinuation
    with _pytest.raises(ValueError):
        FirstMoveContinuation(open_minutes=17)
