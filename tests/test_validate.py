"""Phase 1C: validation stack tests (Monte Carlo, cost stress, deflated
Sharpe, robustness score, honest verdicts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.validate import (
    WEIGHTS,
    cost_stress,
    deflated_sharpe,
    monte_carlo_start_dates,
    monte_carlo_trades,
    validate_strategy,
)
from strategies.base import StrategyConfig
from strategies.library import EMACross


def make_bars(n: int = 1600, seed: int = 8, drift: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(drift, 0.6, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 100.0,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"),
    )


class Loser:
    """Always short on rising data -> must lose; verdict must stay honest."""

    key = "loser"
    warmup = 5

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        cfg.name = "Loser"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(-1.0, index=df.index)


# ---------------------------------------------------------------------------
# Monte Carlo trade resampling
# ---------------------------------------------------------------------------


def test_mc_resampling_positive_edge_survives_resampling():
    rng = np.random.default_rng(0)
    pnls = rng.normal(5, 20, 200)
    mc = monte_carlo_trades(pnls, start_equity=10_000, n_sims=500, seed=42)
    assert mc["sims"] == 500
    assert mc["p5_ret_pct"] > 0
    assert mc["prob_loss"] < 0.05
    assert mc["p50_ret_pct"] > mc["p5_ret_pct"]


def test_mc_resampling_negative_edge_loses_in_most_sims():
    rng = np.random.default_rng(0)
    pnls = rng.normal(-5, 20, 200)
    mc = monte_carlo_trades(pnls, start_equity=10_000, n_sims=500, seed=42)
    assert mc["prob_loss"] > 0.95
    assert mc["p95_ret_pct"] < 0


def test_mc_resampling_deterministic_for_same_seed():
    pnls = np.arange(1, 51, dtype=float)
    a = monte_carlo_trades(pnls, n_sims=200, seed=7)
    b = monte_carlo_trades(pnls, n_sims=200, seed=7)
    assert a == b


def test_mc_resampling_empty_trades_is_honest():
    mc = monte_carlo_trades(np.array([]))
    assert mc["prob_loss"] == 1.0
    assert mc["sims"] == 0


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------


def test_deflated_sharpe_penalizes_many_trials():
    single = deflated_sharpe(0.25, n_obs=200, n_trials=1, skew=0.1, kurtosis=3.0)
    many = deflated_sharpe(0.25, n_obs=200, n_trials=1000, skew=0.1, kurtosis=3.0)
    assert single > many
    assert many < 1.0


def test_deflated_sharpe_negative_or_short_samples_are_low():
    assert deflated_sharpe(-0.1, 200, 10, 0.1, 3.0) < 0.05
    assert deflated_sharpe(0.25, 1, 1, 0.0, 3.0) == 0.0  # too few trades


# ---------------------------------------------------------------------------
# Cost stress
# ---------------------------------------------------------------------------


def test_cost_stress_higher_costs_never_improve_returns():
    df = make_bars()
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0)
    rows = cost_stress(df, EMACross, {"fast": 20, "slow": 60}, cfg, "1m")
    assert [r["multiplier"] for r in rows] == [1.0, 1.5, 2.0]
    rets = [r["Return %"] for r in rows]
    assert rets[2] <= rets[0] + 1e-9  # 2x costs cannot beat 1x costs


# ---------------------------------------------------------------------------
# Randomized start dates
# ---------------------------------------------------------------------------


def test_start_dates_runs_all_starts_and_reports_spread():
    df = make_bars()
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0)
    res = monte_carlo_start_dates(df, EMACross, {"fast": 20, "slow": 60}, cfg, "1m", n_starts=4)
    assert len(res["table"]) == 4
    assert 0.0 <= res["positive_ratio"] <= 1.0
    assert res["worst_ret_pct"] <= res["median_ret_pct"]


# ---------------------------------------------------------------------------
# Full validation + honest verdicts
# ---------------------------------------------------------------------------


def test_validate_strategy_end_to_end_records_trials_and_verdict():
    df = make_bars()
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0)
    rep = validate_strategy(
        df, "ema_cross", EMACross, {"fast": 20, "slow": 60}, cfg, "1m",
        mc_sims=200, n_starts=3,
    )
    assert rep.n_trials > 0                       # every trial recorded
    assert len(rep.trials) == rep.n_trials
    assert 0.0 <= rep.robustness_score <= sum(WEIGHTS.values())
    assert rep.verdict in {"FAIL", "RESEARCH_ONLY", "PAPER_TRADE_ELIGIBLE"}
    d = rep.to_dict()
    assert d["verdict"] == rep.verdict
    assert "OOS Ret %" in d["wf"]


def test_losing_strategy_gets_fail_verdict():
    df = make_bars(drift=0.05)  # rising market, always short -> loses
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0)
    rep = validate_strategy(
        df, "loser", Loser, {}, cfg, "1m", mc_sims=100, n_starts=3,
    )
    assert rep.verdict == "FAIL"
    assert rep.walk_forward["OOS Ret %"] < 0


def test_sensitivity_heatmap_shape_and_values():
    from core.validate import sensitivity_heatmap

    df = make_bars(1600)
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0)
    pivot, trials = sensitivity_heatmap(
        df, EMACross, {"slow": 60}, "fast", [5, 10, 20], "slow", [40, 60, 90],
        cfg, "1min",
    )
    assert pivot.shape == (3, 3)
    assert len(trials) == 9
    # Enough bars + enough trades -> every cell must be numeric, no NaN.
    assert pivot.notna().all().all()


def test_regime_breakdown_tags_and_sums():
    from core.validate import regime_breakdown

    df = make_bars(1600)
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0)
    summary, tagged = regime_breakdown(df, EMACross, {"fast": 10, "slow": 40}, cfg, "1min")
    assert not tagged.empty
    assert {"vol_regime", "dir_regime", "regime"} <= set(tagged.columns)
    # Every trade is tagged with a non-null regime.
    assert tagged["regime"].notna().all()
    assert (tagged["regime"] != "").all()
    # Summary trade counts must add up to the total number of trades.
    assert int(summary["trades"].sum()) == len(tagged)


def test_regime_breakdown_no_trades():
    from core.validate import regime_breakdown

    df = make_bars(400)
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0)

    class Never:
        key = "never"
        warmup = 5

        def __init__(self, cfg, **params):
            self.cfg = cfg

        def generate_signals(self, d):
            import numpy as _np
            import pandas as _pd
            return _pd.Series(_np.zeros(len(d), dtype=int), index=d.index)

    summary, tagged = regime_breakdown(df, Never, {}, cfg, "1min")
    assert tagged.empty
    assert len(summary) == 0


def test_optuna_search_constraint_and_determinism():
    optuna = pytest.importorskip("optuna")
    from scripts.optuna_search import make_objective

    df = make_bars(1600)
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0)

    def run_study():
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(make_objective("ema_cross", EMACross, df, cfg),
                       n_trials=5, show_progress_bar=False)
        return study

    s1, s2 = run_study(), run_study()
    # Deterministic: same seed -> same trial sequence and values.
    assert [t.params for t in s1.trials] == [t.params for t in s2.trials]
    assert [t.value for t in s1.trials] == [t.value for t in s2.trials]
    # The fast < slow constraint holds for every recorded trial.
    for t in s1.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            assert t.params["fast"] < t.params["slow"]


def test_rolling_holdout_windows_are_disjoint_and_recorded():
    from core.validate import rolling_holdout

    df = make_bars(1600)
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0)
    table = rolling_holdout(df, EMACross, {"fast": 10, "slow": 40}, cfg, "1min",
                            n_windows=3)
    assert len(table) == 3
    # Windows must be disjoint and sequential.
    ends = table["end"].tolist()
    starts = table["start"].tolist()
    assert all(ends[i] < starts[i + 1] for i in range(len(ends) - 1))
    # Every window records trades and a profitability flag.
    assert table["profitable"].isin([True, False]).all()
    assert (table["bars"] >= 300).all()


def test_rolling_holdout_rejects_short_data():
    from core.validate import rolling_holdout

    df = make_bars(600)
    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0)
    with pytest.raises(ValueError, match="Not enough data"):
        rolling_holdout(df, EMACross, {"fast": 10, "slow": 40}, cfg, "1min",
                        n_windows=4)
