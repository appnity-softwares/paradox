"""Strategy validation stack (Master Prompt 6.6).

Deterministic, no LLM involvement. Components:

- Walk-forward (rolling and fixed) lives in :mod:`core.optimize`.
- Monte Carlo on trade resampling: bootstrap the trade P&Ls and report the
  distribution of outcomes, not a single lucky path.
- Monte Carlo on randomized start dates: the same strategy started at
  different points of the same data must not collapse.
- Cost stress: 1x / 1.5x / 2x spread+slippage+commission. A strategy that
  only survives baseline costs is not robust.
- Trial-count adjustment: deflated Sharpe (Bailey & Lopez de Prado 2014)
  using the number of parameter trials actually run.
- Robustness score (0-100) and the deployment readiness verdict:
  ``FAIL`` / ``RESEARCH_ONLY`` / ``PAPER_TRADE_ELIGIBLE``. Honest FAIL
  behaviour is preserved: a failing strategy is never softened.

Every trial run is recorded and returned with the report (anti-selection-bias
requirement, Part 6.4).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from core.backtest import run_backtest
from core.metrics import compute_metrics
from core.optimize import PARAM_GRID, fixed_walk_forward, grid_search

_NORM = NormalDist()

# Robustness score weights (sum = 100). Documented, fixed, deterministic.
WEIGHTS = {
    "oos_profitable": 25,      # walk-forward OOS total return > 0, scaled by fold ratio
    "mc_p5_positive": 20,      # Monte Carlo 5th percentile total return > 0
    "cost_stress_2x": 20,      # profitable at 2x costs (partial credit at 1.5x)
    "deflated_sharpe": 20,     # deflated Sharpe > 0
    "sample_size": 15,         # >= 50 OOS trades
}
GAMMA_EULER = 0.5772  # Euler-Mascheroni constant for the expected-max Sharpe


# ---------------------------------------------------------------------------
# Monte Carlo: trade resampling
# ---------------------------------------------------------------------------


def monte_carlo_trades(
    pnls: np.ndarray,
    start_equity: float = 10_000.0,
    n_sims: int = 2_000,
    seed: int = 42,
) -> dict:
    """Bootstrap trade P&Ls (with replacement) into synthetic equity paths.

    Returns percentile bands for total return, drawdown probability stats and
    the probability that the resampled strategy loses money.
    """
    pnls = np.asarray(pnls, dtype=float)
    if pnls.size == 0:
        return {"p5_ret_pct": 0.0, "p50_ret_pct": 0.0, "p95_ret_pct": 0.0,
                "prob_loss": 1.0, "p95_max_dd_pct": 100.0, "sims": 0}
    rng = np.random.default_rng(seed)
    n = pnls.size
    # Matrix of sims x trades; memory-bounded chunking for large sims.
    rets = np.empty(n_sims)
    dds = np.empty(n_sims)
    chunk = 200
    for s0 in range(0, n_sims, chunk):
        s1 = min(s0 + chunk, n_sims)
        sample = rng.choice(pnls, size=(s1 - s0, n), replace=True)
        eq = start_equity + np.cumsum(sample, axis=1)
        peak = np.maximum.accumulate(eq, axis=1)
        dd_pct = (peak - eq) / np.maximum(peak, 1e-9) * 100.0
        rets[s0:s1] = (eq[:, -1] / start_equity - 1.0) * 100.0
        dds[s0:s1] = dd_pct.max(axis=1)
    return {
        "p5_ret_pct": float(np.percentile(rets, 5)),
        "p50_ret_pct": float(np.percentile(rets, 50)),
        "p95_ret_pct": float(np.percentile(rets, 95)),
        "prob_loss": float((rets < 0).mean()),
        "p95_max_dd_pct": float(np.percentile(dds, 95)),
        "sims": int(n_sims),
    }


# ---------------------------------------------------------------------------
# Monte Carlo: randomized start dates
# ---------------------------------------------------------------------------


def monte_carlo_start_dates(
    df: pd.DataFrame,
    strategy_cls,
    params: dict,
    cfg,
    tf_label: str,
    n_starts: int = 6,
    min_bars: int = 300,
) -> dict:
    """Run the same fixed-parameter strategy from different start dates.

    Each start trades from its date to the end of the data. Returns the
    spread of outcomes; a strategy that only works from one lucky start is
    not robust.
    """
    n = len(df)
    usable = n - min_bars
    if usable < n_starts:
        raise ValueError(f"not enough bars ({n}) for {n_starts} start dates")
    starts = np.linspace(0, usable, n_starts, dtype=int)
    rows = []
    for s in starts:
        start_at = df.index[int(s)]
        cfg_c = copy.deepcopy(cfg)
        strat = strategy_cls(cfg_c, **params)
        equity, trades = run_backtest(
            df.iloc[int(s):], strat, cfg_c, start_trading_at=start_at,
        )
        m = compute_metrics(equity, trades, cfg_c, tf_label)
        rows.append({
            "start": str(start_at),
            "Return %": m["Return %"],
            "Sharpe": m["Sharpe"],
            "Trades": m["Total Trades"],
            "Max DD %": m["Max DD %"],
        })
    table = pd.DataFrame(rows)
    rets = table["Return %"].astype(float)
    return {
        "table": table,
        "median_ret_pct": float(rets.median()),
        "worst_ret_pct": float(rets.min()),
        "positive_ratio": float((rets > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Cost stress
# ---------------------------------------------------------------------------


def cost_stress(
    df: pd.DataFrame,
    strategy_cls,
    params: dict,
    cfg,
    tf_label: str,
    multipliers: tuple[float, ...] = (1.0, 1.5, 2.0),
    symbol: str = "XAUUSD",
) -> list[dict]:
    """Re-run with multiplied spread + slippage + commission.

    The baseline (1.0x) uses the configured cost model; higher multipliers
    stress every cost component proportionally.
    """
    from core.config import get_cost_model

    costs = get_cost_model(symbol)
    base_spread = costs.spread_price()
    base_slip = costs.slippage.market_points
    base_comm = costs.commission_per_lot

    rows = []
    for mult in multipliers:
        cfg_c = copy.deepcopy(cfg)
        strat = strategy_cls(cfg_c, **params)
        equity, trades = run_backtest(
            df, strat, cfg_c,
            spread=base_spread * mult,
            slippage=base_slip * mult,
            commission_per_unit=base_comm * mult,
        )
        m = compute_metrics(equity, trades, cfg_c, tf_label)
        rows.append({
            "multiplier": mult,
            "Return %": m["Return %"],
            "Sharpe": m["Sharpe"],
            "Trades": m["Total Trades"],
            "Profit Factor": m.get("Profit Factor", float("nan")),
        })
    return rows


# ---------------------------------------------------------------------------
# Deflated Sharpe (trial-count adjustment)
# ---------------------------------------------------------------------------


def deflated_sharpe(
    sr: float,
    n_obs: int,
    n_trials: int,
    skew: float,
    kurtosis: float,
    var_sharpe_trials: float | None = None,
) -> float:
    """Probability that the true Sharpe exceeds the expected best-of-N-trials.

    ``sr`` is the NON-annualized per-trade Sharpe (mean/std of trade returns),
    ``n_obs`` the number of trades, ``n_trials`` how many parameter sets were
    tested, ``var_sharpe_trials`` the cross-trial variance of Sharpe estimates
    (fallback: the estimator variance from skew/kurtosis).
    """
    if n_obs < 2 or n_trials < 1:
        return 0.0
    sr = float(sr)
    skew = float(skew)
    kurt = float(kurtosis)
    est_var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n_obs - 1)
    if var_sharpe_trials is None or not np.isfinite(var_sharpe_trials) or var_sharpe_trials <= 0:
        var_sr = est_var
    else:
        var_sr = max(var_sharpe_trials, est_var)
    n = max(n_trials, 1)
    if n == 1:
        # No selection effect with a single trial: DSR reduces to the
        # probabilistic Sharpe ratio against SR* = 0.
        sr0 = 0.0
    else:
        z1 = _NORM.inv_cdf(1.0 - 1.0 / n)
        z2 = _NORM.inv_cdf(1.0 - 1.0 / (n * math.e))
        sr0 = math.sqrt(var_sr) * ((1.0 - GAMMA_EULER) * z1 + GAMMA_EULER * z2)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z_dsr = (sr - sr0) * math.sqrt(n_obs - 1) / denom
    return float(_NORM.cdf(z_dsr))


def _per_trade_sharpe(pnls: np.ndarray) -> tuple[float, float, float]:
    """(SR, skew, kurtosis) of per-trade P&L. numpy-native, no scipy."""
    x = np.asarray(pnls, dtype=float)
    if x.size < 3 or np.std(x) == 0:
        return 0.0, 0.0, 3.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    z = (x - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # Pearson (non-excess), normal -> 3.0
    return mu / sd, skew, kurt


# ---------------------------------------------------------------------------
# Full validation report + verdict
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    key: str
    params: dict
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_trials: int = 0
    walk_forward: dict = field(default_factory=dict)
    monte_carlo: dict = field(default_factory=dict)
    start_dates: dict = field(default_factory=dict)
    cost_stress: list = field(default_factory=list)
    deflated_sharpe_value: float = 0.0
    robustness_score: float = 0.0
    verdict: str = "FAIL"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "params": self.params,
            "n_trials": self.n_trials,
            "wf": {k: v for k, v in self.walk_forward.items() if k != "folds_table"},
            "mc": {k: v for k, v in self.monte_carlo.items()},
            "start_dates": {k: v for k, v in self.start_dates.items() if k != "table"},
            "cost_stress": self.cost_stress,
            "deflated_sharpe": round(self.deflated_sharpe_value, 4),
            "robustness_score": round(self.robustness_score, 1),
            "verdict": self.verdict,
        }


def validate_strategy(
    df: pd.DataFrame,
    key: str,
    strategy_cls,
    params: dict,
    cfg,
    tf_label: str,
    start_equity: float = 10_000.0,
    symbol: str = "XAUUSD",
    mc_sims: int = 2_000,
    n_starts: int = 6,
    seed: int = 42,
) -> ValidationReport:
    """Run the full validation battery for one fixed parameter set.

    Grid trials are recorded (anti-selection-bias), the deflated Sharpe uses
    the actual trial count, and the verdict follows the documented gates.
    """
    report = ValidationReport(key=key, params=dict(params))

    # 1) Record every trial from the strategy's parameter grid.
    grid = PARAM_GRID.get(key, {})
    if grid:
        trials_df, best = grid_search(df, strategy_cls, grid, cfg, tf_label)
        report.trials = trials_df
        report.n_trials = int(len(trials_df))
        var_sharpe_trials = float(pd.to_numeric(trials_df["Sharpe"], errors="coerce").var())
    else:
        var_sharpe_trials = None

    # 2) Walk-forward with FIXED params (honest OOS, no per-fold tuning).
    folds_df, pooled, wf_summary = fixed_walk_forward(
        df, strategy_cls, params, cfg, tf_label,
    )
    report.walk_forward = wf_summary

    # 3) Baseline run for the Monte Carlo trade resampling.
    cfg_c = copy.deepcopy(cfg)
    strat = strategy_cls(cfg_c, **params)
    equity, trades = run_backtest(df, strat, cfg_c, start_equity=start_equity)
    pnls = trades["pnl"].to_numpy(dtype=float) if len(trades) else np.array([])
    report.monte_carlo = monte_carlo_trades(pnls, start_equity, mc_sims, seed)
    sr, skew, kurt = _per_trade_sharpe(pnls)
    report.deflated_sharpe_value = deflated_sharpe(
        sr, n_obs=int(pnls.size), n_trials=max(report.n_trials, 1),
        skew=skew, kurtosis=kurt, var_sharpe_trials=var_sharpe_trials,
    )

    # 4) Randomized start dates.
    report.start_dates = monte_carlo_start_dates(
        df, strategy_cls, params, cfg, tf_label, n_starts=n_starts,
    )

    # 5) Cost stress.
    report.cost_stress = cost_stress(df, strategy_cls, params, cfg, tf_label, symbol=symbol)

    # 6) Score + verdict.
    report.robustness_score, report.verdict = _score_and_verdict(report)
    return report


def _score_and_verdict(rep: ValidationReport) -> tuple[float, str]:
    wf = rep.walk_forward
    mc = rep.monte_carlo
    cs = {row["multiplier"]: row for row in rep.cost_stress}

    score = 0.0
    # OOS profitability scaled by fold consistency.
    fold_ratio = float(wf.get("Profitable Folds %", 0.0)) / 100.0
    if float(wf.get("OOS Ret %", 0.0)) > 0:
        score += WEIGHTS["oos_profitable"] * fold_ratio
    # Monte Carlo p5.
    if mc.get("p5_ret_pct", 0.0) > 0:
        score += WEIGHTS["mc_p5_positive"]
    # Cost stress: full credit at 2x, partial at 1.5x.
    if cs.get(2.0, {}).get("Return %", -1e9) > 0:
        score += WEIGHTS["cost_stress_2x"]
    elif cs.get(1.5, {}).get("Return %", -1e9) > 0:
        score += WEIGHTS["cost_stress_2x"] * 0.5
    # Deflated Sharpe.
    if rep.deflated_sharpe_value > 0:
        score += WEIGHTS["deflated_sharpe"]
    # Sample size.
    if int(wf.get("OOS Trades", 0)) >= 50:
        score += WEIGHTS["sample_size"]

    oos_ret = float(wf.get("OOS Ret %", 0.0))
    wf_verdict = str(wf.get("Verdict", "FAIL"))
    if (
        wf_verdict == "ROBUST"
        and mc.get("p5_ret_pct", 0.0) > 0
        and cs.get(2.0, {}).get("Return %", -1e9) > 0
        and rep.deflated_sharpe_value > 0
        and int(wf.get("OOS Trades", 0)) >= 50
    ):
        return score, "PAPER_TRADE_ELIGIBLE"
    if oos_ret > 0 and score >= 40:
        return score, "RESEARCH_ONLY"
    return score, "FAIL"


def sensitivity_heatmap(
    df: pd.DataFrame,
    strategy_cls,
    params: dict,
    x_param: str,
    x_values: list,
    y_param: str,
    y_values: list,
    cfg: "StrategyConfig",
    tf_label: str = "15m",
    metric: str = "Sharpe",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2-D parameter sensitivity grid (Part 6.6).

    Every cell is a full backtest with all other parameters held at
    ``params``. A robust strategy shows a smooth surface; an isolated spike
    is an overfit cell, not an edge. Returns (pivot, all_trials).
    """
    rows = []
    for xv in x_values:
        for yv in y_values:
            p = dict(params)
            p[x_param] = xv
            p[y_param] = yv
            cfg_c = copy.deepcopy(cfg)
            try:
                strat = strategy_cls(cfg_c, **p)
                equity, trades = run_backtest(df, strat, cfg_c)
                m = compute_metrics(equity, trades, cfg_c, tf_label)
            except Exception as exc:  # invalid combo: record, keep going
                rows.append({x_param: xv, y_param: yv, metric: np.nan,
                             "error": str(exc)[:120]})
                continue
            rows.append({x_param: xv, y_param: yv, metric: m.get(metric, np.nan),
                         "Trades": m.get("Total Trades", 0)})
    trials = pd.DataFrame(rows)
    pivot = trials.pivot_table(index=y_param, columns=x_param, values=metric)
    return pivot, trials


def regime_breakdown(
    df: pd.DataFrame,
    strategy_cls,
    params: dict,
    cfg: "StrategyConfig",
    tf_label: str = "15m",
    atr_length: int = 14,
    adx_length: int = 14,
    adx_threshold: float = 20.0,
    vol_lookback: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one strategy run's trades by market regime (Part 6.6).

    A trade's regime is the regime of its ENTRY bar:
    - volatility: ATR percentile terciles over a rolling ``vol_lookback``
      window -> ``vol_low`` / ``vol_mid`` / ``vol_high``
    - direction: ADX >= ``adx_threshold`` -> ``trend``, else ``range``

    Returns (per-regime summary, trades tagged with regime columns).
    """
    from core.indicators import adx, atr

    cfg_c = copy.deepcopy(cfg)
    strat = strategy_cls(cfg_c, **params)
    equity, trades = run_backtest(df, strat, cfg_c)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        empty = pd.DataFrame(columns=[
            "regime", "trades", "win_rate_pct", "total_pnl", "avg_pnl",
        ])
        return empty, trades_df

    a = atr(df, atr_length)
    pct = a.rolling(vol_lookback, min_periods=max(2, vol_lookback // 2)).rank(pct=True)
    vol_regime = pd.Series(
        np.where(pct.isna(), "unknown",
                 np.where(pct <= 1 / 3, "vol_low",
                          np.where(pct <= 2 / 3, "vol_mid", "vol_high"))),
        index=df.index,
    )
    d = adx(df, adx_length)
    dir_regime = pd.Series(
        np.where(d.isna(), "unknown", np.where(d >= adx_threshold, "trend", "range")),
        index=df.index,
    )

    entry_times = pd.DatetimeIndex(trades_df["entry_time"])
    trades_df["vol_regime"] = vol_regime.reindex(entry_times).to_numpy()
    trades_df["dir_regime"] = dir_regime.reindex(entry_times).to_numpy()
    trades_df["regime"] = trades_df["vol_regime"] + "/" + trades_df["dir_regime"]

    summary = trades_df.groupby("regime").agg(
        trades=("pnl", "size"),
        win_rate_pct=("pnl", lambda s: round(100.0 * float((s > 0).mean()), 1)),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).reset_index()
    summary["avg_pnl"] = summary["avg_pnl"].round(2)
    summary["total_pnl"] = summary["total_pnl"].round(2)
    return summary, trades_df


def rolling_holdout(
    df: pd.DataFrame,
    strategy_cls,
    params: dict,
    cfg: "StrategyConfig",
    tf_label: str = "15m",
    n_windows: int = 4,
    min_bars: int = 300,
) -> pd.DataFrame:
    """Multi-period holdout gate (Part 6.6).

    A single holdout window is one sample; this evaluates the SAME fixed
    params on ``n_windows`` successive unseen windows. The config must be
    chosen WITHOUT looking at any window used here (e.g. picked on an
    earlier period, as scripts.verify_cells does). A strategy is only
    consistently profitable if a majority of windows stay positive.

    Returns one row per window: bars, trades, return %, Sharpe, max DD %,
    profitable flag.
    """
    n = len(df)
    seg = n // (n_windows + 1)
    if seg < min_bars:
        raise ValueError(
            f"Not enough data for {n_windows} holdout windows (segment={seg} bars, need >= {min_bars})"
        )
    rows = []
    for w in range(n_windows):
        # Window w is bars[seg*(w+1) : seg*(w+2)]: each window is data the
        # previous windows never saw; window 0 also never saw the front seg
        # bars that stand in for the selection period.
        win = df.iloc[seg * (w + 1): seg * (w + 2)]
        cfg_c = copy.deepcopy(cfg)
        strat = strategy_cls(cfg_c, **params)
        equity, trades = run_backtest(win, strat, cfg_c)
        m = compute_metrics(equity, trades, cfg_c, tf_label)
        rows.append({
            "window": w,
            "start": str(win.index[0]),
            "end": str(win.index[-1]),
            "bars": len(win),
            "trades": m.get("Total Trades", 0),
            "ret_pct": m.get("Return %", 0.0),
            "sharpe": m.get("Sharpe", 0.0),
            "max_dd_pct": m.get("Max DD %", 0.0),
            "profitable": bool(m.get("Return %", 0.0) > 0),
        })
    return pd.DataFrame(rows)
