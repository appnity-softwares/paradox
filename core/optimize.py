"""Grid-search optimization + walk-forward validation (Phase 5 seed).

Anti-overfitting principle:
- Grid search finds the best parameters on IN-SAMPLE (training) data.
- Walk-forward then tests those parameters on OUT-OF-SAMPLE (unseen) data.
- If OOS collapses while IS looks great -> overfit strategy, do not trust it.
"""
import copy
import itertools

import numpy as np
import pandas as pd

from core.backtest import run_backtest
from core.metrics import compute_metrics

PARAM_GRID = {
    "ema_cross": {"fast": [9, 20, 50], "slow": [50, 100, 200]},
    "macd_cross": {"fast": [8, 12, 16], "slow": [21, 26, 34], "signal": [9]},
    "rsi_reversal": {"length": [7, 14], "oversold": [25, 30], "overbought": [70, 75]},
    "bollinger": {"length": [20, 30], "num_std": [1.5, 2.0, 2.5]},
    "vwap_ema": {"ema_length": [20, 50], "slope": [0, 1]},
    "supertrend": {"length": [10, 14], "mult": [2.0, 3.0]},
    "session_breakout": {"buffer": [0.0, 0.3], "range_end": [6, 7]},
    "smc_fvg": {"min_gap_atr": [0.3, 0.5], "max_age": [20, 30]},
    "smc_order_block": {"swing_len": [5, 8], "impulse_mult": [1.2, 1.8]},
    "smc_liquidity_sweep": {"lookback": [15, 25], "sweep_buffer": [0.1, 0.3]},
    # Phase 1C additions
    "psar_trend": {"af_step": [0.02, 0.04], "af_max": [0.2]},
    "stoch_cross": {"length": [9, 14], "oversold": [20, 25], "overbought": [75, 80]},
    "zscore_reversion": {"length": [30, 50], "entry_z": [1.5, 2.0, 2.5]},
    "adx_trend": {"fast": [20, 30], "slow": [60, 100], "adx_threshold": [15, 20, 25]},
    "smc_bos": {"swing_len": [3, 5, 8]},
}


def param_combos(grid: dict):
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        # skip invalid fast/slow combos
        if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
            continue
        yield params


def _run(df, strategy_cls, params, cfg, tf_label, engine_kwargs=None, metrics_start=None):
    cfg_c = copy.deepcopy(cfg)  # constructors mutate cfg.name
    strat = strategy_cls(cfg_c, **params)
    equity, trades = run_backtest(df, strat, cfg_c, **(engine_kwargs or {}))
    metric_equity = equity.loc[metrics_start:] if metrics_start is not None else equity
    metric_trades = trades
    if metrics_start is not None and len(trades):
        metric_trades = trades[trades["entry_time"] >= metrics_start].reset_index(drop=True)
    metrics = compute_metrics(metric_equity, metric_trades, cfg_c, tf_label)
    return equity, trades, metrics


def grid_search(df, strategy_cls, grid, cfg, tf_label, min_trades=15, engine_kwargs=None):
    """Try every param combo; rank by Sharpe, requiring a minimum trade count."""
    rows = []
    for params in param_combos(grid):
        _, _, m = _run(df, strategy_cls, params, cfg, tf_label, engine_kwargs or {})
        rows.append({**params, **m})
    res = pd.DataFrame(rows)
    valid = res[res["Total Trades"] >= min_trades]
    pool = valid if len(valid) else res
    best = pool.loc[pool["Sharpe"].idxmax()]
    return res, best


def _verdict(
    total_oos: float,
    med_sharpe: float,
    profitable_fold_ratio: float,
    trades: int,
    max_dd: float,
) -> str:
    """Conservative prototype verdict; profit alone is not robustness."""
    enough_trades = trades >= 50
    if (
        total_oos > 0
        and med_sharpe >= 0.5
        and profitable_fold_ratio >= 0.60
        and enough_trades
        and max_dd > -25.0
    ):
        return "ROBUST"
    if total_oos > 0 and profitable_fold_ratio >= 0.50 and trades >= 30:
        return "MIXED"
    return "FAIL"


def fixed_walk_forward(df, strategy_cls, params, cfg, tf_label, n_folds=4, engine_kwargs=None):
    """Walk-forward with FIXED params (no per-fold optimization).

    Used by the AI Strategy Builder: it tests the user's exact requested config
    on unseen data instead of finding 'best' params for them.
    """
    seg = len(df) // (n_folds + 1)
    if seg < 300:
        raise ValueError(f"Not enough data for {n_folds} folds (segment={seg} bars, need >=300)")

    probe_cfg = copy.deepcopy(cfg)
    probe = strategy_cls(probe_cfg, **params)
    buffer = getattr(probe, "warmup", 60) + 5

    folds = []
    oos_trades = []
    for f in range(n_folds):
        ts, te = (f + 1) * seg, min((f + 2) * seg, len(df))
        actual_start = df.index[ts]
        test = df.iloc[max(0, ts - buffer): te]
        kwargs = {**(engine_kwargs or {}), "start_trading_at": actual_start}
        _, trades, m = _run(
            test,
            strategy_cls,
            params,
            cfg,
            tf_label,
            kwargs,
            metrics_start=actual_start,
        )
        if len(trades):
            trades = trades.assign(fold=f + 1)
        oos_trades.append(trades)
        folds.append({
            "Fold": f + 1,
            "OOS Sharpe": m["Sharpe"],
            "OOS Ret %": m["Return %"],
            "OOS Trades": m["Total Trades"],
            "OOS Win %": m["Win Rate %"],
            "OOS Max DD %": m["Max DD %"],
        })

    folds_df = pd.DataFrame(folds)
    non_empty = [t for t in oos_trades if len(t)]
    pooled = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

    rets = folds_df["OOS Ret %"].astype(float)
    total_oos = float((np.prod(1 + rets / 100) - 1) * 100)
    med_sharpe = float(folds_df["OOS Sharpe"].median())
    profitable_fold_ratio = float((rets > 0).mean())
    max_dd = float(folds_df["OOS Max DD %"].min())
    summary = {
        "OOS Ret %": round(total_oos, 1),
        "OOS Sharpe (med)": round(med_sharpe, 2),
        "OOS Win %": round(float((pooled["pnl"] > 0).mean() * 100), 1) if len(pooled) else 0.0,
        "OOS Trades": int(len(pooled)),
        "OOS Max DD %": round(max_dd, 1),
        "Profitable Folds %": round(profitable_fold_ratio * 100, 1),
        "Verdict": _verdict(total_oos, med_sharpe, profitable_fold_ratio, len(pooled), max_dd),
    }
    return folds_df, pooled, summary


def walk_forward(df, strategy_cls, grid, cfg, tf_label, n_folds=4, min_trades=15, engine_kwargs=None):
    """Rolling walk-forward: optimize on fold i (train), validate on fold i+1 (test).

    Returns (folds_df, pooled_oos_trades, summary_dict).
    """
    seg = len(df) // (n_folds + 1)
    if seg < 300:
        raise ValueError(f"Not enough data for {n_folds} folds (segment={seg} bars, need >=300)")

    folds = []
    oos_trades = []

    for f in range(n_folds):
        train = df.iloc[f * seg: (f + 1) * seg]

        # 1) optimize on train
        _, best = grid_search(train, strategy_cls, grid, cfg, tf_label, min_trades, engine_kwargs)
        best_params = {k: best[k] for k in grid}
        best_params = {
            k: (int(v) if float(v).is_integer() else float(v)) for k, v in best_params.items()
        }

        # 2) validate on unseen test window (with warmup buffer so no train data leaks)
        probe_cfg = copy.deepcopy(cfg)
        probe = strategy_cls(probe_cfg, **best_params)
        buffer = getattr(probe, "warmup", 60) + 5
        ts, te = (f + 1) * seg, min((f + 2) * seg, len(df))
        actual_start = df.index[ts]
        test = df.iloc[max(0, ts - buffer): te]
        kwargs = {**(engine_kwargs or {}), "start_trading_at": actual_start}

        _, trades, m_oos = _run(
            test,
            strategy_cls,
            best_params,
            cfg,
            tf_label,
            kwargs,
            metrics_start=actual_start,
        )
        if len(trades):
            trades = trades.assign(fold=f + 1)
        oos_trades.append(trades)

        folds.append({
            "Fold": f + 1,
            "Params": ", ".join(f"{k}={v}" for k, v in best_params.items()),
            "IS Sharpe": best["Sharpe"],
            "IS Ret %": best["Return %"],
            "OOS Sharpe": m_oos["Sharpe"],
            "OOS Ret %": m_oos["Return %"],
            "OOS Trades": m_oos["Total Trades"],
            "OOS Win %": m_oos["Win Rate %"],
            "OOS Max DD %": m_oos["Max DD %"],
        })

    folds_df = pd.DataFrame(folds)

    # pooled OOS stats
    non_empty = [t for t in oos_trades if len(t)]
    pooled = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

    rets = folds_df["OOS Ret %"].astype(float)
    total_oos = float((np.prod(1 + rets / 100) - 1) * 100)
    med_sharpe = float(folds_df["OOS Sharpe"].median())
    profitable_fold_ratio = float((rets > 0).mean())
    max_dd = float(folds_df["OOS Max DD %"].min())

    summary = {
        "OOS Ret %": round(total_oos, 1),
        "OOS Sharpe (med)": round(med_sharpe, 2),
        "OOS Win %": round(float((pooled["pnl"] > 0).mean() * 100), 1) if len(pooled) else 0.0,
        "OOS Trades": int(len(pooled)),
        "OOS Max DD %": round(max_dd, 1),
        "Profitable Folds %": round(profitable_fold_ratio * 100, 1),
        "Param Sets Used": int(folds_df["Params"].nunique()),
        "Verdict": _verdict(total_oos, med_sharpe, profitable_fold_ratio, len(pooled), max_dd),
    }
    return folds_df, pooled, summary
