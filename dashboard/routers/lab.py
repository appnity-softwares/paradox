"""Strategy Lab + entry-timer endpoints.

The Strategy Lab is the Strategy Optimization Agent's workbench: it
searches parameter variants on real data, validates them out-of-sample,
shows progress live while it runs, and stores every finished run so
multiple searches accumulate.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import pandas as pd

from dashboard.state import (
    ALL_STRATEGIES,
    PARAM_GRID,
    PAPER_ENGINE,
    STRATEGY_LAB,
    StrategyConfig,
    fetch_market_data,
)

router = APIRouter(tags=["lab"])


@router.post("/api/lab/start")
def lab_start(req: dict):
    strategy = req.get("strategy")
    if not strategy:
        raise HTTPException(422, "strategy is required")
    try:
        run_id = STRATEGY_LAB.start(
            strategy=strategy,
            timeframe=req.get("timeframe", "15m"),
            bars=int(req.get("bars", 8000)),
            risk_pct=float(req.get("risk_pct", 1.0)),
            rr=float(req.get("rr", 2.0)),
            sl_method=req.get("sl_method", "atr"),
            sl_atr_mult=float(req.get("sl_atr_mult", 1.5)),
            sl_fixed=float(req.get("sl_fixed", 5.0)),
            session=req.get("session", "all"),
            max_variants=req.get("max_variants"),
            oos_fraction=float(req.get("oos_fraction", 0.3)),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"started": True, "run_id": run_id}


@router.get("/api/lab/status")
def lab_status():
    """Live progress of the running/last search — poll this."""
    return STRATEGY_LAB.status()


@router.get("/api/lab/runs")
def lab_runs():
    return {"runs": STRATEGY_LAB.list_runs()}


@router.get("/api/lab/run/{run_id}")
def lab_run(run_id: str):
    try:
        return STRATEGY_LAB.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/lab/promote/{run_id}/{variant_index}")
def lab_promote(run_id: str, variant_index: int):
    """Load a stored variant into the live paper engine (paper only — the
    lab never touches a live order path)."""
    try:
        run = STRATEGY_LAB.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    results = run.get("results", [])
    if variant_index < 0 or variant_index >= len(results):
        raise HTTPException(404, f"variant index {variant_index} not in run")
    variant = results[variant_index]
    strategy = run.get("strategy")
    if strategy not in ALL_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {strategy}")
    allowed = set(PARAM_GRID.get(strategy, {}).keys())
    params = {k: v for k, v in variant["params"].items() if k in allowed}
    cfg = StrategyConfig(
        name=f"lab_{run_id}", risk_pct=1.0, rr=2.0,
        sl_method="atr", sl_atr_mult=1.5, sl_fixed=5.0, session="all",
    )
    strategy_obj = ALL_STRATEGIES[strategy](cfg, **params)
    PAPER_ENGINE.start(strategy_obj, cfg, "XAUUSD")
    return {
        "promoted": True,
        "run_id": run_id,
        "variant_index": variant_index,
        "strategy": strategy,
        "params": params,
        "status": variant.get("status"),
        "note": "paper engine reconfigured — simulation only",
    }


@router.get("/api/entry/check")
def entry_check(strategy: str = "ema_cross", timeframe: str = "15m", bars: int = 300):
    """Entry timer: what is the strategy saying RIGHT NOW on live data?

    Returns the last CLOSED bar's signal and the distance from the current
    price to the level that would flip it — the same discipline the
    day-trader entry skill uses: only closed bars trigger, never wicks.
    """
    if strategy not in ALL_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {strategy}")
    snapshot = fetch_market_data(timeframe, bars, allow_synthetic=False)
    df = snapshot.data
    if len(df) < 50:
        raise HTTPException(503, "not enough live data yet")
    cfg = StrategyConfig(name=f"entry_{strategy}")
    strat = ALL_STRATEGIES[strategy](cfg)
    signals = strat.generate_signals(df)
    sig = float(signals.fillna(0.0).iloc[-1])
    last_bar = df.index[-1]
    price = float(df["close"].iloc[-1])
    if sig > 0:
        state, side = "TRIGGERED", "LONG"
    elif sig < 0:
        state, side = "TRIGGERED", "SHORT"
    else:
        state, side = "WAITING", None
    return {
        "state": state,
        "side": side,
        "signal": sig,
        "price": price,
        "last_closed_bar": str(last_bar),
        "strategy": strategy,
        "timeframe": timeframe,
        "note": "signals act on closed bars only; a wick through a level is not a signal",
    }


@router.get("/api/entry/history")
def entry_history(strategy: str = "ema_cross", timeframe: str = "15m", bars: int = 2000):
    """Recent signal flips with timestamps — what the entry timer said and when."""
    if strategy not in ALL_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {strategy}")
    snapshot = fetch_market_data(timeframe, bars, allow_synthetic=False)
    df = snapshot.data
    cfg = StrategyConfig(name=f"entry_{strategy}")
    strat = ALL_STRATEGIES[strategy](cfg)
    signals = strat.generate_signals(df).fillna(0.0)
    flips = []
    prev = 0.0
    for ts, s in signals.items():
        s = float(s)
        if s != prev:
            flips.append({
                "time": str(ts),
                "state": "LONG" if s > 0 else ("SHORT" if s < 0 else "FLAT"),
                "price": float(df["close"].loc[ts]) if ts in df.index else None,
            })
            prev = s
        if len(flips) >= 50:
            break
    return {"strategy": strategy, "timeframe": timeframe, "flips": flips[::-1]}
