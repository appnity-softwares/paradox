"""Bar and tick backtesting endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from dashboard.state import (
    ALL_STRATEGIES,
    BacktestReq,
    PARAM_GRID,
    StrategyConfig,
    TICK_BACKTEST_ENGINE,
    TickBacktestReq,
    compute_metrics,
    equity_to_curve,
    fetch_market_data,
    infer_tf_label,
    run_backtest,
    trades_to_records,
)

router = APIRouter(tags=["backtest"])


@router.post("/api/backtest")
def backtest(req: BacktestReq):
    if req.strategy not in ALL_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {req.strategy}")
    allowed = set(PARAM_GRID.get(req.strategy, {}).keys())
    params = {key: value for key, value in req.params.items() if key in allowed}
    snapshot = fetch_market_data(req.timeframe, 20_000, allow_synthetic=False)
    df = snapshot.data
    tf_label = infer_tf_label(df.index)

    cfg = StrategyConfig(
        risk_pct=req.risk_pct,
        rr=req.rr,
        sl_method=req.sl_method,
        sl_atr_mult=req.sl_atr_mult,
        sl_fixed=req.sl_fixed,
        session=req.session,
        max_trades_per_day=req.max_trades_per_day,
        daily_loss_limit_pct=req.daily_loss_limit_pct,
    )
    try:
        strategy = ALL_STRATEGIES[req.strategy](cfg, **params)
        equity, trades = run_backtest(df, strategy, cfg)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

    return {
        "metrics": compute_metrics(equity, trades, cfg, tf_label),
        "equity": equity_to_curve(equity),
        "trades": trades_to_records(trades),
        "market": snapshot.metadata(),
    }


@router.post("/api/tick_backtest")
def run_tick_backtest(req: TickBacktestReq):
    """Run sub-second tick-by-tick backtesting against compressed vertex dataset."""
    return TICK_BACKTEST_ENGINE.run(
        strategy_key=req.strategy,
        risk_pct=req.risk_pct,
        sl_points=req.sl_points,
        tp_points=req.tp_points,
        max_ticks=req.max_ticks,
    )
