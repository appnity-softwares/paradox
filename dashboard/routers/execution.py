"""Trading desk and paper trading.

U0 (Update Blueprint v2.1): the mutating desk routes
(/api/trade/order|close|close_all|reset_kill_switch|modify) used to call
the legacy ExecutionEngine directly, bypassing ExecutionRouter and
carrying no auth dependency. They are removed: every order must go
through the single safe path (POST /api/platform/orders -> ExecutionRouter,
risk + compliance + mode gate + ApprovedOrder), and kill-switch actions
live at POST /api/platform/kill-switch with role enforcement. The paths
answer 410 Gone so legacy UI callers fail loudly instead of silently
reaching a broker. Read-only desk routes and the paper engine (simulated
fills only, no broker) remain.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from dashboard.state import (
    ALL_STRATEGIES,
    EXECUTION_ENGINE,
    PAPER_ENGINE,
    PAPER_TASK,
    PARAM_GRID,
    PaperReplayReq,
    StrategyConfig,
    _paper_feed_loop,
    asyncio,
    fetch_market_data,
)

router = APIRouter(tags=["execution"])

_REMOVED = {
    "detail": "removed_in_U0",
    "message": ("Order actions must go through POST /api/platform/orders "
                "(ExecutionRouter: risk, compliance, mode gate). Kill-switch "
                "actions live at POST /api/platform/kill-switch."),
}


@router.post("/api/trade/order")
def execute_trade_order():
    """Removed in U0: bypassed the execution router and had no auth."""
    raise HTTPException(status_code=410, detail=_REMOVED)


@router.post("/api/trade/close")
def close_trade_order():
    """Removed in U0: bypassed the execution router and had no auth."""
    raise HTTPException(status_code=410, detail=_REMOVED)


@router.post("/api/trade/close_all")
def close_all_trades():
    """Removed in U0: use POST /api/platform/kill-switch (admin)."""
    raise HTTPException(status_code=410, detail=_REMOVED)


@router.post("/api/trade/reset_kill_switch")
def reset_kill_switch():
    """Removed in U0: use POST /api/platform/kill-switch (admin only)."""
    raise HTTPException(status_code=410, detail=_REMOVED)


@router.post("/api/trade/modify")
def modify_trade():
    """Removed in U0: bypassed the execution router and had no auth."""
    raise HTTPException(status_code=410, detail=_REMOVED)


@router.get("/api/trade/positions")
def get_trade_positions():
    """Get all live positions with calculated floating PnL."""
    return {
        "positions": EXECUTION_ENGINE.get_positions(),
        "account": EXECUTION_ENGINE.get_account_summary(),
    }


@router.get("/api/trade/account")
def get_trade_account():
    """Get account balance, equity, margin, and status."""
    return EXECUTION_ENGINE.get_account_summary()


@router.get("/api/paper/status")
def paper_status():
    """Live paper-trading engine status (simulation execution on real data)."""
    return PAPER_ENGINE.get_status()


@router.post("/api/paper/replay")
def paper_replay(req: PaperReplayReq):
    """Replay a strategy over historical market data through the paper engine."""
    if req.strategy not in ALL_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {req.strategy}")
    snapshot = fetch_market_data(req.timeframe, 20_000, allow_synthetic=False)
    allowed = set(PARAM_GRID.get(req.strategy, {}).keys())
    params = {k: v for k, v in req.params.items() if k in allowed}
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
    strategy = ALL_STRATEGIES[req.strategy](cfg, **params)
    engine = PAPER_ENGINE
    engine.start(strategy, cfg, "XAUUSD")
    result = engine.replay(snapshot.data)
    return {
        "result": result,
        "market": snapshot.metadata(),
    }


@router.post("/api/paper/start")
async def paper_start(req: PaperReplayReq):
    """Start live paper feed for a strategy through the paper engine."""
    if req.strategy not in ALL_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {req.strategy}")
    allowed = set(PARAM_GRID.get(req.strategy, {}).keys())
    params = {k: v for k, v in req.params.items() if k in allowed}
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
    strategy = ALL_STRATEGIES[req.strategy](cfg, **params)
    PAPER_ENGINE.start(strategy, cfg, "XAUUSD")
    task = asyncio.create_task(_paper_feed_loop())
    PAPER_TASK["task"] = task
    return {"started": True, "strategy": req.strategy, "timeframe": req.timeframe, "task_id": task.get_name()}


@router.post("/api/paper/stop")
async def paper_stop():
    task = PAPER_TASK.get("task")
    if task is not None and not task.done():
        task.cancel()
    PAPER_TASK["task"] = None
    return {"stopped": True}
