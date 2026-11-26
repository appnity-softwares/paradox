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
from pydantic import BaseModel

from dashboard.state import (
    ALL_STRATEGIES,
    DATA_MANAGER,
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


# ------------------------------------------------------------------ paper desk
# Manual paper orders for the terminal. Paper only: fills are simulated from
# the provider's real bid/ask, the spread is paid on entry AND exit, and PnL
# is booked at the real contract size (XAUUSD 1 lot = 100 oz). No broker is
# ever touched from these routes.


class PaperOrderReq(BaseModel):
    symbol: str = "XAUUSD"
    side: str
    volume: float = 0.01
    sl: float = 0.0   # distance from entry, in price units
    tp: float = 0.0   # distance from entry, in price units


class PaperCloseReq(BaseModel):
    ticket: int


def _paper_quote(symbol: str) -> tuple[float, float]:
    """Real bid/ask from the active provider. No synthetic fallback: without
    a real quote the desk refuses to invent a fill price."""
    if DATA_MANAGER.active is None:
        raise HTTPException(503, "no market data provider active")
    try:
        tick = DATA_MANAGER.active.fetch_tick(symbol)
    except Exception as e:
        raise HTTPException(503, f"no live quote for {symbol}: {e}")
    if not tick:
        raise HTTPException(503, f"no live quote for {symbol}")
    bid, ask = float(tick["bid"]), float(tick["ask"])
    if bid <= 0 or ask <= 0:
        raise HTTPException(503, f"no live quote for {symbol}")
    return bid, ask


@router.get("/api/paper/positions")
def paper_positions():
    """Open paper positions in the terminal's table shape, with floating PnL
    marked at the last seen price crossed with the exit spread."""
    rows = []
    for p in PAPER_ENGINE.get_positions():
        exit_ref = PAPER_ENGINE._market_exit_price("LONG" if p["side"] == "LONG" else "SHORT")
        if exit_ref is None:
            exit_ref = p["entry_price"]
        direction = 1 if p["side"] == "LONG" else -1
        profit = (exit_ref - p["entry_price"]) * direction * p["volume"] * PAPER_ENGINE.contract_size
        rows.append({
            "ticket": p["ticket"],
            "side": "BUY" if p["side"] == "LONG" else "SELL",
            "volume": p["volume"],
            "open_price": p["entry_price"],
            "profit": round(profit, 2),
            "sl": p["sl"],
            "tp": p["tp"],
            "strategy_name": p["strategy_name"],
            "opened_at": p["opened_at"],
        })
    return {"positions": rows}


@router.get("/api/paper/account")
def paper_account():
    status = PAPER_ENGINE.get_status()
    floating = 0.0
    for p in PAPER_ENGINE.get_positions():
        exit_ref = PAPER_ENGINE._market_exit_price("LONG" if p["side"] == "LONG" else "SHORT")
        if exit_ref is not None:
            direction = 1 if p["side"] == "LONG" else -1
            floating += (exit_ref - p["entry_price"]) * direction * p["volume"] * PAPER_ENGINE.contract_size
    return {
        "balance": round(status["equity"], 2),
        "equity": round(status["equity"] + floating, 2),
        "floating_pnl": round(floating, 2),
        "realized_pnl": status["realized_pnl"],
        "open_positions": status["positions"],
        "kill_switch_active": status["kill_switch_active"],
        "engine": "paper",
    }


@router.post("/api/paper/order")
def paper_order(req: PaperOrderReq):
    bid, ask = _paper_quote(req.symbol)
    side = req.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(422, "side must be BUY or SELL")
    entry = ask if side == "BUY" else bid          # taker pays the spread
    ref = (bid + ask) / 2
    if req.sl <= 0 or req.tp <= 0:
        raise HTTPException(422, "SL and TP distances are required")
    if side == "BUY":
        sl_price, tp_price = entry - abs(req.sl), entry + abs(req.tp)
    else:
        sl_price, tp_price = entry + abs(req.sl), entry - abs(req.tp)
    result = PAPER_ENGINE.manual_order(req.symbol, side, req.volume, entry, sl_price, tp_price)
    if not result.get("ok"):
        raise HTTPException(422, result.get("error", "order refused"))
    result["bid"], result["ask"], result["mid"] = bid, ask, ref
    return result


@router.post("/api/paper/close")
def paper_close(req: PaperCloseReq):
    pos = next((p for p in PAPER_ENGINE.get_positions() if p["ticket"] == req.ticket), None)
    if pos is None:
        raise HTTPException(404, f"unknown ticket {req.ticket}")
    bid, ask = _paper_quote(pos["symbol"])
    exit_price = bid if pos["side"] == "LONG" else ask   # LONG sells at bid
    result = PAPER_ENGINE.manual_close(req.ticket, exit_price)
    if not result.get("ok"):
        raise HTTPException(422, result.get("error", "close refused"))
    trade = result["trade"]
    return {"ok": True, "ticket": req.ticket, "filled_price": trade.exit_price,
            "pnl": round(trade.pnl, 2)}


@router.post("/api/paper/close_all")
def paper_close_all():
    closed, total_pnl = 0, 0.0
    for p in list(PAPER_ENGINE.get_positions()):
        try:
            bid, ask = _paper_quote(p["symbol"])
        except HTTPException:
            continue
        exit_price = bid if p["side"] == "LONG" else ask
        result = PAPER_ENGINE.manual_close(p["ticket"], exit_price)
        if result.get("ok"):
            closed += 1
            total_pnl += result["trade"].pnl
    return {"ok": True, "closed": closed, "total_pnl": round(total_pnl, 2)}
