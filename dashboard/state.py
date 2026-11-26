"""Shared dashboard state: engines, agents, request models and helpers.

Routers import from here; nothing in this module imports a router, so the
dependency direction stays one-way (Master Prompt Part 4).
"""
from __future__ import annotations

import asyncio
import math
import os
from datetime import datetime, timezone
import sys
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
PROTOTYPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROTOTYPE_ROOT)
from ai.builder import run_pipeline  # noqa: E402
from core.backtest import run_backtest  # noqa: E402
from core.data import fetch_market_data  # noqa: E402
from core.data_manager import get_manager  # noqa: E402
from core.execution_engine import ExecutionEngine  # noqa: E402
from core.orchestrator import AgentMessage, AgentOrchestrator, MessageType  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from core.optimization_engine import OptimizationEngine  # noqa: E402
from core.optimize import PARAM_GRID  # noqa: E402
from core.paper_trading import PaperTradingEngine  # noqa: E402
from core.risk_engine import RiskEngine  # noqa: E402
from core.agents import (
    ResearchAgent,
    StrategyAdaptationAgent,
    ComplianceAgent,
    ExecutionAgent as StrategyExecutionAgent,
    MarketIntelligenceAgent,
    StrategyOptimizationAgent,
    PortfolioRiskAgent,
    PerformanceAnalystAgent,
)  # noqa: E402
from run import infer_tf_label  # noqa: E402
from strategies import ALL_STRATEGIES, StrategyConfig  # noqa: E402
from core.prop_firm_simulator import (
    PROP_FIRM_PROFILES,
    PropFirmChallengeSimulator,
    PropFirmProfile,
)  # noqa: E402
from core.vertex_store import VERTEX_STORE  # noqa: E402
from core.tick_backtest import TICK_BACKTEST_ENGINE  # noqa: E402
from core.rag_engine import MARKET_RAG_ENGINE  # noqa: E402
from dataclasses import asdict

class MarketBroadcaster:
    def __init__(self):
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    async def broadcast(self, item: dict):
        for q in list(self._subscribers):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass

MARKET_BROADCASTER = MarketBroadcaster()

Timeframe = Literal["1m", "2m", "5m", "15m", "30m", "1h", "4h", "1d"]
SessionName = Literal["all", "tokyo", "london", "ny", "london_ny"]
StopMethod = Literal["atr", "fixed"]
async def _risk_engine_handler(message: AgentMessage) -> AgentMessage | None:
    """Handle risk check requests from agents."""
    payload = message.payload
    action = payload.get("action", "")

    if action == "check_position_risk":
        result = RISK_ENGINE.check_position_risk(
            symbol=payload.get("symbol", "XAUUSD"),
            side=payload.get("side", "BUY"),
            volume=float(payload.get("volume", 0.01)),
            risk_pct=float(payload.get("risk_pct", 1.0)),
            account_equity=float(payload.get("account_equity", 10000)),
            sl_distance=float(payload.get("sl_distance", 5.0)),
            tp_distance=float(payload.get("tp_distance", 10.0)) if payload.get("tp_distance") else None,
        )
        return AgentMessage(
            sender="risk_engine",
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={
                "allowed": result.allowed,
                "reason": result.reason,
                "adjusted_volume": result.adjusted_volume,
                "risk_amount_usd": result.risk_amount_usd,
                "risk_pct": result.risk_pct,
                "sl_distance": result.sl_distance,
                "tp_distance": result.tp_distance,
                "notional_usd": result.notional_usd,
            },
            correlation_id=message.id,
        )

    if action == "get_status":
        return AgentMessage(
            sender="risk_engine",
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload=RISK_ENGINE.get_status(),
            correlation_id=message.id,
        )

    return AgentMessage(
        sender="risk_engine",
        receiver=message.sender,
        msg_type=MessageType.RESPONSE,
        payload={"error": f"Unknown risk action: {action}"},
        correlation_id=message.id,
    )
async def _execution_engine_handler(message: AgentMessage) -> AgentMessage | None:
    """Handle execution requests from agents.

    U0-hardening: the agent bus is NOT an order path. Orders are only accepted
    through POST /api/platform/orders (the platform order router: risk + compliance +
    mode gate + ApprovedOrder). Mutating actions received here are refused
    with an explicit pointer — they are never forwarded to any engine.
    """
    payload = message.payload
    action = payload.get("action", "")

    if action in ("place_market_order", "close_position", "update_sl_tp",
                  "emergency_kill_switch", "close_all_positions"):
        return AgentMessage(
            sender="execution_engine",
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={
                "success": False,
                "order_id": None,
                "error": ("order actions are not accepted over the agent bus; "
                          "use POST /api/platform/orders (the platform order router)"),
                "refused_action": action,
            },
            correlation_id=message.id,
        )

    if action == "get_positions":
        positions = EXECUTION_ENGINE.get_positions()
        return AgentMessage(
            sender="execution_engine",
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"positions": positions},
            correlation_id=message.id,
        )

    return AgentMessage(
        sender="execution_engine",
        receiver=message.sender,
        msg_type=MessageType.RESPONSE,
        payload={"error": f"Unknown execution action: {action}"},
        correlation_id=message.id,
    )

async def _optimization_engine_handler(message: AgentMessage) -> AgentMessage | None:
    """Handle optimization requests from agents."""
    payload = message.payload
    action = payload.get("action", "")

    if action == "run_optimization":
        strategy_name = payload.get("strategy_name", "ema_cross")
        strategy_cls = ALL_STRATEGIES.get(strategy_name)
        if strategy_cls is None:
            return AgentMessage(
                sender="optimization_engine",
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                payload={"success": False, "reason": f"Unknown strategy: {strategy_name}"},
                correlation_id=message.id,
            )

        cfg = StrategyConfig(
            risk_pct=float(payload.get("risk_pct", 1.0)),
            rr=float(payload.get("rr", 2.0)),
            sl_method=payload.get("sl_method", "atr"),
            sl_atr_mult=float(payload.get("sl_atr_mult", 1.5)),
            session=payload.get("session", "all"),
            max_trades_per_day=int(payload.get("max_trades_per_day", 4)),
            daily_loss_limit_pct=float(payload.get("daily_loss_limit_pct", 3.0)),
        )

        snapshot = fetch_market_data(
            payload.get("timeframe", "15m"),
            bars=10000,
            allow_stale=True,
            allow_synthetic=False,
        )

        result = OPTIMIZATION_ENGINE.run_optimization(
            df=snapshot.data,
            strategy_cls=strategy_cls,
            strategy_name=strategy_name,
            symbol=payload.get("symbol", "XAUUSD"),
            timeframe=payload.get("timeframe", "15m"),
            method=payload.get("method", "walk_forward"),
            param_ranges=payload.get("param_ranges"),
            cfg=cfg,
        )

        return AgentMessage(
            sender="optimization_engine",
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload=result,
            correlation_id=message.id,
        )

    return AgentMessage(
        sender="optimization_engine",
        receiver=message.sender,
        msg_type=MessageType.RESPONSE,
        payload={"error": f"Unknown optimization action: {action}"},
        correlation_id=message.id,
    )
DATA_MANAGER = get_manager()
RISK_ENGINE = RiskEngine(max_risk_pct=2.0, max_daily_loss_pct=5.0, max_leverage=10.0)
EXECUTION_ENGINE = ExecutionEngine(broker_connector=None)  # simulation/paper mode
OPTIMIZATION_ENGINE = OptimizationEngine()
PAPER_ENGINE = PaperTradingEngine()
PAPER_TASK: dict = {"task": None, "last_processed": None}
if DATA_MANAGER.has_execution() and DATA_MANAGER.active is not None:
    EXECUTION_ENGINE.set_broker(DATA_MANAGER.active)
ORCHESTRATOR = AgentOrchestrator()
ORCHESTRATOR.register_handler("risk_engine", _risk_engine_handler)
ORCHESTRATOR.register_handler("execution_engine", _execution_engine_handler)
ORCHESTRATOR.register_handler("optimization_engine", _optimization_engine_handler)
RESEARCH_AGENT = ResearchAgent(ORCHESTRATOR)
STRATEGY_ADAPTATION_AGENT = StrategyAdaptationAgent(ORCHESTRATOR)
COMPLIANCE_AGENT = ComplianceAgent(ORCHESTRATOR)
EXECUTION_AGENT = StrategyExecutionAgent(ORCHESTRATOR)
MARKET_INTELLIGENCE_AGENT = MarketIntelligenceAgent(ORCHESTRATOR)
STRATEGY_OPTIMIZATION_AGENT = StrategyOptimizationAgent(ORCHESTRATOR)
PORTFOLIO_RISK_AGENT = PortfolioRiskAgent(ORCHESTRATOR)
PERFORMANCE_ANALYST_AGENT = PerformanceAnalystAgent(ORCHESTRATOR)
ORCHESTRATOR.register(RESEARCH_AGENT)
ORCHESTRATOR.register(STRATEGY_ADAPTATION_AGENT)
ORCHESTRATOR.register(COMPLIANCE_AGENT)
ORCHESTRATOR.register(EXECUTION_AGENT)
ORCHESTRATOR.register(MARKET_INTELLIGENCE_AGENT)
ORCHESTRATOR.register(STRATEGY_OPTIMIZATION_AGENT)
ORCHESTRATOR.register(PORTFOLIO_RISK_AGENT)
ORCHESTRATOR.register(PERFORMANCE_ANALYST_AGENT)
class MT5ConfigPayload(BaseModel):
    login: str | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None
class MT5LiveTickPayload(BaseModel):
    symbol: str = "XAUUSD"
    bid: float
    ask: float
    last: float | None = None
    volume: float | int = 1
    time_ms: int | None = None
class TradeOrderReq(BaseModel):
    symbol: str = "XAUUSD"
    side: str = "BUY"
    volume: float = 0.01
    sl: float = 0.0
    tp: float = 0.0
    mode: str = "mt5"  # "mt5" or "paper"
    comment: str = "AI-Quant"
class TradeCloseReq(BaseModel):
    ticket: int
    reason: str = "Manual close"
class TradeModifyReq(BaseModel):
    ticket: int
    sl: float = 0.0
    tp: float = 0.0
    break_even: bool = False
class BacktestReq(BaseModel):
    strategy: str
    params: dict[str, float | int] = Field(default_factory=dict)
    timeframe: Timeframe = "15m"
    risk_pct: float = Field(1.0, ge=0.1, le=3.0)
    rr: float = Field(2.0, ge=0.25, le=10.0)
    sl_method: StopMethod = "atr"
    sl_atr_mult: float = Field(1.5, ge=0.1, le=10.0)
    sl_fixed: float = Field(5.0, gt=0, le=1000.0)
    session: SessionName = "all"
    max_trades_per_day: int = Field(4, ge=1, le=50)
    daily_loss_limit_pct: float = Field(3.0, ge=0.1, le=20.0)
class BuildReq(BaseModel):
    text: str = Field(min_length=3, max_length=2000)
    timeframe: Timeframe = "15m"
    folds: int = Field(4, ge=2, le=10)
class AgentMessageReq(BaseModel):
    sender: str
    receiver: str
    action: str
    payload: dict = Field(default_factory=dict)
    priority: int = Field(0, ge=0, le=2)
class PaperReplayReq(BaseModel):
    strategy: str
    params: dict[str, float | int] = Field(default_factory=dict)
    timeframe: Timeframe = "15m"
    risk_pct: float = 1.0
    rr: float = 2.0
    sl_method: StopMethod = "atr"
    sl_atr_mult: float = 1.5
    sl_fixed: float = 5.0
    session: SessionName = "all"
    max_trades_per_day: int = 4
    daily_loss_limit_pct: float = 3.0
import logging  # noqa: E402
logger = logging.getLogger("dashboard")
async def _paper_feed_loop():
    """Poll latest bars and feed them to the paper trading engine."""
    interval_seconds = 30
    while True:
        try:
            snapshot = fetch_market_data("15m", 100, allow_synthetic=False)
            df = snapshot.data
            last = PAPER_TASK.get("last_processed")
            if len(df):
                if last is None:
                    new_df = df
                else:
                    new_df = df[df.index > pd.Timestamp(last)]
                for _, row in new_df.iterrows():
                    await asyncio.to_thread(
                        PAPER_ENGINE.process_bar,
                        {
                            "time": row.name,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row.get("volume") or 0.0),
                        },
                    )
                PAPER_TASK["last_processed"] = df.index[-1]
        except Exception as exc:
            logger.error("paper feed error: %s", exc)
        await asyncio.sleep(interval_seconds)
def get_snapshot(
    timeframe: str,
    bars: int,
    *,
    refresh: bool = False,
    allow_synthetic: bool = False,
):
    try:
        return fetch_market_data(
            timeframe,
            bars,
            force_refresh=refresh,
            allow_stale=True,
            allow_synthetic=allow_synthetic,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
def equity_to_curve(equity: pd.Series, points: int = 800):
    step = max(1, len(equity) // points)
    sub = equity.iloc[::step]
    return [[t.strftime("%m-%d %H:%M"), round(float(v), 2)] for t, v in sub.items()]
def trades_to_records(trades: pd.DataFrame):
    if not len(trades):
        return []
    data = trades.copy()
    data["entry_time"] = data["entry_time"].astype(str)
    data["exit_time"] = data["exit_time"].astype(str)
    return data.to_dict("records")
from core.brokers.mt5_mac import get_mac_mt5_status, sync_from_exported_csv
class SyncMacReq(BaseModel):
    csv_filename: str | None = None
    timeframe: str = "15m"
class AIChatReq(BaseModel):
    message: str = Field(min_length=1)
    context: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)
class SingleAgentExecReq(BaseModel):
    agent_id: str
    symbol: str = "XAUUSD"
    timeframe: str = "15m"
class PropFirmSimulateReq(BaseModel):
    profile_id: str = "ftmo_100k"
    strategy: str = "smc_fvg"
    timeframe: Timeframe = "15m"
    params: dict = Field(default_factory=dict)
    risk_pct: float = Field(1.0, ge=0.1, le=3.0)
    rr: float = Field(2.0, ge=0.5, le=10.0)
    sl_method: StopMethod = "atr"
    sl_atr_mult: float = Field(1.5, ge=0.5, le=5.0)
    sl_fixed: float = Field(5.0, ge=0.5, le=50.0)
    session: SessionName = "all"
    max_trades_per_day: int = Field(4, ge=1, le=50)
    daily_loss_limit_pct: float = Field(3.0, ge=0.1, le=20.0)
    num_simulations: int = Field(1000, ge=100, le=5000)
    custom_profile: dict | None = None
class TickBacktestReq(BaseModel):
    strategy: str = "smc_fvg"
    risk_pct: float = Field(1.0, ge=0.1, le=5.0)
    sl_points: float = Field(3.0, ge=0.5, le=50.0)
    tp_points: float = Field(6.0, ge=0.5, le=100.0)
    max_ticks: int = Field(50000, ge=1000, le=200000)
