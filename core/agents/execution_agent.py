"""Execution Agent - Trade approval, execution and management.

Responsibilities:
- Receive trade proposals from other agents/strategies
- Route through Compliance Agent for permission checks
- Execute via deterministic execution engine (MT5/Broker)
- Manage open positions (SL/TP updates, partial closes, emergency close)
- NEVER makes risk decisions - only executes approved orders
"""
from __future__ import annotations

from typing import Any

from core.orchestrator import AgentCapability, AgentMessage, BaseAgent, MessageType


class ExecutionAgent(BaseAgent):
    def __init__(self, orchestrator):
        super().__init__("execution_agent", "Execution Agent", orchestrator)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="place_order",
                description="Place a trade order after compliance check",
                input_schema={"symbol": "str", "side": "str", "volume": "float", "sl": "float", "tp": "float", "strategy_name": "str"},
                output_schema={"order_id": "str", "status": "str", "filled_price": "float", "compliance_passed": "bool"},
            ),
            AgentCapability(
                name="close_position",
                description="Close an open position",
                input_schema={"position_id": "str", "reason": "str"},
                output_schema={"closed": "bool", "realized_pnl": "float"},
            ),
            AgentCapability(
                name="update_sl_tp",
                description="Update stop loss / take profit on open position",
                input_schema={"position_id": "str", "sl": "float", "tp": "float"},
                output_schema={"updated": "bool"},
            ),
            AgentCapability(
                name="get_positions",
                description="Get all open positions",
                input_schema={},
                output_schema={"positions": "list[dict]"},
            ),
            AgentCapability(
                name="emergency_kill_switch",
                description="Close all positions and stop new orders",
                input_schema={},
                output_schema={"killed": "bool", "positions_closed": "int"},
            ),
        ]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        payload = message.payload
        action = payload.get("action", "")

        if action == "place_order":
            return await self._handle_place_order(message, payload)

        if action == "close_position":
            return await self._handle_close_position(message, payload)

        if action == "update_sl_tp":
            return await self._handle_update_sl_tp(message, payload)

        if action == "get_positions":
            return await self._handle_get_positions(message)

        if action == "emergency_kill_switch":
            return await self._handle_kill_switch(message)

        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown action: {action}", "success": False},
            correlation_id=message.id,
        )

    async def _handle_place_order(self, message: AgentMessage, payload: dict) -> AgentMessage | None:
        # Step 1: Send to Compliance Agent for permission
        compliance_msg = await self.orchestrator.route(
            AgentMessage(
                sender=self.agent_id,
                receiver="compliance_agent",
                msg_type=MessageType.REQUEST,
                payload={
                    "action": "check_trade_permission",
                    "symbol": payload.get("symbol", "XAUUSD"),
                    "side": payload.get("side", "BUY"),
                    "volume": payload.get("volume", 0.01),
                    "risk_pct": payload.get("risk_pct", 1.0),
                    "account_equity": payload.get("account_equity", 10000),
                },
                priority=1,
            )
        )

        if not compliance_msg or not compliance_msg.payload.get("allowed", False):
            return await self.reply_to(
                message,
                {
                    "success": False,
                    "order_id": None,
                    "compliance_passed": False,
                    "reason": (compliance_msg.payload.get("reason", "Compliance check failed") if compliance_msg else "No compliance response"),
                    "adjusted_volume": (compliance_msg.payload.get("adjusted_volume", 0) if compliance_msg else 0),
                },
            )

        # Step 2: Execute via deterministic engine (registered as "execution_engine" in orchestrator)
        exec_msg = await self.orchestrator.route(
            AgentMessage(
                sender=self.agent_id,
                receiver="execution_engine",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "place_market_order",
                    "symbol": payload.get("symbol", "XAUUSD"),
                    "side": payload.get("side", "BUY"),
                    "volume": compliance_msg.payload.get("adjusted_volume", payload.get("volume", 0.01)),
                    "sl": payload.get("sl", 0.0),
                    "tp": payload.get("tp", 0.0),
                    "strategy_name": payload.get("strategy_name", "unknown"),
                    "comment": f"AI-{payload.get('strategy_name', 'strategy')}",
                },
                priority=2,
            )
        )

        if exec_msg and exec_msg.payload.get("success"):
            return await self.reply_to(
                message,
                {
                    "success": True,
                    "order_id": exec_msg.payload.get("order_id"),
                    "compliance_passed": True,
                    "filled_price": exec_msg.payload.get("filled_price"),
                    "volume": compliance_msg.payload.get("adjusted_volume", payload.get("volume", 0.01)),
                    "reason": "Order executed",
                },
            )

        return await self.reply_to(
            message,
            {
                "success": False,
                "order_id": None,
                "compliance_passed": True,
                "reason": (exec_msg.payload.get("error", "Execution engine failed") if exec_msg else "No execution engine"),
            },
        )

    async def _handle_close_position(self, message: AgentMessage, payload: dict) -> AgentMessage | None:
        position_id = payload.get("position_id", "")
        reason = payload.get("reason", "Manual close")

        exec_msg = await self.orchestrator.route(
            AgentMessage(
                sender=self.agent_id,
                receiver="execution_engine",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "close_position",
                    "position_id": position_id,
                    "reason": reason,
                },
                priority=2,
            )
        )

        if exec_msg and exec_msg.payload.get("success"):
            return await self.reply_to(
                message,
                {
                    "success": True,
                    "position_id": position_id,
                    "realized_pnl": exec_msg.payload.get("realized_pnl", 0.0),
                    "reason": reason,
                },
            )

        return await self.reply_to(
            message,
            {
                "success": False,
                "position_id": position_id,
                "reason": exec_msg.payload.get("error", "Close failed") if exec_msg else "No execution engine",
            },
        )

    async def _handle_update_sl_tp(self, message: AgentMessage, payload: dict) -> AgentMessage | None:
        position_id = payload.get("position_id", "")
        sl = float(payload.get("sl", 0.0))
        tp = float(payload.get("tp", 0.0))

        exec_msg = await self.orchestrator.route(
            AgentMessage(
                sender=self.agent_id,
                receiver="execution_engine",
                msg_type=MessageType.COMMAND,
                payload={
                    "action": "update_sl_tp",
                    "position_id": position_id,
                    "sl": sl,
                    "tp": tp,
                },
                priority=1,
            )
        )

        if exec_msg and exec_msg.payload.get("success"):
            return await self.reply_to(
                message,
                {
                    "success": True,
                    "position_id": position_id,
                    "sl": sl,
                    "tp": tp,
                },
            )

        return await self.reply_to(
            message,
            {
                "success": False,
                "position_id": position_id,
                "reason": exec_msg.payload.get("error", "Update failed") if exec_msg else "No execution engine",
            },
        )

    async def _handle_get_positions(self, message: AgentMessage) -> AgentMessage | None:
        exec_msg = await self.orchestrator.route(
            AgentMessage(
                sender=self.agent_id,
                receiver="execution_engine",
                msg_type=MessageType.REQUEST,
                payload={"action": "get_positions"},
                priority=0,
            )
        )

        positions = exec_msg.payload.get("positions", []) if exec_msg else []
        return await self.reply_to(
            message,
            {
                "success": True,
                "positions": positions,
            },
        )

    async def _handle_kill_switch(self, message: AgentMessage) -> AgentMessage | None:
        exec_msg = await self.orchestrator.route(
            AgentMessage(
                sender=self.agent_id,
                receiver="execution_engine",
                msg_type=MessageType.COMMAND,
                payload={"action": "emergency_kill_switch"},
                priority=2,
            )
        )

        if exec_msg and exec_msg.payload.get("success"):
            return await self.reply_to(
                message,
                {
                    "success": True,
                    "killed": exec_msg.payload.get("killed", True),
                    "positions_closed": exec_msg.payload.get("positions_closed", 0),
                    "reason": "Emergency kill switch activated",
                },
            )

        return await self.reply_to(
            message,
            {
                "success": False,
                "positions_closed": 0,
                "reason": exec_msg.payload.get("error", "Kill switch failed") if exec_msg else "No execution engine",
            },
        )
