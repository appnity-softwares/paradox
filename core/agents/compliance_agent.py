"""Compliance Agent - Prop firm rules and account restrictions.

CRITICAL: This agent provides deterministic rule checks. It does NOT execute trades.
It only evaluates whether a proposed trade/strategy configuration complies with
prop firm rules and account restrictions. Final order permissions stay with the
deterministic execution engine.
"""
from __future__ import annotations

from typing import Any

from core.orchestrator import AgentCapability, AgentMessage, BaseAgent, MessageType


class ComplianceAgent(BaseAgent):
    def __init__(self, orchestrator):
        super().__init__("compliance_agent", "Compliance Agent", orchestrator)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="check_prop_rules",
                description="Validate strategy/config against prop firm rules",
                input_schema={"account_size": "float", "daily_loss_limit_pct": "float", "max_daily_trades": "int", "max_risk_pct": "float"},
                output_schema={"compliant": "bool", "violations": "list[str]", "warnings": "list[str]"},
            ),
            AgentCapability(
                name="check_trade_permission",
                description="Validate individual trade against account restrictions",
                input_schema={"symbol": "str", "side": "str", "volume": "float", "risk_pct": "float", "account_equity": "float"},
                output_schema={"allowed": "bool", "reason": "str", "adjusted_volume": "float"},
            ),
            AgentCapability(
                name="check_daily_limits",
                description="Check if daily loss limit or trade count limit breached",
                input_schema={"daily_pnl_pct": "float", "daily_trades": "int", "daily_loss_limit_pct": "float", "max_daily_trades": "int"},
                output_schema={"blocked": "bool", "reason": "str"},
            ),
        ]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        payload = message.payload
        action = payload.get("action", "")

        if action == "check_prop_rules":
            result = self._check_prop_rules(payload)
            return await self.reply_to(message, {"success": True, **result})

        if action == "check_trade_permission":
            result = self._check_trade_permission(payload)
            return await self.reply_to(message, {"success": True, **result})

        if action == "check_daily_limits":
            result = self._check_daily_limits(payload)
            return await self.reply_to(message, {"success": True, **result})

        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown action: {action}", "success": False},
            correlation_id=message.id,
        )

    def _check_prop_rules(self, payload: dict) -> dict:
        account_size = float(payload.get("account_size", 10000))
        daily_loss_limit_pct = float(payload.get("daily_loss_limit_pct", 3.0))
        max_daily_trades = int(payload.get("max_daily_trades", 4))
        max_risk_pct = float(payload.get("max_risk_pct", 1.0))

        violations = []
        warnings = []

        # Prop firm rules (typical)
        if daily_loss_limit_pct > 5.0:
            violations.append(f"Daily loss limit {daily_loss_limit_pct}% exceeds typical prop firm max of 5%")
        if max_daily_trades > 10:
            violations.append(f"Max daily trades {max_daily_trades} exceeds typical prop firm max of 10")
        if max_risk_pct > 2.0:
            violations.append(f"Risk per trade {max_risk_pct}% exceeds typical prop firm max of 2%")
        if account_size < 5000:
            warnings.append(f"Account size {account_size} is below typical prop firm evaluation minimum of $5,000")

        compliant = len(violations) == 0

        return {
            "compliant": compliant,
            "violations": violations,
            "warnings": warnings,
            "account_size": account_size,
            "daily_loss_limit_pct": daily_loss_limit_pct,
            "max_daily_trades": max_daily_trades,
            "max_risk_pct": max_risk_pct,
        }

    def _check_trade_permission(self, payload: dict) -> dict:
        symbol = payload.get("symbol", "XAUUSD")
        side = payload.get("side", "BUY").upper()
        volume = float(payload.get("volume", 0.01))
        risk_pct = float(payload.get("risk_pct", 1.0))
        account_equity = float(payload.get("account_equity", 100000))

        allowed = True
        reason = "Approved"
        warnings = []
        adjusted_volume = volume

        # Risk per trade check
        if risk_pct > 2.0:
            allowed = False
            reason = f"Risk per trade {risk_pct}% exceeds 2% prop firm limit"
        elif risk_pct > 1.0:
            # Warning: reduce to 1%
            adjusted_volume = volume * (1.0 / risk_pct)
            reason = f"Risk reduced from {risk_pct}% to 1.0% per prop firm guidelines"
            warnings.append(reason)

        # Volume sanity check
        if volume <= 0:
            allowed = False
            reason = "Volume must be greater than 0"
        elif volume > 100:
            allowed = False
            reason = f"Volume {volume} exceeds max 100 lots per trade"

        # Symbol check
        if symbol not in ("XAUUSD", "EURUSD", "GBPUSD", "XAUUSDm", "GOLD"):
            warnings.append(f"Symbol {symbol} not in standard prop firm instrument list")

        return {
            "allowed": allowed,
            "reason": reason,
            "warnings": warnings,
            "adjusted_volume": round(adjusted_volume, 4),
            "original_volume": volume,
            "risk_pct": risk_pct,
            "symbol": symbol,
            "side": side,
        }

    def _check_daily_limits(self, payload: dict) -> dict:
        daily_pnl_pct = float(payload.get("daily_pnl_pct", 0.0))
        daily_trades = int(payload.get("daily_trades", 0))
        daily_loss_limit_pct = float(payload.get("daily_loss_limit_pct", 3.0))
        max_daily_trades = int(payload.get("max_daily_trades", 4))

        blocked = False
        reason = "Within limits"

        if daily_pnl_pct <= -daily_loss_limit_pct:
            blocked = True
            reason = f"Daily loss limit breached: {daily_pnl_pct}% <= -{daily_loss_limit_pct}%"
        elif daily_trades >= max_daily_trades:
            blocked = True
            reason = f"Max daily trades reached: {daily_trades} >= {max_daily_trades}"
        elif daily_pnl_pct <= -daily_loss_limit_pct * 0.8:
            blocked = False
            reason = f"Warning: approaching daily loss limit ({daily_pnl_pct}%)"

        return {
            "blocked": blocked,
            "reason": reason,
            "daily_pnl_pct": daily_pnl_pct,
            "daily_trades": daily_trades,
            "daily_loss_limit_pct": daily_loss_limit_pct,
            "max_daily_trades": max_daily_trades,
        }
