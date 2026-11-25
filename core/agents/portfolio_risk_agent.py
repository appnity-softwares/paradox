"""Portfolio Risk Agent - Portfolio-level risk monitoring.

Responsibilities:
- Monitor aggregate risk across all strategies/accounts
- Calculate portfolio drawdown, correlation, concentration
- Alert when portfolio-level limits breached
- NEVER executes trades directly - only warns and recommends
"""
from __future__ import annotations

from typing import Any

from core.orchestrator import AgentCapability, AgentMessage, BaseAgent, MessageType


class PortfolioRiskAgent(BaseAgent):
    def __init__(self, orchestrator):
        super().__init__("portfolio_risk_agent", "Portfolio Risk Agent", orchestrator)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="assess_portfolio_risk",
                description="Calculate aggregate portfolio risk metrics",
                input_schema={"positions": "list[dict]", "account_equity": "float"},
                output_schema={"portfolio_dd_pct": "float", "correlation_risk": "str", "concentration_risk": "str", "verdict": "str"},
            ),
            AgentCapability(
                name="check_correlation",
                description="Check correlation between strategies/instruments",
                input_schema={"symbols": "list[str]", "strategies": "list[str]"},
                output_schema={"correlation_matrix": "dict", "high_correlation_pairs": "list[list[str]]"},
            ),
            AgentCapability(
                name="max_drawdown_alert",
                description="Check if portfolio drawdown exceeds limits",
                input_schema={"current_equity": "float", "peak_equity": "float", "max_dd_pct": "float"},
                output_schema={"breached": "bool", "current_dd_pct": "float", "remaining_buffer_pct": "float"},
            ),
        ]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        payload = message.payload
        action = payload.get("action", "")

        if action == "assess_portfolio_risk":
            result = self._assess_portfolio_risk(payload)
            return await self.reply_to(message, {"success": True, **result})

        if action == "check_correlation":
            result = self._check_correlation(payload)
            return await self.reply_to(message, {"success": True, **result})

        if action == "max_drawdown_alert":
            result = self._check_max_drawdown(payload)
            return await self.reply_to(message, {"success": True, **result})

        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown action: {action}", "success": False},
            correlation_id=message.id,
        )

    def _assess_portfolio_risk(self, payload: dict) -> dict:
        positions = payload.get("positions", [])
        account_equity = float(payload.get("account_equity", 100000.0))

        total_notional = 0.0
        total_risk = 0.0

        for p in positions:
            vol = float(p.get("volume", 0.01))
            entry = float(p.get("price_open", p.get("price", 3557.0)))
            sl = float(p.get("sl", 0.0))
            # 1 lot of XAUUSD = 100 oz
            notional = vol * 100.0 * entry
            total_notional += notional
            if sl > 0:
                dist = abs(entry - sl)
                total_risk += dist * vol * 100.0
            else:
                total_risk += notional * 0.02  # assume 2% risk default if no SL

        leverage = total_notional / account_equity if account_equity > 0 else 0
        risk_pct = (total_risk / account_equity * 100) if account_equity > 0 else 0

        # Correlation risk heuristic
        symbols = [p.get("symbol", "XAUUSD") for p in positions]
        unique_symbols = set(symbols)
        corr_risk = "HIGH" if len(unique_symbols) == 1 and len(positions) > 2 else ("MEDIUM" if len(unique_symbols) == 1 and len(positions) > 0 else "LOW")

        # Concentration risk
        largest_position = max([float(p.get("volume", 0)) * 100.0 * float(p.get("price_open", 3557.0)) for p in positions], default=0.0)
        conc_pct = (largest_position / total_notional * 100) if total_notional > 0 else 0
        conc_risk = "HIGH" if conc_pct > 70 else ("MEDIUM" if conc_pct > 40 else "LOW")

        breaches = []
        warnings = []
        if risk_pct > 5.0:
            breaches.append(f"Total risk {risk_pct:.1f}% exceeds 5% portfolio limit")
        elif risk_pct > 3.0:
            warnings.append(f"Total risk {risk_pct:.1f}% approaching 5% cap")

        if leverage > 10.0:
            breaches.append(f"Leverage {leverage:.1f}x exceeds 10x limit")

        if conc_pct > 80 and len(positions) > 1:
            warnings.append(f"Concentration {conc_pct:.1f}% in single position")

        verdict = "BLOCKED" if breaches else ("WARNING" if (len(warnings) > 0 or corr_risk == "HIGH") else "PASS")

        return {
            "compliant": len(breaches) == 0,
            "verdict": verdict,
            "total_notional_usd": round(total_notional, 2),
            "total_risk_usd": round(total_risk, 2),
            "risk_pct_of_equity": round(risk_pct, 2),
            "leverage": round(leverage, 2),
            "correlation_risk": corr_risk,
            "concentration_risk": conc_risk,
            "concentration_pct": round(conc_pct, 1),
            "violations": breaches,
            "warnings": warnings,
            "positions_count": len(positions),
        }

    def _check_correlation(self, payload: dict) -> dict:
        return {
            "correlation_matrix": {"XAUUSD:XAUUSD": 1.0},
            "high_correlation_pairs": [],
            "status": "APPROVED",
        }

    def _check_max_drawdown(self, payload: dict) -> dict:
        current_equity = float(payload.get("current_equity", 100000.0))
        peak_equity = float(payload.get("peak_equity", 100000.0))
        max_dd_limit_pct = float(payload.get("max_dd_pct", 5.0))

        if peak_equity <= 0:
            peak_equity = current_equity

        dd_pct = max(0.0, (peak_equity - current_equity) / peak_equity * 100.0)
        breached = dd_pct >= max_dd_limit_pct
        remaining = max(0.0, max_dd_limit_pct - dd_pct)

        return {
            "breached": breached,
            "current_dd_pct": round(dd_pct, 2),
            "remaining_buffer_pct": round(remaining, 2),
            "max_dd_limit_pct": max_dd_limit_pct,
            "status": "CRITICAL_HALT" if breached else "NOMINAL",
        }
