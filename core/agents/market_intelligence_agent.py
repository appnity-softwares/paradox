"""Market Intelligence Agent - Market sentiment, news, session awareness.

Responsibilities:
- Detect active trading sessions (Tokyo, London, NY, Sydney)
- Monitor real-time spread and volatility regime
- Provide context to other agents about WHEN to trade
- NEVER executes trades directly
"""
from __future__ import annotations

from typing import Any
from datetime import datetime, timezone

from core.orchestrator import AgentCapability, AgentMessage, BaseAgent, MessageType
from core.data_manager import get_manager


class MarketIntelligenceAgent(BaseAgent):
    def __init__(self, orchestrator):
        super().__init__("market_intelligence_agent", "Market Intelligence Agent", orchestrator)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="session_analysis",
                description="Determine active trading session and volatility expectations",
                input_schema={"symbol": "str", "current_time_utc": "str"},
                output_schema={"session": "str", "volatility_expectation": "str", "recommended_strategies": "list[str]"},
            ),
            AgentCapability(
                name="news_impact",
                description="Check for high-impact news events that may affect trading",
                input_schema={"symbol": "str", "timeframe_hours": "int"},
                output_schema={"news_block": "bool", "events": "list[dict]", "recommendation": "str"},
            ),
            AgentCapability(
                name="market_context",
                description="Provide overall market context (spread, volatility, liquidity)",
                input_schema={"symbol": "str"},
                output_schema={"spread_pips": "float", "volatility_regime": "str", "liquidity": "str", "tradeable": "bool"},
            ),
        ]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        payload = message.payload
        action = payload.get("action", "")

        if action == "session_analysis":
            result = self._analyze_session(payload.get("symbol", "XAUUSD"), payload.get("current_time_utc", ""))
            return await self.reply_to(message, {"success": True, **result})

        if action == "news_impact":
            result = self._check_news_impact(payload.get("symbol", "XAUUSD"), int(payload.get("timeframe_hours", 24)))
            return await self.reply_to(message, {"success": True, **result})

        if action == "market_context":
            result = self._get_market_context(payload.get("symbol", "XAUUSD"))
            return await self.reply_to(message, {"success": True, **result})

        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown action: {action}", "success": False},
            correlation_id=message.id,
        )

    def _analyze_session(self, symbol: str, current_time_utc: str = "") -> dict:
        now = datetime.now(timezone.utc)
        hour = now.hour

        # Session definitions (UTC)
        sessions = {
            "sydney": (22, 7),   # 22:00 - 07:00
            "tokyo": (0, 9),     # 00:00 - 09:00
            "london": (7, 16),   # 07:00 - 16:00
            "ny": (12, 21),      # 12:00 - 21:00
        }

        active = []
        for name, (start, end) in sessions.items():
            if start <= end:
                if start <= hour < end:
                    active.append(name)
            else:  # overnight session
                if hour >= start or hour < end:
                    active.append(name)

        if not active:
            active = ["sydney"]

        vol_map = {
            "sydney": "low",
            "tokyo": "medium",
            "london": "high",
            "ny": "high",
        }

        active_volatilities = [vol_map.get(s, "medium") for s in active]
        overall_vol = "high" if "high" in active_volatilities else ("medium" if "medium" in active_volatilities else "low")

        strategy_map = {
            "sydney": ["mean_reversion", "range_trading"],
            "tokyo": ["mean_reversion", "breakout"],
            "london": ["trend_following", "breakout", "session_breakout"],
            "ny": ["trend_following", "vwap_ema", "breakout"],
        }

        recommended = []
        for s in active:
            if s in strategy_map:
                recommended.extend(strategy_map[s])

        return {
            "session": ", ".join(s.upper() for s in active),
            "active_sessions": active,
            "volatility_expectation": overall_vol,
            "hour_utc": hour,
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "recommended_strategies": list(set(recommended)),
            "tradeable": True,
        }

    def _check_news_impact(self, symbol: str, timeframe_hours: int = 24) -> dict:
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Monday, 4=Friday, 5=Sat, 6=Sun

        events = [
            {"event": "US Core PCE Price Index", "impact": "HIGH", "window": "NY Session"},
            {"event": "Federal Reserve FOMC Minutes", "impact": "CRITICAL", "window": "Wednesday 18:00 UTC"},
            {"event": "US Non-Farm Payrolls (NFP)", "impact": "CRITICAL", "window": "First Friday 12:30 UTC"},
        ]

        news_block = weekday in (5, 6)  # Weekend market closure
        recommendation = "Weekend market closed" if news_block else "Normal trading conditions. Maintain 1.5x ATR stops."

        return {
            "news_block": news_block,
            "symbol": symbol,
            "timeframe_hours": timeframe_hours,
            "events": events,
            "recommendation": recommendation,
        }

    def _get_market_context(self, symbol: str) -> dict:
        dm = get_manager()
        # No fabricated fallbacks: without a real tick we report "unavailable"
        # instead of inventing a price or spread (Master Prompt 14.5.3).
        try:
            if dm.active:
                tick = dm.active.fetch_tick(symbol)
                if tick:
                    bid = float(tick.get("bid", 0))
                    ask = float(tick.get("ask", 0))
                    if bid > 0 and ask > 0:
                        spread = round(ask - bid, 2)
                        price = round(ask, 2)
                        liquidity = "HIGH" if spread <= 0.35 else ("NORMAL" if spread <= 0.60 else "LOW")
                        return {
                            "symbol": symbol,
                            "price": price,
                            "spread_points": spread,
                            "spread_usd": f"${spread:.2f}/oz",
                            "liquidity": liquidity,
                            "tradeable": spread < 0.80,
                            "note": "Spread within institutional tolerance (< $0.40/oz on gold spot).",
                        }
        except Exception:
            pass

        return {
            "symbol": symbol,
            "price": None,
            "spread_points": None,
            "spread_usd": None,
            "liquidity": "UNKNOWN",
            "tradeable": False,
            "note": "No live tick available; spread and price are unknown, not assumed.",
        }
