"""Strategy Adaptation Agent - Configures strategies per account/instrument.

Responsibilities:
- Adapt strategy parameters to specific account size, risk tolerance, instrument
- Map generic strategy signals to account-specific position sizing
- NEVER approves or executes trades - only suggests configurations
"""
from __future__ import annotations

from typing import Any

from core.orchestrator import AgentCapability, AgentMessage, BaseAgent, MessageType


class StrategyAdaptationAgent(BaseAgent):
    def __init__(self, orchestrator):
        super().__init__("strategy_adaptation_agent", "Strategy Adaptation Agent", orchestrator)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="adapt_strategy",
                description="Adapt strategy parameters to account size, instrument, and risk limits",
                input_schema={"strategy_name": "str", "account_size": "float", "risk_pct": "float", "symbol": "str"},
                output_schema={"position_size": "float", "sl_distance": "float", "tp_distance": "float", "max_daily_trades": "int"},
            ),
            AgentCapability(
                name="position_sizing",
                description="Calculate position size based on account equity and stop distance",
                input_schema={"equity": "float", "risk_pct": "float", "sl_distance": "float", "symbol": "str"},
                output_schema={"units": "float", "risk_amount": "float", "notional": "float"},
            ),
            AgentCapability(
                name="instrument_config",
                description="Map strategy to instrument-specific parameters (tick size, contract size)",
                input_schema={"symbol": "str", "strategy_name": "str"},
                output_schema={"tick_size": "float", "contract_size": "float", "min_volume": "float", "max_volume": "float"},
            ),
        ]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        payload = message.payload
        action = payload.get("action", "")

        if action == "adapt_strategy":
            strategy_name = payload.get("strategy_name", "ema_cross")
            account_size = float(payload.get("account_size", 10000))
            risk_pct = float(payload.get("risk_pct", 1.0))
            symbol = payload.get("symbol", "XAUUSD")

            config = self._adapt_strategy(strategy_name, account_size, risk_pct, symbol)
            return await self.reply_to(message, {"success": True, **config})

        if action == "position_sizing":
            equity = float(payload.get("equity", 10000))
            risk_pct = float(payload.get("risk_pct", 1.0))
            sl_distance = float(payload.get("sl_distance", 5.0))
            symbol = payload.get("symbol", "XAUUSD")

            sizing = self._calculate_position_size(equity, risk_pct, sl_distance, symbol)
            return await self.reply_to(message, {"success": True, **sizing})

        if action == "instrument_config":
            symbol = payload.get("symbol", "XAUUSD")
            strategy_name = payload.get("strategy_name", "ema_cross")
            config = self._get_instrument_config(symbol, strategy_name)
            return await self.reply_to(message, {"success": True, **config})

        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown action: {action}", "success": False},
            correlation_id=message.id,
        )

    def _adapt_strategy(self, strategy_name: str, account_size: float, risk_pct: float, symbol: str) -> dict:
        # Default strategy parameters by name
        defaults = {
            "ema_cross": {"fast": 20, "slow": 50, "sl_atr_mult": 1.5, "rr": 2.0},
            "vwap_ema": {"ema_length": 20, "slope": 1, "sl_atr_mult": 1.5, "rr": 1.5},
            "supertrend": {"length": 10, "mult": 3.0, "sl_atr_mult": 2.0, "rr": 2.0},
            "rsi_reversal": {"length": 14, "oversold": 30, "overbought": 70, "sl_atr_mult": 1.5, "rr": 1.5},
            "bollinger": {"length": 20, "num_std": 2.0, "sl_atr_mult": 2.0, "rr": 1.5},
            "macd_cross": {"fast": 12, "slow": 26, "signal": 9, "sl_atr_mult": 1.5, "rr": 2.0},
            "session_breakout": {"buffer": 0.0, "range_end": 7, "sl_atr_mult": 2.0, "rr": 2.0},
        }
        params = defaults.get(strategy_name, defaults["ema_cross"])

        # Risk per trade amount
        risk_amount = account_size * (risk_pct / 100)

        # SL distance from the real 1h ATR when available; never a guessed constant.
        market = self._latest_real_price_atr(symbol)
        if "atr" in market:
            sl_distance = params["sl_atr_mult"] * market["atr"]
        else:
            sl_distance = None
        tp_distance = sl_distance * params["rr"] if sl_distance is not None else None

        # Position size in ounces
        units = risk_amount / sl_distance if sl_distance else 0
        notional = units * market["price"] if "price" in market and sl_distance else None

        # Max trades per day based on account size (smaller account = fewer trades)
        max_daily = min(4, max(1, int(account_size / 5000)))

        return {
            "strategy_name": strategy_name,
            "params": params,
            "account_size": account_size,
            "risk_pct": risk_pct,
            "risk_amount_usd": round(risk_amount, 2),
            "sl_distance_usd": round(sl_distance, 2) if sl_distance is not None else None,
            "tp_distance_usd": round(tp_distance, 2) if tp_distance is not None else None,
            "position_size_units": round(units, 4),
            "position_size_notional": round(notional, 2) if notional is not None else None,
            "max_daily_trades": max_daily,
            "symbol": symbol,
            "data_source": market.get("provider"),
            "data_note": None if market else "No real market data available; SL/TP/notional not estimated.",
        }

    def _latest_real_price_atr(self, symbol: str) -> dict:
        """Last real close and ATR(14) on 1h bars, or {} when no real data.

        Never fabricates a price: without real data the caller reports
        ``unavailable`` instead of guessing (Master Prompt 14.5.3).
        """
        try:
            from core.data_manager import get_manager

            dm = get_manager()
            if dm.active is None:
                return {}
            result = dm.fetch_snapshot(symbol, "1h", 100, allow_synthetic=False)
            df = result.data
            if df is None or len(df) < 15 or result.synthetic:
                return {}
            from core.indicators import atr as atr_series

            atr_value = float(atr_series(df, 14).iloc[-1])
            close = float(df["close"].iloc[-1])
            if not (atr_value > 0 and close > 0):
                return {}
            return {"price": close, "atr": atr_value, "provider": result.provider_key}
        except Exception:
            return {}

    def _calculate_position_size(self, equity: float, risk_pct: float, sl_distance: float, symbol: str) -> dict:
        risk_amount = equity * (risk_pct / 100)
        units = risk_amount / sl_distance if sl_distance > 0 else 0
        market = self._latest_real_price_atr(symbol)
        notional = units * market["price"] if "price" in market else None
        return {
            "equity": equity,
            "risk_pct": risk_pct,
            "risk_amount_usd": round(risk_amount, 2),
            "sl_distance_usd": round(sl_distance, 2),
            "units": round(units, 4),
            "notional_usd": round(notional, 2) if notional is not None else None,
            "price_source": market.get("provider") if market else None,
        }

    def _get_instrument_config(self, symbol: str, strategy_name: str) -> dict:
        # Single source of truth for instrument constants: core.config.SymbolSpec.
        from core.config import get_symbol_spec

        spec = get_symbol_spec(symbol)
        market = self._latest_real_price_atr(symbol)
        return {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "tick_size": spec.tick_size,
            "contract_size": spec.contract_size,
            "min_volume": spec.min_lot,
            "max_volume": spec.max_lot,
            "lot_step": spec.lot_step,
            "last_price": market.get("price"),
            "atr_1h": market.get("atr"),
            "price_source": market.get("provider"),
        }
