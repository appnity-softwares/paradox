"""Deterministic Execution Engine.

Handles ALL order execution via broker connectors (MT5) or institutional paper execution.
Agents request execution, but this engine is the single point of truth for order placement,
position management, real-time floating PnL, margin tracking, and emergency controls.

All symbol constants (contract size, margin mode) are loaded from core.config — never hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import queue

from core.config import DEFAULT_BROKER, get_broker_spec, get_symbol_spec


@dataclass
class Position:
    ticket: int
    symbol: str
    side: str  # "BUY" or "SELL"
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    strategy_name: str
    time: str
    contract_size: float   # always supplied from SymbolSpec


@dataclass
class OrderResult:
    success: bool
    order_id: int | None
    filled_price: float | None
    volume: float
    reason: str


class ExecutionEngine:
    def __init__(self, broker_connector=None, starting_balance: float = 100_000.0, broker_name: str = DEFAULT_BROKER):
        self.broker = broker_connector
        self.broker_spec = get_broker_spec(broker_name)
        self.leverage = self.broker_spec.leverage
        self.positions: dict[int, Position] = {}
        self.order_counter = 100000
        self.kill_switch_active = False
        self.trade_log: list[dict] = []
        self.starting_balance = starting_balance
        self.balance = starting_balance
        # No quote until a real feed calls update_market_price(); simulated
        # fills are rejected while these are None (fail closed).
        self.last_bid: float | None = None
        self.last_ask: float | None = None
        # Command queue for Mac MT5 bridge
        self.mt5_commands = queue.Queue()

    def set_broker(self, broker):
        self.broker = broker

    def update_market_price(self, symbol: str, bid: float, ask: float):
        """Update live price and recalculate floating PnL on all open positions."""
        self.last_bid = bid
        self.last_ask = ask
        for pos in self.positions.values():
            if pos.symbol == symbol:
                pos.price_current = bid if pos.side == "BUY" else ask
                diff = (pos.price_current - pos.price_open) if pos.side == "BUY" else (pos.price_open - pos.price_current)
                pos.profit = round(diff * pos.volume * pos.contract_size, 2)

    def get_equity(self) -> float:
        floating = sum(p.profit for p in self.positions.values())
        return round(self.balance + floating, 2)

    def get_margin_used(self) -> float:
        # Margin = notional / leverage, leverage comes from BrokerSpec.
        margin = 0.0
        for p in self.positions.values():
            margin += (p.volume * p.contract_size * p.price_current) / self.leverage
        return round(margin, 2)

    def get_account_summary(self) -> dict:
        equity = self.get_equity()
        margin = self.get_margin_used()
        free_margin = round(equity - margin, 2)
        margin_level = round((equity / margin * 100), 1) if margin > 0 else 999.0
        floating = round(sum(p.profit for p in self.positions.values()), 2)
        return {
            "balance": round(self.balance, 2),
            "equity": equity,
            "margin": margin,
            "free_margin": free_margin,
            "margin_level": margin_level,
            "floating_pnl": floating,
            "open_positions_count": len(self.positions),
            "kill_switch_active": self.kill_switch_active,
        }

    def place_market_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        strategy_name: str = "Manual",
        comment: str = "AI-Quant",
        force_paper: bool = False,
    ) -> OrderResult:
        """Place a market order. Returns OrderResult."""
        if self.kill_switch_active:
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=None,
                volume=0.0,
                reason="Kill switch active - no new orders allowed",
            )

        side = side.upper()
        if side not in ("BUY", "SELL"):
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=None,
                volume=0.0,
                reason="Invalid side (must be BUY or SELL)",
            )

        if volume <= 0:
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=None,
                volume=0.0,
                reason="Invalid volume",
            )

        try:
            spec = get_symbol_spec(symbol)
        except KeyError as exc:
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=None,
                volume=volume,
                reason=str(exc),
            )

        # Try broker execution if not forced to paper and broker is available
        fill_price = None
        order_id = None

        if not force_paper and self.broker is not None and getattr(self.broker, "is_available", lambda: None)():
            try:
                result = self.broker.place_market_order(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    sl=sl,
                    tp=tp,
                    comment=comment,
                )
                if result:
                    order_id = result.get("ticket", self.order_counter)
                    fill_price = result.get("price")
                    self.order_counter = max(self.order_counter, order_id + 1)
            except Exception as exc:
                return OrderResult(
                    success=False,
                    order_id=None,
                    filled_price=None,
                    volume=volume,
                    reason=f"Broker execution failed: {exc}",
                )

        if fill_price is None:
            if self.last_bid is None or self.last_ask is None:
                return OrderResult(
                    success=False,
                    order_id=None,
                    filled_price=None,
                    volume=volume,
                    reason="No live quote available - update_market_price() must supply a real bid/ask before simulated fills",
                )
            order_id = self.order_counter
            self.order_counter += 1
            fill_price = self.last_ask if side == "BUY" else self.last_bid

        # Queue command for Mac MT5 bridge EA if broker is Mac MT5
        if not force_paper:
            self.mt5_commands.put({
                "action": "OPEN",
                "ticket": order_id,
                "symbol": symbol,
                "side": side,
                "volume": volume,
                "sl": sl,
                "tp": tp,
                "comment": comment,
            })

        # Create position — contract size always comes from the symbol spec
        cs = spec.contract_size
        pos = Position(
            ticket=order_id,
            symbol=symbol,
            side=side,
            volume=volume,
            price_open=fill_price,
            price_current=fill_price,
            sl=sl,
            tp=tp,
            profit=0.0,
            strategy_name=strategy_name,
            time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            contract_size=cs,
        )
        self.positions[order_id] = pos

        self.trade_log.append({
            "action": "OPEN",
            "ticket": order_id,
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "price": fill_price,
            "sl": sl,
            "tp": tp,
            "strategy": strategy_name,
            "time": pos.time,
        })

        return OrderResult(
            success=True,
            order_id=order_id,
            filled_price=fill_price,
            volume=volume,
            reason=f"{side} {volume} {symbol} filled @ {fill_price:.2f}",
        )

    def close_position(self, position_id: int, reason: str = "Manual close") -> dict:
        """Close an open position."""
        pos = self.positions.get(position_id)
        if pos is None:
            return {"success": False, "reason": "Position not found", "realized_pnl": 0.0}

        # Calculate PnL with standard gold contract size (100 oz per lot)
        price_diff = (pos.price_current - pos.price_open) if pos.side == "BUY" else (pos.price_open - pos.price_current)
        realized_pnl = round(price_diff * pos.volume * pos.contract_size, 2)

        # Try broker close
        if self.broker is not None:
            try:
                self.broker.close_position(position_id)
            except Exception:
                pass

        # Also queue for Mac MT5 EA
        self.mt5_commands.put({
            "action": "CLOSE",
            "ticket": position_id,
            "symbol": pos.symbol,
            "volume": pos.volume,
        })

        self.balance += realized_pnl
        del self.positions[position_id]

        self.trade_log.append({
            "action": "CLOSE",
            "ticket": position_id,
            "symbol": pos.symbol,
            "side": pos.side,
            "volume": pos.volume,
            "price": pos.price_current,
            "pnl": realized_pnl,
            "strategy": pos.strategy_name,
            "reason": reason,
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })

        return {
            "success": True,
            "position_id": position_id,
            "realized_pnl": realized_pnl,
            "balance": round(self.balance, 2),
            "reason": reason,
        }

    def update_sl_tp(self, position_id: int, sl: float, tp: float) -> dict:
        """Update SL/TP on open position."""
        pos = self.positions.get(position_id)
        if pos is None:
            return {"success": False, "reason": "Position not found"}

        pos.sl = sl
        pos.tp = tp

        if self.broker is not None:
            try:
                self.broker.update_position_sl_tp(position_id, sl, tp)
            except Exception:
                pass

        self.mt5_commands.put({
            "action": "MODIFY",
            "ticket": position_id,
            "sl": sl,
            "tp": tp,
        })

        return {"success": True, "position_id": position_id, "sl": sl, "tp": tp}

    def set_break_even(self, position_id: int) -> dict:
        """Move Stop Loss to Entry Price (Break Even)."""
        pos = self.positions.get(position_id)
        if pos is None:
            return {"success": False, "reason": "Position not found"}

        # Move SL to entry price (+ spread buffer if in profit)
        pos.sl = pos.price_open
        return self.update_sl_tp(position_id, pos.sl, pos.tp)

    def get_positions(self) -> list[dict]:
        """Get all open positions with live profit."""
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "side": p.side,
                "volume": p.volume,
                "price_open": round(p.price_open, 2),
                "price_current": round(p.price_current, 2),
                "sl": round(p.sl, 2) if p.sl else 0.0,
                "tp": round(p.tp, 2) if p.tp else 0.0,
                "profit": round(p.profit, 2),
                "strategy_name": p.strategy_name,
                "time": p.time,
            }
            for p in self.positions.values()
        ]

    def emergency_kill_switch(self) -> dict:
        """Close ALL open positions immediately and freeze execution."""
        self.kill_switch_active = True
        tickets = list(self.positions.keys())
        total_pnl = 0.0
        for t in tickets:
            res = self.close_position(t, reason="EMERGENCY KILL SWITCH ACTIVATED")
            total_pnl += res.get("realized_pnl", 0.0)

        return {
            "success": True,
            "closed_count": len(tickets),
            "total_realized_pnl": round(total_pnl, 2),
            "kill_switch_active": True,
            "message": "All positions closed. Trading halted.",
        }

    def reset_kill_switch(self) -> dict:
        self.kill_switch_active = False
        return {"success": True, "kill_switch_active": False, "message": "Trading resumed."}
