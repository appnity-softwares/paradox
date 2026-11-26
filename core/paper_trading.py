"""Live Paper Trading Engine - simulation execution on real market data.

Replays the same execution semantics as the backtester without sending real
orders:

1. Signal is taken from the PREVIOUS bar and acted at the CURRENT bar open.
2. Entries pay half the spread plus slippage; exits pay it again (round trip
   = full spread, like a real account).
3. SL is evaluated before TP at the current bar high/low. SL fills AT the
   stop (or at the open if price gapped through it — the worse price);
   TP fills at the target (or at the open if price gapped through it —
   the better price, as a real limit fill would).
4. Opposite signal exits the open position at the open, paying spread.
5. PnL is booked in account currency: price_move x volume(lots) x
   contract_size (XAUUSD 1 lot = 100 oz).

Until the MT5 broker connector is live, this engine runs strategies against
real fetched market data (Yahoo on Mac) and keeps positions/PnL/risk state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.config import CostModel, get_cost_model
from core.risk_engine import RiskEngine
from strategies.base import StrategyConfig


@dataclass
class PaperPosition:
    ticket: int
    symbol: str
    side: str  # "LONG" | "SHORT"
    entry_price: float
    sl: float
    tp: float
    volume: float
    strategy_name: str
    opened_at: str


@dataclass
class PaperTrade:
    ticket: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    reason: str
    strategy_name: str
    entry_time: str
    exit_time: str


class PaperTradingEngine:
    def __init__(
        self,
        risk: RiskEngine | None = None,
        cost_model: CostModel | None = None,
        starting_equity: float = 100_000.0,
        symbol: str = "XAUUSD",
        contract_size: float = 100.0,   # XAUUSD: 1 lot = 100 oz
    ):
        self.risk = risk if risk is not None else RiskEngine()
        # Costs come from the typed CostModel, never from literals here.
        self.cost_model = cost_model if cost_model is not None else get_cost_model(symbol)
        self.spread = self.cost_model.spread_price()
        self.half_spread = self.spread / 2.0
        self.slippage = self.cost_model.slippage.market_points
        self.contract_size = float(contract_size)
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.kill_switch_active = False
        self.positions: list[PaperPosition] = []
        self.trades: list[PaperTrade] = []
        self._strategy = None
        self._cfg: StrategyConfig | None = None
        self._symbol = "XAUUSD"
        self._buffer: list[dict] = []
        self._last_signal = 0.0
        self._warmup_bars = 0

    # ------------------------------------------------------------------ setup
    def start(self, strategy, cfg: StrategyConfig, symbol: str = "XAUUSD"):
        self._strategy = strategy
        self._cfg = cfg
        self._symbol = symbol
        self._buffer = []
        self._last_signal = 0.0
        self.positions = []
        self.trades = []
        self.equity = self.starting_equity
        self.kill_switch_active = False
        warmup = getattr(strategy, "warmup", 60)
        self._warmup_bars = int(warmup)
        return self.get_status()

    def stop(self):
        self._strategy = None
        self._cfg = None
        return self.get_status()

    # ------------------------------------------------------------------ state
    def get_positions(self) -> list[dict]:
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "sl": p.sl,
                "tp": p.tp,
                "volume": p.volume,
                "strategy_name": p.strategy_name,
                "opened_at": p.opened_at,
            }
            for p in self.positions
        ]

    def get_trades(self) -> list[dict]:
        return [
            {
                "ticket": t.ticket,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": round(t.pnl, 2),
                "reason": t.reason,
                "strategy_name": t.strategy_name,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            }
            for t in self.trades
        ]

    def get_status(self) -> dict:
        risk_status = self.risk.get_status()
        return {
            "running": self._strategy is not None,
            "symbol": self._symbol,
            "strategy": getattr(self._strategy, "key", None) if self._strategy else None,
            "kill_switch_active": self.kill_switch_active,
            "positions": len(self.positions),
            "open_position": bool(self.positions),
            "trades_closed": len(self.trades),
            "equity": round(self.equity, 2),
            "realized_pnl": round(sum(t.pnl for t in self.trades), 2),
            "risk": risk_status,
        }

    # ------------------------------------------------------------------ control
    def trigger_kill_switch(self) -> dict:
        self.kill_switch_active = True
        closed = 0
        for p in list(self.positions):
            exit_price = self._market_exit_price(p.side)
            if exit_price is None:
                exit_price = p.sl if p.side == "LONG" else p.sl
            trade = self._close_position(p, reason="KILL_SWITCH", exit_price=exit_price)
            if trade:
                closed += 1
        return {"killed": True, "positions_closed": closed}

    def release_kill_switch(self):
        self.kill_switch_active = False

    def update_sl_tp(self, ticket: int, sl: float, tp: float) -> dict:
        for p in self.positions:
            if p.ticket == ticket:
                p.sl = sl
                p.tp = tp
                return {"success": True, "ticket": ticket, "sl": sl, "tp": tp}
        return {"success": False, "reason": "Position not found"}

    def close_position(self, ticket: int, reason: str = "Manual close") -> dict:
        for p in list(self.positions):
            if p.ticket == ticket:
                exit_price = self._market_exit_price(p.side)
                if exit_price is None:
                    return {"success": False, "reason": "no price seen yet"}
                trade = self._close_position(p, reason=reason, exit_price=exit_price)
                if trade:
                    return {"success": True, "ticket": ticket, "trade": trade}
        return {"success": False, "reason": "Position not found"}

    # ------------------------------------------------------------------ engine
    def _market_exit_price(self, side: str) -> float | None:
        """Last buffered close crossed with the exit spread (bid for a LONG,
        ask for a SHORT). None when no bar has been seen yet."""
        if not self._buffer:
            return None
        last = float(self._buffer[-1]["close"])
        return last - (self.half_spread + self.slippage) if side == "LONG" \
            else last + (self.half_spread + self.slippage)

    # ------------------------------------------------------- manual desk
    def manual_order(self, symbol: str, side: str, volume_lots: float,
                     entry_price: float, sl_price: float, tp_price: float,
                     strategy_name: str = "manual") -> dict:
        """Open a manual paper position at an explicit fill price.

        The caller supplies the real quote side: a BUY fills at the ask and a
        SELL at the bid, so the spread is paid on entry — and again on exit.
        Paper only; this method never touches a broker.
        """
        if self.kill_switch_active:
            return {"ok": False, "error": "kill_switch_active"}
        if self.risk.get_status()["blocked"]:
            return {"ok": False, "error": "risk_blocked"}
        side_u = str(side).upper()
        if side_u not in ("BUY", "SELL"):
            return {"ok": False, "error": "side must be BUY or SELL"}
        try:
            vol = float(volume_lots)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid volume"}
        if not (0 < vol <= 100):
            return {"ok": False, "error": "invalid volume"}
        entry, sl, tp = float(entry_price), float(sl_price), float(tp_price)
        if entry <= 0:
            return {"ok": False, "error": "no live quote"}
        long = side_u == "BUY"
        if long and not (sl < entry < tp):
            return {"ok": False, "error": "SL/TP must bracket the entry for a BUY"}
        if not long and not (tp < entry < sl):
            return {"ok": False, "error": "SL/TP must bracket the entry for a SELL"}
        if len(self.positions) >= 20:
            return {"ok": False, "error": "max open paper positions reached"}
        pos = PaperPosition(
            ticket=self._next_ticket(),
            symbol=str(symbol).upper(),
            side="LONG" if long else "SHORT",
            entry_price=entry,
            sl=sl,
            tp=tp,
            volume=vol,
            strategy_name=strategy_name,
            opened_at=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        self.positions.append(pos)
        return {"ok": True, "ticket": pos.ticket, "filled_price": entry,
                "volume": vol, "sl": sl, "tp": tp, "side": pos.side}

    def manual_close(self, ticket: int, exit_price: float) -> dict:
        """Close a paper position at an explicit exit price (bid for a LONG,
        ask for a SHORT — the caller picks the quote side)."""
        pos = next((p for p in self.positions if p.ticket == int(ticket)), None)
        if pos is None:
            return {"ok": False, "error": "unknown ticket"}
        trade = self._close_position(pos, "MANUAL", float(exit_price))
        return {"ok": True, "trade": trade}

    def _atr(self, highs: list[float], lows: list[float], closes: list[float]) -> float:
        if len(closes) < 2:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(closes[i - 1] - highs[i]), abs(closes[i - 1] - lows[i]))
            trs.append(tr)
        if not trs:
            return 0.0
        return float(np.mean(trs))

    def _compute_signal(self) -> float:
        if self._strategy is None or len(self._buffer) < 2:
            return 0.0
        df = pd.DataFrame(
            {
                "open": [b["open"] for b in self._buffer],
                "high": [b["high"] for b in self._buffer],
                "low": [b["low"] for b in self._buffer],
                "close": [b["close"] for b in self._buffer],
                "volume": [b.get("volume", 0.0) for b in self._buffer],
            },
            index=pd.DatetimeIndex([pd.Timestamp(b["time"]).tz_convert("UTC") if pd.Timestamp(b["time"]).tzinfo is not None else pd.Timestamp(b["time"]).tz_localize("UTC") for b in self._buffer]),
        )
        sig = self._strategy.generate_signals(df)
        last = float(sig.iloc[-1]) if len(sig) else 0.0
        return 0.0 if len(self._buffer) <= int(getattr(self._strategy, "warmup", 60)) else last

    def _close_position(self, pos: PaperPosition, reason: str, exit_price: float | None = None) -> PaperTrade | None:
        if exit_price is None:
            # No explicit exit price given (should not happen anymore): fall
            # back to the level itself, never to a free entry-price exit.
            exit_price = pos.tp if reason == "TP" else pos.sl
        oz = pos.volume * self.contract_size
        pnl = (exit_price - pos.entry_price) * (1 if pos.side == "LONG" else -1) * oz
        trade = PaperTrade(
            ticket=pos.ticket,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=round(pnl, 2),
            reason=reason,
            strategy_name=pos.strategy_name,
            entry_time=pos.opened_at,
            exit_time=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        self.trades.append(trade)
        self.equity += pnl
        try:
            self.risk.record_trade_result(pnl / max(self.equity - pnl, 1e-9) * 100)
        except Exception:
            pass
        self.positions = [p for p in self.positions if p.ticket != pos.ticket]
        return trade

    def process_bar(self, bar: dict) -> PaperTrade | None:
        """Feed one bar (time/open/high/low/close/volume) and execute pending signals."""
        if self.kill_switch_active or self._strategy is None or self._cfg is None:
            return None
        # Signal semantics: act on the PREVIOUS bar's signal at the CURRENT bar open.
        self._last_signal = self._compute_signal()
        self._buffer.append(
            {
                "time": bar["time"],
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume") or 0.0),
            }
        )
        if len(self._buffer) > 5000:
            self._buffer = self._buffer[-5000:]

        if not self.positions and not self.risk.get_status()["blocked"]:
            sig = self._last_signal
            warmup_ok = len(self._buffer) > int(getattr(self._strategy, "warmup", 60))
            if sig != 0 and warmup_ok:
                direction = "LONG" if sig > 0 else "SHORT"
                entry = float(bar["open"]) + (self.half_spread + self.slippage if direction == "LONG"
                                              else -(self.half_spread + self.slippage))
                atr = self._atr(
                    [b["high"] for b in self._buffer], [b["low"] for b in self._buffer], [b["close"] for b in self._buffer]
                )
                sl_dist = atr * float(self._cfg.sl_atr_mult) if self._cfg.sl_method == "atr" else float(self._cfg.sl_fixed)
                sl = entry + sl_dist if direction == "LONG" else entry - sl_dist
                tp = entry + sl_dist * float(self._cfg.rr) if direction == "LONG" else entry - sl_dist * float(self._cfg.rr)
                self.positions.append(
                    PaperPosition(
                        ticket=self._next_ticket(),
                        symbol=self._symbol,
                        side=direction,
                        entry_price=entry,
                        sl=sl,
                        tp=tp,
                        volume=0.1,
                        strategy_name=str(getattr(self._strategy, "key", "strategy")),
                        opened_at=str(bar["time"]),
                    )
                )
                return None

        if not self.positions:
            return None

        o = float(bar["open"])
        lo, hi = float(bar["low"]), float(bar["high"])

        # SL/TP for EVERY open position (manual desk positions included).
        closed_trade = None
        still_open = []
        for pos in self.positions:
            close_price = None
            reason = None
            if pos.side == "LONG":
                # SL first (conservative). Gap through the stop fills at the
                # open (worse); otherwise the stop itself, minus exit costs.
                if o <= pos.sl:
                    close_price, reason = o - self.slippage, "SL"
                elif lo <= pos.sl:
                    close_price, reason = pos.sl - self.half_spread - self.slippage, "SL"
                # TP is a limit: gap through fills at the open (better).
                elif o >= pos.tp:
                    close_price, reason = o, "TP"
                elif hi >= pos.tp:
                    close_price, reason = pos.tp, "TP"
            else:
                if o >= pos.sl:
                    close_price, reason = o + self.slippage, "SL"
                elif hi >= pos.sl:
                    close_price, reason = pos.sl + self.half_spread + self.slippage, "SL"
                elif o <= pos.tp:
                    close_price, reason = o, "TP"
                elif lo <= pos.tp:
                    close_price, reason = pos.tp, "TP"
            if close_price is None:
                still_open.append(pos)
                continue
            trade = self._close_position(pos, reason, close_price)
            if trade is not None and closed_trade is None:
                closed_trade = trade

        # A flipped strategy signal closes only the strategy's own position,
        # never a manual desk position.
        if closed_trade is None and self._last_signal != 0:
            for pos in still_open:
                if pos.strategy_name == "manual":
                    continue
                if (pos.side == "LONG") != (self._last_signal > 0):
                    exit_price = o - (self.half_spread + self.slippage) if pos.side == "LONG" \
                        else o + (self.half_spread + self.slippage)
                    closed_trade = self._close_position(pos, "SIGNAL", exit_price)
                    still_open = [p for p in still_open if p.ticket != pos.ticket]
                    break

        self.positions = still_open
        return closed_trade

    def _next_ticket(self) -> int:
        self._tick = getattr(self, "_tick", 1) + 1
        return self._tick

    def replay(self, df) -> dict:
        """Feed historical bars (pandas DataFrame) through the paper engine."""
        strat = self._strategy
        cfg = self._cfg
        if strat is None or cfg is None:
            return {"ok": False, "error": "no strategy loaded"}
        trades_before = len(self.trades)
        for _, row in df.iterrows():
            bar = {
                "time": row.name,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume") or 0.0),
            }
            trade = self.process_bar(bar)
            if trade is not None:
                self._last_trade = trade
        new_trades = self.trades[trades_before:]
        return {
            "ok": True,
            "trades_executed": len(new_trades),
            "trades": new_trades,
            "status": self.get_status(),
        }
