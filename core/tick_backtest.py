"""Sub-Second Tick-by-Tick High-Fidelity Backtesting Engine.

Replays real bid/ask tick series with zero lookahead bias:
- Long entries fill at ASK, long exits at BID.
- Short entries fill at BID, short exits at ASK.
- SL/TP triggered on every tick at the correct side of the quote.
- Contract size, slippage, and all cost parameters come from core.config.
  NO hardcoded constants for financial calculations.

NOTE: This engine replays ticks from the compressed Vertex Store, which only
accepts real tick data (no synthesis). It is NOT a substitute for a full
2-year Dukascopy tick replay (Phase 1B). See Phase 1B for the production engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.vertex_store import VERTEX_STORE, NoTickDataError
from core.config import get_broker_spec, get_symbol_spec, get_cost_model


@dataclass
class TickTrade:
    id: int
    symbol: str
    side: str
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    exit_reason: str
    duration_sec: float
    ticks_held: int


class TickBacktestEngine:
    """Executes a simple tick-level strategy against vertex-store ticks.

    Uses SymbolSpec from core.config for contract size / tick value.
    Uses CostModel from core.config for slippage.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        leverage: int | None = None,
        symbol: str = "XAUUSD",
        cost_model_key: str = "XAUUSD_ECN",
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage if leverage is not None else get_broker_spec().leverage
        self.symbol = symbol
        self._spec = get_symbol_spec(symbol)
        self._cost = get_cost_model(cost_model_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_key: str = "ema_cross",
        params: Optional[Dict[str, Any]] = None,
        risk_pct: float = 1.0,
        sl_points: float = 3.0,
        tp_points: float = 6.0,
        max_ticks: int = 50_000,
    ) -> Dict[str, Any]:
        """Run the tick backtest and return a performance summary dict."""
        params = params or {}

        try:
            df_ticks = VERTEX_STORE.get_ticks_df()
        except NoTickDataError as exc:
            return {"error": str(exc)}
        if df_ticks.empty:
            return {"error": "No tick data available in vertex store. Rebuild the store first."}

        if len(df_ticks) > max_ticks:
            df_ticks = df_ticks.tail(max_ticks).reset_index(drop=True)

        n = len(df_ticks)
        times = df_ticks["time_ms"].to_numpy(dtype=np.int64)
        bids  = df_ticks["bid"].to_numpy(dtype=np.float64)
        asks  = df_ticks["ask"].to_numpy(dtype=np.float64)
        lasts = df_ticks["last"].to_numpy(dtype=np.float64)

        # Pull cost / symbol constants from config — no magic numbers
        contract_size     = self._spec.contract_size
        mkt_slip_points   = self._cost.slippage.market_points
        stop_slip_points  = self._cost.slippage.stop_points

        balance    = self.initial_balance
        equity     = self.initial_balance
        peak_eq    = self.initial_balance
        max_dd_usd = 0.0

        open_pos: Optional[Dict[str, Any]] = None
        trades: List[TickTrade] = []
        equity_curve: List[List[Any]] = []

        # Simple EMA micro-trend signal (placeholder until Phase 1B strategy interface)
        alpha_fast = 2.0 / (20 + 1)
        alpha_slow = 2.0 / (80 + 1)
        ema_fast = lasts[0]
        ema_slow = lasts[0]

        trade_seq = 1
        warmup = 100

        for i in range(n):
            p   = lasts[i]
            bid = bids[i]
            ask = asks[i]
            t   = times[i]

            ema_fast = alpha_fast * p + (1 - alpha_fast) * ema_fast
            ema_slow = alpha_slow * p + (1 - alpha_slow) * ema_slow

            # ── Manage open position ──────────────────────────────────────
            if open_pos is not None:
                side  = open_pos["side"]
                ep    = open_pos["entry_price"]
                sl    = open_pos["sl"]
                tp    = open_pos["tp"]
                vol   = open_pos["volume"]

                close_reason: Optional[str] = None
                exit_price: float = 0.0

                if side == "BUY":
                    if bid <= sl:
                        close_reason = "SL"
                        exit_price = bid - stop_slip_points
                    elif bid >= tp:
                        close_reason = "TP"
                        exit_price = tp
                    curr_float = (bid - ep) * contract_size * vol
                else:
                    if ask >= sl:
                        close_reason = "SL"
                        exit_price = ask + stop_slip_points
                    elif ask <= tp:
                        close_reason = "TP"
                        exit_price = tp
                    curr_float = (ep - ask) * contract_size * vol

                equity = balance + curr_float
                if equity > peak_eq:
                    peak_eq = equity
                dd = peak_eq - equity
                if dd > max_dd_usd:
                    max_dd_usd = dd

                if close_reason is not None:
                    if side == "BUY":
                        pnl = (exit_price - ep) * contract_size * vol
                    else:
                        pnl = (ep - exit_price) * contract_size * vol
                    balance += pnl
                    equity = balance
                    trades.append(TickTrade(
                        id=int(trade_seq),
                        symbol=self.symbol,
                        side=str(side),
                        entry_time_ms=int(open_pos["entry_time_ms"]),
                        exit_time_ms=int(t),
                        entry_price=round(float(ep), 3),
                        exit_price=round(float(exit_price), 3),
                        volume=round(float(vol), 2),
                        pnl=round(float(pnl), 2),
                        exit_reason=str(close_reason),
                        duration_sec=round(float((t - open_pos["entry_time_ms"]) / 1000.0), 1),
                        ticks_held=int(i - open_pos["entry_tick_idx"]),
                    ))
                    trade_seq += 1
                    open_pos = None

            # ── Generate entry signal (flat only) ─────────────────────────
            if open_pos is None and i > warmup:
                sig = self._signal(strategy_key, i, bids, asks, lasts, ema_fast, ema_slow)

                if sig != 0:
                    # Position sizing from config: risk% of balance / (sl_points * contract)
                    risk_usd = balance * (risk_pct / 100.0)
                    vol = round(
                        max(self._spec.min_lot,
                            min(self._spec.max_lot,
                                risk_usd / (sl_points * contract_size))),
                        2,
                    )
                    # Round to lot step
                    vol = int(vol / self._spec.lot_step) * self._spec.lot_step

                    if sig == 1:
                        ep_   = ask + mkt_slip_points
                        sl_   = ep_ - sl_points
                        tp_   = ep_ + tp_points
                        side_ = "BUY"
                    else:
                        ep_   = bid - mkt_slip_points
                        sl_   = ep_ + sl_points
                        tp_   = ep_ - tp_points
                        side_ = "SELL"

                    open_pos = {
                        "side": side_,
                        "entry_price": ep_,
                        "volume": vol,
                        "sl": sl_,
                        "tp": tp_,
                        "entry_time_ms": t,
                        "entry_tick_idx": i,
                    }

            # Sample equity curve every 200 ticks
            if i % 200 == 0 or i == n - 1:
                ts = datetime.fromtimestamp(t / 1000.0, timezone.utc).strftime("%Y-%m-%d %H:%M")
                equity_curve.append([ts, round(float(equity), 2)])

        # ── Metrics ──────────────────────────────────────────────────────
        pnls    = [tr.pnl for tr in trades]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p < 0]
        total   = sum(pnls)
        wr      = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
        pf      = (sum(wins) / abs(sum(losses))) if losses else (99.0 if wins else 0.0)
        dd_pct  = (max_dd_usd / self.initial_balance) * 100.0

        return {
            "strategy":          str(strategy_key),
            "symbol":            self.symbol,
            "cost_model":        self._cost.symbol,
            "contract_size":     float(contract_size),
            "ticks_replayed":    int(n),
            "total_trades":      int(len(trades)),
            "net_profit_usd":    round(float(total), 2),
            "return_pct":        round(float((total / self.initial_balance) * 100.0), 2),
            "win_rate_pct":      round(float(wr), 1),
            "profit_factor":     round(float(pf), 2),
            "max_drawdown_pct":  round(float(dd_pct), 2),
            "max_drawdown_usd":  round(float(max_dd_usd), 2),
            "avg_trade_sec":     round(float(np.mean([t.duration_sec for t in trades])), 1) if trades else 0.0,
            "avg_ticks_held":    round(float(np.mean([t.ticks_held for t in trades])), 1) if trades else 0.0,
            "equity_curve":      equity_curve,
            "trades":            [asdict(t) for t in trades[-50:]],
        }

    # ------------------------------------------------------------------
    # Strategy signal dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _signal(
        key: str,
        i: int,
        bids: np.ndarray,
        asks: np.ndarray,
        lasts: np.ndarray,
        ema_fast: float,
        ema_slow: float,
    ) -> int:
        """Return +1 (long), -1 (short) or 0 (flat) based on strategy key."""
        p   = lasts[i]
        bid = bids[i]
        ask = asks[i]

        if key in ("ema_cross", "smc_fvg"):
            # EMA crossover with momentum confirmation
            if ema_fast > ema_slow + 0.30 and p > asks[i - 1]:
                return 1
            if ema_fast < ema_slow - 0.30 and p < bids[i - 1]:
                return -1

        elif key == "smc_liquidity_sweep":
            lo = max(0, i - 50)
            recent_high = float(np.max(bids[lo:i]))
            recent_low  = float(np.min(asks[lo:i]))
            # Sweep high then reject → short
            if ask > recent_high + 0.10 and p < recent_high:
                return -1
            # Sweep low then bounce → long
            if bid < recent_low - 0.10 and p > recent_low:
                return 1

        else:
            # Generic EMA cross fallback
            if ema_fast > ema_slow:
                return 1
            if ema_fast < ema_slow:
                return -1

        return 0


# Global singleton
TICK_BACKTEST_ENGINE = TickBacktestEngine()
