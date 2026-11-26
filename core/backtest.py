"""Deterministic bar-by-bar backtester.

Event order for each bar is intentionally explicit:

1. Previous-bar signal is acted on at the current open.
2. Gap-through stops/targets and signal exits are handled at the open.
3. New entries are filled at the open.
4. The current bar's high/low evaluates SL/TP for every position, including a
   position opened on that same bar.

This prevents the earlier time-travel bug where a trade entered at the open
was not exposed to that bar's range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import DEFAULT_SYMBOL, get_cost_model
from core.indicators import atr


REQUIRED_COLUMNS = ("open", "high", "low", "close")


def opening_range_mid(df: pd.DataFrame, or_bars: int) -> np.ndarray:
    """Per-bar opening-range midpoint: (max high, min low) of the first
    `or_bars` bars of each session day, broadcast over the whole day.
    NaN while the range is still forming. Days are calendar days (UTC)."""
    idx = df.index
    day = pd.Series(idx.normalize().values, index=idx)
    rank = day.groupby(day).cumcount()
    oh = df["high"].where(rank < or_bars).groupby(day).transform("max")
    ol = df["low"].where(rank < or_bars).groupby(day).transform("min")
    mid = (oh + ol) / 2.0
    # while the range is still forming there is no completed structure
    mid = mid.where(rank >= or_bars)
    return mid.to_numpy(dtype=float)


def _validate_data(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"market data missing columns: {missing}")
    if len(df) < 2:
        raise ValueError("backtest requires at least two bars")
    if not df.index.is_monotonic_increasing:
        raise ValueError("market data index must be sorted ascending")
    values = df[list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("market data contains NaN or infinite OHLC values")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("high is below open/close on one or more bars")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("low is above open/close on one or more bars")


def run_backtest(
    df: pd.DataFrame,
    strategy,
    cfg,
    start_equity: float = 10_000.0,
    spread: float | None = None,
    slippage: float | None = None,
    commission_per_unit: float | None = None,
    close_open_position: bool = True,
    start_trading_at=None,
    symbol: str = DEFAULT_SYMBOL,
):
    _validate_data(df)
    # Costs come from the typed CostModel, never from literals in this engine.
    costs = get_cost_model(symbol)
    spread = costs.spread_price() if spread is None else spread
    slippage = costs.slippage.market_points if slippage is None else slippage
    commission_per_unit = costs.commission_per_lot if commission_per_unit is None else commission_per_unit
    if start_equity <= 0:
        raise ValueError("start_equity must be positive")
    if spread < 0 or slippage < 0 or commission_per_unit < 0:
        raise ValueError("spread, slippage and commission must be non-negative")

    signals = strategy.generate_signals(df)
    if not signals.index.equals(df.index):
        signals = signals.reindex(df.index)
    sig_np = signals.fillna(0.0).to_numpy(dtype=float)
    atr_np = atr(df, 14).to_numpy(dtype=float)
    or_mid_np = opening_range_mid(df, cfg.or_bars) if cfg.sl_method == "or_mid" else None

    index = df.index
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    is_dt = isinstance(index, pd.DatetimeIndex)
    hours = index.hour.to_numpy(dtype=int) if is_dt else np.full(n, -1)
    if is_dt:
        day_keys = index.normalize().asi8
        day_change = np.empty(n, dtype=bool)
        day_change[0] = True
        day_change[1:] = day_keys[1:] != day_keys[:-1]
    else:
        day_change = np.ones(n, dtype=bool)

    trade_start = pd.Timestamp(start_trading_at) if start_trading_at is not None else None
    if trade_start is not None and is_dt:
        if trade_start.tzinfo is None and index.tz is not None:
            trade_start = trade_start.tz_localize(index.tz)
        elif trade_start.tzinfo is not None and index.tz is not None:
            trade_start = trade_start.tz_convert(index.tz)

    equity = float(start_equity)
    pos = 0
    units = 0.0
    entry = sl = tp = 0.0
    entry_time = None
    side_label = ""
    entry_fee = 0.0

    trades: list[dict] = []
    equity_curve = np.full(n, start_equity, dtype=float)
    trades_today = 0
    day_start_equity = equity
    blocked_day = False

    def floating_at(price: float) -> float:
        if pos == 0:
            return 0.0
        return (price - entry) * units * pos - entry_fee

    def do_close(exit_price: float, i: int, reason: str) -> None:
        nonlocal equity, pos, units, entry_fee
        exit_fee = abs(units) * commission_per_unit
        pnl = (exit_price - entry) * units * pos - entry_fee - exit_fee
        equity += pnl
        trades.append({
            "entry_time": entry_time,
            "exit_time": index[i],
            "side": side_label,
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "exit": round(exit_price, 5),
            "units": round(units, 4),
            "pnl": round(pnl, 2),
            "reason": reason,
            "equity_after": round(equity, 2),
        })
        pos = 0
        units = 0.0
        entry_fee = 0.0

    for i in range(1, n):
        # A prop-style day starts from mark-to-market equity at the new day's open.
        if day_change[i]:
            day_start_equity = equity + floating_at(o[i])
            trades_today = 0
            blocked_day = False

        signal = sig_np[i - 1]
        exited_on_gap = False

        # 1) Handle gaps for positions carried from the previous bar.
        if pos == 1:
            if o[i] <= sl:
                do_close(o[i] - slippage, i, "SL gap")
                exited_on_gap = True
            elif o[i] >= tp:
                do_close(tp - slippage, i, "TP gap")
                exited_on_gap = True
        elif pos == -1:
            if o[i] >= sl:
                do_close(o[i] + slippage, i, "SL gap")
                exited_on_gap = True
            elif o[i] <= tp:
                do_close(tp + slippage, i, "TP gap")
                exited_on_gap = True

        # 2) Exit/reverse at the current open using the previous close's signal.
        if pos == 1 and signal <= 0:
            do_close(o[i] - spread / 2 - slippage, i, "Signal exit")
        elif pos == -1 and signal >= 0:
            do_close(o[i] + spread / 2 + slippage, i, "Signal exit")

        current_equity_open = equity + floating_at(o[i])
        if day_start_equity > 0:
            daily_return = (current_equity_open - day_start_equity) / day_start_equity * 100.0
            blocked_day = blocked_day or daily_return <= -cfg.daily_loss_limit_pct

        can_trade_time = trade_start is None or index[i] >= trade_start
        in_session = cfg.session_hours is None or (
            is_dt and cfg.session_hours[0] <= hours[i] < cfg.session_hours[1]
        )

        # 3) New entry at current open. A gap stop does not immediately re-enter.
        if (
            pos == 0
            and signal != 0
            and not exited_on_gap
            and can_trade_time
            and in_session
            and not blocked_day
            and trades_today < cfg.max_trades_per_day
        ):
            direction = 1 if signal > 0 else -1
            if cfg.sl_method == "or_mid":
                om = or_mid_np[i]
                if not np.isfinite(om):
                    continue   # opening range still forming: no structural stop
                # the stop is where the setup failed; if the range midpoint is
                # on the wrong side of the entry there is no trade
                if (direction == 1 and om >= o[i]) or (direction == -1 and om <= o[i]):
                    continue
                sl_dist = abs(o[i] - om)
            else:
                atr_value = atr_np[i - 1]
                if cfg.sl_method == "atr" and np.isfinite(atr_value):
                    sl_dist = cfg.sl_atr_mult * atr_value
                else:
                    sl_dist = cfg.sl_fixed
            sl_dist = max(float(sl_dist), 1e-9)

            fill = o[i] + direction * (spread / 2 + slippage)
            risk_amount = equity * cfg.risk_pct / 100.0
            units = risk_amount / sl_dist
            pos = direction
            side_label = "LONG" if direction == 1 else "SHORT"
            entry = fill
            sl = entry - direction * sl_dist
            tp = entry + direction * sl_dist * cfg.rr
            entry_time = index[i]
            entry_fee = abs(units) * commission_per_unit
            trades_today += 1

        # 4) Current bar range applies to carried and newly opened positions.
        if pos == 1:
            if l[i] <= sl:
                do_close(sl - slippage, i, "SL")
            elif h[i] >= tp:
                do_close(tp - slippage, i, "TP")
        elif pos == -1:
            if h[i] >= sl:
                do_close(sl + slippage, i, "SL")
            elif l[i] <= tp:
                do_close(tp + slippage, i, "TP")

        current_equity_close = equity + floating_at(c[i])
        if day_start_equity > 0:
            daily_return = (current_equity_close - day_start_equity) / day_start_equity * 100.0
            blocked_day = blocked_day or daily_return <= -cfg.daily_loss_limit_pct
        equity_curve[i] = current_equity_close

    if pos != 0 and close_open_position:
        final_exit = c[-1] - spread / 2 - slippage if pos == 1 else c[-1] + spread / 2 + slippage
        do_close(final_exit, n - 1, "End of data")
        equity_curve[-1] = equity

    return pd.Series(equity_curve, index=index), pd.DataFrame(trades)
