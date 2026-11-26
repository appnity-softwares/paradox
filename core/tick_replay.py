"""Tick-by-tick replay backtester over the FULL recorded tick archive.

Reads data/ticks/XAUUSD/<YYYY>/<MM>/ticks.parquet (time_utc_ms, bid, ask,
volumes, source) — ~207M real recorded ticks, Jan 2024 onward — and fills
every entry, stop, target and signal exit at the actual bid/ask of the first
tick that satisfies the condition.

This is the honesty upgrade over core/backtest.py, which must approximate
intrabar fills from bar high/low. Here:
- Entries fill at the first tick at/after the signal bar's open: buy at ask,
  sell at bid. The spread paid is the spread that was really recorded.
- A stop fills at the tick's price — a gap through the stop fills worse.
  No invented mercy.
- A target is a limit: it fills at the target or better, never worse.
- If the same tick touches both stop and target, the STOP wins
  (conservative; same rule as the bar engine).
- Bars whose window has no recorded ticks cannot fill anything; every skip
  is counted and reported in the coverage stats.

Signals come from the SAME strategy interface as the bar engine
(strategy.generate_signals(df)) computed on the bar series, so a bar
backtest and its tick twin differ ONLY in fill realism — never in signals.

Scaling: months are streamed one parquet file at a time and scans inside a
window are vectorized numpy searches, so multi-year runs stay practical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtest import _validate_data, opening_range_mid
from core.config import get_cost_model, get_symbol_spec

DEFAULT_TICK_DIR = Path("data/ticks/XAUUSD")


@dataclass
class TickCoverage:
    files: int = 0
    ticks_scanned: int = 0
    span_start: str | None = None
    span_end: str | None = None
    bars_without_ticks: int = 0
    entries_skipped_no_tick: int = 0

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "ticks_scanned": self.ticks_scanned,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "bars_without_ticks": self.bars_without_ticks,
            "entries_skipped_no_tick": self.entries_skipped_no_tick,
        }


def month_files(tick_dir: Path) -> list[Path]:
    if not tick_dir.exists():
        return []
    return sorted(tick_dir.glob("*/*/ticks.parquet"))


def _load_chunk(path: Path):
    """(t_ms int64, bid, ask) for one month file, ascending by time."""
    df = pd.read_parquet(path, columns=["time_utc_ms", "bid", "ask"])
    df = df.dropna(subset=["bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    if df.empty:
        return None
    order = np.argsort(df["time_utc_ms"].to_numpy(dtype=np.int64), kind="stable")
    return (
        df["time_utc_ms"].to_numpy(dtype=np.int64)[order],
        df["bid"].to_numpy(dtype=float)[order],
        df["ask"].to_numpy(dtype=float)[order],
    )


def run_tick_replay(
    df: pd.DataFrame,
    strategy,
    cfg,
    start_equity: float = 10_000.0,
    symbol: str = "XAUUSD",
    tick_dir: str | Path = DEFAULT_TICK_DIR,
    slippage: float = 0.0,
    start_trading_at=None,
    spread_ceiling: float | None = None,
) -> dict:
    """Replay the strategy's bar signals against recorded ticks.

    spread_ceiling: optional sanity guard in price units. A tick whose spread
    exceeds it is treated as a data glitch — entries wait for a sane tick.
    None = trust every recorded tick.
    """
    _validate_data(df)
    if start_equity <= 0:
        raise ValueError("start_equity must be positive")
    if slippage < 0:
        raise ValueError("slippage must be non-negative")

    spec = get_symbol_spec(symbol)
    costs = get_cost_model(symbol)
    # Real ticks carry the real spread; only commission and slippage come
    # from the cost model. Commission is USD per standard lot per side,
    # converted here to USD per oz.
    commission_per_oz = costs.commission_per_lot / spec.contract_size

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("tick replay needs a DatetimeIndex")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    index = df.index.tz_convert("UTC")

    signals = strategy.generate_signals(df)
    if not signals.index.equals(df.index):
        signals = signals.reindex(df.index)
    sig_np = signals.fillna(0.0).to_numpy(dtype=float)

    from core.indicators import atr
    atr_np = atr(df, 14).to_numpy(dtype=float)
    or_mid_np = opening_range_mid(df, cfg.or_bars) if cfg.sl_method == "or_mid" else None
    n = len(df)
    if n < 3:
        raise ValueError("need at least 3 bars")

    # Bar i's decision window is [t_open(i), t_open(i+1)); the signal that
    # acts there came from bar i-1 (same semantics as the bar engine).
    # as_unit("ms") normalises whatever resolution pandas chose on load
    # (ns / us / s) so the values are true epoch milliseconds.
    index = index.as_unit("ms")
    t_open_ms = index.asi8
    med_ms = int(np.median(np.diff(t_open_ms))) if n >= 2 else 900_000
    t_end_ms = int(t_open_ms[-1] + med_ms)

    files = month_files(Path(tick_dir))
    cov = TickCoverage(files=len(files))

    equity = float(start_equity)
    pos = 0
    units = 0.0
    entry = sl = tp = 0.0
    entry_fee = 0.0
    entry_time = None
    side_label = ""
    trades: list[dict] = []
    equity_curve = np.full(n, start_equity, dtype=float)

    trades_today = 0
    day_key = None
    day_start_equity = equity
    blocked_day = False

    trade_start = None
    if start_trading_at is not None:
        trade_start = pd.Timestamp(start_trading_at)
        if trade_start.tzinfo is None:
            trade_start = trade_start.tz_localize("UTC")
        trade_start = trade_start.tz_convert("UTC")

    state = {"fi": 0, "chunk": None, "cur": 0}

    def _next_chunk() -> bool:
        while state["fi"] < len(files):
            c = _load_chunk(files[state["fi"]])
            state["fi"] += 1
            if c is not None and len(c[0]):
                state["chunk"] = c
                state["cur"] = 0
                if cov.span_start is None:
                    cov.span_start = pd.Timestamp(int(c[0][0]), unit="ms", tz="UTC").isoformat()
                cov.span_end = pd.Timestamp(int(c[0][-1]), unit="ms", tz="UTC").isoformat()
                return True
        state["chunk"] = None
        return False

    def _ticks_in_window(w_start: int, w_end: int):
        """Advance the cursor and return (t, bid, ask) inside [w_start, w_end).

        A chunk that lies entirely before the window is skipped forward; a
        chunk that straddles the window but has no ticks inside it means a
        real gap in the recording (weekend, holiday) and returns None.
        """
        while True:
            if state["chunk"] is None and not _next_chunk():
                return None
            t = state["chunk"][0]
            cur = max(state["cur"], int(np.searchsorted(t, w_start, side="left")))
            state["cur"] = cur
            if cur < len(t) and t[cur] < w_end:
                j = int(np.searchsorted(t, w_end, side="left"))
                out = (t[cur:j], state["chunk"][1][cur:j], state["chunk"][2][cur:j])
                state["cur"] = j
                cov.ticks_scanned += len(out[0])
                return out
            if cur >= len(t):
                # chunk exhausted before the window: move to the next month
                if not _next_chunk():
                    return None
                continue
            # t[cur] >= w_end inside a chunk that reaches past the window:
            # the window genuinely has no recorded ticks
            return None

    def _close(exit_price: float, ts, reason: str) -> None:
        nonlocal equity, pos, units, entry_fee
        exit_fee = abs(units) * commission_per_oz
        pnl = (exit_price - entry) * units * pos - entry_fee - exit_fee
        equity += pnl
        trades.append({
            "entry_time": entry_time,
            "exit_time": ts,
            "side": side_label,
            "entry": round(float(entry), 5),
            "exit": round(float(exit_price), 5),
            "units": round(float(units), 4),
            "pnl": round(float(pnl), 2),
            "reason": reason,
            "equity_after": round(equity, 2),
        })
        pos = 0
        units = 0.0
        entry_fee = 0.0

    for i in range(1, n):
        w_start = int(t_open_ms[i])
        w_end = int(t_open_ms[i + 1]) if i + 1 < n else t_end_ms

        day = index[i].date()
        if day != day_key:
            day_key = day
            trades_today = 0
            day_start_equity = equity
            blocked_day = False

        sig = sig_np[i - 1]
        win = _ticks_in_window(w_start, w_end)
        if win is not None and len(win[0]):
            t_w, bid_w, ask_w = win
            first_mid = (bid_w[0] + ask_w[0]) / 2.0
        else:
            t_w = bid_w = ask_w = None
            first_mid = None
            cov.bars_without_ticks += 1

        # 1) signal-flip exit at the window's first tick
        if pos != 0 and t_w is not None and (
            (pos == 1 and sig <= 0) or (pos == -1 and sig >= 0)
        ):
            px = bid_w[0] if pos == 1 else ask_w[0]
            _close(px, index[i], "Signal exit")

        # 2) entry at the window's first tick
        if (
            pos == 0 and sig != 0 and t_w is not None
            and not blocked_day and trades_today < cfg.max_trades_per_day
            and (trade_start is None or index[i] >= trade_start)
        ):
            direction = 1 if sig > 0 else -1
            sane = spread_ceiling is None or (ask_w[0] - bid_w[0]) <= spread_ceiling
            if not sane:
                cov.entries_skipped_no_tick += 1
            else:
                raw_fill = ask_w[0] if direction == 1 else bid_w[0]
                if cfg.sl_method == "or_mid":
                    om = or_mid_np[i]
                    # range still forming, or the structural stop sits on the
                    # wrong side of the real fill: no trade
                    if np.isfinite(om) and (
                        (direction == 1 and om < raw_fill) or (direction == -1 and om > raw_fill)
                    ):
                        sl_dist = abs(raw_fill - om)
                        sl = om
                    else:
                        sl_dist = None
                else:
                    atr_value = atr_np[i - 1]
                    if cfg.sl_method == "atr" and np.isfinite(atr_value):
                        sl_dist = cfg.sl_atr_mult * atr_value
                    else:
                        sl_dist = cfg.sl_fixed
                    sl_dist = max(float(sl_dist), 1e-9)
                    sl = raw_fill - direction * sl_dist
                if sl_dist is not None:
                    fill = raw_fill + direction * slippage
                    risk_amount = equity * cfg.risk_pct / 100.0
                    units = risk_amount / sl_dist
                    pos = direction
                    side_label = "LONG" if direction == 1 else "SHORT"
                    entry = fill
                    tp = entry + direction * sl_dist * cfg.rr
                    entry_time = index[i]
                    entry_fee = abs(units) * commission_per_oz
                    trades_today += 1

        # 3) stop / target scan over the window's ticks (vectorized)
        if pos != 0 and t_w is not None and len(t_w):
            if pos == 1:
                stop_hits = np.nonzero(bid_w <= sl)[0]
                tp_hits = np.nonzero(bid_w >= tp)[0]
                stop_px = bid_w[stop_hits[0]] if len(stop_hits) else None
                tp_px = max(tp, bid_w[tp_hits[0]]) if len(tp_hits) else None
            else:
                stop_hits = np.nonzero(ask_w >= sl)[0]
                tp_hits = np.nonzero(ask_w <= tp)[0]
                stop_px = ask_w[stop_hits[0]] if len(stop_hits) else None
                tp_px = min(tp, ask_w[tp_hits[0]]) if len(tp_hits) else None
            si = int(stop_hits[0]) if len(stop_hits) else None
            ti = int(tp_hits[0]) if len(tp_hits) else None
            if si is not None and (ti is None or si <= ti):
                _close(stop_px, pd.Timestamp(int(t_w[si]), unit="ms", tz="UTC"), "SL")
            elif ti is not None:
                _close(tp_px, pd.Timestamp(int(t_w[ti]), unit="ms", tz="UTC"), "TP")

        # 4) daily loss limit on mark-to-market equity
        if first_mid is not None:
            floating = 0.0 if pos == 0 else (first_mid - entry) * units * pos - entry_fee
            mtm = equity + floating
            if day_start_equity > 0 and (
                (mtm - day_start_equity) / day_start_equity * 100.0 <= -cfg.daily_loss_limit_pct
            ):
                blocked_day = True
                if pos != 0:
                    _close(first_mid, index[i], "Daily loss limit")

        equity_curve[i] = equity

    if pos != 0:
        last_mid = None
        if state["chunk"] is not None and state["cur"] > 0:
            c = state["chunk"]
            last_mid = (c[1][state["cur"] - 1] + c[2][state["cur"] - 1]) / 2.0
        if last_mid is None:
            last_mid = float(df["close"].iloc[-1])
        _close(last_mid, index[-1], "End of data")

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(((peak - equity_curve) / peak).max()) if len(equity_curve) else 0.0

    return {
        "engine": "tick_replay",
        "symbol": symbol,
        "start_equity": start_equity,
        "end_equity": round(float(equity), 2),
        "total_pnl": round(float(equity - start_equity), 2),
        "return_pct": round(float((equity - start_equity) / start_equity * 100.0), 3),
        "num_trades": len(trades),
        "win_rate_pct": round(float(100.0 * len(wins) / len(trades)), 2) if trades else 0.0,
        "profit_factor": round(float(gross_win / gross_loss), 3) if gross_loss > 0 else None,
        "max_drawdown_pct": round(float(max_dd * 100.0), 3),
        "trades": trades,
        "equity_curve": [round(float(x), 2) for x in equity_curve],
        "coverage": cov.to_dict(),
    }
