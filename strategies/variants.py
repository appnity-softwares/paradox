"""Session-specific U7 campaign candidates.

Rules are documented OBJECTIVELY in docs/strategy_rules_u7.md BEFORE
this code (U7 step 3). Each strategy: few parameters, single entry,
fixed stop-based risk — no martingale / grid / averaging_down.

All three run on the tick engine's M5 bar path (bar_ms = 300_000).
The opening range uses whole M5 bars, so `range_minutes` must be a
multiple of 5.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core.strategy_api import (Context, Signal, Strategy, SIDE_CLOSE,
                               SIDE_LONG, SIDE_SHORT)

BAR_MS = 300_000        # M5


def _wilder_atr_arr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                    period: int) -> float:
    """ATR from the tail of closed-bar arrays (simple mean of TR over
    `period` — deterministic and cheap for small periods)."""
    n = len(closes)
    if n < period + 1:
        return 0.0
    prev = closes[-period - 1:-1]
    tr = np.maximum(highs[-period:] - lows[-period:],
                    np.maximum(np.abs(highs[-period:] - prev),
                               np.abs(lows[-period:] - prev)))
    return float(np.mean(tr))


def _percentile(values: deque, pct: float) -> float:
    if len(values) < 5:
        return 0.0
    return float(np.percentile(np.asarray(values), pct))


class _SessionDayTracker:
    """Groups bars into session days (a gap > 2 hours starts a new day
    because the U2 session prefilter removes out-of-session ticks)."""

    def __init__(self):
        self.day_start_ms: int | None = None
        self.day_index = 0
        self.bars_in_day = 0

    def advance(self, bar_time_ms: int) -> None:
        if self.day_start_ms is None or bar_time_ms - self.day_start_ms > 7_200_000:
            self.day_start_ms = bar_time_ms
            self.day_index += 1
            self.bars_in_day = 0
        self.bars_in_day += 1


class LondonOpeningRange(Strategy):
    """V1: opening-range breakout. See docs/strategy_rules_u7.md."""
    key = "london_opening_range"
    version = "1"
    warmup_bars = 60

    def __init__(self, range_minutes: int = 15, buffer: float = 0.10,
                 stop_buffer_atr: float = 0.25, rr: float = 1.5,
                 max_range_atr: float = 2.0):
        if range_minutes % 5 != 0 or range_minutes < 5:
            raise ValueError("range_minutes must be a positive multiple of 5")
        self.range_minutes = int(range_minutes)
        self.buffer = float(buffer)
        self.stop_buffer_atr = float(stop_buffer_atr)
        self.rr = float(rr)
        self.max_range_atr = float(max_range_atr)
        self.warmup_bars = self.range_minutes // 5 + 20
        self._tracker = _SessionDayTracker()
        self._day_bars: list[tuple[float, float]] = []   # (high, low) today
        self._range_hi: float | None = None
        self._range_lo: float | None = None
        self._day_done: dict[int, set[int]] = {}     # day -> tried directions
        self._day_skipped: set[int] = set()

    def on_bar(self, ctx: Context, bar) -> list[Signal]:
        t = int(bar.name) if hasattr(bar, "name") else int(bar["time_utc_ms"])
        prev_day = self._tracker.day_index
        self._tracker.advance(t)
        day = self._tracker.day_index
        if day != prev_day:
            self._day_bars = []          # new session day
        closes = ctx.bars.closes(30)
        highs = ctx.bars.highs(30)
        lows = ctx.bars.lows(30)
        atr = _wilder_atr_arr(closes, highs, lows, 14)
        self._day_bars.append((float(bar["high"]), float(bar["low"])))

        # Build the range from the first k bars of THIS day only.
        k = self.range_minutes // 5
        if len(self._day_bars) <= k:
            if len(self._day_bars) == k:
                seg_hi = float(np.max([h for h, _ in self._day_bars]))
                seg_lo = float(np.min([l for _, l in self._day_bars]))
                self._range_hi, self._range_lo = seg_hi, seg_lo
                if atr > 0 and (seg_hi - seg_lo) > self.max_range_atr * atr:
                    self._day_skipped.add(day)
            return []

        if day in self._day_skipped or ctx.position.side != 0:
            return []
        tried = self._day_done.setdefault(day, set())
        if self._range_hi is None or atr <= 0:
            return []
        close = float(bar["close"])
        out: list[Signal] = []
        if close > self._range_hi + self.buffer and SIDE_LONG not in tried:
            risk = close - (self._range_lo - self.stop_buffer_atr * atr)
            if risk > 0:
                tried.add(SIDE_LONG)
                out.append(Signal(SIDE_LONG, sl=self._range_lo
                                  - self.stop_buffer_atr * atr,
                                  tp=close + self.rr * risk,
                                  tag="orb_long"))
        elif close < self._range_lo - self.buffer and SIDE_SHORT not in tried:
            risk = (self._range_hi + self.stop_buffer_atr * atr) - close
            if risk > 0:
                tried.add(SIDE_SHORT)
                out.append(Signal(SIDE_SHORT, sl=self._range_hi
                                  + self.stop_buffer_atr * atr,
                                  tp=close - self.rr * risk,
                                  tag="orb_short"))
        return out


class TrendPullback(Strategy):
    """V2: pullback in trend. See docs/strategy_rules_u7.md."""
    key = "trend_pullback"
    version = "1"
    warmup_bars = 260

    def __init__(self, fast: int = 50, slow: int = 200, atr_mult: float = 1.5,
                 rr: float = 1.5, touch_bps: float = 5.0,
                 dead_market_pct: float = 20.0):
        if fast >= slow:
            raise ValueError("fast EMA must be below slow EMA")
        self.fast, self.slow = int(fast), int(slow)
        self.warmup_bars = self.slow + 20
        self.atr_mult = float(atr_mult)
        self.rr = float(rr)
        self.touch_bps = float(touch_bps)
        self.dead_market_pct = float(dead_market_pct)
        self._tracker = _SessionDayTracker()
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._atr_hist: deque = deque(maxlen=240)   # ~5 sessions of M5
        self._day_done: dict[int, set[int]] = {}
        self._count = 0

    def on_bar(self, ctx: Context, bar) -> list[Signal]:
        t = int(bar.name) if hasattr(bar, "name") else int(bar["time_utc_ms"])
        self._tracker.advance(t)
        self._count += 1
        close = float(bar["close"])
        alpha_f = 2.0 / (self.fast + 1.0)
        alpha_s = 2.0 / (self.slow + 1.0)
        self._ema_fast = close if self._ema_fast is None else \
            alpha_f * close + (1 - alpha_f) * self._ema_fast
        self._ema_slow = close if self._ema_slow is None else \
            alpha_s * close + (1 - alpha_s) * self._ema_slow
        atr = _wilder_atr_arr(ctx.bars.closes(30), ctx.bars.highs(30),
                              ctx.bars.lows(30), 14)
        if atr > 0:
            self._atr_hist.append(atr)
        if self._count <= self.slow + 10 or atr <= 0:
            return []
        # Dead-market filter: ATR below its trailing 20th percentile.
        if atr < _percentile(self._atr_hist, self.dead_market_pct):
            return []
        day = self._tracker.day_index
        if ctx.position.side != 0:
            return []
        tried = self._day_done.setdefault(day, set())
        lo = float(bar["low"])
        hi = float(bar["high"])
        tol = close * self.touch_bps / 10_000.0
        out: list[Signal] = []
        uptrend = self._ema_fast is not None and self._ema_fast > self._ema_slow
        if (uptrend and SIDE_LONG not in tried
                and abs(lo - self._ema_fast) <= tol and close > self._ema_fast):
            sl = lo - self.atr_mult * atr
            risk = close - sl
            if risk > 0:
                tried.add(SIDE_LONG)
                out.append(Signal(SIDE_LONG, sl=sl,
                                  tp=close + self.rr * risk,
                                  tag="pullback_long"))
        elif (not uptrend and SIDE_SHORT not in tried
                and abs(hi - self._ema_fast) <= tol and close < self._ema_fast):
            sl = hi + self.atr_mult * atr
            risk = sl - close
            if risk > 0:
                tried.add(SIDE_SHORT)
                out.append(Signal(SIDE_SHORT, sl=sl,
                                  tp=close - self.rr * risk,
                                  tag="pullback_short"))
        return out


class EmaCrossVolFilter(Strategy):
    """V3: ema_cross signal logic + volatility gate. See
    docs/strategy_rules_u7.md (ATR(14) on M15 built from M5 bars,
    percentile over the trailing 10 sessions)."""
    key = "ema_cross_vol_filter"
    version = "1"
    warmup_bars = 320

    def __init__(self, fast: int = 20, slow: int = 50, atr_pct: float = 50.0):
        if fast >= slow:
            raise ValueError("fast EMA must be below slow EMA")
        self.fast, self.slow = int(fast), int(slow)
        self.warmup_bars = self.slow + 20
        self.atr_pct = float(atr_pct)
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._prev_dir = 0
        self._m15_count = 0
        self._m15_close: float | None = None
        self._atr_hist: deque = deque(maxlen=480)   # 10 sessions of M15
        self._m5_in_m15 = 0

    def on_bar(self, ctx: Context, bar) -> list[Signal]:
        close = float(bar["close"])
        alpha_f = 2.0 / (self.fast + 1.0)
        alpha_s = 2.0 / (self.slow + 1.0)
        self._ema_fast = close if self._ema_fast is None else \
            alpha_f * close + (1 - alpha_f) * self._ema_fast
        self._ema_slow = close if self._ema_slow is None else \
            alpha_s * close + (1 - alpha_s) * self._ema_slow
        # M15 resample: every 3rd M5 bar closes an M15 bar.
        self._m5_in_m15 += 1
        if self._m5_in_m15 == 3:
            self._m5_in_m15 = 0
            self._m15_count += 1
            atr15 = self._atr15(ctx)
            if atr15 > 0:
                self._atr_hist.append(atr15)
        if self._ema_fast is None or self._ema_slow is None:
            return []
        dir_now = 1 if self._ema_fast > self._ema_slow else \
            (-1 if self._ema_fast < self._ema_slow else 0)
        out: list[Signal] = []
        if dir_now != self._prev_dir and dir_now != 0:
            # Volatility gate: ATR above its trailing percentile.
            atr_now = self._atr_hist[-1] if self._atr_hist else 0.0
            floor = _percentile(self._atr_hist, self.atr_pct)
            gate_ok = atr_now > 0 and atr_now >= floor
            if gate_ok:
                if ctx.position.side == -dir_now:
                    out.append(Signal(SIDE_CLOSE, tag="flip_close"))
                elif ctx.position.side == 0:
                    sl_dist = max(1.5 * atr_now, 0.5)
                    if dir_now == 1:
                        out.append(Signal(SIDE_LONG, sl=close - sl_dist,
                                          tp=close + 1.5 * sl_dist,
                                          tag="ema_vol_long"))
                    else:
                        out.append(Signal(SIDE_SHORT, sl=close + sl_dist,
                                          tp=close - 1.5 * sl_dist,
                                          tag="ema_vol_short"))
            self._prev_dir = dir_now
        return out

    def _atr15(self, ctx: Context) -> float:
        closes = ctx.bars.closes(150)       # 150 M5 bars -> 50 M15 bars
        highs = ctx.bars.highs(150)
        lows = ctx.bars.lows(150)
        if len(closes) < 45:
            return 0.0
        # Take every 3rd bar (the M15 boundary bars) as a proxy series.
        c = closes[2::3]
        h = highs[2::3]
        l = lows[2::3]
        return _wilder_atr_arr(c, h, l, 14)


# ---------------------------------------------------------------------------
# Round 2 candidates (rules documented FIRST in docs/strategy_rules_u7.md)
# ---------------------------------------------------------------------------

class AsiaRangeBreakout(Strategy):
    """V4: Asia-range breakout traded in the London window.

    Runs on an EXTENDED data window (00:00-11:00 UTC) so the strategy
    can see the Asia range; entries are gated to >= trade_start_min
    (07:00 UTC). See docs/strategy_rules_u7.md V4."""
    key = "asia_range_breakout"
    version = "1"
    warmup_bars = 80

    def __init__(self, buffer: float = 0.10, rr: float = 1.5,
                 max_range_atr: float = 2.0, trade_start_min: int = 420):
        self.buffer = float(buffer)
        self.rr = float(rr)
        self.max_range_atr = float(max_range_atr)
        self.trade_start_min = int(trade_start_min)
        self._tracker = _SessionDayTracker()
        self._asia: dict[int, tuple[float, float]] = {}   # day -> (hi, lo)
        self._day_done: dict[int, set[int]] = {}
        self._day_skipped: set[int] = set()
        self._bars_today: list[tuple[float, float, float, int]] = []

    def on_bar(self, ctx: Context, bar) -> list[Signal]:
        t = int(bar.name) if hasattr(bar, "name") else int(bar["time_utc_ms"])
        prev_day = self._tracker.day_index
        self._tracker.advance(t)
        day = self._tracker.day_index
        if day != prev_day:
            self._bars_today = []
        minute_of_day = (t // 60_000) % 1440
        hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        # Accumulate Asia-range bars (before the trade window).
        if minute_of_day < self.trade_start_min:
            self._bars_today.append((hi, lo, close, minute_of_day))
            return []
        # Asia range must be complete (we saw bars up to 06:00 UTC).
        if not self._bars_today or self._bars_today[-1][3] < 330:
            return []                      # no Asia data for this day
        if day not in self._asia:
            asia_hi = max(b[0] for b in self._bars_today)
            asia_lo = min(b[1] for b in self._bars_today)
            self._asia[day] = (asia_hi, asia_lo)
            atr = _wilder_atr_arr(ctx.bars.closes(30), ctx.bars.highs(30),
                                  ctx.bars.lows(30), 14)
            if atr > 0 and (asia_hi - asia_lo) > self.max_range_atr * atr:
                self._day_skipped.add(day)
        if day in self._day_skipped or ctx.position.side != 0:
            return []
        tried = self._day_done.setdefault(day, set())
        asia_hi, asia_lo = self._asia[day]
        out: list[Signal] = []
        if close > asia_hi + self.buffer and SIDE_LONG not in tried:
            risk = close - asia_lo
            if risk > 0:
                tried.add(SIDE_LONG)
                out.append(Signal(SIDE_LONG, sl=asia_lo,
                                  tp=close + self.rr * risk,
                                  tag="asia_long"))
        elif close < asia_lo - self.buffer and SIDE_SHORT not in tried:
            risk = asia_hi - close
            if risk > 0:
                tried.add(SIDE_SHORT)
                out.append(Signal(SIDE_SHORT, sl=asia_hi,
                                  tp=close - self.rr * risk,
                                  tag="asia_short"))
        return out


class SqueezeBreakout(Strategy):
    """V5: volatility-compression breakout. See docs V5."""
    key = "squeeze_breakout"
    version = "1"
    warmup_bars = 220

    def __init__(self, period: int = 20, squeeze_pct: float = 30.0,
                 atr_mult: float = 1.5, rr: float = 1.5):
        self.period = int(period)
        self.squeeze_pct = float(squeeze_pct)
        self.atr_mult = float(atr_mult)
        self.rr = float(rr)
        self._tracker = _SessionDayTracker()
        self._widths: deque = deque(maxlen=200)
        self._day_done: dict[int, set[int]] = {}

    def on_bar(self, ctx: Context, bar) -> list[Signal]:
        t = int(bar.name) if hasattr(bar, "name") else int(bar["time_utc_ms"])
        self._tracker.advance(t)
        closes = ctx.bars.closes(self.period + 1)
        if len(closes) < self.period + 1:
            return []
        ma = float(np.mean(closes[-self.period:]))
        sd = float(np.std(closes[-self.period:]))
        width = 4.0 * sd
        atr = _wilder_atr_arr(ctx.bars.closes(30), ctx.bars.highs(30),
                              ctx.bars.lows(30), 14)
        if atr <= 0:
            return []
        floor = _percentile(self._widths, self.squeeze_pct)
        was_squeezed = len(self._widths) >= 50 and width <= floor
        self._widths.append(width)
        close = float(bar["close"])
        upper, lower = ma + 2 * sd, ma - 2 * sd
        day = self._tracker.day_index
        if ctx.position.side != 0 or not was_squeezed:
            return []
        tried = self._day_done.setdefault(day, set())
        out: list[Signal] = []
        if close > upper and SIDE_LONG not in tried:
            sl = close - self.atr_mult * atr
            risk = close - sl
            tried.add(SIDE_LONG)
            out.append(Signal(SIDE_LONG, sl=sl, tp=close + self.rr * risk,
                              tag="squeeze_long"))
        elif close < lower and SIDE_SHORT not in tried:
            sl = close + self.atr_mult * atr
            risk = sl - close
            tried.add(SIDE_SHORT)
            out.append(Signal(SIDE_SHORT, sl=sl, tp=close - self.rr * risk,
                              tag="squeeze_short"))
        return out


class FirstMoveContinuation(Strategy):
    """V6: first-30-minutes direction continuation. See docs V6."""
    key = "first_move_continuation"
    version = "1"
    warmup_bars = 40

    def __init__(self, open_minutes: int = 30, min_move_atr: float = 0.5,
                 buffer: float = 0.05, rr: float = 1.5,
                 max_range_atr: float = 2.5):
        if open_minutes % 5 != 0:
            raise ValueError("open_minutes must be a multiple of 5")
        self.open_minutes = int(open_minutes)
        self.min_move_atr = float(min_move_atr)
        self.buffer = float(buffer)
        self.rr = float(rr)
        self.max_range_atr = float(max_range_atr)
        self._tracker = _SessionDayTracker()
        self._day_done: set[int] = set()
        self._open_bars: list[tuple[float, float, float]] = []
        self._decided: dict[int, int] = {}      # day -> direction (+1/-1)

    def on_bar(self, ctx: Context, bar) -> list[Signal]:
        t = int(bar.name) if hasattr(bar, "name") else int(bar["time_utc_ms"])
        prev_day = self._tracker.day_index
        self._tracker.advance(t)
        day = self._tracker.day_index
        if day != prev_day:
            self._open_bars = []
        k = self.open_minutes // 5
        hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        if len(self._open_bars) < k:
            self._open_bars.append((hi, lo, close))
            if len(self._open_bars) == k:
                o_hi = max(b[0] for b in self._open_bars)
                o_lo = min(b[1] for b in self._open_bars)
                o_close = self._open_bars[-1][2]
                o_open = self._open_bars[0][2]
                atr = _wilder_atr_arr(ctx.bars.closes(30), ctx.bars.highs(30),
                                      ctx.bars.lows(30), 14)
                if atr <= 0 or (o_hi - o_lo) > self.max_range_atr * atr:
                    return []
                move = o_close - o_open
                if abs(move) >= self.min_move_atr * atr:
                    self._decided[day] = 1 if move > 0 else -1
            return []
        direction = self._decided.get(day, 0)
        if direction == 0 or day in self._day_done or ctx.position.side != 0:
            return []
        o_hi = max(b[0] for b in self._open_bars)
        o_lo = min(b[1] for b in self._open_bars)
        out: list[Signal] = []
        if direction == 1 and close > o_hi + self.buffer:
            sl = o_lo
            risk = close - sl
            if risk > 0:
                self._day_done.add(day)
                out.append(Signal(SIDE_LONG, sl=sl,
                                  tp=close + self.rr * risk,
                                  tag="fm_long"))
        elif direction == -1 and close < o_lo - self.buffer:
            sl = o_hi
            risk = sl - close
            if risk > 0:
                self._day_done.add(day)
                out.append(Signal(SIDE_SHORT, sl=sl,
                                  tp=close - self.rr * risk,
                                  tag="fm_short"))
        return out
