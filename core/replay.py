"""Time Machine — Monte Carlo replay of a trade before it is placed.

Adapted from the day-trader-skill "Time Machine" idea (MIT, Jakeschincariol)
for a 24-hour instrument: instead of replaying "the rest of the stock
session", each replay walks the next N bars and, for every step, borrows the
relative high/low/close move of a RANDOM past bar from the same time-of-day
slot. That keeps gold's intraday volatility shape (quiet Asia, loud London/NY)
while sampling the market's own real behaviour.

Honesty rules carried over from the original:
- Simulated from the instrument's own recent bars — NOT a prediction.
- If a bar's range touches both stop and target, the STOP counts first
  (conservative).
- The result is odds, never a recommendation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass
class ReplayResult:
    replays: int
    horizon_bars: int
    side: str
    entry: float
    stop: float
    target: float
    target_first: int
    stop_first: int
    neither: int          # horizon ended with the trade still open
    close_win: int        # of "neither", how many closed in favour
    avg_r: float
    median_close: float
    p05_close: float
    p95_close: float
    bars_used: int
    days_used: int
    data_from: str
    data_to: str

    def to_dict(self) -> dict:
        n = max(1, self.replays)
        return {
            "replays": self.replays,
            "horizon_bars": self.horizon_bars,
            "side": self.side,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "target_first": self.target_first,
            "stop_first": self.stop_first,
            "neither": self.neither,
            "close_win": self.close_win,
            "odds_target": round(self.target_first / n, 4),
            "odds_stop": round(self.stop_first / n, 4),
            "odds_neither": round(self.neither / n, 4),
            "avg_r": round(self.avg_r, 3),
            "median_close": round(self.median_close, 4),
            "p05_close": round(self.p05_close, 4),
            "p95_close": round(self.p95_close, 4),
            "bars_used": self.bars_used,
            "days_used": self.days_used,
            "data_from": self.data_from,
            "data_to": self.data_to,
            "verdict": _verdict(self),
        }


def _verdict(r: "ReplayResult") -> str:
    """One honest line. High odds are still not a recommendation."""
    n = max(1, r.replays)
    t, s = r.target_first / n, r.stop_first / n
    if t >= 0.55 and t > 2 * s:
        return "odds favour the target — but a 45%+ stop side is still a real outcome"
    if s >= 0.55 and s > 2 * t:
        return "odds favour the STOP — this plan risks more than it collects"
    if abs(t - s) < 0.05:
        return "near a coin flip — the edge is not in these odds"
    return "mixed odds — size for the stop side, not the target side"


def build_slot_pool(df: pd.DataFrame) -> dict[int, list[tuple[float, float, float]]]:
    """Relative (high, low, close) moves grouped by time-of-day slot.

    Ratios are against each bar's own open, so the pool is price-level
    independent. Bars with a non-positive open are dropped.
    """
    pool: dict[int, list[tuple[float, float, float]]] = {}
    times = df.index
    for i, ts in enumerate(times):
        o = float(df["open"].iloc[i])
        if o <= 0:
            continue
        h = float(df["high"].iloc[i]) / o
        l = float(df["low"].iloc[i]) / o
        c = float(df["close"].iloc[i]) / o
        slot = ts.hour * 60 + ts.minute
        pool.setdefault(slot, []).append((h, l, c))
    return pool


def replay_trade(
    df: pd.DataFrame,
    entry: float,
    stop: float,
    target: float,
    side: str = "buy",
    horizon_bars: int = 16,
    replays: int = 2000,
    seed: int = 19,
    window: int = 1,
) -> ReplayResult:
    """Replay the trade `replays` times over the next `horizon_bars` bars.

    df must be a UTC-indexed OHLCV frame (any timeframe; use the timeframe
    you actually trade). Returns odds, R stats and close percentiles.
    """
    if df is None or len(df) < 50:
        raise ValueError("need at least 50 bars of real data to replay")
    long = side.lower() != "sell"
    entry, stop, target = float(entry), float(stop), float(target)
    if entry <= 0 or stop <= 0 or target <= 0:
        raise ValueError("entry, stop and target must be positive prices")
    risk = (entry - stop) if long else (stop - entry)
    if risk <= 0:
        raise ValueError("stop must be beyond entry in the trade's direction")
    reward = (target - entry) if long else (entry - target)
    if reward <= 0:
        raise ValueError("target must be beyond entry in the trade's direction")
    r_target = round(reward / risk, 4)

    horizon_bars = max(1, min(int(horizon_bars), 96))
    replays = max(100, min(int(replays), 20000))

    pool = build_slot_pool(df)
    if not pool:
        raise ValueError("no usable bars in the data")
    slots = sorted(pool)
    pos = {s: i for i, s in enumerate(slots)}
    last_slot = len(slots) - 1

    def slot_at(i: int, rng: random.Random) -> int:
        """Time-of-day slot for step i, with +/- window jitter."""
        base = pos[slots[i % len(slots)]]
        j = base if window == 0 else min(last_slot, max(0, base + rng.randint(-window, window)))
        return slots[j]

    rng = random.Random(seed)
    choice = rng.choice
    outcomes: list[str] = []
    closes: list[float] = []
    for _ in range(replays):
        p = entry
        outcome = None
        for step in range(horizon_bars):
            slot = slot_at(step, rng)
            h, l, c = choice(pool[slot])
            hi, lo, cl = p * h, p * l, p * c
            if outcome is None:
                if long:
                    if lo <= stop:
                        outcome = "stop"      # both inside one bar: stop counts (conservative)
                    elif hi >= target:
                        outcome = "target"
                else:
                    if hi >= stop:
                        outcome = "stop"
                    elif lo <= target:
                        outcome = "target"
            p = cl
        outcomes.append(outcome or "close")
        closes.append(p)

    cnt = {x: outcomes.count(x) for x in ("target", "stop", "close")}
    green = sum(
        1 for o, p in zip(outcomes, closes)
        if o == "target" or (o == "close" and (p > entry if long else p < entry))
    )
    rs = [
        r_target if o == "target"
        else -1.0 if o == "stop"
        else ((p - entry) if long else (entry - p)) / risk
        for o, p in zip(outcomes, closes)
    ]
    srt = sorted(closes)

    def pct(q: float) -> float:
        k = (len(srt) - 1) * q
        f = int(k)
        return srt[f] + (srt[min(f + 1, len(srt) - 1)] - srt[f]) * (k - f)

    avg_r = sum(rs) / len(rs)
    return ReplayResult(
        replays=replays,
        horizon_bars=horizon_bars,
        side="buy" if long else "sell",
        entry=entry, stop=stop, target=target,
        target_first=cnt["target"],
        stop_first=cnt["stop"],
        neither=cnt["close"],
        close_win=green - cnt["target"],
        avg_r=avg_r,
        median_close=pct(0.5),
        p05_close=pct(0.05),
        p95_close=pct(0.95),
        bars_used=len(df),
        days_used=len(pd.Series(df.index.date).unique()),
        data_from=str(df.index[0]),
        data_to=str(df.index[-1]),
    )


def suggest_stops(
    df: pd.DataFrame,
    entry: float,
    side: str = "buy",
    atr_mult: float = 1.5,
    r_multiple: float = 2.0,
) -> dict:
    """Exit Planner: ATR-based stop and R-multiple target for the ticket.

    Uses Wilder ATR(14) on the frame's own timeframe. Returns rounded prices
    plus the risk/reward actually implied — never a recommendation to place
    the order.
    """
    long = side.lower() != "sell"
    entry = float(entry)
    if entry <= 0:
        raise ValueError("entry must be a positive price")
    if df is None or len(df) < 15:
        raise ValueError("need at least 15 bars for ATR")
    trs = []
    prev_close = float(df["close"].iloc[0])
    for i in range(1, len(df)):
        h, l, c = float(df["high"].iloc[i]), float(df["low"].iloc[i]), float(df["close"].iloc[i])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        if tr > 0:                     # skip flat placeholder bars (zero range)
            trs.append(tr)
        prev_close = c
    if len(trs) < 5:
        raise ValueError("not enough bars with real range to compute ATR")
    atr = sum(trs[-14:]) / min(14, len(trs))
    if atr <= 0:
        raise ValueError("ATR is zero on this data")
    dist = max(atr * float(atr_mult), entry * 0.0005)
    stop = entry - dist if long else entry + dist
    target = entry + dist * float(r_multiple) if long else entry - dist * float(r_multiple)
    digits = 2 if entry >= 20 else 5
    return {
        "entry": round(entry, digits),
        "stop": round(stop, digits),
        "target": round(target, digits),
        "atr": round(atr, digits),
        "atr_mult": float(atr_mult),
        "r_multiple": float(r_multiple),
        "risk_pct": round(dist / entry * 100, 3),
        "side": "buy" if long else "sell",
    }
