"""Time Machine — replay a planned trade before risking money.

Ported from the day-trader-skill project (trader-timemachine, MIT) and
adapted for XAUUSD's near-24-hour session:

1. SIMULATION: thousands of replays of the rest of the day from the entry.
   Each replay walks bar by bar, drawing each bar from a pool of the SAME
   time-of-day bars taken from past sessions, scaled as a ratio to the
   running price. Stop is checked before target in every bar (the
   conservative call).
2. HISTORY: the same rule re-run on every past session — how many times it
   fired, how many hit the target, the stop, or neither.

The verdict compares the target's share of decided replays with the
break-even share for the plan's reward:risk. "Close to a coin flip" is a
real possible answer — the point is to look before you risk a dollar.

A simulation built from an instrument's own recent bars is NOT a prediction:
news, data releases and halts can do what the last N days never did.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd

COIN_FLIP = 0.05     # verdict thresholds in percentage points
STRONG = 0.15
MANY_NEITHER = 0.40


class TimeMachineError(Exception):
    pass


@dataclass
class Plan:
    entry: float
    stop: float
    target: float
    side: str = "buy"          # "buy" | "sell"
    entry_time: pd.Timestamp | None = None

    def __post_init__(self):
        if self.side not in ("buy", "sell"):
            raise TimeMachineError("side must be 'buy' or 'sell'")
        long = self.side == "buy"
        if long and self.stop >= self.entry:
            raise TimeMachineError("a buy's stop must be below its entry")
        if not long and self.stop <= self.entry:
            raise TimeMachineError("a sell's stop must be above its entry")
        risk = (self.entry - self.stop) if long else (self.stop - self.entry)
        if risk <= 0:
            raise TimeMachineError("risk must be positive")
        if long and self.target <= self.entry:
            raise TimeMachineError("a buy's target must be above its entry")
        if not long and self.target >= self.entry:
            raise TimeMachineError("a sell's target must be below its entry")
        self.risk = risk
        self.rr = abs(self.target - self.entry) / risk

    @property
    def r_at_target(self) -> float:
        return round(self.rr, 2)


def _session_key(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.date())


def build_pool(
    bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    *,
    bar_minutes: int | None = None,
    max_sessions: int = 120,
) -> tuple[dict[int, list[tuple[float, float, float, float]]], int, int, int]:
    """Pool of same-slot-of-day bars as ratios to the previous close.

    Returns (pool, entry_slot, last_slot, sessions_used). Slot = bars since
    midnight UTC. Only FULL sessions before the entry's session feed the pool.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TimeMachineError("bars need a DatetimeIndex")
    if bars.index.tz is None:
        bars = bars.tz_localize("UTC")
    bars = bars.sort_index()
    idx = bars.index.tz_convert("UTC")

    if bar_minutes is None:
        if len(idx) < 3:
            raise TimeMachineError("need at least 3 bars")
        bar_minutes = int(pd.Series(idx).diff().dt.total_seconds().dropna().median() // 60)
    slots_per_day = 1440 // bar_minutes

    entry_session = _session_key(entry_time.tz_convert("UTC") if entry_time.tzinfo else entry_time.tz_localize("UTC"))
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)

    sessions: dict[pd.Timestamp, list[int]] = {}
    for i in range(len(idx)):
        sessions.setdefault(_session_key(idx[i]), []).append(i)

    past = sorted(d for d in sessions if d < entry_session)
    # keep only full sessions (a partial first/last day would distort the pool)
    full = [d for d in past if len(sessions[d]) >= slots_per_day * 0.9]
    full = full[-max_sessions:]
    if len(full) < 20:
        raise TimeMachineError(
            f"only {len(full)} full past sessions in the data — need at least 20 for a fair pool")

    pool: dict[int, list[tuple[float, float, float, float]]] = {}
    for d in full:
        rows = sessions[d]
        for j, i in enumerate(rows):
            prev_c = c[rows[j - 1]] if j > 0 else o[i]   # first bar: ratio to its own open
            if prev_c <= 0:
                continue
            slot = int((idx[i] - idx[i].normalize()).total_seconds() // 60) // bar_minutes
            pool.setdefault(slot, []).append((o[i] / prev_c, h[i] / prev_c, l[i] / prev_c, c[i] / prev_c))

    e_ts = entry_time.tz_convert("UTC") if entry_time.tzinfo else entry_time.tz_localize("UTC")
    e_slot = int((e_ts - e_ts.normalize()).total_seconds() // 60) // bar_minutes
    last_slot = max(pool) if pool else e_slot
    return pool, e_slot, last_slot, len(full)


def replay(
    plan: Plan,
    bars: pd.DataFrame,
    *,
    replays: int = 10_000,
    seed: int = 19,
    window: int = 2,
    flat_by: str | None = None,
    bar_minutes: int | None = None,
) -> dict:
    """Simulate the plan's outcome distribution. Stop checked before target."""
    if replays < 100:
        raise TimeMachineError("replays must be at least 100")
    entry_ts = plan.entry_time if plan.entry_time is not None else pd.Timestamp.now(tz="UTC")
    pool, e_slot, last_slot, n_sessions = build_pool(bars, entry_ts, bar_minutes=bar_minutes)
    long = plan.side == "buy"

    if flat_by:
        hh, mm = map(int, flat_by.split(":"))
        cut = (hh * 60 + mm) // bar_minutes if bar_minutes else (hh * 60 + mm) // 5
    else:
        cut = last_slot
    last = min(last_slot, max(cut, e_slot + 1))
    if last <= e_slot:
        raise TimeMachineError("no bars left in the session after the entry — nothing to replay")

    rng = random.Random(seed)
    choice = rng.choice

    def slot_at(i: int) -> int:
        return min(max(i, 0), last_slot)

    outcomes: list[str] = []
    closes: list[float] = []
    for _ in range(replays):
        p = plan.entry
        outcome = None
        for i in range(e_slot + 1, last + 1):
            bo, bh, bl, bc = choice(pool[slot_at(i + rng.randint(-window, window))])
            # candidate prices scaled to the running price
            o_p, h_p, l_p, c_p = p * bo, p * bh, p * bl, p * bc
            # stop first (conservative)
            if (l_p <= plan.stop) if long else (h_p >= plan.stop):
                outcome, p = "stop", plan.stop
                break
            if (h_p >= plan.target) if long else (l_p <= plan.target):
                outcome, p = "target", plan.target
                break
            p = c_p
        if outcome is None:
            outcome = "neither"
        outcomes.append(outcome)
        closes.append(p)

    cnt = {k: outcomes.count(k) for k in ("target", "stop", "neither")}
    green = sum(1 for o_, p in zip(outcomes, closes)
                if o_ == "target" or (o_ == "neither" and (p > plan.entry if long else p < plan.entry)))
    rs = [plan.r_at_target if o_ == "target" else -1.0 if o_ == "stop"
          else ((p - plan.entry) if long else (plan.entry - p)) / plan.risk
          for o_, p in zip(outcomes, closes)]
    srt = sorted(closes)

    def pct(a, q):
        return a[min(int(q * len(a)), len(a) - 1)]

    decided = cnt["target"] + cnt["stop"]
    p_target_decided = cnt["target"] / decided if decided else None
    break_even = 1.0 / (1.0 + plan.rr)
    verdict = None
    if p_target_decided is not None:
        gap = (p_target_decided - break_even) * 100.0
        if abs(gap) < COIN_FLIP:
            verdict = "Close to a coin flip" if abs(plan.rr - 1.0) < 1e-9 else "Close to break-even"
        elif gap > 0:
            verdict = "Strongly toward the target" if gap >= STRONG else "Leans toward the target"
        else:
            verdict = "Strongly toward the stop" if gap <= -STRONG else "Leans toward the stop"

    return {
        "replays": replays,
        "seed": seed,
        "window": window,
        "bars_left": last - e_slot,
        "p_target": round(cnt["target"] / replays, 4),
        "p_stop": round(cnt["stop"] / replays, 4),
        "p_neither": round(cnt["neither"] / replays, 4),
        "p_green": round(green / replays, 4),
        "avg_R": round(float(statistics.mean(rs)), 3) if rs else None,
        "r_at_target": plan.r_at_target,
        "median_close": round(float(pct(srt, 0.5)), 2),
        "p5_close": round(float(pct(srt, 0.05)), 2),
        "p95_close": round(float(pct(srt, 0.95)), 2),
        "p_target_decided": round(p_target_decided, 4) if p_target_decided is not None else None,
        "break_even_share": round(break_even, 4),
        "verdict": verdict,
        "many_neither": cnt["neither"] / replays > MANY_NEITHER,
        "pool_sessions": n_sessions,
    }


def history_replay(
    bars: pd.DataFrame,
    *,
    or_minutes: int = 15,
    stop_rule: str = "or_mid",
    target_r: float = 2.0,
    side: str = "buy",
    bar_minutes: int | None = None,
    flat_by: str | None = None,
    max_sessions: int = 120,
) -> dict:
    """The same entry rule on every past full session: first bar after the
    opening range to CLOSE beyond it is the entry; then stop-first, then
    target, bar by bar; else the close at flat_by."""
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TimeMachineError("bars need a DatetimeIndex")
    if bars.index.tz is None:
        bars = bars.tz_localize("UTC")
    bars = bars.sort_index()
    idx = bars.index.tz_convert("UTC")
    long = side == "buy"

    if bar_minutes is None:
        bar_minutes = int(pd.Series(idx).diff().dt.total_seconds().dropna().median() // 60)
    slots_per_day = 1440 // bar_minutes
    n_or = max(1, or_minutes // bar_minutes)

    if flat_by:
        hh, mm = map(int, flat_by.split(":"))
        last_slot = (hh * 60 + mm) // bar_minutes
    else:
        last_slot = slots_per_day - 1

    sessions: dict[pd.Timestamp, list[int]] = {}
    for i in range(len(idx)):
        sessions.setdefault(_session_key(idx[i]), []).append(i)
    full = [d for d in sorted(sessions) if len(sessions[d]) >= slots_per_day * 0.9][-max_sessions:]
    if len(full) < 20:
        raise TimeMachineError(
            f"only {len(full)} full past sessions — need at least 20")

    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)

    trades = []
    for d in full:
        rows = sessions[d]
        or_rows = rows[:n_or]
        oh = max(h[i] for i in or_rows)
        ol = min(l[i] for i in or_rows)
        kk = next((r for r in rows[n_or:last_slot + 1]
                   if (c[r] > oh if long else c[r] < ol)), None)
        if kk is None:
            continue
        e = c[kk]
        if stop_rule == "or_mid":
            s = (oh + ol) / 2
        elif stop_rule == "or_low":
            s = ol if long else oh
        else:
            raise TimeMachineError(f"unknown stop_rule {stop_rule!r}")
        risk = (e - s) if long else (s - e)
        if risk <= 0:
            continue
        t = e + target_r * risk if long else e - target_r * risk
        oc = None
        for b in rows[rows.index(kk) + 1: last_slot + 1]:
            if (l[b] <= s) if long else (h[b] >= s):
                oc = "stop"
                break
            if (h[b] >= t) if long else (l[b] <= t):
                oc = "target"
                break
        cl = c[rows[min(last_slot, len(rows) - 1)]]
        R = ((t - e) if long else (e - t)) / risk if oc == "target" else \
            -1.0 if oc == "stop" else ((cl - e) if long else (e - cl)) / risk
        trades.append({
            "date": d.date().isoformat(),
            "entry": round(float(e), 3),
            "stop": round(float(s), 3),
            "target": round(float(t), 3),
            "outcome": oc or "neither",
            "R": round(float(R), 2),
        })

    decided = [x for x in trades if x["outcome"] in ("target", "stop")]
    tgt = sum(1 for x in trades if x["outcome"] == "target")
    return {
        "sessions": len(full),
        "triggered": len(trades),
        "target": tgt,
        "stop": sum(1 for x in trades if x["outcome"] == "stop"),
        "neither": sum(1 for x in trades if x["outcome"] == "neither"),
        "green": sum(1 for x in trades if x["R"] > 0),
        "avg_R": round(float(statistics.mean(x["R"] for x in trades)), 3) if trades else None,
        "p_target_decided": round(tgt / len(decided), 4) if decided else None,
        "rule": {"or_minutes": or_minutes, "stop_rule": stop_rule,
                 "target_r": target_r, "side": side, "flat_by": flat_by},
        "trades": trades,
    }
