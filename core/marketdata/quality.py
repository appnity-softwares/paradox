"""Tick data quality report with quarantine thresholds (Master Prompt 6.1).

A dataset that fails the thresholds is flagged ``quarantined`` and must not be
used for validation runs. Nothing is repaired silently: every counter is
reported so the caller can see exactly what was found.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# Thresholds are explicit and tunable per dataset, never hidden in the engine.
THRESHOLDS = {
    "max_duplicate_pct": 0.5,
    "max_out_of_order_pct": 0.1,
    "max_crossed_pct": 0.05,
    "max_nonpositive_spread_pct": 5.0,
    "max_spike_pct": 0.5,
    "max_gap_hours": 72.0,       # weekend + holiday allowance
    "min_ticks": 1000,
    "spike_jump_multiple": 40.0,  # vs median absolute tick-to-tick move
}

# XAUUSD trading week (UTC): opens Sunday 22:00, closes Friday evening
# (21:00 UTC in the northern summer, 22:00 in winter). A gap that does
# not intersect this window is a scheduled market closure (weekend,
# Good Friday, Christmas...), NOT missing data — it is reported but
# does not quarantine the dataset. Any open-hours hole above
# max_gap_hours still quarantines.
WEEKLY_OPEN = {"sun_from": 22, "fri_until": 21}
CLOSURE_TOLERANCE_H = 2.0   # pre-close quiet around the Friday close


def _easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


_HOLIDAY_CACHE: dict[int, set[dt.date]] = {}


def market_holidays(year: int) -> set[dt.date]:
    """UTC dates on which XAUUSD does not trade (full-day closures):
    New Year's Day, Good Friday, Easter Monday, Christmas Day."""
    if year not in _HOLIDAY_CACHE:
        easter = _easter_sunday(year)
        _HOLIDAY_CACHE[year] = {
            dt.date(year, 1, 1), dt.date(year, 12, 25),
            easter - dt.timedelta(days=2),          # Good Friday
            easter + dt.timedelta(days=1),          # Easter Monday
        }
    return _HOLIDAY_CACHE[year]


def _early_close_dates(year: int) -> set[dt.date]:
    """Trading days with an early ~21:00 UTC close: the day before
    Good Friday, Christmas Eve, New Year's Eve."""
    easter = _easter_sunday(year)
    return {easter - dt.timedelta(days=3), dt.date(year, 12, 24),
            dt.date(year, 12, 31)}


def _open_hours_inside(prev_ms: int, cur_ms: int) -> float:
    """Hours of [prev_ms, cur_ms] that fall inside the XAUUSD weekly
    open window (Sun 22:00 UTC -> Fri ~21:00 UTC), minus holidays."""
    total = 0.0
    prev_dt = dt.datetime.fromtimestamp(prev_ms / 1000, tz=dt.timezone.utc)
    day = prev_dt
    end = dt.datetime.fromtimestamp(cur_ms / 1000, tz=dt.timezone.utc)
    while day < end:
        next_midnight = (day + dt.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        seg_start = max(day, prev_dt)   # first segment starts at prev_dt
        seg_end = min(next_midnight, end)
        wd = day.weekday()              # Mon=0 ... Sun=6
        if day.date() in market_holidays(day.year):
            day = seg_end               # full holiday: no trading hours
            continue
        if wd == 6:                     # Sunday: open from 22:00
            lo, hi = day.replace(hour=22, minute=0), seg_end
        elif wd == 4:                   # Friday: open until ~21:00
            lo, hi = day, day.replace(hour=21, minute=0)
        elif wd == 5:                   # Saturday: fully closed
            day = seg_end
            continue
        elif day.date() in _early_close_dates(day.year):
            lo, hi = day, day.replace(hour=21, minute=0)
        else:                           # Monday-Thursday: fully open
            lo, hi = day, seg_end
        total += max(0.0, (min(hi, seg_end) - max(lo, seg_start)
                           ).total_seconds() / 3600.0)
        day = seg_end
    return total


def quality_report(df: pd.DataFrame, thresholds: dict | None = None) -> dict:
    limits = {**THRESHOLDS, **(thresholds or {})}
    n = int(len(df))
    if n == 0:
        return {"ticks": 0, "status": "quarantined", "reasons": ["empty dataset"], "checks": {}}

    t = df["time_utc_ms"].to_numpy(dtype=np.int64)
    bid = df["bid"].to_numpy(dtype=float)
    ask = df["ask"].to_numpy(dtype=float)
    spread = ask - bid
    mid = (ask + bid) / 2.0

    duplicates = int(pd.Series(t).duplicated().sum())
    out_of_order = int((np.diff(t) < 0).sum())
    crossed = int((bid > ask).sum())
    nonpositive_spread = int((spread <= 0).sum())
    zero_price = int(((bid <= 0) | (ask <= 0)).sum())

    gaps = np.diff(t) / 3_600_000.0
    max_gap_h = float(gaps.max()) if len(gaps) else 0.0
    big_gap_idx = np.where(gaps > limits["max_gap_hours"])[0]
    closures, data_gaps = [], []
    for i in big_gap_idx:
        prev_ms, cur_ms = int(t[i]), int(t[i + 1])
        open_h = _open_hours_inside(prev_ms, cur_ms)
        entry = {"from_ms": prev_ms, "to_ms": cur_ms,
                 "hours": round(float(gaps[i]), 2),
                 "open_hours_inside": round(open_h, 2)}
        (closures if open_h <= CLOSURE_TOLERANCE_H else data_gaps
         ).append(entry)
    big_gaps = len(data_gaps)          # only unexplained holes quarantine

    jumps = np.abs(np.diff(mid))
    median_jump = float(np.median(jumps)) if len(jumps) else 0.0
    spike_cut = max(median_jump * limits["spike_jump_multiple"], 1e-9)
    spikes = int((jumps > spike_cut).sum())

    span_hours = float((t[-1] - t[0]) / 3_600_000.0)
    per_hour = round(n / span_hours, 1) if span_hours > 0 else 0.0

    def pct(x: int) -> float:
        return round(100.0 * x / n, 4)

    checks = {
        "duplicates": {"count": duplicates, "pct": pct(duplicates), "limit_pct": limits["max_duplicate_pct"]},
        "out_of_order": {"count": out_of_order, "pct": pct(out_of_order), "limit_pct": limits["max_out_of_order_pct"]},
        "crossed_quotes": {"count": crossed, "pct": pct(crossed), "limit_pct": limits["max_crossed_pct"]},
        "nonpositive_spread": {"count": nonpositive_spread, "pct": pct(nonpositive_spread), "limit_pct": limits["max_nonpositive_spread_pct"]},
        "zero_or_negative_price": {"count": zero_price, "pct": pct(zero_price), "limit_pct": 0.0},
        "price_spikes": {"count": spikes, "pct": pct(spikes), "limit_pct": limits["max_spike_pct"]},
        "gaps_over_limit": {"count": big_gaps, "max_gap_hours": round(max_gap_h, 2), "limit_hours": limits["max_gap_hours"],
                            "market_closures": closures, "data_gaps": data_gaps},
        "density": {"ticks_per_hour": per_hour, "span_hours": round(span_hours, 2)},
        "spread": {
            "median": round(float(np.median(spread)), 5),
            "p95": round(float(np.percentile(spread, 95)), 5),
            "max": round(float(spread.max()), 5),
        },
    }

    reasons = []
    if n < limits["min_ticks"]:
        reasons.append(f"only {n} ticks (minimum {limits['min_ticks']})")
    for key in ("duplicates", "out_of_order", "crossed_quotes", "nonpositive_spread", "price_spikes"):
        if checks[key]["pct"] > checks[key]["limit_pct"]:
            reasons.append(f"{key} {checks[key]['pct']}% exceeds {checks[key]['limit_pct']}%")
    if zero_price:
        reasons.append(f"{zero_price} ticks with a non-positive price")
    if big_gaps:
        reasons.append(f"{big_gaps} data gaps longer than "
                       f"{limits['max_gap_hours']}h during trading hours")

    return {
        "ticks": n,
        "status": "quarantined" if reasons else "ok",
        "reasons": reasons,
        "checks": checks,
        "thresholds": limits,
    }
