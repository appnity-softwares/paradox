"""Market-closure-aware gap classification (Part 6.1 honesty rule):
scheduled XAUUSD closures (weekend, Good Friday) are reported but do
NOT quarantine the dataset; a genuine missing week during trading
hours still does. Uses constructed timestamps to test the checker
itself — never presented as market data.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from core.marketdata.quality import THRESHOLDS, _open_hours_inside, \
    quality_report


def _ts(y, mo, d, h=0, mi=0):
    return int(dt.datetime(y, mo, d, h, mi,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)


def _frame(*times_ms):
    rows = [{"time_utc_ms": t, "bid": 2600.0, "ask": 2600.2,
             "bid_volume": 1.0, "ask_volume": 1.0, "source": "test"}
            for t in times_ms]
    return pd.DataFrame(rows)


def _pad(df, before=600, after=401):
    """Add dense ticks at both edges so min_ticks (1000) is satisfied
    without touching the gap under test."""
    head = [{"time_utc_ms": int(df["time_utc_ms"].iloc[0]) - i * 1000,
             "bid": 2600.0, "ask": 2600.2, "bid_volume": 1.0,
             "ask_volume": 1.0, "source": "test"}
            for i in range(before, 0, -1)]
    tail_t = int(df["time_utc_ms"].iloc[-1])
    tail = [{"time_utc_ms": tail_t + i * 1000, "bid": 2600.0,
             "ask": 2600.2, "bid_volume": 1.0, "ask_volume": 1.0,
             "source": "test"} for i in range(1, after + 1)]
    return pd.DataFrame(head + df.to_dict("records") + tail)


def test_easter_weekend_gap_does_not_quarantine():
    # Good Friday 2025-04-18 20:58 -> Easter Sunday 2025-04-20 22:00:
    # the exact closure observed in the real 2025-H1 ingest.
    df = _pad(_frame(_ts(2025, 4, 17, 20, 58), _ts(2025, 4, 20, 22, 0),
                     _ts(2025, 4, 20, 22, 1)))
    rep = quality_report(df)
    assert rep["status"] == "ok"
    check = rep["checks"]["gaps_over_limit"]
    assert check["count"] == 0                       # no DATA gaps
    assert len(check["market_closures"]) == 1
    closure = check["market_closures"][0]
    assert closure["hours"] > 72                     # still reported
    assert closure["open_hours_inside"] <= 2.0


def test_missing_trading_week_still_quarantines():
    # Tuesday -> the following Tuesday: a full trading week vanished.
    df = _pad(_frame(_ts(2025, 6, 3, 12), _ts(2025, 6, 10, 12),
                     _ts(2025, 6, 10, 12, 1)))
    rep = quality_report(df)
    assert rep["status"] == "quarantined"
    check = rep["checks"]["gaps_over_limit"]
    assert check["count"] == 1
    assert check["data_gaps"][0]["open_hours_inside"] > 100


def test_friday_close_to_sunday_open_is_closure():
    # Regular weekly close: Fri 21:00 -> Sun 22:00 (~49 h).
    assert _open_hours_inside(_ts(2025, 6, 6, 21), _ts(2025, 6, 8, 22)) \
        <= 2.0


def test_midweek_gap_has_open_hours():
    # Monday 12:00 -> Wednesday 12:00 sits fully inside the open window.
    assert _open_hours_inside(_ts(2025, 6, 2, 12), _ts(2025, 6, 4, 12)) \
        > 40.0


def test_reason_wording_distinguishes_data_gaps():
    df = _pad(_frame(_ts(2025, 6, 3, 12), _ts(2025, 6, 10, 12),
                     _ts(2025, 6, 10, 12, 1)))
    rep = quality_report(df)
    assert any("data gaps" in r and "trading hours" in r
               for r in rep["reasons"])
    assert "max_gap_hours" in THRESHOLDS
