#!/usr/bin/env python3
"""Dukascopy cross-check (U8 step 3): ingest one week and compare it
with the research MT5 store over the same period. Reports differences;
respects the provider's terms (docs/data_licensing.md).

Usage: python scripts/compare_dukascopy.py 2025 10
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.marketdata.compare import compare_feeds, write_report
from core.marketdata.dukascopy_ingest import ingest_week
from core.marketdata.store import read_range


def main() -> int:
    year, week = int(sys.argv[1]), int(sys.argv[2])
    res = ingest_week(year, week, "XAUUSD")
    if res.get("status") != "ok":
        print(f"ingest failed: {res.get('error', res.get('status'))}")
        print("NOTE: datafeed.dukascopy.com must be reachable; the "
              "check is an OPEN ITEM until network access exists.")
        return 1
    duk = res["data"] if "data" in res else res.get("df")
    start = dt.datetime.fromisocalendar(year, week, 1)
    start_ms = int(start.replace(tzinfo=dt.UTC).timestamp() * 1000)
    end_ms = start_ms + 7 * 86_400_000
    research = read_range("XAUUSD", start_ms, end_ms)
    report = compare_feeds(research, duk)
    date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    out = write_report(report, {"note": "cross-check only, no cost "
                                "suggestion"}, "dukascopy", date)
    print(f"report -> {out}")
    print(f"spread p50 research={report['research']['spread']['p50']} "
          f"dukascopy={report['firm']['spread']['p50']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
