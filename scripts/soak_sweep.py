"""Month-by-month soak sweep: runs the replay soak over every ingested
month and aggregates integrity. Writes reports/soak_sweep.json.

Usage: python -m scripts.soak_sweep [max_months]   (default: all)
"""
from __future__ import annotations

import calendar
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.soak_replay import run_soak


def ingested_months() -> list[tuple[int, int]]:
    root = Path(__file__).resolve().parent.parent / "data" / "ticks" / "XAUUSD"
    months = sorted((int(y.name), int(m.name)) for y in root.iterdir()
                    if y.is_dir() for m in y.iterdir())
    return months


def month_bounds(year: int, month: int) -> tuple[int, int]:
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp()
                * 1000)
    last_day = calendar.monthrange(year, month)[1]
    end = int(datetime(year, month, last_day, 23, 59, 59,
                       tzinfo=timezone.utc).timestamp() * 1000)
    return start, end


def sweep(max_months: int | None = None) -> dict:
    months = ingested_months()
    if max_months:
        months = months[:max_months]
    results = []
    t0 = time.time()
    for year, month in months:
        start, end = month_bounds(year, month)
        try:
            s = run_soak(days=31, start_ms=start)
            row = {"month": f"{year}-{month:02d}", "ticks": s["window"]["ticks"],
                   "end_equity": s["end_equity"],
                   "orders": s["stats"]["orders_submitted"],
                   "filled": s["stats"]["filled"],
                   "rejected": s["stats"]["rejected"],
                   "rejection_reasons": s["rejection_reasons"],
                   "integrity_problems": s["integrity_problems"]}
        except SystemExit as exc:
            row = {"month": f"{year}-{month:02d}", "error": str(exc)}
        results.append(row)
        print(f"{row['month']}: ticks={row.get('ticks', 0):,} "
              f"orders={row.get('orders', 0)} "
              f"integrity={'CLEAN' if not row.get('integrity_problems') else 'PROBLEMS'}")
    bad = [r for r in results if r.get("integrity_problems") or
           r.get("error")]
    summary = {
        "source_label": "REPLAY_SOAK_SWEEP",
        "source_detail": "real historical ticks replayed through the "
                         "live pipeline — not live trading, not a "
                         "forecast",
        "months": results,
        "n_months": len(results),
        "n_clean": len(results) - len(bad),
        "n_problem": len(bad),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = Path("reports/soak_sweep.json")
    out.write_text(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    max_months = int(sys.argv[1]) if len(sys.argv) > 1 else None
    s = sweep(max_months)
    print(f"\n{s['n_clean']}/{s['n_months']} months clean "
          f"in {s['elapsed_s']}s")
    return 1 if s["n_problem"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
