"""Background Runner: Ingest 2 Years of Real XAUUSD Ticks.

Downloads genuine Dukascopy ticks for 2024, 2025, and 2026.
Writes into partitioned Parquet with zstd compression.
Runs completely in the background with progress reporting to a status file.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Setup paths
PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROTOTYPE_ROOT))

from core.marketdata import store
from core.marketdata.dukascopy_ingest import ingest_week

STATUS_FILE = PROTOTYPE_ROOT / "data" / "ticks" / "ingest_progress.json"
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_two_years")


def write_progress(state: str, current_year: int, current_week: int, total_weeks: int, ticks_so_far: int, err: str = ""):
    data = {
        "state": state,
        "current_year": current_year,
        "current_week": current_week,
        "total_weeks": total_weeks,
        "progress_pct": round((((current_year - 2024) * 52 + current_week) / total_weeks) * 100, 1) if total_weeks > 0 else 0,
        "ticks_so_far": ticks_so_far,
        "error": err,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    STATUS_FILE.write_text(json.dumps(data, indent=2))


def main():
    logger.info("Starting automated 2-year XAUUSD tick ingestion (2024 - 2025)...")
    total_weeks = 52 * 2  # 2024 and 2025
    ticks_total = store.coverage("XAUUSD")["ticks"]
    write_progress("RUNNING", 2024, 1, total_weeks, ticks_total)

    years = [2024, 2025]
    for year in years:
        for week in range(1, 53):
            try:
                logger.info("Downloading %d Week %d ...", year, week)
                res = ingest_week(year, week, symbol="XAUUSD")
                ticks_in_week = res.get("ticks", 0)
                ticks_total += ticks_in_week
                write_progress("RUNNING", year, week, total_weeks, ticks_total)
            except Exception as exc:
                logger.error("Error on %d w%d: %s", year, week, exc)
                write_progress("RUNNING", year, week, total_weeks, ticks_total, err=str(exc))
            # Brief pause to keep CPU/network well-paced
            time.sleep(0.1)

    cov = store.coverage("XAUUSD")
    write_progress("COMPLETED", 2025, 52, total_weeks, cov["ticks"])
    logger.info("Successfully finished 2-year tick ingestion! Total ticks: %d", cov["ticks"])


if __name__ == "__main__":
    main()
