"""CLI for the tick data pipeline (Master Prompt 6.1).

    python3 -m core.marketdata.cli ingest --symbol XAUUSD
    python3 -m core.marketdata.cli ingest --file /path/XAUUSD_ticks_2025_01.csv
    python3 -m core.marketdata.cli quality-report --symbol XAUUSD
    python3 -m core.marketdata.cli coverage --symbol XAUUSD
    python3 -m core.marketdata.cli watch --symbol XAUUSD --interval 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from core.marketdata import ingest, store
from core.marketdata.quality import quality_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marketdata", description="Real tick data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="ingest MT5 tick exports into Parquet")
    p_ing.add_argument("--symbol", default="XAUUSD")
    p_ing.add_argument("--file", type=Path, help="a single exported CSV")
    p_ing.add_argument("--folder", type=Path, help="folder of exported CSVs")
    p_ing.add_argument("--keep-csv", action="store_true")

    p_q = sub.add_parser("quality-report", help="quality report for the stored dataset")
    p_q.add_argument("--symbol", default="XAUUSD")
    p_q.add_argument("--limit", type=int, default=2_000_000)

    p_c = sub.add_parser("coverage", help="what real tick data is on disk")
    p_c.add_argument("--symbol", default="XAUUSD")

    p_w = sub.add_parser("watch", help="ingest new exports as the MT5 script writes them")
    p_w.add_argument("--symbol", default="XAUUSD")
    p_w.add_argument("--interval", type=int, default=30)
    p_w.add_argument("--once", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        if args.file:
            print(json.dumps(ingest.ingest_csv(args.file, args.symbol, keep_csv=args.keep_csv), indent=2))
        else:
            results = ingest.ingest_folder(args.folder, args.symbol)
            print(json.dumps({"ingested": results, "coverage": store.coverage(args.symbol)}, indent=2))
        return 0

    if args.command == "coverage":
        print(json.dumps(store.coverage(args.symbol), indent=2))
        return 0

    if args.command == "quality-report":
        cov = store.coverage(args.symbol)
        if not cov["ticks"]:
            print(json.dumps({"status": "no_data", "coverage": cov}, indent=2))
            return 1
        df = store.read_range(args.symbol, cov["first_ms"], cov["last_ms"], limit=args.limit)
        print(json.dumps({"coverage": cov, "dataset_hash": store.dataset_hash(args.symbol),
                          "quality": quality_report(df)}, indent=2))
        return 0

    if args.command == "watch":
        while True:
            results = ingest.ingest_folder(symbol=args.symbol)
            for r in results:
                print(f"[{r.get('status')}] {r.get('file')} ticks={r.get('ticks', 0)}", flush=True)
            if results:
                print(json.dumps(store.coverage(args.symbol)), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)

    return 1


if __name__ == "__main__":
    sys.exit(main())
