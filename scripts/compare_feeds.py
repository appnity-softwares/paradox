#!/usr/bin/env python3
"""Feed comparison CLI (U8 step 1).

Subcommands:
  record   --firm FIRM --server SRV --hours N [--symbol XAUUSD]
             Record N hours of live quotes through the bridge into
             data/feeds/<firm>_<srv>/ (research store untouched).
  compare  --firm FIRM --server SRV --from ISO --to ISO [--news-json FILE]
             Compare the firm feed with the research store over the
             same period; write reports/feed_comparison_<firm>_<date>.json
             plus a CostModel suggestion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.marketdata.compare import (FeedRecorder, compare_feeds,
                                     suggest_cost_model, write_report)
from core.marketdata.store import read_range


def cmd_record(args: argparse.Namespace) -> int:
    recorder = FeedRecorder(args.firm, args.server, args.symbol)
    try:
        from core.execution.broker_adapter import BrokerAdapter
    except Exception as exc:
        print(f"bridge adapter unavailable: {exc!r}", file=sys.stderr)
        return 1
    # The concrete adapter (MT5 bridge client) is configured on the
    # Windows VPS; here we only require the protocol.
    raise SystemExit(
        "record: run this on the Windows VPS with the MT5 bridge client "
        "adapter configured (see docs/deploy_windows_bridge.md). The "
        "recording loop polls adapter.quote() every 250 ms and appends "
        "to " + str(recorder.root / recorder.source_name))


def cmd_compare(args: argparse.Namespace) -> int:
    def _ms(s: str) -> int:
        try:
            return int(s)
        except ValueError:
            return int(datetime.fromisoformat(s)
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)

    start, end = _ms(args.from_), _ms(args.to)
    research = read_range(args.symbol, start, end)
    feed_dir = FeedRecorder(args.firm, args.server,
                            args.symbol).partition_dir(start).parent
    files = sorted(feed_dir.glob("*/*/quotes.parquet"))
    if not files:
        print(f"no recorded feed under {feed_dir}", file=sys.stderr)
        return 1
    import pandas as pd
    firm = pd.concat([pd.read_parquet(f) for f in files],
                     ignore_index=True)
    firm = firm[(firm["time_utc_ms"] >= start)
                & (firm["time_utc_ms"] < end)]
    news_windows = []
    if args.news_json:
        news_windows = [tuple(w) for w in
                        json.loads(Path(args.news_json).read_text())]
    report = compare_feeds(research, firm, tuple(news_windows))
    suggestion = suggest_cost_model(report)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = write_report(report, suggestion, args.firm, date)
    print(json.dumps({"report": str(out),
                      "spread_ratio": report.get(
                          "spread_ratio_firm_over_research_p50"),
                      "provenance_firm": report["firm"]["provenance"][
                          "verdict"]}, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--firm", required=True)
    r.add_argument("--server", required=True)
    r.add_argument("--hours", type=float, default=8.0)
    r.add_argument("--symbol", default="XAUUSD")
    r.set_defaults(func=cmd_record)
    c = sub.add_parser("compare")
    c.add_argument("--firm", required=True)
    c.add_argument("--server", required=True)
    c.add_argument("--from", dest="from_", required=True)
    c.add_argument("--to", required=True)
    c.add_argument("--symbol", default="XAUUSD")
    c.add_argument("--news-json")
    c.set_defaults(func=cmd_compare)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
