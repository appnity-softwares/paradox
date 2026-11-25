"""Normalized tick schema and partitioned Parquet storage (Master Prompt 6.1).

One schema for every provider:
    time_utc_ms : int64   milliseconds since epoch, UTC
    bid         : float64
    ask         : float64
    bid_volume  : float64
    ask_volume  : float64
    source      : string  provider key, never blank

Ticks are stored as Parquet partitioned by symbol/year/month so a two-year
dataset can be read a month at a time without loading everything into memory.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent.parent
TICK_ROOT = PROTOTYPE_ROOT / "data" / "ticks"
TICK_ROOT.mkdir(parents=True, exist_ok=True)

TICK_SCHEMA = pa.schema([
    ("time_utc_ms", pa.int64()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("bid_volume", pa.float64()),
    ("ask_volume", pa.float64()),
    ("source", pa.string()),
])

REQUIRED = ["time_utc_ms", "bid", "ask", "source"]


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Coerce any provider frame into the normalized schema, or raise."""
    if df is None or len(df) == 0:
        raise ValueError("cannot normalize an empty tick frame")
    out = df.copy()

    if "time_utc_ms" not in out.columns:
        if "time_msc" in out.columns:
            out["time_utc_ms"] = out["time_msc"].astype("int64")
        elif "time_ms" in out.columns:
            out["time_utc_ms"] = out["time_ms"].astype("int64")
        elif isinstance(out.index, pd.DatetimeIndex):
            out["time_utc_ms"] = (out.index.asi8 // 1_000_000).astype("int64")
        else:
            raise ValueError("tick frame has no usable timestamp column")

    if "bid" not in out.columns or "ask" not in out.columns:
        raise ValueError("real ticks must carry both bid and ask; they are never inferred")

    out["bid_volume"] = out.get("bid_volume", out.get("volume", 0.0))
    out["ask_volume"] = out.get("ask_volume", 0.0)
    out["source"] = source

    out = out[["time_utc_ms", "bid", "ask", "bid_volume", "ask_volume", "source"]]
    out = out.astype({
        "time_utc_ms": "int64", "bid": "float64", "ask": "float64",
        "bid_volume": "float64", "ask_volume": "float64", "source": "string",
    })
    return out.sort_values("time_utc_ms").reset_index(drop=True)


def partition_path(symbol: str, year: int, month: int) -> Path:
    folder = TICK_ROOT / symbol.upper() / f"{year:04d}" / f"{month:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "ticks.parquet"


def write_partition(df: pd.DataFrame, symbol: str, year: int, month: int) -> Path:
    """Append to a month partition, de-duplicating on timestamp."""
    path = partition_path(symbol, year, month)
    if path.exists():
        existing = pq.read_table(path).to_pandas()
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["time_utc_ms"], keep="last").sort_values("time_utc_ms")
    table = pa.Table.from_pandas(df.reset_index(drop=True), schema=TICK_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    return path


def read_range(symbol: str, start_ms: int, end_ms: int, limit: int | None = None) -> pd.DataFrame:
    """Read ticks in [start_ms, end_ms] month-partition by month-partition."""
    root = TICK_ROOT / symbol.upper()
    if not root.exists():
        return pd.DataFrame(columns=REQUIRED)
    start = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
    end = datetime.fromtimestamp(end_ms / 1000, timezone.utc)
    frames = []
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if year < start.year or year > end.year:
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            month = int(month_dir.name)
            if (year, month) < (start.year, start.month) or (year, month) > (end.year, end.month):
                continue
            path = month_dir / "ticks.parquet"
            if not path.exists():
                continue
            frame = pq.read_table(path).to_pandas()
            frame = frame[(frame["time_utc_ms"] >= start_ms) & (frame["time_utc_ms"] <= end_ms)]
            if len(frame):
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=REQUIRED)
    out = pd.concat(frames, ignore_index=True).sort_values("time_utc_ms")
    return out.head(limit) if limit else out


def coverage(symbol: str) -> dict:
    """What real tick data is actually on disk."""
    root = TICK_ROOT / symbol.upper()
    months, total, first, last = [], 0, None, None
    if root.exists():
        for path in sorted(root.glob("*/*/ticks.parquet")):
            meta = pq.read_metadata(path)
            rows = meta.num_rows
            if rows == 0:
                continue
            col = pq.read_table(path, columns=["time_utc_ms"]).column("time_utc_ms")
            lo = col[0].as_py()
            hi = col[-1].as_py()
            first = lo if first is None else min(first, lo)
            last = hi if last is None else max(last, hi)
            total += rows
            months.append({
                "partition": f"{path.parent.parent.name}-{path.parent.name}",
                "ticks": rows,
                "size_mb": round(path.stat().st_size / 1_048_576, 2),
            })
    span_days = round((last - first) / 86_400_000, 1) if first and last else 0
    return {
        "symbol": symbol.upper(),
        "ticks": total,
        "months": months,
        "first_ms": first,
        "last_ms": last,
        "span_days": span_days,
        "span_years": round(span_days / 365.25, 2),
        "meets_two_year_minimum": span_days >= 730,
    }


def dataset_hash(symbol: str) -> str:
    """Content hash over every partition, so a backtest can pin the dataset."""
    digest = hashlib.sha256()
    root = TICK_ROOT / symbol.upper()
    for path in sorted(root.glob("*/*/ticks.parquet")) if root.exists() else []:
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()
