"""Ingest real tick exports into the partitioned Parquet store.

Supported source: the MT5 terminal export produced by
``scripts/Export_XAUUSD_Ticks.mq5`` (columns: time_msc,bid,ask,last,volume,flags).

Flow per file: normalize -> quality report -> write month partitions ->
manifest entry with content hash -> remove the CSV so disk usage stays bounded.
A file that fails quality is quarantined, never silently repaired.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.marketdata import store
from core.marketdata.quality import quality_report

PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent.parent
QUARANTINE = PROTOTYPE_ROOT / "data" / "ticks_quarantine"
QUARANTINE.mkdir(parents=True, exist_ok=True)
MANIFEST = store.TICK_ROOT / "datasets.json"

MT5_FILES = Path.home() / (
    "Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/"
    "Program Files/MetaTrader 5/MQL5/Files/ticks"
)


def _load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except (OSError, ValueError):
            pass
    return {"datasets": []}


def _save_manifest(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def ingest_csv(path: Path, symbol: str = "XAUUSD", source: str = "mt5_export",
               keep_csv: bool = False) -> dict:
    """Ingest one exported CSV. Returns the dataset record."""
    path = Path(path)
    df = pd.read_csv(path)
    if "time_msc" not in df.columns:
        raise ValueError(f"{path.name}: expected a time_msc column from the MT5 tick export")

    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    if df.empty:
        raise ValueError(f"{path.name}: no usable bid/ask rows")

    ticks = store.normalize(df, source=source)
    report = quality_report(ticks)

    record = {
        "file": path.name,
        "symbol": symbol.upper(),
        "provider": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "ticks": int(len(ticks)),
        "first_ms": int(ticks["time_utc_ms"].iloc[0]),
        "last_ms": int(ticks["time_utc_ms"].iloc[-1]),
        "quality": report,
        "licence_note": "Broker terminal export (MetaQuotes-Demo). Check broker terms before commercial redistribution.",
    }

    if report["status"] == "quarantined":
        target = QUARANTINE / path.name
        shutil.move(str(path), target)
        record["status"] = "quarantined"
        record["quarantined_to"] = str(target)
        manifest = _load_manifest()
        manifest["datasets"].append(record)
        _save_manifest(manifest)
        return record

    times = pd.to_datetime(ticks["time_utc_ms"], unit="ms", utc=True)
    written = []
    for (year, month), group in ticks.groupby([times.dt.year, times.dt.month]):
        out = store.write_partition(group, symbol, int(year), int(month))
        written.append(str(out.relative_to(store.TICK_ROOT)))

    record["status"] = "ok"
    record["partitions"] = written
    record["dataset_hash"] = store.dataset_hash(symbol)
    manifest = _load_manifest()
    manifest["datasets"].append(record)
    manifest["coverage"] = store.coverage(symbol)
    _save_manifest(manifest)

    if not keep_csv:
        path.unlink(missing_ok=True)
    return record


def ingest_folder(folder: Path | None = None, symbol: str = "XAUUSD") -> list[dict]:
    """Ingest every exported CSV in the MT5 files folder, oldest first."""
    folder = Path(folder) if folder else MT5_FILES
    if not folder.exists():
        return []
    results = []
    for path in sorted(folder.glob(f"{symbol.upper()}_ticks_*.csv")):
        try:
            results.append(ingest_csv(path, symbol=symbol))
        except Exception as exc:
            results.append({"file": path.name, "status": "error", "error": str(exc)})
    return results


def export_progress() -> dict:
    """Read the MT5 export script's status file, if present."""
    status_file = MT5_FILES / "_export_status.txt"
    if not status_file.exists():
        return {"state": "NOT_STARTED"}
    out = {}
    for line in status_file.read_text(errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out
