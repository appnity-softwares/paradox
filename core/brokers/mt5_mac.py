"""MetaTrader 5 Mac Integration.

Directly reads, decodes and synchronizes real MT5 spot data from the native macOS MT5 installation:
~/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5
"""
import os
import struct
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

MT5_MAC_DIR = Path(os.path.expanduser(
    "~/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5"
))

PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROTOTYPE_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


def decode_mt5_hc_file(file_path: Path) -> pd.DataFrame:
    """Decode MetaQuotes binary HistoryCache (.hc) format into a clean pandas DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"MT5 history cache file not found: {file_path}")

    with open(file_path, "rb") as f:
        # Seek past header to count and timestamps offset (0x01ac)
        f.seek(0x01ac)
        c_time = struct.unpack("<I", f.read(4))[0]
        times = struct.unpack(f"<{c_time}q", f.read(c_time * 8))

        c_open = struct.unpack("<I", f.read(4))[0]
        opens = struct.unpack(f"<{c_open}d", f.read(c_open * 8))

        c_high = struct.unpack("<I", f.read(4))[0]
        highs = struct.unpack(f"<{c_high}d", f.read(c_high * 8))

        c_low = struct.unpack("<I", f.read(4))[0]
        lows = struct.unpack(f"<{c_low}d", f.read(c_low * 8))

        c_close = struct.unpack("<I", f.read(4))[0]
        closes = struct.unpack(f"<{c_close}d", f.read(c_close * 8))

        c_vol = struct.unpack("<I", f.read(4))[0]
        vols = struct.unpack(f"<{c_vol}q", f.read(c_vol * 8))

    df = pd.DataFrame({
        "time": [datetime.fromtimestamp(t, timezone.utc) for t in times],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })
    df = df.sort_values("time").drop_duplicates(subset=["time"])
    return df


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1m or 1h OHLCV bars into higher timeframe bars."""
    df_idx = df.set_index("time")
    res = df_idx.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return res


def extract_and_sync_all_mt5_bars() -> dict:
    """Read M1.hc and H1.hc directly from MT5 Mac installation and generate all real timeframes."""
    cache_dir = MT5_MAC_DIR / "bases" / "MetaQuotes-Demo" / "history" / "XAUUSD" / "cache"
    m1_file = cache_dir / "M1.hc"
    h1_file = cache_dir / "H1.hc"

    if not m1_file.exists():
        # Fallback check under any bases subdirectory
        cand = list(MT5_MAC_DIR.glob("bases/*/history/XAUUSD/cache/M1.hc"))
        if cand:
            m1_file = cand[0]
            h1_file = m1_file.parent / "H1.hc"

    if not m1_file.exists():
        return {"success": False, "message": f"M1.hc not found in {cache_dir}"}

    # 1. Decode 1m bars
    df_1m = decode_mt5_hc_file(m1_file)
    df_1m.to_csv(DATA_DIR / "xauusd_1m_real.csv", index=False)

    # 2. Resample 5m, 15m, 30m
    df_5m = resample_bars(df_1m, "5min")
    df_5m.to_csv(DATA_DIR / "xauusd_5m_real.csv", index=False)

    df_15m = resample_bars(df_1m, "15min")
    df_15m.to_csv(DATA_DIR / "xauusd_15m_real.csv", index=False)

    df_30m = resample_bars(df_1m, "30min")
    df_30m.to_csv(DATA_DIR / "xauusd_30m_real.csv", index=False)

    # 3. Decode or resample 1h, 4h, 1d
    if h1_file.exists():
        df_1h = decode_mt5_hc_file(h1_file)
    else:
        df_1h = resample_bars(df_1m, "1h")
    df_1h.to_csv(DATA_DIR / "xauusd_1h_real.csv", index=False)

    df_4h = resample_bars(df_1h, "4h")
    df_4h.to_csv(DATA_DIR / "xauusd_4h_real.csv", index=False)

    df_1d = resample_bars(df_1h, "1D")
    df_1d.to_csv(DATA_DIR / "xauusd_1d_real.csv", index=False)

    # Provenance marker: MacMT5Broker only serves bar files written here, so a
    # file produced by any other feed can never be labelled as MT5 data.
    import json
    (DATA_DIR / "mt5_mac_provenance.json").write_text(json.dumps({
        "source": "mt5_mac_history_cache",
        "symbol": "XAUUSD",
        "m1_file": str(m1_file),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "bars_1m": len(df_1m),
        "data_as_of": df_1m["time"].iloc[-1].isoformat() if "time" in df_1m.columns else None,
    }, indent=2))

    return {
        "success": True,
        "source": "MetaTrader 5 (Mac Cache)",
        "bars_1m": len(df_1m),
        "bars_15m": len(df_15m),
        "start": df_15m["time"].iloc[0].isoformat(),
        "end": df_15m["time"].iloc[-1].isoformat(),
        "last_price": float(df_15m["close"].iloc[-1]),
    }


def get_mac_mt5_status() -> dict:
    """Scan and report macOS MT5 installation and available market data."""
    installed = MT5_MAC_DIR.exists()
    terminal_exe = MT5_MAC_DIR / "terminal64.exe"
    mql5_files = MT5_MAC_DIR / "MQL5" / "Files"
    mql5_scripts = MT5_MAC_DIR / "MQL5" / "Scripts"
    mql5_experts = MT5_MAC_DIR / "MQL5" / "Experts"
    ticks_dir = MT5_MAC_DIR / "bases" / "MetaQuotes-Demo" / "ticks" / "XAUUSD"
    history_dir = MT5_MAC_DIR / "bases" / "MetaQuotes-Demo" / "history" / "XAUUSD"

    files_found = []
    if ticks_dir.exists():
        for f in ticks_dir.glob("*"):
            if f.is_file():
                files_found.append({
                    "name": f.name,
                    "type": "tick_data",
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(),
                })

    if history_dir.exists():
        for f in history_dir.rglob("*.hc*"):
            if f.is_file():
                files_found.append({
                    "name": f.name,
                    "type": "bar_data",
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(),
                })

    local_real_files = list(DATA_DIR.glob("xauusd_*_real.csv"))

    return {
        "installed": installed,
        "path": str(MT5_MAC_DIR),
        "terminal_exe_exists": terminal_exe.exists(),
        "scripts_dir_exists": mql5_scripts.exists(),
        "export_script_installed": (mql5_scripts / "Export_XAUUSD.mq5").exists() if mql5_scripts.exists() else False,
        "bridge_ea_installed": (mql5_experts / "AI_Quant_Live_Bridge.mq5").exists() if mql5_experts.exists() else False,
        "real_ticks_detected": any(f["type"] == "tick_data" for f in files_found),
        "real_bars_detected": any(f["type"] == "bar_data" for f in files_found),
        "files": files_found,
        "synced_real_files": [f.name for f in local_real_files],
    }


def sync_from_exported_csv(csv_path: str | Path, timeframe: str = "15m", symbol: str = "XAUUSD") -> dict:
    """Import and normalize an exported MT5 CSV into the prototype data directory."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    df = pd.read_csv(p)
    col_map = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if "time" in c_low or "date" in c_low:
            col_map[col] = "time"
        elif "open" in c_low:
            col_map[col] = "open"
        elif "high" in c_low:
            col_map[col] = "high"
        elif "low" in c_low:
            col_map[col] = "low"
        elif "close" in c_low:
            col_map[col] = "close"
        elif "vol" in c_low:
            col_map[col] = "volume"

    df = df.rename(columns=col_map)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"])

    out_path = DATA_DIR / f"{symbol.lower()}_{timeframe.lower()}_real.csv"
    df.to_csv(out_path, index=False)

    return {
        "success": True,
        "bars": len(df),
        "start": df["time"].iloc[0].isoformat(),
        "end": df["time"].iloc[-1].isoformat(),
        "output_path": str(out_path),
        "source": "MT5 Mac Direct",
    }


def find_mt5_ticks_file(symbol: str = "XAUUSD") -> Path | None:
    """Locate the terminal's raw tick cache for a symbol, if it exists."""
    candidates = list(MT5_MAC_DIR.glob(f"bases/*/ticks/{symbol}/ticks.dat"))
    candidates += list(MT5_MAC_DIR.glob(f"Bases/*/ticks/{symbol}/ticks.dat"))
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def decode_mt5_ticks_dat(file_path: Path) -> pd.DataFrame:
    """Decode real bid/ask ticks from MT5's raw 60-byte tick records.

    Returns columns: time (UTC DatetimeIndex), bid, ask, last, volume, flags.
    These are genuine broker ticks; nothing is synthesised here.
    """
    if file_path is None or not Path(file_path).exists():
        return pd.DataFrame()

    records = []
    with open(file_path, "rb") as handle:
        handle.seek(0x01AC)
        raw_count = handle.read(4)
        if len(raw_count) < 4:
            return pd.DataFrame()
        count = struct.unpack("<I", raw_count)[0]
        for _ in range(count):
            chunk = handle.read(60)
            if len(chunk) < 60:
                break
            t_sec, bid, ask, last, volume, t_msc, flags = struct.unpack("<qdddQqI8x", chunk)
            if bid <= 0 or ask <= 0:
                continue
            records.append({
                "time_ms": int(t_msc if t_msc > 0 else t_sec * 1000),
                "bid": float(bid),
                "ask": float(ask),
                "last": float(last if last > 0 else (bid + ask) / 2),
                "volume": float(volume if volume > 0 else 0),
                "flags": int(flags),
            })

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
    return df.set_index("time").sort_index()
