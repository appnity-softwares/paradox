"""Market-data access with provenance-aware, provider-specific caching.

Real, stale and synthetic data are never silently mixed. Dashboard code should
use :func:`fetch_market_data` and display the returned metadata.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from core.broker import MarketDataResult


PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROTOTYPE_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CACHE_LOCK = threading.RLock()

# Freshness controls provider calls without pretending delayed data is a tick feed.
TIMEFRAME_TTL_SECONDS = {
    "1m": 30,
    "2m": 45,
    "5m": 90,
    "15m": 180,
    "30m": 300,
    "1h": 600,
    "4h": 1800,
    "1d": 14_400,
    "1w": 43_200,
    "1mo": 86_400,
}


def _safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def _cache_synchronized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with CACHE_LOCK:
            return function(*args, **kwargs)
    return wrapped


def _cache_path(name: str) -> str:
    """Legacy path helper retained for compatibility."""
    return str(DATA_DIR / name)


def _provider_paths(provider_key: str, symbol: str, timeframe: str) -> tuple[Path, Path]:
    folder = CACHE_DIR / _safe(provider_key) / _safe(symbol)
    folder.mkdir(parents=True, exist_ok=True)
    stem = _safe(timeframe)
    return folder / f"{stem}.csv", folder / f"{stem}.meta.json"


def _read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        if "dt" in df.columns:
            df = df.rename(columns={"dt": "time"})
        elif "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "time"})
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"OHLCV file missing columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601", errors="raise")
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated(keep="last")]


def _write_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = df.copy().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "time"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)


def _interval_to_min(interval: str) -> int:
    if interval.endswith("h"):
        return int(float(interval[:-1]) * 60)
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("d"):
        return int(float(interval[:-1]) * 1440)
    if interval.endswith("w"):
        return int(float(interval[:-1]) * 10080)
    return 15


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def generate_synthetic(
    n_bars: int = 12_000,
    start_price: float = 2400.0,
    seed: int = 42,
    interval_min: int = 15,
) -> pd.DataFrame:
    """Generate deterministic XAUUSD-like bars for explicit offline testing."""
    rng = np.random.default_rng(seed)
    timestamps = []
    timestamp = pd.Timestamp("2025-01-06 00:00", tz="UTC")
    while len(timestamps) < n_bars:
        if timestamp.day_of_week < 5:
            timestamps.append(timestamp)
        timestamp += pd.Timedelta(minutes=interval_min)
    index = pd.DatetimeIndex(timestamps)

    hours = index.hour.to_numpy()
    vol_mult = np.where((hours >= 7) & (hours < 21), 1.6, 0.7).astype(float)
    base_vol = 0.00045 * (interval_min / 15) ** 0.5
    drift = np.zeros(n_bars)
    state = 0.0
    for i in range(n_bars):
        if i % 400 == 0:
            state = rng.choice([-1.0, 0.0, 0.6, -0.6])
        drift[i] = state

    returns = drift * base_vol * 0.35 + base_vol * vol_mult * rng.standard_normal(n_bars)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.empty(n_bars)
    open_[0] = start_price
    open_[1:] = close[:-1]
    noise = np.abs(rng.standard_normal(n_bars)) * base_vol * vol_mult * close * 0.8
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = (rng.integers(500, 3000, n_bars) * vol_mult).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _sanitize_ohlc(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clamp physically implausible wicks (bad ticks) while preserving the body.

    A wick is treated as a bad tick only when it extends more than 6x the local
    median bar range beyond the candle body; it is then clamped to 3x that range.
    The number of repaired bars is reported in metadata so nothing is hidden.
    """
    if df is None or len(df) < 5:
        return df, 0
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    span = (h - l).rolling(200, min_periods=20).median()
    span = span.fillna((h - l).median()).clip(lower=1e-6)
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    bad_hi = h > body_hi + span * 6
    bad_lo = l < body_lo - span * 6
    count = int((bad_hi | bad_lo).sum())
    if count == 0:
        return df, 0
    out = df.copy()
    out.loc[bad_hi, "high"] = (body_hi + span * 3)[bad_hi]
    out.loc[bad_lo, "low"] = (body_lo - span * 3)[bad_lo]
    out["high"] = np.maximum.reduce([out["high"].to_numpy(), out["open"].to_numpy(), out["close"].to_numpy()])
    out["low"] = np.minimum.reduce([out["low"].to_numpy(), out["open"].to_numpy(), out["close"].to_numpy()])
    return out, count


def _bars_stale(result: MarketDataResult, timeframe: str) -> bool:
    """True when the newest bar is older than ~2 intervals (honest freshness)."""
    data = getattr(result, "data", None)
    if data is None or len(data) == 0:
        return False
    last = data.index[-1]
    if getattr(last, "tzinfo", None) is None:
        last = last.tz_localize("UTC")
    interval_min = _interval_to_min(timeframe)
    max_age = max(interval_min * 2 * 60, 900)
    return (_utc_now() - last.to_pydatetime()).total_seconds() > max_age


def _synthetic_result(symbol: str, timeframe: str, bars: int) -> MarketDataResult:
    df = generate_synthetic(
        n_bars=bars,
        interval_min=_interval_to_min(timeframe),
    )
    return MarketDataResult(
        data=df,
        provider="Synthetic generator",
        provider_key="synthetic",
        symbol=symbol,
        timeframe=timeframe,
        real_data=False,
        spot=False,
        delayed=False,
        synthetic=True,
    )


def _read_cached_result(provider_key: str, symbol: str, timeframe: str) -> MarketDataResult | None:
    csv_path, meta_path = _provider_paths(provider_key, symbol, timeframe)
    if not csv_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text())
        df = _read_csv(csv_path)
        return MarketDataResult(
            data=df,
            provider=metadata["provider"],
            provider_key=metadata.get("provider_key", provider_key),
            symbol=metadata.get("symbol", symbol),
            timeframe=metadata.get("timeframe", timeframe),
            real_data=bool(metadata.get("real_data", False)),
            spot=bool(metadata.get("spot", False)),
            delayed=bool(metadata.get("delayed", False)),
            synthetic=bool(metadata.get("synthetic", False)),
            stale=bool(metadata.get("stale", False)),
            fetched_at=_parse_dt(metadata.get("fetched_at")),
            data_as_of=_parse_dt(metadata.get("data_as_of")),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_cached_result(result: MarketDataResult, *, max_rows: int = 100_000) -> None:
    csv_path, meta_path = _provider_paths(
        result.provider_key,
        result.symbol,
        result.timeframe,
    )
    existing = _read_csv(csv_path) if csv_path.exists() else None
    merged = result.data if existing is None else pd.concat([existing, result.data])
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")].tail(max_rows)
    _write_csv(merged, csv_path)
    cached_result = replace(result, data=merged, stale=False)
    meta_path.write_text(json.dumps(cached_result.metadata(), indent=2))


def _is_fresh(result: MarketDataResult, max_age_seconds: int) -> bool:
    if result.fetched_at is None:
        return False
    fetched_at = result.fetched_at
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return (_utc_now() - fetched_at).total_seconds() <= max_age_seconds


def _legacy_cache(timeframe: str, symbol: str) -> MarketDataResult | None:
    path = DATA_DIR / f"xauusd_{timeframe}_real.csv"
    if not path.exists() or symbol.upper() != "XAUUSD":
        return None
    df = _read_csv(path)
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return MarketDataResult(
        data=df,
        provider="Legacy real cache (provider unverified)",
        provider_key="legacy",
        symbol=symbol,
        timeframe=timeframe,
        real_data=True,
        spot=False,
        delayed=True,
        stale=True,
        fetched_at=fetched_at,
    )


@_cache_synchronized
def fetch_market_data(
    timeframe: str = "15m",
    bars: int = 5000,
    *,
    symbol: str = "XAUUSD",
    force_refresh: bool = False,
    max_age_seconds: int | None = None,
    allow_stale: bool = True,
    allow_synthetic: bool = False,
) -> MarketDataResult:
    """Return market data with truthful source and freshness metadata."""
    if bars <= 0 or bars > 100_000:
        raise ValueError("bars must be between 1 and 100000")

    from core.data_manager import get_manager

    manager = get_manager()
    manager.refresh_primary()
    active_key = manager.active.key() if manager.active is not None else "none"
    cached = _read_cached_result(active_key, symbol, timeframe)
    ttl = max_age_seconds if max_age_seconds is not None else TIMEFRAME_TTL_SECONDS.get(timeframe, 300)

    required_cached_rows = min(bars, 1000)
    if (
        cached is not None
        and len(cached.data) >= required_cached_rows
        and not force_refresh
        and _is_fresh(cached, ttl)
    ):
        clean, repaired = _sanitize_ohlc(cached.data.tail(bars))
        return replace(
            cached,
            data=clean,
            stale=cached.stale or _bars_stale(cached, timeframe),
            sanitized_bars=repaired,
        )

    try:
        fresh = manager.fetch_snapshot(
            symbol,
            timeframe,
            bars,
            allow_synthetic=False,
        )
        _write_cached_result(fresh)
        clean, repaired = _sanitize_ohlc(fresh.data.tail(bars))
        return replace(
            fresh,
            data=clean,
            stale=_bars_stale(fresh, timeframe),
            sanitized_bars=repaired,
        )
    except RuntimeError:
        if allow_stale:
            fallback = cached or _legacy_cache(timeframe, symbol)
            if fallback is not None:
                clean, repaired = _sanitize_ohlc(fallback.data.tail(bars))
                return replace(fallback, data=clean, stale=True, sanitized_bars=repaired)
        if allow_synthetic:
            return _synthetic_result(symbol, timeframe, bars)
        raise


def load_data_result(
    csv_path: str | None = None,
    force_synthetic: bool = False,
    interval: str = "15m",
) -> MarketDataResult:
    """Load CLI data while preserving source metadata."""
    if csv_path:
        path = Path(csv_path)
        return MarketDataResult(
            data=_read_csv(path),
            provider="User CSV (unverified)",
            provider_key="user_csv",
            symbol="XAUUSD",
            timeframe=interval,
            real_data=False,
            spot=False,
            delayed=False,
            fetched_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
        )
    if force_synthetic:
        return _synthetic_result("XAUUSD", interval, 12_000)
    # Research/validation CLIs never fall back to synthetic data silently:
    # if no real feed is reachable the caller gets a clear RuntimeError.
    return fetch_market_data(
        interval,
        5000,
        allow_stale=True,
        allow_synthetic=False,
    )


def load_data(
    csv_path: str | None = None,
    force_synthetic: bool = False,
    interval: str = "15m",
) -> pd.DataFrame:
    """DataFrame-only compatibility loader."""
    return load_data_result(csv_path, force_synthetic, interval).data


def fetch_data(
    timeframe: str = "15m",
    bars: int = 5000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """DataFrame-only compatibility wrapper. Always respects ``bars``."""
    return fetch_market_data(
        timeframe,
        bars,
        force_refresh=force_refresh,
        allow_stale=True,
        allow_synthetic=True,
    ).data


def get_data_source(interval: str = "15m") -> str:
    """Return the newest cached provider label for a timeframe."""
    candidates = list(CACHE_DIR.glob(f"*/*/{_safe(interval)}.meta.json"))
    if candidates:
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        try:
            metadata = json.loads(newest.read_text())
            label = metadata.get("provider", "Unknown")
            if metadata.get("stale"):
                label += " (stale)"
            return label
        except (OSError, json.JSONDecodeError):
            pass
    legacy = _legacy_cache(interval, "XAUUSD")
    return legacy.provider if legacy else "Unknown"


def fetch_yfinance(
    symbol: str = "GC=F",
    interval: str = "15m",
    period: str | None = None,
):
    """Backward-compatible direct Yahoo fetch."""
    from core.brokers.yahoo_broker import YahooBroker

    return YahooBroker().fetch_ohlcv(timeframe=interval, bars=5000)
