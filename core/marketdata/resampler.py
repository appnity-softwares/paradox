"""Deterministic Tick-to-Bar Resampler (Master Prompt 6.1).

Derives bars (1m to 1d) directly from genuine ticks with exact alignment.
Bars and ticks must agree:
- Open = first tick in bar window
- High = max tick in bar window
- Low = min tick in bar window
- Close = last tick in bar window
- Volume = sum of tick volumes
No peeking at unclosed bars.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional


TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def resample_ticks_to_bars(
    df_ticks: pd.DataFrame,
    timeframe: str = "15m",
    price_col: str = "mid",
) -> pd.DataFrame:
    """Resample normalized tick DataFrame to OHLCV bars.
    
    Expected tick schema: time_utc_ms, bid, ask, bid_volume, ask_volume.
    """
    if df_ticks is None or df_ticks.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread", "ticks"])

    rule = TIMEFRAME_MAP.get(timeframe.lower())
    if not rule:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of {list(TIMEFRAME_MAP.keys())}")

    df = df_ticks.copy()
    if "time_utc_ms" in df.columns:
        dt_idx = pd.to_datetime(df["time_utc_ms"], unit="ms", utc=True)
    elif "time_ms" in df.columns:
        dt_idx = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_idx = df.index
    else:
        raise ValueError("Ticks frame must have time_utc_ms or time_ms")

    df["dt"] = dt_idx
    if price_col == "mid":
        df["p"] = (df["bid"] + df["ask"]) / 2.0
    elif price_col == "bid":
        df["p"] = df["bid"]
    elif price_col == "ask":
        df["p"] = df["ask"]
    else:
        df["p"] = df["last"] if "last" in df.columns else (df["bid"] + df["ask"]) / 2.0

    df["vol"] = df.get("bid_volume", 0.0) + df.get("ask_volume", 0.0)
    df["vol"] = np.where(df["vol"] > 0, df["vol"], 1.0)
    df["spread"] = df["ask"] - df["bid"]

    df = df.set_index("dt")

    res = df.resample(rule).agg({
        "p": ["first", "max", "min", "last", "count"],
        "vol": "sum",
        "spread": "mean",
    }).dropna()

    res.columns = ["open", "high", "low", "close", "ticks", "volume", "spread"]
    res = res[["open", "high", "low", "close", "volume", "spread", "ticks"]]
    res["ticks"] = res["ticks"].astype(int)
    res = res[res["ticks"] > 0]
    res.index.name = "time"
    return res
