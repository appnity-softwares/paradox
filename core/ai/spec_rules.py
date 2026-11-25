"""Declarative entry rules for the StrategySpec interpreter (Part 11.3).

Each rule uses only the project indicator library; semantics match the
legacy signal convention used across the platform (a signal at bar i-1
is acted on at bar i's open).
"""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    out = 100 - 100 / (1 + rs)
    return out.astype(float)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


SUPPORTED_ENTRIES = {
    "ema_cross_up": ("fast", "slow"),          # params: fast, slow
    "ema_cross_down": ("fast", "slow"),
    "rsi_below": ("level", "period"),
    "rsi_above": ("level", "period"),
    "close_above_ema": ("period",),
    "close_below_ema": ("period",),
}


def apply_entry(indicator: str, params: dict,
                bars: pd.DataFrame) -> pd.Series:
    close = bars["close"]
    if indicator == "ema_cross_up":
        fast = ema(close, int(params.get("fast", 20)))
        slow = ema(close, int(params.get("slow", 60)))
        sig = (fast > slow) & (fast.shift() <= slow.shift())
    elif indicator == "ema_cross_down":
        fast = ema(close, int(params.get("fast", 20)))
        slow = ema(close, int(params.get("slow", 60)))
        sig = (fast < slow) & (fast.shift() >= slow.shift())
    elif indicator == "rsi_below":
        sig = rsi(close, int(params.get("period", 14))) < \
            float(params.get("level", 30))
    elif indicator == "rsi_above":
        sig = rsi(close, int(params.get("period", 14))) > \
            float(params.get("level", 70))
    elif indicator == "close_above_ema":
        sig = close > ema(close, int(params.get("period", 50)))
    elif indicator == "close_below_ema":
        sig = close < ema(close, int(params.get("period", 50)))
    else:
        raise ValueError(f"unsupported entry rule {indicator!r}")
    sig = sig.fillna(False).astype(bool)
    sig.iloc[: max(2, int(params.get("warmup", 5)))] = False   # warmup
    return sig
