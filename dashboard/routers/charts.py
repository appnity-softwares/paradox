"""Chart data endpoints (Part 14.1): candlesticks from REAL ingested
ticks, plus equity/drawdown curves and trade markers from a
deterministic backtest on those same real bars. Labels are honest:
chart data is DELAYED (historical), never live.
"""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd

from dashboard.platform_state import PLATFORM
from dashboard.routers.platform import require
from core.marketdata.resampler import resample_ticks_to_bars
from core.marketdata.store import read_range
from strategies.base import StrategyConfig
from strategies.library import EMACross

router = APIRouter(prefix="/api/platform/charts", tags=["charts"])

TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
MAX_BARS = 2_000


def _bars(symbol: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    if tf not in TIMEFRAMES:
        raise HTTPException(status_code=422, detail="unknown timeframe")
    if end_ms <= start_ms:
        raise HTTPException(status_code=422, detail="empty range")
    ticks = read_range(symbol, start_ms, end_ms)
    if ticks.empty:
        raise HTTPException(status_code=404, detail="no data in range")
    return resample_ticks_to_bars(ticks, tf).tail(MAX_BARS)


@router.get("/bars")
def bars(symbol: str = "XAUUSD", tf: str = "15m",
         start_ms: int = Query(...), end_ms: int = Query(...),
         user: dict = Depends(require("viewer"))) -> dict:
    df = _bars(symbol, tf, start_ms, end_ms)
    rows = [{"t": str(ts), "o": float(r["open"]), "h": float(r["high"]),
             "l": float(r["low"]), "c": float(r["close"]),
             "v": float(r["volume"])} for ts, r in df.iterrows()]
    return {"symbol": symbol, "timeframe": tf, "source_label": "DELAYED",
            "source_detail": "real ingested ticks, historical by "
                             "definition — never live",
            "bars": rows}


@router.get("/backtest")
def backtest(symbol: str = "XAUUSD", tf: str = "15m",
             start_ms: int = Query(...), end_ms: int = Query(...),
             fast: int = 20, slow: int = 60, sl_atr_mult: float = 1.5,
             rr: float = 1.5, session: str = "london_ny",
             risk_pct: float = 0.5,
             user: dict = Depends(require("viewer"))) -> dict:
    df = _bars(symbol, tf, start_ms, end_ms)
    cfg = StrategyConfig(sl_method="atr", sl_atr_mult=sl_atr_mult, rr=rr,
                         session=session, risk_pct=risk_pct)
    equity, trades = _run(df, EMACross(cfg, fast=fast, slow=slow), cfg,
                          symbol)
    eq = equity.values
    hwm = eq[0]
    dd = [0.0]
    for v in eq[1:]:
        hwm = max(hwm, v)
        dd.append((hwm - v) / hwm * 100.0)
    markers = [{"entry_time": str(t["entry_time"]),
                "exit_time": str(t["exit_time"]), "side": str(t["side"]),
                "entry": float(t["entry"]), "exit": float(t["exit"]),
                "pnl": float(t["pnl"]), "reason": str(t["reason"])}
               for _, t in trades.iterrows()]
    return {"symbol": symbol, "timeframe": tf,
            "source_label": "BACKTEST",
            "source_detail": "deterministic backtest on real bars — "
                             "not live and not a forecast",
            "equity": [{"t": str(ts), "v": float(v)}
                       for ts, v in equity.items()],
            "drawdown_pct": [{"t": str(ts), "v": float(v)}
                             for ts, v in zip(equity.index, dd)],
            "trades": markers,
            "summary": {"n_trades": len(markers),
                        "final_equity": float(eq[-1]),
                        "max_drawdown_pct": max(dd)}}


@router.get("/replay")
def replay(symbol: str = "XAUUSD", tf: str = "15m",
           start_ms: int = Query(...), end_ms: int = Query(...),
           fast: int = 20, slow: int = 60, sl_atr_mult: float = 1.5,
           rr: float = 1.5, session: str = "london_ny",
           risk_pct: float = 0.5,
           user: dict = Depends(require("viewer"))) -> dict:
    """Bar-by-bar replay state: for every bar index, which trades are
    open, which closed on that bar, and the equity at that point.
    Deterministic — the same data and config always yield the same
    replay."""
    df = _bars(symbol, tf, start_ms, end_ms)
    cfg = StrategyConfig(sl_method="atr", sl_atr_mult=sl_atr_mult, rr=rr,
                         session=session, risk_pct=risk_pct)
    equity, trades = _run(df, EMACross(cfg, fast=fast, slow=slow), cfg,
                          symbol)
    times = [str(ts) for ts in df.index]
    index_of = {t: i for i, t in enumerate(times)}
    # events: (bar_index, kind, trade)
    open_at: list[list[int]] = [[] for _ in times]      # trade ids open
    closed_on: list[list[dict]] = [[] for _ in times]   # trades closed here
    for tid, (_, t) in enumerate(trades.iterrows()):
        ei = index_of.get(str(t["entry_time"]))
        xi = index_of.get(str(t["exit_time"]))
        if ei is None or xi is None:
            continue
        for i in range(ei, min(xi + 1, len(times))):
            open_at[i].append(tid)
        closed_on[xi].append({
            "id": tid, "side": str(t["side"]), "pnl": float(t["pnl"]),
            "reason": str(t["reason"]), "entry": float(t["entry"]),
            "exit": float(t["exit"])})
    eq = equity.values
    hwm, dd = eq[0], [0.0]
    for v in eq[1:]:
        hwm = max(hwm, v)
        dd.append((hwm - v) / hwm * 100.0)
    return {"symbol": symbol, "timeframe": tf,
            "source_label": "BACKTEST_REPLAY",
            "source_detail": "deterministic replay of a backtest on "
                             "real bars — not live, not a forecast",
            "times": times,
            "ohlc": [[float(r["open"]), float(r["high"]), float(r["low"]),
                      float(r["close"])] for _, r in df.iterrows()],
            "open_trades": open_at,
            "closed_on_bar": closed_on,
            "equity": [float(v) for v in eq],
            "drawdown_pct": dd}


def _run(df, strategy, cfg, symbol):
    from core.backtest import run_backtest
    return run_backtest(df, strategy, cfg, start_equity=10_000.0,
                        symbol=symbol)
