"""Time Machine API — replay a trade's odds before it is placed."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.data import fetch_market_data
from core.replay import replay_trade, suggest_stops

router = APIRouter(tags=["replay"])


class ReplayRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "15m"
    side: str = "buy"
    entry: float
    stop: float
    target: float
    horizon_bars: int = 16
    replays: int = 2000
    seed: int = 19


class SuggestRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "15m"
    side: str = "buy"
    entry: float
    atr_mult: float = 1.5
    r_multiple: float = 2.0


@router.post("/api/replay")
def run_replay(req: ReplayRequest):
    """Odds for entry/stop/target from the instrument's own recent bars.

    Simulated from real recorded data — not a prediction, never a
    recommendation. Synthetic fallback is never used here: no real bars
    means an honest error, not fabricated odds.
    """
    try:
        snapshot = fetch_market_data(req.timeframe, 6000, symbol=req.symbol,
                                     force_refresh=False, allow_synthetic=False)
    except Exception as e:
        raise HTTPException(503, f"no real data for {req.symbol}: {e}")
    df = snapshot.data
    if df is None or len(df) < 50:
        raise HTTPException(503, f"not enough real bars for {req.symbol} to replay")
    try:
        result = replay_trade(
            df, entry=req.entry, stop=req.stop, target=req.target,
            side=req.side, horizon_bars=req.horizon_bars,
            replays=req.replays, seed=req.seed,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    out = result.to_dict()
    out["provider"] = snapshot.provider
    out["synthetic"] = False
    return out


@router.post("/api/suggest-stops")
def suggest(req: SuggestRequest):
    """ATR-based stop + R-multiple target suggestion for the ticket."""
    try:
        snapshot = fetch_market_data(req.timeframe, 300, symbol=req.symbol,
                                     force_refresh=False, allow_synthetic=False)
    except Exception as e:
        raise HTTPException(503, f"no real data for {req.symbol}: {e}")
    try:
        out = suggest_stops(snapshot.data, entry=req.entry, side=req.side,
                            atr_mult=req.atr_mult, r_multiple=req.r_multiple)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["provider"] = snapshot.provider
    return out
