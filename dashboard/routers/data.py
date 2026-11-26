"""Market data: bars, quotes, ticks and the live tick websocket."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, WebSocket, WebSocketDisconnect

from dashboard.state import (
    ALL_STRATEGIES,
    DATA_MANAGER,
    EXECUTION_ENGINE,
    MARKET_BROADCASTER,
    PARAM_GRID,
    Timeframe,
    asyncio,
    datetime,
    fetch_market_data,
    math,
    pd,
    timezone,
)

router = APIRouter(tags=["data"])


def _live_tick_available() -> bool:
    """True when the active provider can actually return a quote right now."""
    provider = DATA_MANAGER.active
    if provider is None:
        return False
    try:
        return provider.fetch_tick("XAUUSD") is not None
    except Exception:
        return False


@router.get("/api/info")
def info():
    snapshot = fetch_market_data("15m", 5000, allow_synthetic=True)
    metadata = snapshot.metadata()
    return {
        "bars": int(len(snapshot.data)),
        "start": str(snapshot.data.index[0]),
        "end": str(snapshot.data.index[-1]),
        "tf": "15m",
        "strategies": list(ALL_STRATEGIES.keys()),
        "grids": PARAM_GRID,
        "data_source": metadata["provider"],
        "data_manager": DATA_MANAGER.summary(),
        "is_real": metadata["real_data"],
        "is_spot": metadata["spot"],
        "delayed": metadata["delayed"],
        "synthetic": metadata["synthetic"],
        "stale": metadata["stale"],
        "proxy": metadata.get("proxy", False),
        "sanitized_bars": metadata.get("sanitized_bars", 0),
        "data_as_of": metadata["data_as_of"],
        "has_execution": DATA_MANAGER.has_execution(),
        "tick_feed": _live_tick_available(),
    }


@router.get("/api/data/coverage")
def data_coverage(symbol: str = "XAUUSD"):
    from core.marketdata import store
    cov = store.coverage(symbol)
    return {
        "symbol": symbol,
        "coverage": cov,
        "dataset_hash": store.dataset_hash(symbol),
        "meets_two_year_minimum": cov.get("meets_two_year_minimum", False),
    }


@router.get("/api/data/ingest/status")
def data_ingest_status():
    from pathlib import Path
    status_file = Path(__file__).resolve().parent.parent.parent / "data" / "ticks" / "ingest_progress.json"
    if status_file.exists():
        try:
            import json
            return json.loads(status_file.read_text())
        except Exception:
            pass
    return {"state": "IDLE", "progress_pct": 0, "ticks_so_far": 0}


@router.get("/api/data")
def get_data(response: Response, tf: Timeframe = "15m", bars: int = Query(2000, ge=1, le=100_000), refresh: bool = False, symbol: str = "XAUUSD"):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    snapshot = fetch_market_data(tf, bars, symbol=symbol, force_refresh=refresh, allow_synthetic=False)
    records = []
    for timestamp, row in snapshot.data.iterrows():
        records.append({
            "time": int(timestamp.timestamp()),
            "open": round(float(row["open"]), 5),
            "high": round(float(row["high"]), 5),
            "low": round(float(row["low"]), 5),
            "close": round(float(row["close"]), 5),
            "volume": int(row["volume"]) if math.isfinite(float(row["volume"])) else 0,
        })
    return {
        "tf": tf,
        "symbol": symbol,
        "bars": len(snapshot.data),
        "data": records,
        "meta": snapshot.metadata(),
    }


@router.get("/api/quote")
def quote(response: Response, symbol: str = "XAUUSD", refresh: bool = True):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    if DATA_MANAGER.active is not None and DATA_MANAGER.is_spot():
        try:
            tick = DATA_MANAGER.active.fetch_tick(symbol)
            if tick:
                bid = float(tick["bid"])
                ask = float(tick["ask"])
                tick_time = tick["time"]
                # Only real broker XAUUSD ticks feed simulated fills — the
                # PAXG spot proxy is a different instrument and must not.
                if DATA_MANAGER.active.key() == "mt5_mac":
                    EXECUTION_ENGINE.update_market_price(symbol, bid, ask)
                age_seconds = (datetime.now(timezone.utc) - tick_time.astimezone(timezone.utc)).total_seconds()
                return {
                    "symbol": symbol,
                    "time": tick_time.isoformat(),
                    "bid": bid,
                    "ask": ask,
                    "price": (bid + ask) / 2,
                    "provider": DATA_MANAGER.active_name,
                    "delayed": age_seconds > 60,
                    "stale": age_seconds > 300,
                    "age_seconds": round(age_seconds, 1),
                }
        except Exception as exc:
            raise HTTPException(503, f"MT5 quote failed: {exc}") from exc

    snapshot = None
    # Synthetic fallback is a gold-only feature; another symbol with no
    # local data must answer "no data", never a fabricated price.
    allow_syn = symbol.upper() in ("XAUUSD",)
    for tf in ("1m", "15m", "1h", "1d"):
        try:
            snapshot = fetch_market_data(tf, 2, symbol=symbol, force_refresh=False,
                                         allow_stale=True, allow_synthetic=allow_syn)
            break
        except Exception:
            continue
    if snapshot is None:
        raise HTTPException(503, f"no local data for {symbol}")
    last = snapshot.data.iloc[-1]
    return {
        "symbol": symbol,
        "time": snapshot.data.index[-1].isoformat(),
        "bid": None,
        "ask": None,
        "price": float(last["close"]),
        "provider": snapshot.provider,
        "delayed": snapshot.delayed,
        "stale": snapshot.stale,
    }


def _real_tick_source(symbol: str, count: int):
    """Return (provider, DataFrame) from the first provider with REAL ticks.

    Bar-derived pseudo-ticks are never returned: the validated path must not
    contain synthesised data (Master Prompt Part 6.1).
    """
    ordered = []
    if DATA_MANAGER.active is not None:
        ordered.append(DATA_MANAGER.active)
    ordered.extend(p for p, _ in DATA_MANAGER.statuses if p not in ordered)

    errors = []
    for provider in ordered:
        try:
            df = provider.fetch_ticks(symbol, count)
        except Exception as exc:
            errors.append(f"{provider.name()}: {exc}")
            continue
        if df is not None and len(df) > 0:
            return provider, df
    detail = "No real tick source available."
    if errors:
        detail += " " + " | ".join(errors)
    raise HTTPException(503, detail)


def _tick_records(df, provider):
    records = []
    for timestamp, row in df.iterrows():
        bid, ask = float(row["bid"]), float(row["ask"])
        last = float(row.get("last", 0.0))
        records.append({
            "time": timestamp.isoformat(),
            "time_ms": int(timestamp.timestamp() * 1000),
            "bid": bid,
            "ask": ask,
            "last": last if last > 0 else (bid + ask) / 2,
            "volume": float(row.get("volume", 0.0)),
            "source": provider.name(),
        })
    return records


def _tick_payload(symbol: str, df, provider, extra: dict | None = None):
    newest = df.index[-1].to_pydatetime()
    age = (datetime.now(timezone.utc) - newest).total_seconds()
    payload = {
        "symbol": symbol,
        "count": len(df),
        "ticks": _tick_records(df, provider),
        "source": provider.name(),
        "is_real_tick": True,
        "proxy": bool(getattr(provider.is_available(), "proxy", False)),
        "data_as_of": newest.isoformat(),
        "age_seconds": round(age, 1),
        "stale": age > 300,
    }
    if payload["stale"]:
        payload["note"] = (
            f"Real ticks, but the newest is {round(age / 3600, 1)}h old. "
            "Live ticks stream separately over the websocket."
        )
    if extra:
        payload.update(extra)
    return payload


@router.get("/api/ticks")
def get_ticks(
    response: Response,
    symbol: str = "XAUUSD",
    count: int = Query(1000, ge=1, le=20_000),
    from_time: str | None = None,
    to_time: str | None = None,
):
    """Real historical bid/ask ticks. Fails closed when no real source exists."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    provider, df = _real_tick_source(symbol, count)
    if from_time:
        df = df[df.index >= pd.Timestamp(from_time, tz="UTC")]
    if to_time:
        df = df[df.index <= pd.Timestamp(to_time, tz="UTC")]
    df = df.tail(count)
    if df.empty:
        raise HTTPException(404, "No real ticks inside the requested range.")
    return _tick_payload(symbol, df, provider)


@router.get("/api/ticks/history")
def get_ticks_history(
    symbol: str = "XAUUSD",
    from_time: str = Query(..., description="ISO datetime e.g. 2026-11-01T00:00:00Z"),
    to_time: str = Query(..., description="ISO datetime e.g. 2026-11-21T00:00:00Z"),
    tf: str = Query("1m", description="Unused: kept for client compatibility"),
    max_bars: int = Query(5000, ge=1, le=50_000),
):
    """Real ticks inside a time range. Never substitutes OHLCV bars for ticks."""
    try:
        ft = pd.Timestamp(from_time)
        ft = ft.tz_localize("UTC") if ft.tzinfo is None else ft.tz_convert("UTC")
        tt = pd.Timestamp(to_time)
        tt = tt.tz_localize("UTC") if tt.tzinfo is None else tt.tz_convert("UTC")
    except Exception as exc:
        raise HTTPException(422, f"Invalid time format: {exc}") from exc

    provider, df = _real_tick_source(symbol, max_bars)
    covered = df[(df.index >= ft) & (df.index <= tt)]
    if covered.empty:
        raise HTTPException(
            404,
            "No real ticks in that range. Available real tick coverage: "
            f"{df.index[0].isoformat()} to {df.index[-1].isoformat()} ({provider.name()}).",
        )
    return _tick_payload(symbol, covered, provider, {"from": ft.isoformat(), "to": tt.isoformat()})


@router.websocket("/ws/market/{symbol}")
async def market_stream(websocket: WebSocket, symbol: str):
    await websocket.accept()
    if not DATA_MANAGER.is_spot():
        await websocket.send_json({
            "type": "unavailable",
            "message": "Spot gold tick feed is not active",
            "provider": DATA_MANAGER.active_name,
        })
        return

    q = MARKET_BROADCASTER.subscribe()
    try:
        # Immediately send current quote upon connection
        if DATA_MANAGER.active is not None and DATA_MANAGER.is_spot():
            try:
                initial_tick = await asyncio.to_thread(DATA_MANAGER.active.fetch_tick, symbol)
                if initial_tick:
                    bid = float(initial_tick["bid"])
                    ask = float(initial_tick["ask"])
                    await websocket.send_json({
                        "type": "tick",
                        "symbol": symbol,
                        "time": initial_tick["time"].isoformat(),
                        "time_ms": int(initial_tick["time"].timestamp() * 1000),
                        "bid": bid,
                        "ask": ask,
                        "price": initial_tick.get("last", (bid + ask) / 2),
                        "volume": float(initial_tick.get("volume", 1.0)),
                        "provider": DATA_MANAGER.active_name,
                        "is_real_tick": True,
                        "delayed": False,
                        "stale": False,
                    })
            except Exception:
                pass

        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=2.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Send periodic heartbeat so client connection never drops
                await websocket.send_json({
                    "type": "ping",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "provider": DATA_MANAGER.active_name,
                })
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        pass
    finally:
        MARKET_BROADCASTER.unsubscribe(q)
