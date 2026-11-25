"""MT5 bridge: status, configuration, command polling and account info."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path

from dashboard.state import (
    DATA_MANAGER,
    EXECUTION_ENGINE,
    MARKET_BROADCASTER,
    MT5ConfigPayload,
    MT5LiveTickPayload,
    PROTOTYPE_ROOT,
    SyncMacReq,
    datetime,
    get_mac_mt5_status,
    os,
    sync_from_exported_csv,
    timezone,
)

router = APIRouter(tags=["mt5"])


@router.get("/api/mt5/status")
def mt5_status():
    """Detailed MT5 / data-provider status for the UI."""
    statuses = DATA_MANAGER.all_statuses()
    return {
        "active": DATA_MANAGER.active_name,
        "has_execution": DATA_MANAGER.has_execution(),
        "is_spot": DATA_MANAGER.is_spot(),
        "providers": [
            {
                "name": name,
                "available": st.available,
                "reason": st.reason,
                "real_data": st.real_data,
                "spot": st.spot,
                "execution": st.execution,
                "delayed": st.delayed,
            }
            for name, st in statuses
        ],
    }


@router.get("/api/account")
def account_info():
    """MT5 account info if connected, else placeholder."""
    if DATA_MANAGER.active is not None and DATA_MANAGER.has_execution():
        try:
            info = DATA_MANAGER.active.account_info()
            if info:
                return {"connected": True, "source": "mt5", **info}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
    return {"connected": False, "note": "MT5 not connected. Using Yahoo Finance (read-only, no execution)."}


@router.post("/api/mt5/config")
def mt5_config(payload: MT5ConfigPayload):
    """Save MT5 settings to .env, update runtime credentials, and probe MT5 connection."""
    from pathlib import Path
    env_path = Path(PROTOTYPE_ROOT) / ".env"
    existing_lines = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    env_dict = {}
    for line in existing_lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            env_dict[k.strip()] = v.strip()

    if payload.login is not None:
        val = str(payload.login).strip()
        os.environ["MT5_LOGIN"] = val
        env_dict["MT5_LOGIN"] = val
    if payload.password is not None:
        val = str(payload.password)
        os.environ["MT5_PASSWORD"] = val
        env_dict["MT5_PASSWORD"] = val
    if payload.server is not None:
        val = str(payload.server).strip()
        os.environ["MT5_SERVER"] = val
        env_dict["MT5_SERVER"] = val
    if payload.path is not None:
        val = str(payload.path).strip()
        if val:
            os.environ["MT5_PATH"] = val
            env_dict["MT5_PATH"] = val
        elif "MT5_PATH" in env_dict:
            del env_dict["MT5_PATH"]
            os.environ.pop("MT5_PATH", None)

    out_lines = []
    keys_written = set()
    for line in existing_lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in env_dict:
                out_lines.append(f"{k}={env_dict[k]}")
                keys_written.add(k)
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
    for k, v in env_dict.items():
        if k not in keys_written:
            out_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # Update MT5 broker instance if present in providers
    for provider, _ in DATA_MANAGER.statuses:
        if getattr(provider, "name", None) and provider.name() == "MetaTrader5":
            if payload.login is not None:
                provider.login = str(payload.login).strip()
            if payload.password is not None:
                provider.password = str(payload.password)
            if payload.server is not None:
                provider.server = str(payload.server).strip()
            if payload.path is not None:
                provider.path = str(payload.path).strip()

    # Re-probe primary provider
    DATA_MANAGER.refresh_primary()

    return {
        "success": True,
        "active": DATA_MANAGER.active_name,
        "has_execution": DATA_MANAGER.has_execution(),
        "is_spot": DATA_MANAGER.is_spot(),
        "statuses": [
            {"name": name, "available": st.available, "reason": st.reason}
            for name, st in DATA_MANAGER.all_statuses()
        ],
    }


@router.post("/api/mt5/tick")
async def mt5_live_tick_webhook(payload: MT5LiveTickPayload):
    """Receive real-time live tick directly from MT5 Expert Advisor (AI_Quant_Live_Bridge.mq5)."""
    for p, _ in DATA_MANAGER.statuses:
        if getattr(p, "record_live_tick", None):
            p.record_live_tick(payload.dict())
    
    bid = float(payload.bid or (payload.last - 0.15 if payload.last else 3945.60))
    ask = float(payload.ask or (payload.last + 0.15 if payload.last else 3945.90))
    EXECUTION_ENGINE.update_market_price(payload.symbol, bid, ask)

    now = datetime.now(timezone.utc)
    tick_msg = {
        "type": "tick",
        "symbol": payload.symbol,
        "time": now.isoformat(),
        "time_ms": payload.time_ms or int(now.timestamp() * 1000),
        "bid": bid,
        "ask": ask,
        "price": payload.last or (bid + ask) / 2,
        "volume": float(payload.volume or 1.0),
        "provider": DATA_MANAGER.active_name,
        "is_real_tick": True,
        "delayed": False,
        "stale": False,
    }
    await MARKET_BROADCASTER.broadcast(tick_msg)

    return {"status": "ok", "symbol": payload.symbol, "last": payload.last or payload.bid}


@router.get("/api/mt5/commands")
def mt5_poll_commands():
    """Poll for pending execution commands queued by dashboard for Mac MT5 bridge."""
    try:
        import queue
        cmd = EXECUTION_ENGINE.mt5_commands.get_nowait()
        return cmd
    except Exception:
        return {}


@router.get("/api/mt5/mac_status")
def mt5_mac_status_endpoint():
    """Scan and report macOS MT5 Wine installation, real tick files, and scripts."""
    return get_mac_mt5_status()


@router.post("/api/mt5/sync_mac")
def mt5_sync_mac_endpoint(req: SyncMacReq):
    """Sync an exported MT5 CSV into the real data cache for immediate backtesting."""
    status = get_mac_mt5_status()
    exported = status.get("exported_csvs", [])
    if not exported and not req.csv_filename:
        return {
            "success": False,
            "message": "No exported CSV found in MT5 MQL5/Files yet. In MT5, run the 'Export_XAUUSD' script or right-click chart > Save as CSV into MQL5/Files.",
            "mt5_status": status,
        }

    target_path = None
    if req.csv_filename:
        for f in exported:
            if f["name"] == req.csv_filename:
                target_path = f["path"]
                break
    if not target_path and exported:
        target_path = exported[0]["path"]

    if not target_path:
        raise HTTPException(404, "Target CSV not found")

    res = sync_from_exported_csv(target_path, timeframe=req.timeframe)
    return res
