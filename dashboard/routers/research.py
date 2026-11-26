"""Prop-firm simulation, vertex tick store and market RAG."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from dashboard.state import (
    ALL_STRATEGIES,
    MARKET_RAG_ENGINE,
    PARAM_GRID,
    PROP_FIRM_PROFILES,
    PropFirmChallengeSimulator,
    PropFirmProfile,
    PropFirmSimulateReq,
    StrategyConfig,
    VERTEX_STORE,
    asdict,
    fetch_market_data,
    run_backtest,
)
from core.vertex_store import NoTickDataError

router = APIRouter(tags=["research"])


@router.get("/api/prop_firm/profiles")
def get_prop_firm_profiles():
    """Return all pre-configured institutional prop firm challenge profiles."""
    return {
        "profiles": [asdict(p) for p in PROP_FIRM_PROFILES.values()],
        "default": "ftmo_100k",
        "disclaimer": ("Rule values are an UNVERIFIED snapshot from public "
                       "information. Rules change often — confirm every "
                       "number on the firm's own rules page before "
                       "purchasing a challenge."),
    }


@router.post("/api/prop_firm/simulate")
def simulate_prop_firm_challenge(req: PropFirmSimulateReq):
    """Run Monte Carlo simulation of prop firm challenge against strategy backtest trades."""
    if req.custom_profile:
        try:
            profile = PropFirmProfile(
                id="custom",
                name=req.custom_profile.get("name", "Custom Challenge"),
                initial_balance=float(req.custom_profile.get("initial_balance", 100_000.0)),
                profit_target_pct=float(req.custom_profile.get("profit_target_pct", 10.0)),
                max_daily_loss_pct=float(req.custom_profile.get("max_daily_loss_pct", 5.0)),
                max_total_loss_pct=float(req.custom_profile.get("max_total_loss_pct", 10.0)),
                min_trading_days=int(req.custom_profile.get("min_trading_days", 4)),
            )
        except Exception as exc:
            raise HTTPException(400, f"Invalid custom profile: {exc}")
    else:
        if req.profile_id not in PROP_FIRM_PROFILES:
            raise HTTPException(404, f"Prop firm profile '{req.profile_id}' not found")
        profile = PROP_FIRM_PROFILES[req.profile_id]

    # Run backtest to generate trade list
    if req.strategy not in ALL_STRATEGIES:
        raise HTTPException(400, f"unknown strategy: {req.strategy}")
    allowed = set(PARAM_GRID.get(req.strategy, {}).keys())
    params = {key: value for key, value in req.params.items() if key in allowed}
    snapshot = fetch_market_data(req.timeframe, 20_000, allow_synthetic=False)
    df = snapshot.data

    cfg = StrategyConfig(
        risk_pct=req.risk_pct,
        rr=req.rr,
        sl_method=req.sl_method,
        sl_atr_mult=req.sl_atr_mult,
        sl_fixed=req.sl_fixed,
        session=req.session,
        max_trades_per_day=req.max_trades_per_day,
        daily_loss_limit_pct=req.daily_loss_limit_pct,
    )
    try:
        strategy = ALL_STRATEGIES[req.strategy](cfg, **params)
        _, trades = run_backtest(df, strategy, cfg, start_equity=profile.initial_balance)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

    simulator = PropFirmChallengeSimulator(profile)
    sim_result = simulator.simulate_from_trades(
        trades=trades,
        num_simulations=req.num_simulations,
    )
    sim_result["strategy"] = req.strategy
    sim_result["trades_in_sample"] = len(trades)
    sim_result["market"] = snapshot.metadata()
    return sim_result


@router.get("/api/vertex/status")
def get_vertex_status():
    """Return storage size, tick count, compression ratio, and vector count."""
    return VERTEX_STORE.get_status()


@router.post("/api/vertex/convert")
def convert_vertex_data():
    """Ingest real MT5 ticks into compressed vertex parquet and embeddings."""
    try:
        return VERTEX_STORE.build_and_compress_vertex_store(force_refresh=True)
    except NoTickDataError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/api/rag/search")
def search_rag_patterns(top_k: int = Query(5, ge=1, le=20)):
    """Search for the most similar historical microstructural patterns."""
    return {
        "context": MARKET_RAG_ENGINE.get_market_pattern_rag_context(),
        "matches": MARKET_RAG_ENGINE.search_similar_patterns(top_k=top_k),
    }
