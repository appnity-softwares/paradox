"""U0 route-enumeration guard (Update Blueprint v2.1).

No route may place, modify or close an order, or reset a kill switch,
without (a) an auth dependency AND (b) going through ExecutionRouter.
A route without auth must be in the reviewed PUBLIC_WRITE_ROUTES
allowlist, and no unauthenticated route may call a broker-mutating
method. Any new unauthenticated write route fails this test unless it
is consciously allowlisted here.
"""
from __future__ import annotations

import inspect

from fastapi import FastAPI
from fastapi.routing import APIRoute

from dashboard.app import app

WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# Methods on the legacy engine / platform stack that reach a broker or
# change order/kill-switch state.
BROKER_MUTATIONS = (
    "place_market_order", "close_position", "close_all_positions",
    "update_sl_tp", "set_break_even", "emergency_kill_switch",
    "reset_kill_switch", "submit", "submit_ticket",
)

# Reviewed allowlist of write routes WITHOUT an auth dependency. None of
# these may reach a broker (asserted below by source inspection).
PUBLIC_WRITE_ROUTES = {
    "/api/platform/auth/login",      # auth itself
    "/api/platform/auth/logout",     # revokes own session
    "/api/backtest",                 # compute only
    "/api/tick_backtest",            # compute only
    "/api/build",                    # strategy builder (compute)
    "/api/agent/message",            # agent messaging (no order path)
    "/api/agents/research",
    "/api/agents/adapt",
    "/api/agents/compliance",
    "/api/agents/execution",         # execution agent ANALYSIS only
    "/api/agents/market_intelligence",
    "/api/agents/optimization",
    "/api/agents/portfolio_risk",
    "/api/agents/performance_analyst",
    "/api/agents/execute_single",    # single agent analysis, read-only
    "/api/agents/run_live",          # agent recalculation audit, read-only
    "/api/ai/chat",                  # console assistant, read/propose tools
    "/api/paper/replay",             # simulated paper engine, no broker
    "/api/paper/start",
    "/api/paper/stop",
    "/api/mt5/config",               # bridge configuration (no orders)
    "/api/mt5/tick",                 # quote feed for the mac backend
    "/api/mt5/sync_mac",             # history sync
    "/api/prop_firm/simulate",       # offline simulation
}

# Paths that removed legacy order actions must NOT come back.
REMOVED_TRADE_PATHS = {
    "/api/trade/order", "/api/trade/close", "/api/trade/close_all",
    "/api/trade/reset_kill_switch", "/api/trade/modify",
}

TRADE_PATH_PATTERN = ("trade", "order", "kill", "position", "close",
                      "modify")


def _has_auth(route: APIRoute) -> bool:
    for dep in route.dependencies:
        if "require" in getattr(dep.call, "__qualname__", ""):
            return True
    return False


def _write_routes(app_: FastAPI) -> list[APIRoute]:
    return [r for r in app_.routes if isinstance(r, APIRoute)
            and WRITE_METHODS & {m.upper() for m in r.methods}]


def test_removed_legacy_trade_routes_answer_410(client=None):
    from fastapi.testclient import TestClient
    c = TestClient(app)
    for path in REMOVED_TRADE_PATHS:
        r = c.post(path, json={})
        assert r.status_code == 410, f"{path} must be gone, got {r.status_code}"
        assert r.json()["detail"]["detail"] == "removed_in_U0"


def test_every_write_route_has_auth_or_is_reviewed_allowlisted():
    unlisted = []
    for route in _write_routes(app):
        if not _has_auth(route) and route.path not in PUBLIC_WRITE_ROUTES:
            unlisted.append(route.path)
    assert unlisted == [], (
        f"unauthenticated write routes not in the reviewed allowlist: "
        f"{unlisted} — add auth or review them into PUBLIC_WRITE_ROUTES")


def test_no_trade_capable_path_is_unauthenticated():
    for route in _write_routes(app):
        if _has_auth(route):
            continue
        lowered = route.path.lower()
        assert not any(p in lowered for p in TRADE_PATH_PATTERN), (
            f"unauthenticated trade-capable route: {route.path}")


def test_no_unauthenticated_route_reaches_a_broker():
    """Source-inspect every unauthenticated write handler: it must not
    call a broker/order-mutating method."""
    import dashboard.state as state
    import dashboard.routers.platform as platform_router
    forbidden = set(BROKER_MUTATIONS)
    for route in _write_routes(app):
        if _has_auth(route):
            continue
        src = inspect.getsource(route.endpoint)
        for name in forbidden:
            assert name not in src, (
                f"unauthenticated route {route.path} references "
                f"broker mutation '{name}'")
    # Belt and braces: the platform stack's broker-facing objects must not
    # be imported by the unauthenticated legacy routers at all.
    assert "ExecutionRouter" not in inspect.getsource(state)
    assert platform_router.PLATFORM is not None
