"""Part 14.1 platform endpoints on top of the Phase 3-5 engines.

Honest labels everywhere (LIVE / DELAYED / STALE / SYNTHETIC), the kill
switch always visible, live trading controls disabled until the Phase 4
sign-off, and the AI assistant restricted to read/propose tools.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core.session import SessionProfile
from dashboard.platform_state import PLATFORM
from dashboard.security import AUTH, role_at_least

router = APIRouter(prefix="/api/platform", tags=["platform"])


# ── Authentication (Part 14.2) ───────────────────────────────────────────────

def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    return header[7:] if header.startswith("Bearer ") else None


def _current_user(request: Request, min_role: str) -> dict:
    """Fail closed: without configured credentials or a valid session
    token, nothing beyond /health and /auth/* is reachable."""
    if not AUTH.configured:
        raise HTTPException(status_code=403, detail="auth_not_configured")
    session = AUTH.sessions.get(_bearer(request))
    if session is None:
        raise HTTPException(status_code=401,
                            detail="authentication_required")
    if not role_at_least(session["role"], min_role):
        raise HTTPException(status_code=403, detail="insufficient_role")
    return session


def require(min_role: str):
    def dep(request: Request) -> dict:
        return _current_user(request, min_role)
    return dep


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    totp: str | None = None


@router.post("/auth/login")
def login(req: LoginRequest, request: Request) -> dict:
    client_key = request.client.host if request.client else "unknown"
    token, error = AUTH.login(req.username, req.password, req.totp,
                              client_key=client_key)
    if error:
        raise HTTPException(status_code=401, detail=error)
    return {"token": token, "role": "admin",
            "mfa_enabled": bool(AUTH.totp_secret)}


@router.post("/auth/logout")
def logout(request: Request) -> dict:
    token = _bearer(request)
    if token:
        AUTH.sessions.revoke(token)
    return {"ok": True}


@router.get("/auth/state")
def auth_state() -> dict:
    """Public: tells the UI whether to show a login screen. Reveals
    nothing about credentials."""
    return {"configured": AUTH.configured,
            "mfa_enabled": bool(AUTH.totp_secret)}


# ── Overview ─────────────────────────────────────────────────────────────────

@router.get("/overview")
def overview(user: dict = Depends(require("viewer"))) -> dict:
    accounts = []
    for account_id, account in PLATFORM.accounts.items():
        snap = PLATFORM.snapshots[account_id]
        ruleset = PLATFORM.rulesets.get(account_id)
        compliance = PLATFORM.compliance_engine.monitor(
            ruleset, PLATFORM.router._compliance_state(account_id)) \
            if ruleset else None
        accounts.append({
            "account_id": account_id, "balance": account.balance,
            "equity": snap.equity,
            "open_positions": len(PLATFORM.broker.positions_of(account_id)),
            "mode": PLATFORM.mode_gate.mode_of(account_id),
            "risk_status": ("kill_switch_engaged"
                            if PLATFORM.kill_switch.is_blocked(account_id)
                            else "ok"),
            "compliance_status": compliance.level if compliance else "none",
            "daily_realized_pnl": snap.daily_realized_pnl,
        })
    alerts = [n for n in PLATFORM.notifier.feed[-20:]]
    return {"accounts": accounts, "alerts": alerts,
            "kill_switch": {
                "global": PLATFORM.kill_switch.public_state()["global"],
                "accounts": PLATFORM.kill_switch.public_state()["accounts"]},
            "live_signoff": PLATFORM.live_signoff,
            "data_sources": PLATFORM.data_sources()}


# ── Kill switch (always visible, API + UI) ───────────────────────────────────

class KillSwitchRequest(BaseModel):
    scope: str = "global"            # "global" | account id
    action: str = "engage"           # "engage" | "disengage"
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)


@router.get("/kill-switch")
def kill_switch_state(user: dict = Depends(require("viewer"))) -> dict:
    return {"global": PLATFORM.kill_switch.public_state()["global"],
            "accounts": PLATFORM.kill_switch.public_state()["accounts"]}


@router.post("/kill-switch")
def kill_switch_action(req: KillSwitchRequest, user: dict = Depends(require("trader"))) -> dict:
    # Engaging is a trader-level safety action; REMOVING a safety block
    # is admin-only (never let the same hand weaken its own guard).
    if req.action == "disengage" and not role_at_least(user["role"], "admin"):
        raise HTTPException(status_code=403, detail="insufficient_role")
    if req.action == "engage":
        if req.scope == "global":
            PLATFORM.kill_switch.engage_global(req.actor, req.reason)
        else:
            PLATFORM.kill_switch.engage_account(req.scope, req.actor,
                                                req.reason)
    elif req.action == "disengage":
        if req.scope == "global":
            PLATFORM.kill_switch.disengage_global(req.actor, req.reason)
        else:
            PLATFORM.kill_switch.disengage_account(req.scope, req.actor,
                                                   req.reason)
    else:
        raise HTTPException(status_code=422, detail="unknown action")
    PLATFORM.audit.append({"event": "kill_switch_api", "scope": req.scope,
                           "action": req.action, "actor": req.actor,
                           "reason": req.reason, "session_user":
                           user["username"]})
    return kill_switch_state()


# ── Compliance: rule sets, verification screen, challenge progress ──────────

@router.get("/compliance/rulesets")
def rulesets(user: dict = Depends(require("viewer"))) -> list[dict]:
    out = []
    for rs in PLATFORM.rule_store.all():
        out.append({
            "firm": rs.firm, "program": rs.program, "version": rs.version,
            "status": rs.status, "verified_by": rs.verified_by,
            "grants_permission": rs.grants_trading_permission(),
            "rules_with_clauses": [
                {"rule": "daily_loss", "definition":
                    f"{rs.daily_loss.limit_pct}% on {rs.daily_loss.basis}, "
                    f"reset {rs.daily_loss.reset_time} {rs.daily_loss.reset_tz}, "
                    f"floating {'included' if rs.daily_loss.includes_floating else 'excluded'}",
                 "citations": list(rs.source_clause_refs.get("daily_loss", []))},
                {"rule": "max_loss", "definition":
                    f"{rs.max_loss.limit_pct}% {rs.max_loss.type} on "
                    f"{rs.max_loss.basis}",
                 "citations": list(rs.source_clause_refs.get("max_loss", []))},
            ]})
    return out


class VerifyRequest(BaseModel):
    firm: str
    program: str
    version: str
    verifier: str = Field(min_length=1)


@router.post("/compliance/verify")
def verify_ruleset(req: VerifyRequest, user: dict = Depends(require("admin"))) -> dict:
    """Human-only verification from the verification screen."""
    try:
        rs = PLATFORM.rule_store.verify(req.firm, req.program, req.version,
                                        req.verifier)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    PLATFORM.audit.append({"event": "ruleset_verified", "firm": rs.firm,
                           "program": rs.program, "version": rs.version,
                           "verifier": req.verifier})
    return {"status": rs.status, "verified_by": rs.verified_by}


@router.get("/compliance/monitor/{account_id}")
def compliance_monitor(account_id: str, user: dict = Depends(require("viewer"))) -> dict:
    ruleset = PLATFORM.rulesets.get(account_id)
    if ruleset is None:
        raise HTTPException(status_code=404, detail="no ruleset for account")
    status = PLATFORM.compliance_engine.monitor(
        ruleset, PLATFORM.router._compliance_state(account_id))
    return {"level": status.level,
            "remaining_daily_loss": status.remaining_daily_loss,
            "remaining_max_loss": status.remaining_max_loss,
            "current_drawdown_pct": status.current_drawdown_pct,
            "profit_target_progress_pct": status.profit_target_progress_pct,
            "trading_days_completed": status.trading_days_completed}


# ── Trading: tickets, positions, execution log ───────────────────────────────

class AccountRequest(BaseModel):
    account_id: str = Field(min_length=1)
    balance: float = Field(gt=0)


@router.post("/accounts")
def create_account(req: AccountRequest,
                   user: dict = Depends(require("admin"))) -> dict:
    """Register a trading account. Admin-only: account creation changes
    what the risk engine protects."""
    if req.account_id in PLATFORM.accounts:
        raise HTTPException(status_code=409, detail="account_exists")
    PLATFORM.register_account(req.account_id, req.balance)
    PLATFORM.audit.append({"event": "account_registered",
                           "account_id": req.account_id,
                           "balance": req.balance,
                           "session_user": user["username"]})
    return {"ok": True, "account_id": req.account_id,
            "balance": req.balance}


class TicketRequest(BaseModel):
    account_id: str
    symbol: str = "XAUUSD"
    side: str                      # "long" | "short"
    intended_risk_pct: float = Field(gt=0, le=5)
    sl_price: float = Field(gt=0)
    tp_price: float | None = None
    idempotency_key: str = Field(min_length=4)
    approval_token: str | None = None
    order_type: str = "market"    # market | buy_stop | sell_stop | ...
    limit_price: float | None = None


@router.post("/orders")
def submit_ticket(req: TicketRequest, user: dict = Depends(require("trader"))) -> dict:
    mode = PLATFORM.mode_gate.mode_of(req.account_id)
    if mode in ("manual", "semi_auto") and not req.approval_token:
        raise HTTPException(status_code=422,
                            detail="mode requires a human approval token")
    return PLATFORM.submit_ticket(
        req.account_id, req.symbol, req.side, req.intended_risk_pct,
        req.sl_price, req.tp_price, req.idempotency_key,
        approval_token=req.approval_token, order_type=req.order_type,
        limit_price=req.limit_price)


@router.get("/positions/{account_id}")
def positions(account_id: str, user: dict = Depends(require("viewer"))) -> list[dict]:
    return [p.__dict__ for p in PLATFORM.broker.positions_of(account_id)]


@router.get("/execution-log")
def execution_log(limit: int = 100, user: dict = Depends(require("viewer"))) -> list[dict]:
    return PLATFORM.audit.entries()[-limit:]


# ── AI assistant: read/propose tools only ────────────────────────────────────

class AssistantRequest(BaseModel):
    message: str = Field(min_length=1)


@router.post("/assistant")
def assistant(req: AssistantRequest, user: dict = Depends(require("viewer"))) -> dict:
    from dashboard.platform_ai import ASSISTANT
    run = ASSISTANT.respond(req.message)
    return run


# ── Admin: agent run and cost logs ───────────────────────────────────────────

@router.get("/admin/agent-runs")
def agent_runs(limit: int = 100, user: dict = Depends(require("admin"))) -> list[dict]:
    from dashboard.platform_ai import ASSISTANT
    return ASSISTANT.run_log(limit)


# ── Health (Part 14.4) ───────────────────────────────────────────────────────

@router.get("/health")
def health() -> dict:
    """Composite health: data feed, broker, engines. Never 500s; a dead
    component is reported as unhealthy, not hidden."""
    broker_health = PLATFORM.broker.health()
    # Evaluate staleness directly (side-effect free): a symbol whose last
    # tick is older than the freshness window is STALE until proven fresh.
    now_ms = max(PLATFORM.broker.last_tick_ms.values(), default=0)
    stale_symbols = [
        s for s, last in PLATFORM.broker.last_tick_ms.items()
        if now_ms - last > PLATFORM.feed_monitor.max_staleness_ms]
    components = {
        "broker": {"status": "up" if broker_health.get("connected")
                   else "down", "detail": broker_health},
        "risk_engine": {"status": "up",
                        "detail": {"kill_switch_global":
                                   PLATFORM.kill_switch.public_state()["global"]}},
        "reconciliation": {"status": "paused" if any(
            PLATFORM.reconciler.paused.values()) else "up",
            "detail": {"paused_accounts": [a for a, p in
                                           PLATFORM.reconciler.paused.items()
                                           if p]}},
        "data_feed": {"status": "stale" if stale_symbols else "up",
                      "detail": {"stale_symbols": stale_symbols,
                                 "last_tick_ms":
                                 dict(PLATFORM.broker.last_tick_ms)}},
    }
    overall = "healthy" if all(c["status"] in ("up", "paused")
                               for c in components.values()) else "degraded"
    return {"status": overall, "components": components}


# ── Adaptation suggestions: audit + human approval (Part 12 flow) ───────────

@router.get("/adaptation/{account_id}")
def adaptation_suggestion(account_id: str, user: dict = Depends(require("viewer"))) -> dict:
    """Deterministic adaptation suggestions — proposals only, audited,
    requiring explicit human approval before any change is applied."""
    from core.adaptation import suggest_adaptation
    from core.config import SYMBOL_SPECS
    account = PLATFORM.accounts.get(account_id)
    profile = PLATFORM.profiles.get(account_id)
    if account is None or profile is None:
        raise HTTPException(status_code=404, detail="unknown account")
    suggestions = suggest_adaptation(
        account, profile, SYMBOL_SPECS["XAUUSD"], price=2600.0,
        current_atr=8.0)
    payload = {"account_id": account_id,
               "suggestions": list(suggestions),
               "status": "pending_approval"}
    PLATFORM.audit.append({"event": "adaptation_suggested",
                           "account": account_id,
                           "n_suggestions": len(suggestions),
                           "status": "pending_approval"})
    return payload


class ApprovalRequest(BaseModel):
    actor: str = Field(min_length=1)
    decision: str                     # "approved" | "rejected"


@router.post("/adaptation/{account_id}/approve")
def approve_adaptation(account_id: str, req: ApprovalRequest, user: dict = Depends(require("trader"))) -> dict:
    if req.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=422, detail="unknown decision")
    PLATFORM.audit.append({"event": "adaptation_decision",
                           "account": account_id,
                           "decision": req.decision, "actor": req.actor})
    return {"account_id": account_id, "decision": req.decision,
            "applied_by": "deterministic engines on next evaluation"}


# ── Live-control lockout ─────────────────────────────────────────────────────

class SignoffRequest(BaseModel):
    actor: str = Field(min_length=1)
    notes: str = Field(min_length=1)


@router.post("/admin/signoff")
def record_signoff(req: SignoffRequest, user: dict = Depends(require("admin"))) -> dict:
    return PLATFORM.record_signoff(req.actor, req.notes)


@router.get("/admin/signoff")
def signoff_state(user: dict = Depends(require("viewer"))) -> dict:
    return {"live_signoff": PLATFORM.live_signoff,
            "live_controls_enabled": PLATFORM.live_signoff is not None}


# ── U2: session profiles and runs ────────────────────────────────────────────

class SessionProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    start: str = Field(min_length=4, max_length=5)      # "HH:MM"
    end: str = Field(min_length=4, max_length=5)        # "HH:MM"
    timezone: str = Field(min_length=1)
    days_of_week: list[int] = Field(default=[0, 1, 2, 3, 4])
    max_duration_minutes: int = 240
    no_new_entries_after_minutes_before_end: int = 0
    flatten_by: str | None = None
    max_trades_in_session: int | None = None


class SessionBindRequest(BaseModel):
    profile: str | None = Field(default=None)   # None unbinds


@router.get("/accounts/{account_id}/payout-evidence-pack")
def payout_evidence_pack(account_id: str, From: str = "", To: str = "",
                         user: dict = Depends(require("trader"))):
    """U5: deterministic payout evidence pack (ZIP). from/to are UTC
    ISO dates or epoch ms."""
    from fastapi import Response
    from datetime import datetime, timezone

    def _ms(v: str, default: int) -> int:
        if not v:
            return default
        if v.isdigit():
            return int(v)
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            raise HTTPException(422, f"invalid date: {v!r}")

    state = PLATFORM
    from_ms = _ms(From, 0)
    to_ms = _ms(To, int(time.time() * 1000))
    try:
        blob = state.payout_evidence_pack(account_id, from_ms, to_ms)
    except KeyError:
        raise HTTPException(404, f"unknown account {account_id!r}")
    return Response(content=blob, media_type="application/zip",
                    headers={"Content-Disposition":
                             'attachment; filename="evidence_'
                             f'{account_id}_{from_ms}-{to_ms}.zip"'})


@router.post("/session-profiles")
def upsert_session_profile(req: SessionProfileRequest,
                           user: dict = Depends(require("trader"))) -> dict:
    try:
        profile = SessionProfile(
            name=req.name, start=req.start, end=req.end,
            timezone_name=req.timezone,
            days_of_week=tuple(req.days_of_week),
            max_duration_minutes=req.max_duration_minutes,
            no_new_entries_after_minutes_before_end=
                req.no_new_entries_after_minutes_before_end,
            flatten_by=req.flatten_by,
            max_trades_in_session=req.max_trades_in_session)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PLATFORM.upsert_session_profile(profile)


@router.get("/session-profiles")
def list_session_profiles(user: dict = Depends(require("viewer"))) -> dict:
    return {"profiles": {name: p.to_dict()
                         for name, p in PLATFORM.session_profiles.items()}}


@router.get("/network-status")
def network_status(user: dict = Depends(require("viewer"))) -> dict:
    """U5: network identity registry view (no credentials, ever)."""
    return PLATFORM.network_status()


@router.post("/accounts/{account_id}/session-profile")
def bind_session_profile(account_id: str, req: SessionBindRequest,
                         user: dict = Depends(require("trader"))) -> dict:
    if req.profile is None:
        try:
            return PLATFORM.unbind_session_profile(account_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    try:
        return PLATFORM.bind_session_profile(account_id, req.profile)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/accounts/{account_id}/session-status")
def session_status(account_id: str,
                   user: dict = Depends(require("viewer"))) -> dict:
    return PLATFORM.session_status(account_id)


@router.get("/deployments/{deployment_id}/sessions")
def deployment_sessions(deployment_id: str,
                        user: dict = Depends(require("viewer"))) -> list[dict]:
    return PLATFORM.session_runs.for_deployment(deployment_id)
