"""Part 14.2 security tests: session auth, role enforcement, TOTP MFA,
rate limiting, fail-closed behaviour when auth is unconfigured, and
credential hygiene.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import dashboard.security as sec
from dashboard.app import app
from dashboard.platform_state import PlatformState
from dashboard.security import (Authenticator, RateLimiter, SessionStore,
                                hash_password, verify_password, totp_code,
                                verify_totp)


@pytest.fixture()
def auth_env(monkeypatch):
    monkeypatch.setenv("DSH_ADMIN_PASSWORD", "hunter2-secret")
    monkeypatch.setenv("DSH_TOTP_SECRET", sec.generate_totp_secret())
    auth = Authenticator()
    monkeypatch.setattr(sec, "AUTH", auth)
    import dashboard.routers.platform as platform_router
    monkeypatch.setattr(platform_router, "AUTH", auth)
    return auth


@pytest.fixture()
def anon_client(tmp_path, monkeypatch, auth_env):
    monkeypatch.setattr("dashboard.platform_state.PLATFORM",
                        PlatformState(state_dir=tmp_path))
    import dashboard.routers.platform as platform_router
    monkeypatch.setattr(platform_router, "PLATFORM",
                        __import__("dashboard.platform_state",
                                   fromlist=["PLATFORM"]).PLATFORM)
    return TestClient(app)


def login(client, auth, password="hunter2-secret", totp=None, username="admin"):
    return client.post("/api/platform/auth/login", json={
        "username": username, "password": password,
        "totp": totp if totp is not None
        else totp_code(auth.totp_secret)})


# ── Password hashing ─────────────────────────────────────────────────────────

def test_password_hashing_roundtrip_and_salt():
    h1 = hash_password("s3cret")
    h2 = hash_password("s3cret")
    assert h1 != h2                      # unique salt per hash
    assert verify_password("s3cret", h1)
    assert not verify_password("wrong", h1)
    assert not verify_password("s3cret", "garbage")


# ── TOTP ─────────────────────────────────────────────────────────────────────

def test_totp_accepts_current_and_adjacent_steps_rejects_wrong():
    secret = sec.generate_totp_secret()
    now = 1_700_000_000
    assert verify_totp(secret, totp_code(secret, now), at_time=now)
    assert verify_totp(secret, totp_code(secret, now - 30), at_time=now)
    assert not verify_totp(secret, totp_code(secret, now - 120), at_time=now)
    assert not verify_totp(secret, "000000", at_time=now) or \
        totp_code(secret, now) == "000000"   # collision is astronomically unlikely but honest
    assert not verify_totp(secret, "abc")


def test_totp_is_rfc6238_shape():
    secret = sec.generate_totp_secret()
    code = totp_code(secret, 59)             # RFC test vector time bucket
    assert len(code) == 6 and code.isdigit()


# ── Login flow ───────────────────────────────────────────────────────────────

def test_login_success_returns_session(anon_client, auth_env):
    r = login(anon_client, auth_env)
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "admin" and body["token"]
    assert body["mfa_enabled"] is True


def test_login_wrong_password_rejected(anon_client, auth_env):
    r = login(anon_client, auth_env, password="nope")
    assert r.status_code == 401 and r.json()["detail"] == "invalid_credentials"


def test_login_without_totp_rejected_when_mfa_enabled(anon_client, auth_env):
    r = login(anon_client, auth_env, totp="")
    assert r.status_code == 401 and r.json()["detail"] == "invalid_totp"


def test_revoked_session_is_rejected(anon_client, auth_env):
    token = login(anon_client, auth_env).json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    assert anon_client.get("/api/platform/overview", headers=h).status_code == 200
    anon_client.post("/api/platform/auth/logout", headers=h)
    assert anon_client.get("/api/platform/overview", headers=h).status_code == 401


def test_sessions_expire(monkeypatch):
    store = SessionStore(ttl_s=60)
    token = store.create("op", "admin")
    real_now = time.time()
    monkeypatch.setattr(time, "time", lambda: real_now + 3600)
    assert store.get(token) is None


# ── Fail closed when auth is unconfigured ────────────────────────────────────

def test_unconfigured_auth_blocks_mutating_endpoints(tmp_path, monkeypatch):
    monkeypatch.delenv("DSH_ADMIN_PASSWORD", raising=False)
    auth = Authenticator()
    monkeypatch.setattr(sec, "AUTH", auth)
    import dashboard.routers.platform as platform_router
    monkeypatch.setattr(platform_router, "AUTH", auth)
    monkeypatch.setattr("dashboard.platform_state.PLATFORM",
                        PlatformState(state_dir=tmp_path))
    c = TestClient(app)
    assert auth.configured is False
    r = c.post("/api/platform/orders", json={
        "account_id": "x", "side": "long", "intended_risk_pct": 0.5,
        "sl_price": 2588.0, "idempotency_key": "blocked-1"})
    assert r.status_code == 403
    assert r.json()["detail"] == "auth_not_configured"
    # Health stays reachable for operations monitoring.
    assert c.get("/api/platform/health").status_code == 200


# ── Role enforcement ─────────────────────────────────────────────────────────

def make_session(auth, role):
    return {"Authorization": f"Bearer {auth.sessions.create('op', role)}"}


def test_viewer_cannot_submit_orders(anon_client, auth_env):
    h = make_session(auth_env, "viewer")
    r = anon_client.post("/api/platform/orders", headers=h, json={
        "account_id": "x", "side": "long", "intended_risk_pct": 0.5,
        "sl_price": 2588.0, "idempotency_key": "v-1"})
    assert r.status_code == 403 and r.json()["detail"] == "insufficient_role"


def test_viewer_can_read_overview(anon_client, auth_env):
    h = make_session(auth_env, "viewer")
    assert anon_client.get("/api/platform/overview", headers=h).status_code == 200


def test_trader_can_trade_but_not_administer(anon_client, auth_env):
    trader = make_session(auth_env, "trader")
    # Kill-switch ENGAGE is a trader-level safety action...
    r = anon_client.post("/api/platform/kill-switch", headers=trader, json={
        "scope": "global", "action": "engage", "actor": "trader-1",
        "reason": "halt"})
    assert r.status_code == 200 and r.json()["global"] is True
    # ...but DISENGAGE and sign-off are admin-only.
    r = anon_client.post("/api/platform/kill-switch", headers=trader, json={
        "scope": "global", "action": "disengage", "actor": "trader-1",
        "reason": "resume"})
    assert r.status_code == 403
    r = anon_client.post("/api/platform/admin/signoff", headers=trader,
                         json={"actor": "trader-1", "notes": "self-approve"})
    assert r.status_code == 403
    r = anon_client.post("/api/platform/compliance/verify", headers=trader,
                         json={"firm": "f", "program": "p", "version": "v",
                               "verifier": "trader-1"})
    assert r.status_code == 403


def test_admin_can_disengage_and_signoff(anon_client, auth_env):
    admin = make_session(auth_env, "admin")
    anon_client.post("/api/platform/kill-switch", headers=admin, json={
        "scope": "global", "action": "engage", "actor": "a", "reason": "halt"})
    r = anon_client.post("/api/platform/kill-switch", headers=admin, json={
        "scope": "global", "action": "disengage", "actor": "a",
        "reason": "reviewed"})
    assert r.status_code == 200 and r.json()["global"] is False
    assert anon_client.post("/api/platform/admin/signoff", headers=admin,
                            json={"actor": "a", "notes": "ok"}).status_code == 200


def test_admin_agent_runs_is_admin_only(anon_client, auth_env):
    assert anon_client.get("/api/platform/admin/agent-runs",
                           headers=make_session(auth_env, "viewer")
                           ).status_code == 403
    assert anon_client.get("/api/platform/admin/agent-runs",
                           headers=make_session(auth_env, "admin")
                           ).status_code == 200


def test_missing_token_is_401_everywhere(anon_client, auth_env):
    for method, url in [("get", "/api/platform/overview"),
                        ("post", "/api/platform/orders"),
                        ("get", "/api/platform/execution-log")]:
        r = getattr(anon_client, method)(url)
        assert r.status_code == 401, url


# ── Rate limiting ────────────────────────────────────────────────────────────

def test_login_rate_limited_fail_closed(anon_client, auth_env):
    for _ in range(5):
        login(anon_client, auth_env, password="wrong")
    r = login(anon_client, auth_env)          # correct creds still blocked
    assert r.status_code == 401 and r.json()["detail"] == "rate_limited"


def test_rate_limiter_window_semantics():
    rl = RateLimiter(max_events=3, window_s=10)
    assert rl.allow("k") and rl.allow("k") and rl.allow("k")
    assert not rl.allow("k")
    assert rl.allow("other")                 # per-key isolation


# ── Credential hygiene ───────────────────────────────────────────────────────

def test_password_never_stored_in_plain_or_echoed(anon_client, auth_env):
    auth = auth_env
    assert auth._password_raw is None        # raw discarded after hashing
    r = login(anon_client, auth_env)
    assert "hunter2" not in r.text
    state = anon_client.get("/api/platform/auth/state").json()
    assert state == {"configured": True, "mfa_enabled": True}
