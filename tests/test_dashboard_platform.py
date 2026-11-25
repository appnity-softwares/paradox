"""Part 14.1 dashboard tests: the platform console API — honest labels,
kill switch via API, compliance verification screen, paper tickets,
live-control lockout, assistant restrictions, admin logs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import dashboard.platform_state as ps
from dashboard.app import app
from dashboard.platform_state import PlatformState


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Fresh isolated platform state per test.
    monkeypatch.setattr(ps, "PLATFORM", PlatformState(state_dir=tmp_path))
    import dashboard.routers.platform as platform_router
    monkeypatch.setattr(platform_router, "PLATFORM", ps.PLATFORM)
    import dashboard.platform_ai as pai
    monkeypatch.setattr(pai, "PLATFORM", ps.PLATFORM)
    pai.ASSISTANT = pai.ConsoleAssistant()   # rebuild against the fresh state
    # Fresh authenticator with test credentials (env-only by design).
    import dashboard.security as sec
    monkeypatch.setenv("DSH_ADMIN_PASSWORD", "test-pass-1234")
    monkeypatch.setenv("DSH_TOTP_SECRET", sec.generate_totp_secret())
    test_auth = sec.Authenticator()
    monkeypatch.setattr(sec, "AUTH", test_auth)
    monkeypatch.setattr(platform_router, "AUTH", test_auth)
    c = TestClient(app)
    # Login once; every request carries the session token.
    r = c.post("/api/platform/auth/login", json={
        "username": "admin", "password": "test-pass-1234",
        "totp": sec.totp_code(test_auth.totp_secret)})
    assert r.status_code == 200, r.text
    c.headers.update({"Authorization": f"Bearer {r.json()['token']}"})
    return c


def setup_account(client, account_id="paper-10k", verified=True):
    """Register an account with a ruleset via the state singleton."""
    platform = ps.PLATFORM
    from core.compliance import FirmRuleSet
    from core.compliance.rules import DailyLossRule, MaxLossRule
    rs = FirmRuleSet(
        firm="ftmo_style", program="100k", version="v1",
        status="verified" if verified else "draft",
        ea_policy="allowed",
        verified_by="human-1" if verified else None,
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=10_000.0, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
        min_trading_days=0, weekend_holding="allowed")
    platform.rule_store.add_draft(rs)          # drafts always stored
    if verified:
        platform.rule_store.verify("ftmo_style", "100k", "v1", "human-1")
    platform.register_account(account_id, 10_000.0, ruleset=rs)


# ── Overview and honest labels ───────────────────────────────────────────────

def test_overview_reports_accounts_and_labels(client):
    setup_account(client)
    r = client.get("/api/platform/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["accounts"][0]["account_id"] == "paper-10k"
    assert body["accounts"][0]["mode"] == "paper"
    labels = {s["source"]: s["label"] for s in body["data_sources"]}
    # Honest labels: the parquet store is DELAYED, paper quotes SYNTHETIC,
    # the unconnected bridge STALE. Nothing is labelled LIVE falsely.
    assert labels["Dukascopy tick store (parquet)"] == "DELAYED"
    assert labels["Paper broker quotes"] == "SYNTHETIC"
    assert labels["MT5 bridge feed"] == "STALE"
    assert all(s["label"] != "LIVE" for s in body["data_sources"])


def test_live_controls_disabled_until_signoff(client):
    r = client.get("/api/platform/admin/signoff")
    assert r.json()["live_controls_enabled"] is False
    # Recording the sign-off flips the flag and audits it.
    r2 = client.post("/api/platform/admin/signoff",
                     json={"actor": "human-1", "notes": "paper accepted"})
    assert r2.status_code == 200
    r3 = client.get("/api/platform/admin/signoff")
    assert r3.json()["live_controls_enabled"] is True


# ── Kill switch via API ──────────────────────────────────────────────────────

def test_kill_switch_engage_and_disengage_via_api(client):
    setup_account(client)
    r = client.post("/api/platform/kill-switch", json={
        "scope": "global", "action": "engage", "actor": "human-1",
        "reason": "halt everything"})
    assert r.json()["global"] is True
    # An engaged kill switch blocks new orders.
    t = client.post("/api/platform/orders", json={
        "account_id": "paper-10k", "side": "long", "intended_risk_pct": 0.5,
        "sl_price": 2588.0, "idempotency_key": "ks-test-1"})
    assert t.json()["status"] == "rejected"
    assert "kill_switch_engaged" in t.json()["reasons"]
    # Disengage requires actor + reason and clears the block.
    d = client.post("/api/platform/kill-switch", json={
        "scope": "global", "action": "disengage", "actor": "human-1",
        "reason": "reviewed"})
    assert d.json()["global"] is False


def test_kill_switch_action_requires_actor_and_reason(client):
    r = client.post("/api/platform/kill-switch", json={
        "scope": "global", "action": "engage", "actor": "", "reason": ""})
    assert r.status_code == 422


# ── Compliance screens ───────────────────────────────────────────────────────

def test_ruleset_verification_screen_shows_rules_next_to_clauses(client):
    setup_account(client, verified=False)
    r = client.get("/api/platform/compliance/rulesets")
    entry = r.json()[0]
    assert entry["status"] == "draft"
    assert entry["grants_permission"] is False      # draft grants NOTHING
    daily = next(x for x in entry["rules_with_clauses"]
                 if x["rule"] == "daily_loss")
    assert daily["citations"] == ["3.2"]
    # Human verification from the screen flips it to verified.
    v = client.post("/api/platform/compliance/verify", json={
        "firm": "ftmo_style", "program": "100k", "version": "v1",
        "verifier": "human-2"})
    assert v.json()["status"] == "verified"
    entry2 = client.get("/api/platform/compliance/rulesets").json()[0]
    assert entry2["grants_permission"] is True


def test_compliance_monitor_reports_remaining_allowance(client):
    setup_account(client)
    r = client.get("/api/platform/compliance/monitor/paper-10k")
    body = r.json()
    assert body["level"] in ("ok", "warning", "critical", "breached")
    assert body["remaining_daily_loss"] > 0
    assert body["remaining_max_loss"] > 0


# ── Order tickets through the real pipeline ──────────────────────────────────

def test_paper_ticket_flows_through_full_pipeline(client):
    setup_account(client)
    r = client.post("/api/platform/orders", json={
        "account_id": "paper-10k", "side": "long", "intended_risk_pct": 0.5,
        "sl_price": 2588.0, "tp_price": 2624.0,
        "idempotency_key": "ticket-1"})
    body = r.json()
    assert body["status"] == "filled"
    steps = {s["step"] for s in body["steps"]}
    assert {"mode_gate", "sizing", "risk", "compliance", "order_check",
            "send"} <= steps
    positions = client.get("/api/platform/positions/paper-10k").json()
    assert len(positions) == 1


def test_duplicate_ticket_is_suppressed(client):
    setup_account(client)
    payload = {"account_id": "paper-10k", "side": "long",
               "intended_risk_pct": 0.5, "sl_price": 2588.0,
               "idempotency_key": "dup-1"}
    r1 = client.post("/api/platform/orders", json=payload)
    r2 = client.post("/api/platform/orders", json=payload)
    assert r1.json()["status"] == "filled"
    assert r2.json()["status"] == "duplicate"
    assert len(client.get("/api/platform/positions/paper-10k").json()) == 1


def test_risky_ticket_reduced_to_risk_cap(client):
    """A 5%-risk ticket is REDUCED to the 1% cap by the risk engine —
    the designed behaviour (reduced-lot offer), not a rejection."""
    setup_account(client)
    r = client.post("/api/platform/orders", json={
        "account_id": "paper-10k", "side": "long", "intended_risk_pct": 5.0,
        "sl_price": 2500.0, "idempotency_key": "risky-1"})
    body = r.json()
    assert body["status"] == "filled"
    assert body["lot"] <= 0.01            # $100 risk cap / $100 stop distance
    risk_step = next(s for s in body["steps"] if s["step"] == "risk")
    assert risk_step["outcome"] in ("approved", "reduced")


def test_execution_log_records_the_ticket(client):
    setup_account(client)
    client.post("/api/platform/orders", json={
        "account_id": "paper-10k", "side": "long", "intended_risk_pct": 0.5,
        "sl_price": 2588.0, "idempotency_key": "log-1"})
    log = client.get("/api/platform/execution-log").json()
    assert any(e["event"] == "execution_result" and e["status"] == "filled"
               for e in log)


# ── AI assistant restrictions ────────────────────────────────────────────────

def test_assistant_responds_and_cannot_execute(client):
    setup_account(client)
    r = client.post("/api/platform/assistant",
                    json={"message": "Summarize my positions"})
    body = r.json()
    assert body["status"] == "ok"
    assert "proposal" in body          # any proposal is inert data
    # The assistant's tool registry contains no execute-level tool.
    from dashboard.platform_ai import ASSISTANT
    assert all(t.permission in ("read", "propose")
               for t in ASSISTANT.registry._tools.values())


def test_admin_agent_runs_expose_costs(client):
    setup_account(client)
    client.post("/api/platform/assistant", json={"message": "hi"})
    runs = client.get("/api/platform/admin/agent-runs").json()
    assert runs and runs[-1]["cost_usd"] >= 0 and "status" in runs[-1]


# ── Health (Part 14.4) ───────────────────────────────────────────────────────

def test_health_reports_components_honestly(client):
    setup_account(client)
    # Advance the clock on one symbol only: XAUUSD's last tick (t=1ms)
    # is now far behind the freshness window — honestly STALE.
    ps.PLATFORM.broker.on_tick("EURUSD", bid=1.08, ask=1.0802, ts_ms=30_000)
    r = client.get("/api/platform/health")
    body = r.json()
    assert body["status"] in ("healthy", "degraded")
    comps = body["components"]
    assert comps["broker"]["status"] == "up"          # paper broker connected
    assert comps["data_feed"]["status"] == "stale"
    assert "XAUUSD" in comps["data_feed"]["detail"]["stale_symbols"]
    # A degraded component is reported, never hidden.
    assert "risk_engine" in comps and "reconciliation" in comps


def test_adaptation_suggestions_are_pending_approval(client):
    setup_account(client)
    r = client.get("/api/platform/adaptation/paper-10k")
    body = r.json()
    assert body["status"] == "pending_approval"
    assert isinstance(body["suggestions"], list)
    # The suggestion event is audited.
    log = client.get("/api/platform/execution-log").json()
    assert any(e["event"] == "adaptation_suggested" for e in log)


def test_adaptation_decision_is_audited_with_actor(client):
    setup_account(client)
    r = client.post("/api/platform/adaptation/paper-10k/approve",
                    json={"actor": "human-1", "decision": "rejected"})
    assert r.json()["decision"] == "rejected"
    log = client.get("/api/platform/execution-log").json()
    entry = next(e for e in log if e["event"] == "adaptation_decision")
    assert entry["actor"] == "human-1" and entry["decision"] == "rejected"


def test_adaptation_unknown_account_404(client):
    r = client.get("/api/platform/adaptation/nope")
    assert r.status_code == 404
