"""Account registration endpoint + unknown-account fail-closed behaviour.

Browser verification (Playwright, 13/13) exposed two gaps this pins:
1. There was no API to register an account — tests called the state
   method directly, so the console's ticket form referenced an account
   that could never exist through the UI.
2. A ticket against an unknown account raised KeyError -> HTTP 500.
   It must fail closed with a clean rejection instead.
"""
from __future__ import annotations

from tests.test_dashboard_platform import client, setup_account  # noqa: F401


def test_admin_registers_account_via_api(client):
    r = client.post("/api/platform/accounts",
                    json={"account_id": "new-1", "balance": 25_000.0})
    assert r.status_code == 200
    assert r.json()["account_id"] == "new-1"
    # Audited
    log = client.get("/api/platform/execution-log").json()
    assert any(e["event"] == "account_registered" for e in log)
    # Visible in overview
    ov = client.get("/api/platform/overview").json()
    assert any(a["account_id"] == "new-1" for a in ov["accounts"])


def test_duplicate_registration_conflicts(client):
    client.post("/api/platform/accounts",
                json={"account_id": "dup-1", "balance": 1_000.0})
    r = client.post("/api/platform/accounts",
                    json={"account_id": "dup-1", "balance": 1_000.0})
    assert r.status_code == 409


def test_registration_rejects_bad_input(client):
    assert client.post("/api/platform/accounts",
                       json={"account_id": "", "balance": 1_000.0}
                       ).status_code == 422
    assert client.post("/api/platform/accounts",
                       json={"account_id": "neg", "balance": -5}
                       ).status_code == 422


def test_ticket_on_unknown_account_fails_closed_cleanly(client):
    """Was a KeyError -> 500; must be a structured rejection."""
    r = client.post("/api/platform/orders", json={
        "account_id": "ghost", "side": "long",
        "intended_risk_pct": 0.5, "sl_price": 2588.0,
        "idempotency_key": "unknown-acct-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "rejected"
    assert "unknown_account" in body["reasons"]


def test_registered_account_ticket_then_priced(client):
    """End-to-end through the API: register -> ticket gets a real
    verdict (not a server error)."""
    client.post("/api/platform/accounts",
                json={"account_id": "flow-1", "balance": 10_000.0})
    r = client.post("/api/platform/orders", json={
        "account_id": "flow-1", "side": "long",
        "intended_risk_pct": 0.5, "sl_price": 2588.0,
        "idempotency_key": "flow-1"})
    assert r.status_code == 200
    assert r.json()["status"] in ("filled", "rejected", "pending_accepted")
