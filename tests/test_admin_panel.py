"""Admin panel tests: UI routing config, AI provider settings, auth gating.

Fail-closed rules verified: no token -> 401; invalid page -> 422; the API
key is never returned by the config view; env var wins over the config file.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import dashboard.platform_state as ps
from dashboard.app import app
from dashboard.platform_state import PlatformState


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PLATFORM", PlatformState(state_dir=tmp_path))
    import dashboard.routers.platform as platform_router
    monkeypatch.setattr(platform_router, "PLATFORM", ps.PLATFORM)
    import dashboard.platform_ai as pai
    monkeypatch.setattr(pai, "PLATFORM", ps.PLATFORM)
    pai.ASSISTANT = pai.ConsoleAssistant()
    import dashboard.security as sec
    monkeypatch.setenv("DSH_ADMIN_PASSWORD", "test-pass-1234")
    monkeypatch.delenv("DSH_TOTP_SECRET", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    test_auth = sec.Authenticator()
    monkeypatch.setattr(sec, "AUTH", test_auth)
    monkeypatch.setattr(platform_router, "AUTH", test_auth)
    # Isolate the admin config file per test.
    import dashboard.admin_config as ac
    monkeypatch.setattr(ac, "config_path",
                        lambda: str(tmp_path / "admin_config.json"))
    c = TestClient(app)
    r = c.post("/api/platform/auth/login", json={
        "username": "admin", "password": "test-pass-1234"})
    assert r.status_code == 200, r.text
    c.headers.update({"Authorization": f"Bearer {r.json()['token']}"})
    return c


def test_admin_config_requires_auth():
    c = TestClient(app)
    # 401 when auth is configured but no token, 403 when auth is not
    # configured at all — both are fail-closed.
    assert c.get("/api/platform/admin/config").status_code in (401, 403)
    assert c.post("/api/platform/admin/config", json={}).status_code in (401, 403)


def test_default_config_and_admin_page(client):
    r = client.get("/api/platform/admin/config")
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["default_page"] == "terminal"
    assert set(cfg["valid_pages"]) == {"terminal", "client", "legacy"}
    assert cfg["ai_provider"] == "rule_based_fallback"
    assert cfg["ai_key_set"] is False
    # The admin panel page itself is served without auth (login happens in-page).
    page = client.get("/admin")
    assert page.status_code == 200
    assert "Admin Panel" in page.text


def test_default_page_switch_round_trip(client):
    r = client.post("/api/platform/admin/config",
                    json={"default_page": "legacy"})
    assert r.status_code == 200
    assert r.json()["default_page"] == "legacy"
    # `/` now serves the legacy page.
    page = client.get("/")
    assert page.status_code == 200
    assert "Gold Trading Terminal" in page.text
    # And back.
    r = client.post("/api/platform/admin/config",
                    json={"default_page": "terminal"})
    assert r.json()["default_page"] == "terminal"
    assert "Trading Terminal" in client.get("/").text


def test_invalid_default_page_rejected(client):
    r = client.post("/api/platform/admin/config",
                    json={"default_page": "http://evil.example"})
    assert r.status_code == 422


def test_gemini_key_set_and_clear(client):
    r = client.post("/api/platform/admin/config",
                    json={"gemini_api_key": "  test-key-123  "})
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["ai_key_set"] is True
    assert cfg["ai_provider"] == "gemini"
    assert cfg["ai_key_source"] == "config"
    # The raw key is never echoed back.
    assert "test-key-123" not in r.text
    assert "test-key-123" not in client.get("/api/platform/admin/config").text
    # get_llm now resolves through the config file.
    from core.llm import get_llm
    llm = get_llm()
    assert llm is not None and llm.api_key == "test-key-123"
    # Clear it.
    r = client.post("/api/platform/admin/config", json={"gemini_api_key": ""})
    assert r.json()["ai_key_set"] is False
    assert get_llm() is None


def test_env_key_wins_over_config(client, monkeypatch):
    client.post("/api/platform/admin/config",
                json={"gemini_api_key": "config-key"})
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    r = client.get("/api/platform/admin/config")
    assert r.json()["ai_key_source"] == "env"
    # Changing the key through the panel is refused while env is set.
    r = client.post("/api/platform/admin/config",
                    json={"gemini_api_key": "other"})
    assert r.status_code == 409


def test_config_survives_missing_file(client, tmp_path, monkeypatch):
    import dashboard.admin_config as ac
    monkeypatch.setattr(ac, "config_path",
                        lambda: str(tmp_path / "does_not_exist.json"))
    cfg = ac.load_config()
    assert cfg["default_page"] == "terminal"
    assert cfg["gemini_api_key"] == ""
