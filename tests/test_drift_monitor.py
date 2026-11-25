"""U6 tests: drift detection to auto-pause (Update Blueprint v2.1).

Covers: band comparison (inside/outside), severity ladder, critical ->
auto-pause to paper with audit + alert, operational pause triggers,
resume requires authentication, deterministic report data, and the
no-LLM guarantee.
"""
from __future__ import annotations

import json

import pytest

from core.execution.modes import AUTO, ModeGate
from core.monitoring import (DriftAlertStore, DriftMonitor, DriftThresholds,
                             ValidationBand)
from tests.test_execution_gate import env, setup_auto  # noqa: F401
from tests.test_dashboard_platform import client  # noqa: F401  (fixture)

BAND = ValidationBand(
    strategy_version_id="ema_cross:aa:bb",
    win_rate=0.45, win_rate_tol=0.10,
    avg_r=0.20, avg_r_tol=0.25,
    trades_per_day=3.0, trade_freq_tol=1.0,
    max_drawdown_pct=10.0, max_dd_tol_pct=5.0,
    avg_slippage=0.02, slippage_tol=0.05,
    avg_spread_at_entry=0.20, spread_tol=0.10)


def monitor(tmp_path, mode_gate=None):
    from core.execution.router import AuditLog, Notifier
    store = DriftAlertStore(tmp_path / "alerts.jsonl")
    mon = DriftMonitor(mode_gate or ModeGate(), AuditLog(":memory:"),
                       Notifier(), store)
    mon.set_band(BAND)
    return mon


def inside_stats(**over):
    stats = {"win_rate": 0.44, "avg_r": 0.18, "trades_per_day": 3.2,
             "max_drawdown_pct": 9.0, "avg_slippage": 0.02,
             "avg_spread_at_entry": 0.21, "losing_streak": 0}
    stats.update(over)
    return stats


# ── inside the band: no alert ───────────────────────────────────────────

def test_inside_band_no_alert(tmp_path):
    mon = monitor(tmp_path)
    assert mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                        inside_stats(), now_ms=1000) is None
    assert mon.alert_store.entries() == []


def test_no_band_configured_no_alert(tmp_path):
    mon = monitor(tmp_path)
    mon.bands.clear()
    assert mon.evaluate("a1", "dep-1", "x", inside_stats(),
                        now_ms=1000) is None


# ── outside the band: severity ladder ───────────────────────────────────

def test_warning_severity_no_pause(tmp_path):
    mon = monitor(tmp_path)
    # win_rate 0.30 vs 0.45: deviation 0.15 = 1.5x tol -> warning.
    alert = mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                         inside_stats(win_rate=0.30), now_ms=1000)
    assert alert.severity == "warning"
    assert alert.action == "none"
    assert mon.mode_gate.mode_of("a1") != "paper"


def test_critical_severity_pauses_to_paper(tmp_path):
    mon = monitor(tmp_path)
    mon.mode_gate.set_mode("a1", AUTO, verdict="PAPER_TRADE_ELIGIBLE",
                           paper_days=30, rules_verified=True,
                           confirmed_by="human")
    # win_rate 0.20 vs 0.45: deviation 0.25 = 2.5x tol -> critical.
    alert = mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                         inside_stats(win_rate=0.20), now_ms=1000)
    assert alert.severity == "critical"
    assert alert.action == "paused_to_paper"
    assert mon.mode_gate.mode_of("a1") == "paper"
    # Persisted + audited.
    assert mon.alert_store.entries()[0]["severity"] == "critical"
    assert any(e["event"] == "drift_alert"
               for e in mon.audit.entries())


def test_drawdown_beyond_band_is_critical(tmp_path):
    mon = monitor(tmp_path)
    alert = mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                         inside_stats(max_drawdown_pct=25.0), now_ms=1000)
    assert alert.severity == "critical"      # 15 = 3x tol of 5


def test_losing_streak_probability(tmp_path):
    mon = monitor(tmp_path)
    # p=0.45 -> P(10 losses) = 0.55^10 ~= 0.0025 < 0.01 -> critical.
    alert = mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                         inside_stats(losing_streak=10), now_ms=1000)
    assert any(f["metric"] == "losing_streak" and f["severity"] == "critical"
               for f in alert.findings)
    # P(6 losses) = 0.55^6 ~= 0.028 < 0.05 -> warning.
    mon2 = monitor(tmp_path / "w2")
    alert2 = mon2.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                           inside_stats(losing_streak=6), now_ms=1000)
    assert alert2 is not None and alert2.severity == "warning"
    # P(5 losses) = 0.55^5 ~= 0.0503 >= 0.05 -> inside the band.
    mon3 = monitor(tmp_path / "w3")
    assert mon3.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                         inside_stats(losing_streak=5), now_ms=1000) is None


def test_slippage_and_spread_drift(tmp_path):
    mon = monitor(tmp_path)
    alert = mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                         inside_stats(avg_slippage=0.20,
                                      avg_spread_at_entry=0.60),
                         now_ms=1000)
    metrics = {f["metric"] for f in alert.findings}
    assert {"avg_slippage", "avg_spread_at_entry"} <= metrics


# ── operational pauses ──────────────────────────────────────────────────

def test_operational_pause_triggers(tmp_path):
    mon = monitor(tmp_path)
    mon.mode_gate.set_mode("a1", AUTO, verdict="PAPER_TRADE_ELIGIBLE",
                           paper_days=30, rules_verified=True,
                           confirmed_by="human")
    alert = mon.check_operational_pause(
        "a1", "dep-1", daily_loss_stop=True, feed_stale_ms=120_000,
        bridge_disconnected_ms=400_000, now_ms=1000)
    assert alert.severity == "critical"
    hits = {f["metric"] for f in alert.findings}
    assert {"daily_loss_stop", "stale_feed", "bridge_disconnect"} <= hits
    assert mon.mode_gate.mode_of("a1") == "paper"


def test_operational_pause_below_thresholds_noop(tmp_path):
    mon = monitor(tmp_path)
    assert mon.check_operational_pause(
        "a1", "dep-1", feed_stale_ms=1_000, now_ms=1000) is None


def test_reconciliation_mismatch_pauses(tmp_path):
    mon = monitor(tmp_path)
    alert = mon.check_operational_pause(
        "a1", "dep-1", reconciliation_mismatch=True, now_ms=1000)
    assert alert.action == "paused_to_paper"


# ── resume requires authenticated human action ──────────────────────────

def test_resume_requires_auth(client, monkeypatch):  # dashboard fixture
    from fastapi.testclient import TestClient
    from dashboard.app import app
    anon = TestClient(app)                      # no Authorization header
    r = anon.post("/api/platform/accounts/paper-10k/resume-deployment",
                  json={"mode": "semi_auto"})
    assert r.status_code == 401                 # no token -> fail closed


def test_resume_with_trader_token(client):
    # The fixture client is already authenticated as admin (trader).
    r = client.post("/api/platform/accounts/paper-10k/resume-deployment",
                    json={"mode": "semi_auto"})
    assert r.status_code == 200
    assert r.json()["resumed_by"]


def test_drift_alerts_endpoint_requires_auth(client):
    from fastapi.testclient import TestClient
    from dashboard.app import app
    anon = TestClient(app)
    r = anon.get("/api/platform/drift-alerts")
    assert r.status_code == 401


# ── report data ─────────────────────────────────────────────────────────

def test_report_data_deterministic(tmp_path):
    mon = monitor(tmp_path)
    mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                 inside_stats(win_rate=0.30), now_ms=1000)
    mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                 inside_stats(win_rate=0.10), now_ms=2000)
    data = mon.report_data(0, 3000)
    assert data["total_alerts"] == 2
    assert data["by_severity"] == {"info": 0, "warning": 1, "critical": 1}
    assert data["pauses"] == 1
    data2 = mon.report_data(0, 3000)
    assert data == data2


# ── no LLM in the monitor ───────────────────────────────────────────────

def test_no_llm_call_in_monitor(tmp_path, monkeypatch):
    import core.llm as llm

    def boom(*a, **k):
        raise AssertionError("LLM must not be called by DriftMonitor")

    monkeypatch.setattr(llm, "get_llm", boom)
    mon = monitor(tmp_path)
    mon.mode_gate.set_mode("a1", AUTO, verdict="PAPER_TRADE_ELIGIBLE",
                           paper_days=30, rules_verified=True,
                           confirmed_by="human")
    alert = mon.evaluate("a1", "dep-1", "ema_cross:aa:bb",
                         inside_stats(win_rate=0.10), now_ms=1000)
    assert alert.action == "paused_to_paper"
    mon.check_operational_pause("a1", "dep-1", daily_loss_stop=True,
                                now_ms=2000)
