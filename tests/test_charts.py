"""Chart endpoint tests (Part 14.1): real-data candlesticks, honest
source labels, deterministic backtest curves and trade markers.
"""
from __future__ import annotations

import pytest

from tests.test_dashboard_platform import client, setup_account  # noqa: F401


from tests.conftest import requires_dataset

pytestmark = requires_dataset

START_MS = 1735776000000     # 2025-01-02
END_MS = 1736294400000       # 2025-01-08


def test_bars_come_from_real_data_with_honest_label(client):
    r = client.get(f"/api/platform/charts/bars?tf=15m&start_ms={START_MS}"
                   f"&end_ms={END_MS}")
    body = r.json()
    assert body["source_label"] == "DELAYED"
    assert "never live" in body["source_detail"]
    bars = body["bars"]
    assert len(bars) > 100
    for b in bars:
        assert b["h"] >= max(b["o"], b["c"]) - 1e-9
        assert b["l"] <= min(b["o"], b["c"]) + 1e-9
        assert b["v"] >= 0
    # Real XAUUSD price levels in this window, not placeholders.
    assert 2500 < bars[0]["o"] < 2800


def test_backtest_chart_payload_is_deterministic(client):
    url = (f"/api/platform/charts/backtest?tf=15m&start_ms={START_MS}"
           f"&end_ms={END_MS}")
    r1 = client.get(url).json()
    r2 = client.get(url).json()
    assert r1 == r2                       # same data + config, same curve
    assert r1["source_label"] == "BACKTEST"
    assert "not a forecast" in r1["source_detail"]
    assert r1["summary"]["n_trades"] == len(r1["trades"]) > 0
    # Equity starts at start_equity and the drawdown curve is consistent.
    assert r1["equity"][0]["v"] == 10_000.0
    peak = 10_000.0
    for point, dd in zip(r1["equity"], r1["drawdown_pct"]):
        peak = max(peak, point["v"])
        expected = (peak - point["v"]) / peak * 100.0
        assert dd["v"] == pytest.approx(expected, abs=1e-6)
    assert r1["summary"]["max_drawdown_pct"] == \
        pytest.approx(max(p["v"] for p in r1["drawdown_pct"]))


def test_trade_markers_reference_chart_times(client):
    body = client.get(f"/api/platform/charts/backtest?tf=15m"
                      f"&start_ms={START_MS}&end_ms={END_MS}").json()
    bar_times = {b["t"] for b in client.get(
        f"/api/platform/charts/bars?tf=15m&start_ms={START_MS}"
        f"&end_ms={END_MS}").json()["bars"]}
    for t in body["trades"]:
        assert t["entry_time"] in bar_times    # markers align with bars
        assert t["side"] in ("LONG", "SHORT")
        assert t["reason"] in ("TP", "SL", "signal", "session_end",
                               "max_hold", "eod")


def test_charts_public_read(client):
    # Chart reads are market data only (no account info) and are public
    # so the client-facing portal needs no login. Everything else
    # (accounts, orders, kill switch, admin) stays auth-gated.
    from fastapi.testclient import TestClient
    from dashboard.app import app
    anon = TestClient(app)
    assert anon.get(f"/api/platform/charts/bars?tf=15m&start_ms={START_MS}"
                    f"&end_ms={END_MS}").status_code == 200
    # Writes and account data stay protected.
    assert anon.get("/api/platform/overview").status_code in (401, 403)
    assert anon.post("/api/platform/kill-switch", json={}).status_code in (401, 403)


def test_charts_reject_bad_ranges(client):
    assert client.get(f"/api/platform/charts/bars?tf=15m&start_ms={END_MS}"
                      f"&end_ms={START_MS}").status_code == 422
    assert client.get(f"/api/platform/charts/bars?tf=7m&start_ms={START_MS}"
                      f"&end_ms={END_MS}").status_code == 422
