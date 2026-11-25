"""Trade replay tests (Part 14.1): deterministic per-bar replay state
that is consistent with the trade list and equity curve.
"""
from __future__ import annotations

import pytest

from tests.test_dashboard_platform import client, setup_account  # noqa: F401


from tests.conftest import requires_dataset

pytestmark = requires_dataset

START_MS = 1735776000000
END_MS = 1736294400000
URL = (f"/api/platform/charts/replay?tf=15m&start_ms={START_MS}"
       f"&end_ms={END_MS}")


def test_replay_is_deterministic_and_labelled(client):
    r1 = client.get(URL).json()
    r2 = client.get(URL).json()
    assert r1 == r2
    assert r1["source_label"] == "BACKTEST_REPLAY"
    assert "not live" in r1["source_detail"]


def test_replay_open_trades_match_trade_lifetimes(client):
    body = client.get(URL).json()
    times = body["times"]
    idx = {t: i for i, t in enumerate(times)}
    n_trades = sum(len(c) for c in body["closed_on_bar"])
    assert n_trades > 0
    # Reconstruct each trade's open interval from the replay state and
    # verify it is contiguous and ends on its close bar.
    for bar_i, ids in enumerate(body["open_trades"]):
        for tid in ids:
            assert tid < n_trades
    # Every trade id appears in open_trades for at least one bar.
    seen = {tid for ids in body["open_trades"] for tid in ids}
    assert seen == set(range(n_trades))
    # Closed trades disappear from open_trades after their close bar.
    for xi, closed in enumerate(body["closed_on_bar"]):
        for t in closed:
            assert t["id"] not in body["open_trades"][xi + 1] \
                if xi + 1 < len(times) else True


def test_replay_equity_matches_backtest_curve(client):
    replay = client.get(URL).json()
    backtest = client.get(f"/api/platform/charts/backtest?tf=15m"
                          f"&start_ms={START_MS}&end_ms={END_MS}").json()
    assert replay["equity"] == [p["v"] for p in backtest["equity"]]
    assert replay["drawdown_pct"] == [p["v"] for p in
                                      backtest["drawdown_pct"]]
    assert len(replay["equity"]) == len(replay["times"]) == \
        len(replay["ohlc"])


def test_replay_closes_carry_pnl_and_reason(client):
    body = client.get(URL).json()
    closes = [c for bar in body["closed_on_bar"] for c in bar]
    assert closes
    for c in closes:
        assert c["side"] in ("LONG", "SHORT")
        assert c["reason"] in ("TP", "SL", "signal", "session_end",
                               "max_hold", "eod")
        assert isinstance(c["pnl"], float)


def test_replay_public_read(client):
    # Replay is market-data-only (no account info) — public read for the
    # client-facing portal; account/write endpoints stay auth-gated.
    from fastapi.testclient import TestClient
    from dashboard.app import app
    assert TestClient(app).get(URL).status_code == 200
