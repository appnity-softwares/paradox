"""Daily-PnL wiring tests: the compliance daily-loss rule must see
realized losses since the 17:00 UTC reset. This wiring was missing
(snapshot froze at registration values) — a bug the replay soak
exposed; these tests pin the fix.
"""
from __future__ import annotations

import time

import pytest

from tests.test_dashboard_platform import client, setup_account  # noqa: F401


def _force_closes(platform, account_id, pnls, now_ms):
    for i, pnl in enumerate(pnls):
        platform.broker.closed.append({
            "position_id": f"forced-{account_id}-{i}",
            "account_id": account_id, "pnl": pnl,
            "exit_price": 2600.0, "reason": "stop_loss", "ts_ms": now_ms})


def _platform():
    import dashboard.platform_state as ps
    return ps.PLATFORM


def test_daily_losses_reach_the_snapshot_and_block_trading(client):
    platform = _platform()
    setup_account(client)
    account_id = "paper-10k"
    now_ms = int(time.time() * 1000)
    # 6 losing closes of -900 each = -5,400 on a 10,000 account:
    # far beyond the 5% daily loss limit.
    # A fresh tick so the feed monitor does not (correctly) block on
    # staleness before the daily-loss rule is even reached.
    platform.broker.on_tick("XAUUSD", 2599.8, 2600.2, now_ms)
    _force_closes(platform, account_id, [-900.0] * 6, now_ms)
    platform._refresh_snapshot(account_id, now_ms)
    snap = platform.snapshots[account_id]
    assert snap.daily_realized_pnl == pytest.approx(-5400.0)
    # The next ticket must be blocked by the daily-loss rule.
    res = platform.submit_ticket(account_id, "XAUUSD", "long", 0.5,
                                 2588.0, None, "dl-block-1")
    assert res["status"] == "rejected"
    assert any("daily_loss" in r for r in res["reasons"])


def test_daily_reset_window_excludes_yesterdays_losses(client):
    platform = _platform()
    setup_account(client)
    account_id = "paper-10k"
    now_ms = int(time.time() * 1000)
    import datetime as dt
    now_dt = dt.datetime.fromtimestamp(now_ms / 1000, tz=dt.timezone.utc)
    # A loss at 10:00 UTC today counts; one at 10:00 UTC yesterday
    # (before the most recent 17:00 reset) does not. Note 18:00 UTC
    # yesterday WOULD count: the window runs yesterday 17:00 -> today
    # 17:00.
    today_morning = int(now_dt.replace(hour=10, microsecond=0)
                        .timestamp() * 1000)
    prev_evening = int((now_dt.replace(hour=10, microsecond=0)
                        - dt.timedelta(days=1)).timestamp() * 1000)
    platform.broker.closed.append({
        "position_id": "old-loss", "account_id": account_id,
        "pnl": -900.0, "exit_price": 2600.0, "reason": "stop_loss",
        "ts_ms": prev_evening})
    platform.broker.closed.append({
        "position_id": "new-loss", "account_id": account_id,
        "pnl": -250.0, "exit_price": 2600.0, "reason": "stop_loss",
        "ts_ms": today_morning})
    platform._refresh_snapshot(account_id, now_ms)
    snap = platform.snapshots[account_id]
    assert snap.daily_realized_pnl == pytest.approx(-250.0)


def test_other_accounts_losses_do_not_leak(client):
    platform = _platform()
    setup_account(client)
    account_id = "paper-10k"
    now_ms = int(time.time() * 1000)
    _force_closes(platform, "some-other-account", [-5000.0], now_ms)
    platform._refresh_snapshot(account_id, now_ms)
    assert platform.snapshots[account_id].daily_realized_pnl == 0.0


def test_equity_and_high_watermark_track_live_state(client):
    platform = _platform()
    setup_account(client)
    account_id = "paper-10k"
    now_ms = int(time.time() * 1000)
    _force_closes(platform, account_id, [250.0, -100.0], now_ms)
    platform._refresh_snapshot(account_id, now_ms)
    snap = platform.snapshots[account_id]
    assert snap.balance == pytest.approx(10_000.0 + 150.0)
    assert snap.equity == pytest.approx(10_150.0)
    assert snap.equity_high_watermark == pytest.approx(10_150.0)
