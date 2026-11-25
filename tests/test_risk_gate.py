"""Part 8 tests: deterministic risk engine, fail-closed inputs, kill switch."""
from __future__ import annotations

import json

import pytest

from core.config import SYMBOL_SPECS
from core.risk import (
    AccountSnapshot,
    DeterministicRiskEngine,
    KillSwitch,
    OrderProposal,
    RiskConfig,
)

XAU = SYMBOL_SPECS["XAUUSD"]
CFG = RiskConfig(version="test-v1")


def snapshot(**over):
    base = dict(
        account_id="a1", equity=10_000.0, balance=10_000.0, used_margin=0.0,
        open_positions=0, exposure_lots_by_symbol={},
        daily_realized_pnl=0.0, daily_floating_pnl=0.0,
        equity_high_watermark=10_000.0,
        last_tick_ms={"XAUUSD": 1_000}, spread={"XAUUSD": 0.20},
        server_time_ms=2_000,
    )
    base.update(over)
    return AccountSnapshot(**base)


def proposal(**over):
    base = dict(account_id="a1", symbol="XAUUSD", side=1, lot=0.10,
                price=2600.0, sl_price=2588.0, time_ms=2_000)  # $12 stop
    base.update(over)
    return OrderProposal(**base)


def evaluate(prop=None, snap=None, cfg=None, **kw):
    return DeterministicRiskEngine(cfg or CFG).evaluate(
        prop or proposal(), snap or snapshot(), **kw)


# ── Happy path and the numbers ───────────────────────────────────────────────

def test_basic_approval_with_exact_numbers():
    # 0.08 lot x $12 stop x 100 contract = $96 = 0.96% of 10k <= 1% cap.
    d = evaluate(proposal(lot=0.08))
    assert d.outcome == "approved"
    assert d.lot == 0.08
    assert d.risk_money == pytest.approx(96.0)
    assert d.risk_pct == pytest.approx(0.96)
    assert d.margin_required == pytest.approx(0.08 * 100 * 2600 / 100)
    assert d.config_version == "test-v1"


def test_all_checks_run_and_report_every_reason():
    # Wide spread AND stale price AND outside hours at once.
    cfg = RiskConfig(version="v", spread_max=0.10, stale_price_max_ms=100,
                     trading_hours_utc=(10, 11))
    d = evaluate(proposal(lot=0.08, time_ms=2_000),
                 snapshot(last_tick_ms={"XAUUSD": 0}, spread={"XAUUSD": 0.40}),
                 cfg=cfg)
    assert set(d.reasons) == {"stale_price", "spread_too_wide",
                              "outside_trading_hours"}


# ── Fail closed ──────────────────────────────────────────────────────────────

def test_missing_stop_loss_rejects():
    assert "missing_stop_loss" in evaluate(proposal(sl_price=None)).reasons


def test_unknown_symbol_rejects():
    assert "unknown_symbol" in evaluate(proposal(symbol="NOPE")).reasons


def test_missing_market_data_rejects():
    d = evaluate(snapshot(last_tick_ms={}, spread={}))
    assert "missing_market_data" in d.reasons


def test_incomplete_snapshot_rejects():
    assert "incomplete_account_snapshot" in evaluate(snapshot(equity=float("nan"))).reasons


def test_stale_price_rejects():
    d = evaluate(snapshot(last_tick_ms={"XAUUSD": 0}, server_time_ms=20_000))
    assert "stale_price" in d.reasons


# ── Limit checks ─────────────────────────────────────────────────────────────

def test_risk_cap_reduces_oversized_trade():
    # 0.5 lot with a $12 stop = $600 risk = 6% of 10k > 1% cap -> the
    # engine offers the largest lot that fits (0.08) instead of flat-rejecting.
    d = evaluate(proposal(lot=0.50))
    assert d.outcome == "reduced"
    assert "risk_cap" in d.reasons
    assert d.lot == 0.08


def test_risk_cap_offers_reduced_lot():
    # 1% of 10k = $100 cap; $12 stop, contract 100 -> max lot 0.0833 -> 0.08.
    d = evaluate(proposal(lot=0.50))
    assert d.outcome == "reduced"
    assert d.lot == 0.08
    assert d.risk_money == pytest.approx(96.0)
    assert d.risk_pct == pytest.approx(0.96)


def test_reduced_lot_below_min_lot_rejects():
    # Equity 1,100 -> cap $11 -> raw max lot 0.00916 -> below the 0.01
    # floor -> the risk cap becomes a hard rejection.
    d = evaluate(proposal(lot=0.50), snapshot(equity=1_100.0))
    assert d.outcome == "rejected"
    assert "risk_cap" in d.reasons


def test_daily_loss_limit_blocks():
    d = evaluate(snapshot(daily_realized_pnl=-250.0, daily_floating_pnl=-60.0))
    assert "daily_loss_limit" in d.reasons  # -310 <= -300


def test_daily_loss_limit_exactly_at_limit_blocks():
    d = evaluate(snapshot(daily_realized_pnl=-300.0))
    assert "daily_loss_limit" in d.reasons  # exactly at the limit


def test_daily_loss_safety_buffer_acts_early():
    cfg = RiskConfig(version="v", max_daily_loss_pct=3.0, safety_buffer_pct=80.0)
    d = evaluate(snapshot(daily_realized_pnl=-250.0), cfg=cfg)
    assert "daily_loss_limit" in d.reasons  # -250 <= -240


def test_max_drawdown_blocks():
    d = evaluate(snapshot(equity=8_400.0, equity_high_watermark=10_000.0))
    assert "max_drawdown" in d.reasons  # 16% >= 15%


def test_max_open_positions_blocks():
    d = evaluate(snapshot(open_positions=3))
    assert "max_open_positions" in d.reasons


def test_total_exposure_blocks():
    d = evaluate(snapshot(exposure_lots_by_symbol={"XAUUSD": 4.95}))
    assert "max_total_exposure" in d.reasons


def test_correlated_bucket_blocks_silver_via_gold_bucket():
    cfg = RiskConfig(version="v",
                     correlated_buckets={"metals": ("XAUUSD", "XAGUSD")},
                     correlated_max_lots=1.0)
    d = evaluate(snapshot(exposure_lots_by_symbol={"XAGUSD": 0.95}), cfg=cfg)
    assert "correlated_exposure" in d.reasons


def test_margin_usage_blocks():
    # 0.9 lot margin = 0.9*100*2600/100 = 2340 = 23.4%; add used 2800 -> 51.4%.
    d = evaluate(proposal(lot=0.90), snapshot(used_margin=2_800.0))
    assert "margin_usage" in d.reasons


def test_leverage_cap_blocks():
    # 5 lots notional = 5*100*2600 = 1.3M / 10k equity = 130x > 100x.
    d = evaluate(proposal(lot=5.0))
    assert "max_leverage" in d.reasons


def test_broker_stops_level_blocks_tight_sl():
    cfg = RiskConfig(version="v", stops_level=0.50)
    d = evaluate(proposal(sl_price=2599.8), cfg=cfg)  # 0.20 distance
    assert "sl_inside_stops_level" in d.reasons


def test_freeze_level_blocks():
    cfg = RiskConfig(version="v", freeze_level=5.0)
    d = evaluate(proposal(sl_price=2598.0), cfg=cfg)  # 2.0 < 5.0
    assert "sl_inside_freeze_level" in d.reasons


def test_lot_off_grid_rejects():
    assert "lot_off_grid" in evaluate(proposal(lot=0.085)).reasons


def test_below_min_lot_rejects():
    assert "below_min_lot" in evaluate(proposal(lot=0.005)).reasons


def test_overnight_trading_window():
    cfg = RiskConfig(version="v", trading_hours_utc=(22, 6))
    eng = DeterministicRiskEngine(cfg)
    # 23:00 UTC allowed, 12:00 UTC blocked.
    assert eng._hours_ok(proposal(time_ms=23 * 3_600_000), snapshot())
    assert not eng._hours_ok(proposal(time_ms=12 * 3_600_000), snapshot())


# ── Kill switch ──────────────────────────────────────────────────────────────

def test_kill_switch_blocks_and_persists(tmp_path):
    ks = KillSwitch(tmp_path / "kill.json")
    ks.engage_account("a1", actor="human-1", reason="manual halt")
    engine = DeterministicRiskEngine(CFG, kill_switch=ks)
    d = engine.evaluate(proposal(), snapshot())
    assert d.outcome == "rejected" and "kill_switch_engaged" in d.reasons
    # Other accounts unaffected.
    d2 = engine.evaluate(proposal(account_id="a2", lot=0.08),
                         snapshot(account_id="a2"))
    assert d2.outcome == "approved"
    # Survives restart from the same file.
    ks2 = KillSwitch(tmp_path / "kill.json")
    assert ks2.is_blocked("a1") and not ks2.is_blocked("a2")
    # Only a human action disengages.
    ks2.disengage_account("a1", actor="human-1", reason="resume")
    assert not KillSwitch(tmp_path / "kill.json").is_blocked("a1")


def test_kill_switch_global_and_audit_log(tmp_path):
    ks = KillSwitch(tmp_path / "kill.json")
    ks.engage_global(actor="human-1", reason="news event")
    assert ks.is_blocked("any-account")
    ks.disengage_global(actor="human-1", reason="event over")
    actions = [e["action"] for e in ks.audit_log]
    assert actions == ["engage_global", "disengage_global"]
    assert all(e["actor"] == "human-1" for e in ks.audit_log)


def test_kill_switch_corrupt_state_fails_closed(tmp_path):
    p = tmp_path / "kill.json"
    p.write_text("{not json")
    assert KillSwitch(p).is_blocked("a1")


def test_kill_switch_state_file_is_json(tmp_path):
    p = tmp_path / "kill.json"
    KillSwitch(p).engage_account("a1", actor="h", reason="r")
    data = json.loads(p.read_text())
    assert data["accounts"]["a1"] is True


# ── Compliance auto-protection gate (Part 10.3 x Part 8) ─────────────────────

from types import SimpleNamespace  # noqa: E402


def test_compliance_critical_blocks_new_orders():
    status = SimpleNamespace(level="critical")
    d = evaluate(proposal(lot=0.08), compliance_status=status)
    assert "compliance_auto_protection" in d.reasons


def test_compliance_breached_blocks_new_orders():
    status = SimpleNamespace(level="breached")
    d = evaluate(proposal(lot=0.08), compliance_status=status)
    assert "compliance_auto_protection" in d.reasons


def test_compliance_ok_and_warning_allow():
    for level in ("ok", "warning"):
        d = evaluate(proposal(lot=0.08),
                     compliance_status=SimpleNamespace(level=level))
        assert d.outcome == "approved"
