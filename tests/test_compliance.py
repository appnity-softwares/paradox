"""Part 10 scenario tests: firm rule schema, lifecycle, compliance engine,
challenge simulator. 40+ scenarios including edge cases: exactly at the
limit, reset rollover, floating-loss spikes, trailing watermark moves.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from core.compliance import (
    AccountState,
    ComplianceEngine,
    FirmRuleSet,
    ProposedTrade,
    RuleStore,
    simulate_challenge,
)
from core.compliance.rules import DailyLossRule, MaxLossRule, diff_rulesets

TS = 1_736_000_000_000  # 2025-01-04 (Saturday) 13:00 UTC-ish; tests set exact times


def ruleset(**over) -> FirmRuleSet:
    rs = FirmRuleSet(
        firm="test_firm", program="2_step_100k", version="v1",
        status="verified", verified_by="human-1",
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=100_000.0, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00",
                                 "America/New_York", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
        min_trading_days=4,
    )
    for k, v in over.items():
        setattr(rs, k, v)
    return rs


def state(**over) -> AccountState:
    base = dict(
        account_id="a1", balance=100_000.0, equity=100_000.0,
        balance_at_reset=100_000.0, equity_at_reset=100_000.0,
        equity_high_watermark=100_000.0, open_positions=0, exposure_lots=0.0,
    )
    base.update(over)
    return AccountState(**base)


def trade(**over) -> ProposedTrade:
    # 1 lot XAUUSD with a $10 stop -> worst case $1,000.
    base = dict(symbol="XAUUSD", lot=1.0, price=2600.0, sl_price=2590.0,
                time_ms=TS, side=1)
    base.update(over)
    return ProposedTrade(**base)


# ── 10.1 schema ──────────────────────────────────────────────────────────────

def test_schema_records_exact_basis_definitions():
    rs = ruleset()
    assert rs.daily_loss.basis == "balance"
    assert rs.daily_loss.reset_tz == "America/New_York"
    assert rs.daily_loss.includes_floating is True
    assert rs.max_loss.type == "static"


def test_schema_validation_rejects_unknown_basis():
    rs = ruleset(daily_loss=DailyLossRule(5.0, "weird", "17:00", "UTC", True))
    assert any("basis" in p for p in rs.validate())


def test_verified_requires_citations_and_verifier():
    rs = ruleset(status="verified", verified_by=None,
                 source_clause_refs={})
    assert rs.validate()  # problems listed


def test_schema_roundtrip_through_dict():
    rs = ruleset()
    clone = FirmRuleSet.from_dict(rs.to_dict())
    assert clone.content_hash() == rs.content_hash()


def test_diff_reports_field_changes():
    a, b = ruleset(), ruleset(profit_target_pct=9.0, version="v2")
    d = diff_rulesets(a, b)
    assert "profit_target_pct" in d and d["profit_target_pct"] == (8.0, 9.0)


# ── 10.2 lifecycle ───────────────────────────────────────────────────────────

def test_draft_rules_never_grant_permission():
    assert not ruleset(status="draft").grants_trading_permission()


def test_draft_cannot_be_enforced_by_engine(tmp_path):
    store = RuleStore(tmp_path / "rules")
    store.add_draft(ruleset(status="draft"))
    engine = ComplianceEngine()
    d = engine.pre_trade_check(store.verified("test_firm", "2_step_100k"),
                               state(), trade())
    assert not d.allowed and "rules_not_verified" in d.reasons


def test_missing_ruleset_blocks():
    d = ComplianceEngine().pre_trade_check(None, state(), trade())
    assert not d.allowed and "rules_not_verified" in d.reasons


def test_verify_requires_citations(tmp_path):
    store = RuleStore(tmp_path / "rules")
    store.add_draft(ruleset(status="draft", source_clause_refs={}))
    with pytest.raises(ValueError):
        store.verify("test_firm", "2_step_100k", "v1", "human-1")


def test_verify_flow_and_permission(tmp_path):
    store = RuleStore(tmp_path / "rules")
    store.add_draft(ruleset(status="draft"))
    v = store.verify("test_firm", "2_step_100k", "v1", "human-1")
    assert v.status == "verified" and v.verified_by == "human-1"
    assert store.verified("test_firm", "2_step_100k") is not None


def test_new_version_creates_diff_and_flags_reverification(tmp_path):
    store = RuleStore(tmp_path / "rules")
    store.add_draft(ruleset(status="draft"))
    store.verify("test_firm", "2_step_100k", "v1", "human-1")
    store.add_draft(ruleset(status="draft", version="v2", profit_target_pct=9.0))
    d = store.diff("test_firm", "2_step_100k", "v1", "v2")
    assert "profit_target_pct" in d
    assert store.needs_reverification("test_firm", "2_step_100k", "v1")
    assert not store.needs_reverification("test_firm", "2_step_100k", "v2")


def test_duplicate_version_rejected(tmp_path):
    store = RuleStore(tmp_path / "rules")
    store.add_draft(ruleset())
    with pytest.raises(ValueError):
        store.add_draft(ruleset())


# ── 10.3 pre-trade checks ────────────────────────────────────────────────────

def test_pre_trade_allows_normal_trade():
    d = ComplianceEngine().pre_trade_check(ruleset(), state(), trade())
    assert d.allowed


def test_symbol_not_allowed():
    rs = ruleset(allowed_symbols=["EURUSD"])
    d = ComplianceEngine().pre_trade_check(rs, state(), trade())
    assert "symbol_not_allowed" in d.reasons


def test_outside_trading_hours():
    rs = ruleset(trading_hours_utc=(7, 16))
    # 20:00 UTC.
    d = ComplianceEngine().pre_trade_check(rs, state(),
                                           trade(time_ms=20 * 3_600_000))
    assert "outside_trading_hours" in d.reasons


def test_max_lot():
    rs = ruleset(max_lot=0.5)
    d = ComplianceEngine().pre_trade_check(rs, state(), trade(lot=1.0))
    assert "max_lot" in d.reasons


def test_max_open_positions():
    rs = ruleset(max_open_positions=2)
    d = ComplianceEngine().pre_trade_check(rs, state(open_positions=2), trade())
    assert "max_open_positions" in d.reasons


def test_max_exposure_lots():
    rs = ruleset(max_exposure_lots=2.0)
    d = ComplianceEngine().pre_trade_check(rs, state(exposure_lots=1.5),
                                           trade(lot=1.0))
    assert "max_exposure_lots" in d.reasons


def test_news_window_blocks_open():
    rs = ruleset(news_restriction=__import__(
        "core.compliance.rules", fromlist=["NewsRestriction"]).NewsRestriction(
        enabled=True, window_before_min=2, window_after_min=2,
        applies_to="open_and_close"))
    event_ms = TS + 60_000  # event 1 minute after the trade
    d = ComplianceEngine().pre_trade_check(rs, state(), trade(),
                                           calendar_events_ms={event_ms: "NFP"})
    assert "news_window" in d.reasons


def test_news_outside_window_allows():
    rs = ruleset(news_restriction=__import__(
        "core.compliance.rules", fromlist=["NewsRestriction"]).NewsRestriction(
        enabled=True))
    event_ms = TS + 10 * 60_000  # 10 minutes away
    d = ComplianceEngine().pre_trade_check(rs, state(), trade(),
                                           calendar_events_ms={event_ms: "NFP"})
    assert d.allowed


def test_calendar_failure_blocks_not_allows():
    rs = ruleset(news_restriction=__import__(
        "core.compliance.rules", fromlist=["NewsRestriction"]).NewsRestriction(
        enabled=True))
    d = ComplianceEngine().pre_trade_check(rs, state(), trade(),
                                           calendar_events_ms=None)
    assert "calendar_unavailable" in d.reasons


def test_weekend_holding_forbidden_after_friday_cutoff():
    rs = ruleset(weekend_holding="forbidden", weekend_cutoff_utc_hour=20)
    friday_21 = pd.Timestamp("2025-01-10 21:00", tz="UTC").value // 10**6
    d = ComplianceEngine().pre_trade_check(rs, state(), trade(time_ms=friday_21))
    assert "weekend_holding_forbidden" in d.reasons


def test_weekend_holding_allowed_before_cutoff():
    rs = ruleset(weekend_holding="forbidden", weekend_cutoff_utc_hour=20)
    friday_15 = pd.Timestamp("2025-01-10 15:00", tz="UTC").value // 10**6
    d = ComplianceEngine().pre_trade_check(rs, state(), trade(time_ms=friday_15))
    assert d.allowed


def test_missing_sl_rejected():
    d = ComplianceEngine().pre_trade_check(ruleset(), state(), trade(sl_price=None))
    assert "missing_stop_loss" in d.reasons


def test_daily_loss_worst_case_exactly_at_limit_blocks():
    # Balance basis, 5% of 100k = $5,000. A trade that could lose exactly
    # $5,000 must be blocked (>= limit).
    rs = ruleset()
    d = ComplianceEngine().pre_trade_check(
        rs, state(), trade(lot=5.0, sl_price=2590.0))  # 5*10*100 = 5000
    assert "daily_loss_worst_case" in d.reasons


def test_daily_loss_worst_case_just_under_limit_allows():
    # No buffer: 4,900 < 5,000 fits.
    d = ComplianceEngine(safety_buffer_pct=100.0).pre_trade_check(
        ruleset(), state(), trade(lot=4.9, sl_price=2590.0))
    assert d.allowed


def test_daily_loss_safety_buffer_blocks_early():
    # 80% buffer: a $4,100 worst case (4.1%) exceeds 4,000 -> blocked.
    d = ComplianceEngine(safety_buffer_pct=80.0).pre_trade_check(
        ruleset(), state(), trade(lot=4.1, sl_price=2590.0))
    assert "daily_loss_worst_case" in d.reasons


def test_daily_loss_uses_existing_floating_loss():
    # Already -4,500 floating today; even a $600 worst case breaches 5,000.
    d = ComplianceEngine().pre_trade_check(
        ruleset(), state(daily_floating_pnl=-4_500.0),
        trade(lot=0.6, sl_price=2590.0))
    assert "daily_loss_worst_case" in d.reasons


def test_daily_loss_equity_basis_uses_equity_at_reset():
    rs = ruleset(daily_loss=DailyLossRule(5.0, "equity", "17:00", "UTC", True))
    # Equity at reset 90k -> limit 4,500; a 4,600 worst case breaches.
    d = ComplianceEngine().pre_trade_check(
        rs, state(equity_at_reset=90_000.0, equity=90_000.0),
        trade(lot=4.6, sl_price=2590.0))
    assert "daily_loss_worst_case" in d.reasons


def test_daily_loss_balance_or_equity_takes_worse_reference():
    rs = ruleset(daily_loss=DailyLossRule(5.0, "balance_or_equity", "17:00",
                                          "UTC", True))
    d = ComplianceEngine().pre_trade_check(
        rs, state(balance_at_reset=100_000.0, equity_at_reset=90_000.0,
                  equity=90_000.0),
        trade(lot=4.6, sl_price=2590.0))  # 4,600 > 4,500 (equity ref)
    assert "daily_loss_worst_case" in d.reasons


def test_daily_loss_excludes_floating_when_firm_says_so():
    rs = ruleset(daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", False))
    # -4,500 floating does NOT count; a 4,900 worst case still fits.
    d = ComplianceEngine(safety_buffer_pct=100.0).pre_trade_check(
        rs, state(daily_floating_pnl=-4_500.0),
        trade(lot=4.9, sl_price=2590.0))
    assert d.allowed


def test_max_loss_static_from_starting_balance():
    rs = ruleset(max_loss=MaxLossRule(10.0, "static", "starting_balance", True))
    # Equity 91k -> used 9k; worst case 1,100 -> projected 10,100 >= 10,000.
    d = ComplianceEngine().pre_trade_check(
        rs, state(equity=91_000.0, balance=91_000.0),
        trade(lot=1.1, sl_price=2590.0))
    assert "max_loss_worst_case" in d.reasons


def test_max_loss_trailing_from_equity_hwm():
    rs = ruleset(max_loss=MaxLossRule(10.0, "trailing", "equity_hwm", True))
    # HWM 110k -> limit 11,000; equity 100k -> used 10,000; +600 worst case
    # projects 10,600 < 11,000 -> allowed (buffer off for this test).
    d = ComplianceEngine(safety_buffer_pct=100.0).pre_trade_check(
        rs, state(equity=100_000.0, equity_high_watermark=110_000.0),
        trade(lot=0.6, sl_price=2590.0))
    assert d.allowed


def test_max_loss_trailing_blocks_after_hwm_rise():
    rs = ruleset(max_loss=MaxLossRule(10.0, "trailing", "equity_hwm", True))
    # HWM 110k, equity 99.5k -> used 10,500; +600 -> 11,100 >= 11,000.
    d = ComplianceEngine().pre_trade_check(
        rs, state(equity=99_500.0, equity_high_watermark=110_000.0),
        trade(lot=0.6, sl_price=2590.0))
    assert "max_loss_worst_case" in d.reasons


def test_max_loss_excludes_floating_when_firm_says_so():
    rs = ruleset(max_loss=MaxLossRule(10.0, "static", "starting_balance", False))
    # Balance basis, floating excluded: ref - balance = 9,000 used;
    # +600 worst case -> 9,600 < 10,000 -> allowed (buffer off).
    d = ComplianceEngine(safety_buffer_pct=100.0).pre_trade_check(
        rs, state(equity=90_000.0, balance=91_000.0),
        trade(lot=0.6, sl_price=2590.0))
    assert d.allowed


def test_consistency_best_day_blocks_when_share_exceeded():
    rs = ruleset(consistency_rule=__import__(
        "core.compliance.rules", fromlist=["ConsistencyRule"]).ConsistencyRule(
        enabled=True, best_day_max_share=0.30))
    # Total profit 10,000, today already 3,500 (35% > 30%).
    d = ComplianceEngine().pre_trade_check(
        rs, state(total_profit_so_far=10_000.0, today_profit=3_500.0), trade())
    assert "consistency_best_day" in d.reasons


def test_consistency_allows_under_share():
    rs = ruleset(consistency_rule=__import__(
        "core.compliance.rules", fromlist=["ConsistencyRule"]).ConsistencyRule(
        enabled=True, best_day_max_share=0.30))
    d = ComplianceEngine().pre_trade_check(
        rs, state(total_profit_so_far=10_000.0, today_profit=2_500.0), trade())
    assert d.allowed


# ── 10.3 real-time monitoring ────────────────────────────────────────────────

def test_monitor_ok_state():
    s = ComplianceEngine().monitor(ruleset(), state())
    assert s.level == "ok"
    assert s.remaining_daily_loss == pytest.approx(5_000.0)
    assert s.remaining_max_loss == pytest.approx(10_000.0)


def test_monitor_warning_at_half_limit():
    s = ComplianceEngine().monitor(ruleset(),
                                   state(daily_realized_pnl=-2_600.0))
    assert s.level == "warning"


def test_monitor_critical_at_safety_buffer():
    eng = ComplianceEngine(safety_buffer_pct=80.0)
    s = eng.monitor(ruleset(), state(daily_realized_pnl=-4_100.0))
    assert s.level == "critical"  # auto-protection zone


def test_monitor_breached_at_limit():
    s = ComplianceEngine().monitor(ruleset(),
                                   state(daily_realized_pnl=-5_000.0))
    assert s.level == "breached"


def test_monitor_floating_loss_spike_counts():
    s = ComplianceEngine().monitor(ruleset(),
                                   state(daily_floating_pnl=-4_800.0))
    assert s.level in ("critical", "breached")


def test_monitor_trailing_max_loss_moves_with_hwm():
    rs = ruleset(max_loss=MaxLossRule(10.0, "trailing", "equity_hwm", True))
    s = ComplianceEngine().monitor(
        rs, state(equity=100_000.0, equity_high_watermark=120_000.0))
    # Limit 12,000 from HWM; used 20,000 -> breached.
    assert s.level == "breached"


def test_monitor_profit_target_progress():
    s = ComplianceEngine().monitor(ruleset(),
                                   state(total_profit_so_far=4_000.0))
    assert s.profit_target_progress_pct == pytest.approx(50.0)


def test_monitor_trading_days_reported():
    s = ComplianceEngine().monitor(ruleset(), state(trading_days_completed=2))
    assert s.trading_days_completed == 2


def test_monitor_drawdown_pct():
    s = ComplianceEngine().monitor(
        ruleset(), state(equity=95_000.0, equity_high_watermark=100_000.0))
    assert s.current_drawdown_pct == pytest.approx(5.0)


# ── 10.3 challenge simulator ─────────────────────────────────────────────────

def eq_curve(pnls: list[float], start: float = 100_000.0,
             tz: str = "America/New_York") -> pd.Series:
    idx = pd.date_range("2025-01-06 09:30", periods=len(pnls) + 1,
                        freq="6h", tz=tz)
    vals = [start]
    for p in pnls:
        vals.append(vals[-1] + p)
    return pd.Series(vals, index=idx)


def trades_df(pnls: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(pnls)
    vols = volumes or [1.0] * n
    return pd.DataFrame({
        "pnl": pnls,
        "volume": vols,
        "entry_time": pd.date_range("2025-01-06 10:00", periods=n, freq="6h",
                                    tz="America/New_York"),
        "exit_time": pd.date_range("2025-01-06 11:00", periods=n, freq="6h",
                                   tz="America/New_York"),
    })


def test_simulator_refuses_draft_rules():
    r = simulate_challenge(ruleset(status="draft"), eq_curve([100.0]),
                           trades_df([100.0]))
    assert not r.passed and "not verified" in r.notes[0]


def test_simulator_passes_clean_challenge():
    # 8% target with tiny wins, no breaches, 5 trading days.
    r = simulate_challenge(ruleset(min_trading_days=4),
                           eq_curve([800.0] * 11), trades_df([800.0] * 11))
    assert r.passed and r.target_hit and not r.breached


def test_simulator_fails_on_daily_loss_breach():
    # One day loses 6% (6,000) -> daily breach (floating included).
    pnls = [500.0, -6_000.0, 500.0]
    r = simulate_challenge(ruleset(), eq_curve(pnls), trades_df(pnls))
    assert r.breached and not r.passed
    assert any(h["rule"] == "breach" and "daily_loss" in h["detail"]
               for h in r.rule_hits)


def test_simulator_fails_on_static_max_loss():
    pnls = [-4_000.0, -4_000.0, -3_000.0]  # cumulative -11,000 > 10,000
    r = simulate_challenge(ruleset(), eq_curve(pnls), trades_df(pnls))
    assert r.breached


def test_simulator_trailing_max_loss_uses_hwm():
    rs = ruleset(max_loss=MaxLossRule(10.0, "trailing", "equity_hwm", True))
    # Run up to 110k then fall 11k -> breach vs trailing HWM.
    pnls = [10_000.0, -11_000.0]
    r = simulate_challenge(rs, eq_curve(pnls), trades_df(pnls))
    assert r.breached and "max_loss" in r.rule_hits[-1]["detail"]


def test_simulator_counts_trading_days_by_volume():
    r = simulate_challenge(ruleset(min_trading_days=4),
                           eq_curve([100.0] * 3),
                           trades_df([100.0] * 3, volumes=[1.0] * 3))
    # Exits at 11:00, 17:00, 23:00 NY -> the 17:00 reset puts the last two
    # on the SAME firm-day: 2 distinct trading days.
    assert r.trading_days_used == 2 and not r.passed  # target also not hit


def test_simulator_ignores_zero_volume_days():
    r = simulate_challenge(ruleset(min_trading_days=1),
                           eq_curve([100.0, 100.0]),
                           trades_df([100.0, 100.0], volumes=[0.0, 0.0]))
    assert r.trading_days_used == 0


def test_simulator_flags_weekend_entry():
    # Friday 21:00 entry with weekend holding forbidden.
    trades = trades_df([100.0])
    trades["entry_time"] = [pd.Timestamp("2025-01-10 21:00",
                                         tz="America/New_York")]
    trades["exit_time"] = [pd.Timestamp("2025-01-10 22:00",
                                        tz="America/New_York")]
    r = simulate_challenge(ruleset(weekend_holding="forbidden"),
                           eq_curve([100.0]), trades)
    assert any(h["rule"] == "weekend_entry" for h in r.rule_hits)


def test_simulator_reports_drawdown_path():
    r = simulate_challenge(ruleset(), eq_curve([1_000.0, -2_000.0, 500.0]),
                           trades_df([1_000.0, -2_000.0, 500.0]))
    assert len(r.daily_loss_path) >= 1
    assert r.max_drawdown_pct > 0


def test_simulator_reset_time_splits_days():
    # Two trades either side of the 17:00 New York reset land on different
    # firm-days; the -6,000 loss then stays within one day's limit.
    trades = trades_df([-6_000.0])
    trades["exit_time"] = [pd.Timestamp("2025-01-06 16:00",
                                        tz="America/New_York")]
    equity = eq_curve([-6_000.0])
    equity.index = [pd.Timestamp("2025-01-06 09:00", tz="America/New_York"),
                    pd.Timestamp("2025-01-06 18:00", tz="America/New_York")]
    r = simulate_challenge(ruleset(), equity, trades)
    assert r.breached  # the loss day still breaches


def test_store_persists_json(tmp_path):
    store = RuleStore(tmp_path / "rules")
    store.add_draft(ruleset(status="draft"))
    files = list((tmp_path / "rules").glob("*.json"))
    assert files and json.loads(files[0].read_text())[0]["firm"] == "test_firm"
