"""U1 tests: behaviour controls (Update Blueprint v2.1).

Covers: schema extension + backward compatibility, ea_policy → Auto
refusal, min-hold boundaries (exactly at the limit, one second under),
per-day and per-minute caps with sliding-window rollover, prohibited-
style detection true/false positives, hedging block, and backtest
parity (same strategy/data with and without controls).
"""
from __future__ import annotations

import json

import pytest

from core.compliance.behaviour import (
    BehaviourConfig,
    BehaviourControls,
    R_HEDGING,
    R_MAX_MODS,
    R_MAX_ORDERS_DAY,
    R_MAX_REQUESTS,
    R_MIN_HOLD,
)
from core.compliance.rules import FirmRuleSet, InactivityRule

H = 3_600_000  # one hour in ms


def mk_rules(**over) -> FirmRuleSet:
    base = dict(firm="Synthetic", program="1-step", version="v1",
                status="verified", verified_by="human",
                ea_policy="allowed")
    base.update(over)
    return FirmRuleSet(**base)


# ── schema ──────────────────────────────────────────────────────────────

def test_new_fields_default_safe_and_old_json_loads():
    rs = FirmRuleSet(firm="Synthetic", program="1-step", version="v1",
                     status="verified", verified_by="human")
    assert rs.ea_policy == "unknown"
    assert rs.min_hold_seconds is None
    assert rs.prohibited_styles == ["martingale", "grid", "averaging_down"]
    assert rs.inactivity_rule.enabled is False
    assert rs.news_restriction.block_open and rs.news_restriction.block_sltp
    # Old JSON without any U1 fields still loads.
    old = json.loads(json.dumps(rs.to_dict()))
    for k in ("ea_policy", "min_hold_seconds", "max_orders_per_day",
              "max_modifications_per_trade", "max_requests_per_minute",
              "vps_policy", "ip_policy", "hedging_policy",
              "copy_trading_policy", "inactivity_rule", "prohibited_styles",
              "platform", "server", "policy_notes"):
        old.pop(k, None)
    old["news_restriction"] = {
        k: v for k, v in old["news_restriction"].items()
        if k in ("enabled", "window_before_min", "window_after_min",
                 "applies_to")}
    rs2 = FirmRuleSet.from_dict(old)
    assert rs2.ea_policy == "unknown" and rs2.min_hold_seconds is None


def test_validate_rejects_bad_policy_values_and_limits():
    rs = mk_rules(ea_policy="maybe")
    assert any("ea_policy" in p for p in rs.validate())
    rs = mk_rules(min_hold_seconds=-5)
    assert any("min_hold_seconds" in p for p in rs.validate())
    rs = mk_rules(inactivity_rule=InactivityRule(enabled=True))
    assert any("inactivity_rule" in p for p in rs.validate())
    rs = mk_rules(prohibited_styles=["martingale", "nonsense"])
    assert any("prohibited styles" in p for p in rs.validate())
    clean = mk_rules(source_clause_refs={"daily_loss": ["3.2"],
                                         "max_loss": ["5.1"]})
    assert clean.validate() == []


def test_allows_auto_rule():
    assert not mk_rules(ea_policy="unknown").allows_auto()
    assert not mk_rules(ea_policy="forbidden").allows_auto()
    assert mk_rules(ea_policy="allowed").allows_auto()
    assert mk_rules(ea_policy="conditional").allows_auto()


# ── min hold ────────────────────────────────────────────────────────────

def test_min_hold_boundary_exact_and_one_second_under():
    bc = BehaviourControls(BehaviourConfig(min_hold_seconds=60))
    bc.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    # One second under: blocked.
    d = bc.check_close("a", "p1", 59_000)
    assert not d.allowed and R_MIN_HOLD in d.reasons
    # Exactly at the limit: allowed.
    assert bc.check_close("a", "p1", 60_000).allowed
    # One ms under: blocked.
    assert not bc.check_close("a", "p1", 59_999).allowed


def test_min_hold_protective_close_is_exception_not_block():
    bc = BehaviourControls(BehaviourConfig(min_hold_seconds=3600))
    bc.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    d = bc.check_close("a", "p1", 1_000, protective=True)
    assert d.allowed and d.details.get("protective_exception") is True
    assert len(bc.exceptions) == 1


def test_modify_cap_boundary():
    bc = BehaviourControls(BehaviourConfig(max_modifications_per_trade=2))
    bc.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    assert bc.check_modify("a", "p1", 1_000).allowed    # 1st
    assert bc.check_modify("a", "p1", 2_000).allowed    # 2nd (at cap)
    d = bc.check_modify("a", "p1", 3_000)               # 3rd → blocked
    assert not d.allowed and R_MAX_MODS in d.reasons
    assert d.details["modifications"] == 2


# ── caps and sliding window ─────────────────────────────────────────────

def test_max_orders_per_day_resets_at_utc_midnight():
    bc = BehaviourControls(BehaviourConfig(max_orders_per_day=2))
    day0 = 1_700_000_000_000 - (1_700_000_000_000 % 86_400_000)
    # The router records each accepted fill; two fills exhaust the cap.
    for i in range(2):
        assert bc.check_entry("a", "XAUUSD", 1, 1.0, 2600.0, day0).allowed
        bc.record_entry("a", f"p{i}", "XAUUSD", 1, 1.0, 2600.0, day0)
    d = bc.check_entry("a", "XAUUSD", 1, 1.0, 2600.0, day0)
    assert not d.allowed and R_MAX_ORDERS_DAY in d.reasons
    # Next UTC day: fresh count.
    assert bc.check_entry("a", "XAUUSD", 1, 1.0, 2600.0,
                          day0 + 86_400_000).allowed


def test_requests_per_minute_sliding_window_rollover():
    bc = BehaviourControls(BehaviourConfig(max_requests_per_minute=2))
    t0 = 1_700_000_000_000
    bc.record_request("a", t0)
    bc.record_request("a", t0 + 30_000)
    # Third request inside the 60 s window: blocked.
    d = bc.check_request_rate("a", t0 + 45_000)
    assert not d.allowed and R_MAX_REQUESTS in d.reasons
    # After the first request rolls out of the window: allowed again.
    assert bc.check_request_rate("a", t0 + 60_001).allowed


# ── prohibited styles ───────────────────────────────────────────────────

def test_martingale_detection_true_positive():
    bc = BehaviourControls(BehaviourConfig())
    # First entry 1.0 lot, closed at a loss; escalation to 1.5+ blocked.
    bc.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    bc.record_close("a", "p1", H, pnl=-100.0)
    d = bc.check_entry("a", "XAUUSD", 1, 1.6, 2600.0, H + 1)
    assert not d.allowed
    assert "prohibited_style_martingale" in d.reasons


def test_martingale_false_positive_is_not_flagged():
    bc = BehaviourControls(BehaviourConfig())
    bc.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    bc.record_close("a", "p1", H, pnl=-100.0)
    # Same size after a loss: NOT escalation.
    assert bc.check_entry("a", "XAUUSD", 1, 1.0, 2600.0, H + 1).allowed
    # Escalation after a WIN is not martingale (no loss to recover).
    bc.record_entry("a", "p2", "XAUUSD", 1, 1.0, 2600.0, H + 2)
    bc.record_close("a", "p2", 2 * H, pnl=+200.0)
    assert bc.check_entry("a", "XAUUSD", 1, 2.0, 2600.0, 2 * H + 1).allowed


def test_grid_and_averaging_detection():
    bc = BehaviourControls(BehaviourConfig())
    # Two longs at worsening (lower) prices within the window.
    bc.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    bc.record_entry("a", "p2", "XAUUSD", 1, 1.0, 2590.0, 60_000)
    d = bc.check_entry("a", "XAUUSD", 1, 1.0, 2580.0, 120_000)
    reasons = set(d.reasons)
    assert "prohibited_style_grid" in reasons
    assert "prohibited_style_averaging_down" in reasons
    # A long at a BETTER price is neither.
    bc2 = BehaviourControls(BehaviourConfig())
    bc2.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    bc2.record_entry("a", "p2", "XAUUSD", 1, 1.0, 2590.0, 60_000)
    assert bc2.check_entry("a", "XAUUSD", 1, 1.0, 2610.0, 120_000).allowed


def test_hedging_block():
    bc = BehaviourControls(BehaviourConfig(hedging_forbidden=True))
    bc.record_entry("a", "p1", "XAUUSD", 1, 1.0, 2600.0, 0)
    d = bc.check_hedging("a", "XAUUSD", -1)
    assert not d.allowed and R_HEDGING in d.reasons
    assert bc.check_hedging("a", "XAUUSD", 1).allowed   # same direction ok


def test_config_for_merges_ruleset_over_defaults():
    bc = BehaviourControls(BehaviourConfig(min_hold_seconds=30))
    rs = mk_rules(min_hold_seconds=120, max_orders_per_day=10,
                  hedging_policy="forbidden")
    cfg = bc.config_for(rs)
    assert cfg.min_hold_seconds == 120          # ruleset wins
    assert cfg.max_orders_per_day == 10
    assert cfg.hedging_forbidden is True
    assert cfg.prohibited_styles == ("martingale", "grid", "averaging_down")
    # Unknown hedging policy does NOT silently enable the block.
    cfg2 = bc.config_for(mk_rules())
    assert cfg2.hedging_forbidden is False
    assert cfg2.min_hold_seconds == 30          # base default kept


# ── backtest parity ─────────────────────────────────────────────────────

MIN = 60_000


def _ticks():
    """Deterministic rising-then-falling tick series: enough bars for a
    long, a losing close, a re-entry, and a winning close."""
    from tests.test_tick_engine import make_ticks
    rows = []
    price = 2000.0
    for i in range(24):                      # 24 minutes
        if i < 12:
            price = 2000.0 + i * 2.0         # up
        else:
            price = 2022.0 - (i - 12) * 3.0  # down
        rows.append((i * MIN, price, price + 0.5))
    return make_ticks(rows)


def _run(script: dict, over: dict) -> dict:
    from core.tick_engine import BacktestConfig, TickEngine
    from tests.test_tick_engine import FixedSignalStrategy
    strat = FixedSignalStrategy(script)
    cfg = BacktestConfig(symbol="XAUUSD", initial_balance=10_000.0,
                         risk_pct=1.0, dataset_hash="bh-test",
                         **over)
    return TickEngine().run(strat, _ticks(), cfg)


# Script: enter long on bar 2, close on bar 5 (young), re-enter bar 7,
# close on bar 10. With no controls both trades complete.
SCRIPT = {
    2: "ENTER",
    5: "CLOSE",
    7: "ENTER",
    10: "CLOSE",
}


def _script_strategy():
    from core.strategy_api import Signal, SIDE_CLOSE, SIDE_LONG
    from tests.test_tick_engine import FixedSignalStrategy
    out = {}
    for bar, what in SCRIPT.items():
        if what == "ENTER":
            out[bar] = [Signal(SIDE_LONG, sl=1950.0)]
        else:
            out[bar] = [Signal(SIDE_CLOSE)]
    return FixedSignalStrategy(out)


def _run_s(over: dict) -> dict:
    from core.tick_engine import BacktestConfig, TickEngine
    cfg = BacktestConfig(symbol="XAUUSD", initial_balance=10_000.0,
                         risk_pct=1.0, dataset_hash="bh-test", **over)
    return TickEngine().run(_script_strategy(), _ticks(), cfg)


def test_parity_no_controls_baseline_and_block_report():
    base = _run_s({})
    assert base["valid"]
    assert base["behaviour_blocks"] == {
        "min_hold_time": 0, "max_orders_per_day": 0,
        "max_modifications_per_trade": 0,
        "session_no_new_entries": 0,
        "prohibited_style_martingale": 0}
    assert base["total_trades"] == 2


def test_parity_min_hold_blocks_the_young_exit():
    # 10-minute min hold: the bar-5 close (3 minutes after entry) is
    # blocked; the bar-10 close (8 minutes after the re-entry) too.
    held = _run_s({"min_hold_seconds": 600})
    assert held["behaviour_blocks"]["min_hold_time"] == 2
    assert held["total_trades"] < 2   # both exits blocked → no booked trades


def test_parity_min_hold_exact_boundary_allows_the_close():
    # Entries fill at decision+latency; with a 60 s min hold the bar-5
    # close (~3 min after entry) is allowed: boundary sanity.
    ok = _run_s({"min_hold_seconds": 60})
    assert ok["behaviour_blocks"]["min_hold_time"] == 0
    assert ok["total_trades"] == 2


def test_parity_max_orders_per_day():
    capped = _run_s({"max_orders_per_day": 1})
    # Second entry of the day is blocked.
    assert capped["behaviour_blocks"]["max_orders_per_day"] >= 1
    assert capped["total_trades"] < 2


def test_parity_deterministic_same_config_same_result():
    a = _run_s({"min_hold_seconds": 600})
    b = _run_s({"min_hold_seconds": 600})
    assert a["config_hash"] == b["config_hash"]
    assert a["trades"] == b["trades"]
    assert a["behaviour_blocks"] == b["behaviour_blocks"]
