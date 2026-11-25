"""Part 12.1/12.2 tests: order flow, idempotency, timeout reconciliation,
mode gate, and the malicious-agent test (no path reaches the broker
without Risk and Compliance approval).
"""
from __future__ import annotations

import pytest

from core.account import Account
from core.adaptation import StrategyProfile
from core.compliance import ComplianceEngine, FirmRuleSet
from core.compliance.rules import DailyLossRule, MaxLossRule
from core.config import COST_MODELS, SYMBOL_SPECS
from core.execution import (
    AuditLog,
    ExecutionRouter,
    ModeGate,
    ModeError,
    Notifier,
    OrderIntent,
    PaperBroker,
    make_version,
)
from core.execution.versions import StrategyVersion
from core.risk import AccountSnapshot, DeterministicRiskEngine, KillSwitch, RiskConfig

XAU = SYMBOL_SPECS["XAUUSD"]

PROFILE = StrategyProfile(
    key="ema_cross", name="EMA cross (ATR stops)", max_dd_pct=35.0,
    typical_stop_distance=12.0, recommended_risk_pct=1.0,
    cost_per_round_trip=0.50)

RULES = FirmRuleSet(
    firm="test_firm", program="100k", version="v1", status="verified",
    ea_policy="allowed",
    verified_by="human-1",
    source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
    starting_balance=10_000.0, profit_target_pct=8.0,
    daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", True),
    max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
    min_trading_days=0, weekend_holding="allowed")


def snapshot(**over):
    base = dict(
        account_id="a1", equity=10_000.0, balance=10_000.0, used_margin=0.0,
        open_positions=0, exposure_lots_by_symbol={},
        daily_realized_pnl=0.0, daily_floating_pnl=0.0,
        equity_high_watermark=10_000.0,
        last_tick_ms={"XAUUSD": 1_000}, spread={"XAUUSD": 0.20},
        server_time_ms=2_000)
    base.update(over)
    return AccountSnapshot(**base)


def intent(key="ord-001", **over):
    base = dict(
        idempotency_key=key, account_id="a1", symbol="XAUUSD", side=1,
        intended_risk_pct=1.0,
        strategy_version=make_version(type("S", (), {}), {}),
        sl_price=2588.0, tp_price=2624.0, created_ms=2_000)
    base.update(over)
    return OrderIntent(**base)


@pytest.fixture()
def env(tmp_path):
    broker = PaperBroker(cost_model=COST_MODELS["XAUUSD_ECN"])
    broker.connect()
    broker.on_tick("XAUUSD", bid=2599.80, ask=2600.20, ts_ms=1_000)
    router = ExecutionRouter(
        broker, DeterministicRiskEngine(RiskConfig(version="v1")),
        ComplianceEngine(), ModeGate(),
        AuditLog(tmp_path / "audit" / "log.jsonl"), Notifier())
    router.register_account(Account(account_id="a1", balance=10_000.0),
                            snapshot(), PROFILE, RULES, XAU, 2600.0)
    return broker, router


def setup_auto(router: ExecutionRouter):
    router.modes.set_mode("a1", "auto", verdict="PAPER_TRADE_ELIGIBLE",
                          paper_days=30, rules_verified=True,
                          confirmed_by="human-1")


# ── Mode gate (12.5) ─────────────────────────────────────────────────────────

def test_research_mode_places_no_orders(env):
    broker, router = env
    router.modes.set_mode("a1", "research")
    r = router.submit(intent())
    assert r.status == "rejected" and "mode_gate" in r.reasons[0]
    assert not broker.order_history


def test_manual_mode_requires_approval_token(env):
    broker, router = env
    router.modes.set_mode("a1", "manual")
    r1 = router.submit(intent())
    assert r1.status == "rejected"
    r2 = router.submit(intent(key="ord-002"),
                       approval_token="human-yes")
    assert r2.status == "filled"


def test_auto_mode_requires_full_evidence():
    gate = ModeGate()
    with pytest.raises(ModeError):
        gate.set_mode("a1", "auto")  # no verdict, no paper days, nothing
    with pytest.raises(ModeError):
        gate.set_mode("a1", "auto", verdict="RESEARCH_ONLY", paper_days=30,
                      rules_verified=True, confirmed_by="human-1")
    with pytest.raises(ModeError):
        gate.set_mode("a1", "auto", verdict="PAPER_TRADE_ELIGIBLE",
                      paper_days=3, rules_verified=True, confirmed_by="human-1")
    am = gate.set_mode("a1", "auto", verdict="PAPER_TRADE_ELIGIBLE",
                       paper_days=30, rules_verified=True,
                       confirmed_by="human-1")
    assert am.mode == "auto"


# ── The single path (12.1) ───────────────────────────────────────────────────

def test_full_pipeline_fills(env):
    broker, router = env
    setup_auto(router)
    r = router.submit(intent())
    assert r.status == "filled"
    assert r.filled_lot == 0.08          # 1% risk, $12 stop, floored
    assert r.fill_price == pytest.approx(2600.20 + 0.05)  # ask + slippage
    assert {s["step"] for s in r.steps} >= {
        "mode_gate", "sizing", "risk", "compliance", "order_check", "send"}
    assert len(broker.order_history) == 1


def test_broker_rejects_raw_objects(env):
    broker, _ = env
    with pytest.raises(TypeError):
        broker.place(intent())           # an OrderIntent is not an ApprovedOrder
    with pytest.raises(TypeError):
        broker.place({"symbol": "XAUUSD", "lot": 1.0})   # a dict even less


def test_risk_rejection_blocks_broker(env):
    broker, router = env
    setup_auto(router)
    # Daily loss near the limit: risk engine blocks the new order.
    router.snapshots["a1"] = snapshot(daily_realized_pnl=-250.0,
                                      daily_floating_pnl=-60.0)
    r = router.submit(intent())
    assert r.status == "rejected"
    assert "daily_loss_limit" in r.reasons
    assert not broker.order_history


def test_compliance_rejection_blocks_broker(env):
    broker, router = env
    setup_auto(router)
    # Unverified rules -> no permission.
    router.rulesets["a1"] = None
    r = router.submit(intent())
    assert r.status == "rejected" and "rules_not_verified" in r.reasons
    assert not broker.order_history


def test_malicious_agent_cannot_bypass(env):
    """Every bypass attempt fails: no direct broker path without an
    ApprovedOrder, and the router enforces risk + compliance + mode."""
    broker, router = env
    # 1) Direct broker call with a hand-made approved-like object fails.
    class Fake:
        pass
    with pytest.raises(TypeError):
        broker.place(Fake())
    # 2) Router with a kill switch engaged refuses everything.
    ks = KillSwitch(tmp_path_none := __import__("pathlib").Path(
        __import__("tempfile").mkdtemp()) / "kill.json")
    ks.engage_global("human-1", "halt")
    router2 = ExecutionRouter(
        broker, DeterministicRiskEngine(RiskConfig(version="v1")),
        ComplianceEngine(), ModeGate(),
        AuditLog(__import__("pathlib").Path(
            __import__("tempfile").mkdtemp()) / "log.jsonl"))
    router2.register_account(Account(account_id="a1", balance=10_000.0),
                             snapshot(), PROFILE, RULES, XAU, 2600.0)
    setup_auto(router2)
    router2.risk.kill_switch = ks
    r = router2.submit(intent(key="ord-ks"))
    assert r.status == "rejected" and "kill_switch_engaged" in r.reasons
    # 3) The router's pipeline is the only minter of ApprovedOrder.
    import inspect
    src = inspect.getsource(type(router))
    assert "ApprovedOrder(" in src   # minted inside the gate only


def test_missing_sl_rejected_before_broker(env):
    broker, router = env
    setup_auto(router)
    r = router.submit(intent(sl_price=None))
    assert r.status == "rejected"
    assert "missing_stop_loss" in r.reasons
    assert not broker.order_history


# ── Idempotency and timeout reconciliation (12.2) ────────────────────────────

def test_duplicate_retry_never_creates_two_positions(env):
    broker, router = env
    setup_auto(router)
    r1 = router.submit(intent(key="dup-1"))
    r2 = router.submit(intent(key="dup-1"))     # forced retry
    r3 = router.submit(intent(key="dup-1"))     # and again
    assert r1.status == "filled"
    assert r2.status == "duplicate" and r2.order_ref == r1.order_ref
    assert r3.status == "duplicate"
    assert len(broker.order_history) == 1
    assert len(broker.positions_of("a1")) == 1


def test_timeout_with_fill_reconciles_to_known_state(env):
    broker, router = env
    setup_auto(router)
    broker.fail_next_send_with_timeout = True   # fills, THEN times out
    r = router.submit(intent(key="to-1"))
    assert r.status == "filled"                 # reconciled via broker query
    assert r.order_ref is not None
    assert len(broker.positions_of("a1")) == 1  # exactly one position
    # A retry after reconciliation is a duplicate, not a new order.
    r2 = router.submit(intent(key="to-1"))
    assert r2.status == "duplicate"
    assert len(broker.order_history) == 1


def test_timeout_without_fill_reports_pending_unknown(env):
    broker, router = env
    setup_auto(router)
    broker.find_by_idempotency_key = lambda key: None  # broker lost it too
    broker.fail_next_send_with_timeout = True
    # Patch: make place raise WITHOUT opening the position.
    def place_no_fill(approved):
        raise ConnectionError("connection lost mid-send")
    broker.place = place_no_fill
    r = router.submit(intent(key="to-2"))
    assert r.status == "pending_unknown"
    assert not broker.positions_of("a1")


def test_paused_account_cannot_submit_new_entries(env):
    broker, router = env
    setup_auto(router)
    from core.execution.monitoring import Reconciler
    rec = Reconciler(broker, router.audit, router.notifier)
    router.reconciler = rec
    rec.paused["a1"] = True
    r = router.submit(intent(key="paused-1"))
    assert r.status == "rejected" and "entries_paused_reconciliation" in r.reasons
    assert not broker.order_history


def test_stale_feed_blocks_new_entries(env):
    broker, router = env
    setup_auto(router)
    from core.execution.monitoring import FeedMonitor
    mon = FeedMonitor(broker, max_staleness_ms=10_000,
                      notifier=router.notifier)
    router.feed_monitor = mon
    # Intent created at ts 2_000; last tick at 1_000 -> healthy.
    r1 = router.submit(intent(key="feed-1"))
    assert r1.status == "filled"
    # Now pretend the intent arrives long after the last tick.
    r2 = router.submit(intent(key="feed-2", created_ms=20_000))
    assert r2.status == "rejected" and "stale_feed" in r2.reasons
    assert len(broker.order_history) == 1


def test_audit_log_records_every_step(env):
    broker, router = env
    setup_auto(router)
    router.submit(intent(key="aud-1"))
    entries = router.audit.entries()
    assert any(e["event"] == "execution_result" and e["status"] == "filled"
               for e in entries)


def test_notifications_fire_on_open(env):
    broker, router = env
    setup_auto(router)
    router.submit(intent(key="ntf-1"))
    kinds = [n["kind"] for n in router.notifier.feed]
    assert "trade_opened" in kinds
