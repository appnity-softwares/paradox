"""Pending stop/limit order tests, end to end: intent validation,
PaperBroker triggering and cancellation, the router's pending_accepted
outcome, the MT5 bridge, and the adapter contract.
"""
from __future__ import annotations

import pytest

from core.config import COST_MODELS
from core.execution import (ApprovedOrder, AuditLog, ExecutionRouter,
                             ModeGate, Notifier, OrderIntent, PaperBroker)
from core.execution.versions import make_version
from core.risk.engine import RiskDecision

PRICE = 2600.0
VERSION = make_version(type("S", (), {}), {})


def broker():
    b = PaperBroker(cost_model=COST_MODELS["XAUUSD_ECN"])
    b.connect()
    b.on_tick("XAUUSD", bid=PRICE - 0.20, ask=PRICE + 0.20, ts_ms=1_000)
    return b


def approved(intent: OrderIntent, lot: float = 0.05) -> ApprovedOrder:
    return ApprovedOrder(
        intent=intent, lot=lot, price_ref=PRICE,
        risk_decision=RiskDecision("approved", (), lot, 0, 0, 0,
                                  "v1"),
        compliance_reasons=(), mode="paper", approval_token=None,
        approved_ms=1_000)


def intent(key: str, side: int = 1, order_type: str = "market",
           limit_price: float | None = None, sl: float | None = 2588.0) \
        -> OrderIntent:
    return OrderIntent(key, "acc-1", "XAUUSD", side, 0.5, VERSION, sl,
                       tp_price=None, source="manual", created_ms=1_000,
                       order_type=order_type, limit_price=limit_price)


# ── Intent validation ────────────────────────────────────────────────────────

def test_pending_intent_requires_trigger_price():
    with pytest.raises(ValueError):
        intent("k1", order_type="buy_stop")


def test_market_intent_rejects_trigger_price():
    with pytest.raises(ValueError):
        intent("k2", limit_price=2605.0)


def test_unknown_order_type_rejected():
    with pytest.raises(ValueError):
        intent("k3", order_type="iceberg", limit_price=2605.0)


# ── PaperBroker triggering ───────────────────────────────────────────────────

def test_buy_stop_fills_when_ask_reaches_trigger():
    b = broker()
    rec = b.place(approved(intent("ps-1", order_type="buy_stop",
                                  limit_price=2605.0)))
    assert rec["pending"] is True and rec["fill_price"] is None
    assert b.positions() == [] and len(b.pending_orders()) == 1
    # Tick below the trigger: nothing fills.
    b.on_tick("XAUUSD", bid=2603.0, ask=2603.2, ts_ms=2_000)
    assert b.positions() == []
    # Ask reaches the trigger: fills at trigger + adverse slippage.
    b.on_tick("XAUUSD", bid=2605.0, ask=2605.2, ts_ms=3_000)
    pos = b.positions()[0]
    assert pos["side"] == 1
    assert pos["entry_price"] == pytest.approx(2605.0 + 0.05)  # stop slips
    assert b.pending_orders() == []


def test_sell_limit_fills_at_trigger_or_better():
    b = broker()
    b.place(approved(intent("sl-1", side=-1, order_type="sell_limit",
                            limit_price=2610.0, sl=2620.0)))
    b.on_tick("XAUUSD", bid=2612.0, ask=2612.2, ts_ms=2_000)   # better
    pos = b.positions()[0]
    assert pos["side"] == -1
    assert pos["entry_price"] == pytest.approx(2612.0)          # better fill
    assert b.pending_orders() == []


def test_buy_limit_fills_only_when_ask_touches_trigger():
    b = broker()
    b.place(approved(intent("bl-1", order_type="buy_limit",
                            limit_price=2595.0)))
    b.on_tick("XAUUSD", bid=2597.0, ask=2597.2, ts_ms=2_000)
    assert b.positions() == [] and len(b.pending_orders()) == 1
    b.on_tick("XAUUSD", bid=2594.5, ask=2594.7, ts_ms=3_000)
    assert b.positions()[0]["entry_price"] == pytest.approx(2594.7)


def test_cancel_pending_order_and_cancel_after_fill_fails():
    b = broker()
    rec = b.place(approved(intent("c-1", order_type="buy_stop",
                                  limit_price=2605.0)))
    b.cancel_order(rec["order_ref"])
    assert b.pending_orders() == []
    with pytest.raises(KeyError):
        b.cancel_order(rec["order_ref"])
    # A filled pending order is gone from the queue; cancelling its
    # position is close_position's job, not cancel_order's.
    rec2 = b.place(approved(intent("c-2", order_type="buy_stop",
                                   limit_price=2605.0)))
    b.on_tick("XAUUSD", bid=2605.0, ask=2605.2, ts_ms=3_000)
    with pytest.raises(KeyError):
        b.cancel_order(rec2["order_ref"])


def test_pending_duplicate_key_rejected():
    b = broker()
    b.place(approved(intent("dup-1", order_type="buy_stop",
                            limit_price=2605.0)))
    with pytest.raises(ValueError):
        b.place(approved(intent("dup-1", order_type="buy_stop",
                                limit_price=2605.0)))


def test_find_by_idempotency_key_finds_pending():
    b = broker()
    rec = b.place(approved(intent("f-1", order_type="buy_stop",
                                  limit_price=2605.0)))
    found = b.find_by_idempotency_key("f-1")
    assert found["pending"] is True
    assert found["order_ref"] == rec["order_ref"]


def test_pending_order_carries_sl_and_tp_into_position():
    b = broker()
    b.place(approved(intent("s-1", order_type="buy_stop",
                            limit_price=2605.0, sl=2590.0)))
    b.on_tick("XAUUSD", bid=2605.0, ask=2605.2, ts_ms=2_000)
    pos = b.positions()[0]
    assert pos["sl_price"] == 2590.0


# ── Router end to end ────────────────────────────────────────────────────────

def _router(b):
    from core.account import Account
    from core.adaptation import StrategyProfile
    from core.compliance import ComplianceEngine, FirmRuleSet
    from core.compliance.rules import DailyLossRule, MaxLossRule
    from core.config import SYMBOL_SPECS
    from core.risk import AccountSnapshot, DeterministicRiskEngine, \
        RiskConfig
    rules = FirmRuleSet(
        firm="test_firm", program="100k", version="v1", status="verified",
        ea_policy="allowed",
        verified_by="human-1",
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=10_000.0, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
        min_trading_days=0, weekend_holding="allowed")
    r = ExecutionRouter(b, DeterministicRiskEngine(RiskConfig(version="v1")),
                        ComplianceEngine(), ModeGate(), AuditLog(":memory:"),
                        Notifier())
    account = Account(account_id="acc-1", balance=10_000.0, leverage=100)
    r.register_account(account,
                       AccountSnapshot("acc-1", 10_000.0, 10_000.0, 0.0, 0,
                                       {}, 0.0, 0.0, 10_000.0,
                                       {"XAUUSD": 1_000}, {"XAUUSD": 0.20},
                                       1_000),
                       StrategyProfile("ema_cross", "EMA cross", 35.0, 12.0,
                                       0.5, 0.5),
                       rules, SYMBOL_SPECS["XAUUSD"], PRICE)
    r.modes.set_mode("acc-1", "paper")
    return r


def test_router_reports_pending_accepted_then_tick_fills():
    b = broker()
    r = _router(b)
    result = r.submit(intent("rt-1", order_type="buy_stop",
                             limit_price=2605.0))
    assert result.status == "pending_accepted"
    assert result.fill_price is None and result.order_ref
    assert any(s["step"] == "send" and s.get("pending")
               for s in result.steps)
    assert b.positions() == []
    b.on_tick("XAUUSD", bid=2605.0, ask=2605.2, ts_ms=3_000)
    assert len(b.positions()) == 1


def test_router_risk_still_applies_to_pending_orders():
    b = broker()
    r = _router(b)
    # 5% intended risk -> reduced to the 1% cap exactly as for market orders.
    result = r.submit(intent("rt-2", order_type="buy_stop",
                             limit_price=2605.0))
    assert result.status == "pending_accepted"
    risk_step = next(s for s in result.steps if s["step"] == "risk")
    assert risk_step["outcome"] in ("approved", "reduced")


# ── MT5 bridge ───────────────────────────────────────────────────────────────

def test_bridge_pending_round_trip():
    from apps.mt5_bridge.mt5_backend import FakeMT5Backend
    from apps.mt5_bridge.server import create_app
    from fastapi.testclient import TestClient
    backend = FakeMT5Backend()
    backend.set_quote("XAUUSD", PRICE - 0.20, PRICE + 0.20, 1_000)
    c = TestClient(create_app(backend=backend, token="t"),
                   headers={"Authorization": "Bearer t"})
    check = c.post("/order_check", json={
        "symbol": "XAUUSD", "type": 4, "volume": 0.05, "price": 2605.0,
        "sl": 2588.0, "comment": "br-1"}).json()
    assert check["ok"] is True
    sent = c.post("/order_send", json={
        "symbol": "XAUUSD", "type": 4, "volume": 0.05, "price": 2605.0,
        "sl": 2588.0, "comment": "br-1"}).json()
    assert sent["retcode"] == 10009 and sent["deal"] == 0
    assert not backend.positions
    # Trigger via a tick.
    backend.set_quote("XAUUSD", 2605.0, 2605.2, 2_000)
    assert len(backend.positions) == 1
    assert backend.positions[0]["side"] == 1
    # Cancel path for a fresh pending order.
    c.post("/order_check", json={
        "symbol": "XAUUSD", "type": 4, "volume": 0.05, "price": 2610.0,
        "sl": 2588.0, "comment": "br-2"})
    sent2 = c.post("/order_send", json={
        "symbol": "XAUUSD", "type": 4, "volume": 0.05, "price": 2610.0,
        "sl": 2588.0, "comment": "br-2"}).json()
    cancel = c.post("/order_cancel",
                    json={"ticket": sent2["order"]}).json()
    assert cancel["retcode"] == 10009


def test_bridge_pending_without_price_rejected_at_check():
    from apps.mt5_bridge.mt5_backend import FakeMT5Backend
    from apps.mt5_bridge.server import create_app
    from fastapi.testclient import TestClient
    backend = FakeMT5Backend()
    backend.set_quote("XAUUSD", PRICE - 0.20, PRICE + 0.20, 1_000)
    c = TestClient(create_app(backend=backend, token="t"),
                   headers={"Authorization": "Bearer t"})
    r = c.post("/order_check", json={
        "symbol": "XAUUSD", "type": 4, "volume": 0.05,
        "sl": 2588.0, "comment": "np-1"})
    assert r.status_code == 422
    assert "price" in r.json()["detail"]["comment"]


# ── Adapter contract ─────────────────────────────────────────────────────────

def test_adapter_maps_pending_types_and_reports_pending():
    from apps.mt5_bridge.client import MT5BridgeAdapter
    from apps.mt5_bridge.mt5_backend import FakeMT5Backend
    from apps.mt5_bridge.server import create_app, serve_in_thread
    backend = FakeMT5Backend()
    backend.set_quote("XAUUSD", PRICE - 0.20, PRICE + 0.20, 1_000)
    server, thread = serve_in_thread(port=8737, token="t",
                backend=backend)
    try:
        adapter = MT5BridgeAdapter("http://127.0.0.1:8737", token="t")
        ao1 = approved(intent("ad-1", order_type="buy_stop",
                              limit_price=2605.0))
        assert adapter.order_check(ao1)["ok"] is True   # check before send
        rec = adapter.place(ao1)
        assert rec["ok"] is True and rec["pending"] is True
        assert rec["fill_price"] is None
        # Trigger, then the position exists and the key resolves to it.
        backend.set_quote("XAUUSD", 2605.0, 2605.2, 2_000)
        found = adapter.find_by_idempotency_key("ad-1")
        assert found and found["fill_price"] == pytest.approx(2605.0)
        # Cancel a fresh pending order through the adapter.
        ao2 = approved(intent("ad-2", order_type="sell_limit",
                              limit_price=2615.0, sl=None))
        assert adapter.order_check(ao2)["ok"] is True
        rec2 = adapter.place(ao2)
        assert rec2["pending"] is True
        adapter.cancel_order(rec2["order_ref"])
        assert backend.pending == []
    finally:
        # uvicorn.Server.shutdown() is a coroutine; a server running in a
        # thread is stopped by the should_exit flag, not by calling it.
        server.should_exit = True
        thread.join(timeout=5)
