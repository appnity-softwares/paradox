"""Part 12.4 tests: the MT5 bridge — auth, order_check enforcement, the
client adapter, contract parity with the paper broker, and the
security rules (no token/credential leaks, no unauthenticated mode).
"""
from __future__ import annotations

import pytest

from apps.mt5_bridge import FakeMT5Backend, MT5BridgeAdapter, MT5BridgeError
from apps.mt5_bridge.mt5_backend import MT5Unavailable, RealMT5Backend
from apps.mt5_bridge.server import create_app, run, serve_in_thread
from core.execution import PaperBroker
from core.execution.broker_adapter import BrokerAdapter, assert_conforms
from core.execution.intent import ApprovedOrder

TOKEN = "test-token-9f2c"


@pytest.fixture(scope="module")
def bridge():
    backend = FakeMT5Backend()
    backend.set_quote("XAUUSD", bid=2599.80, ask=2600.20, ts_ms=1_000)
    server, thread = serve_in_thread(port=8731, token=TOKEN, backend=backend)
    yield {"base": "http://127.0.0.1:8731", "backend": backend}
    server.should_exit = True
    thread.join(timeout=5)


def make_approved(key="k1", lot=0.10, side=1, sl=2588.0):
    from core.execution import OrderIntent, make_version
    from core.risk.engine import RiskDecision
    it = OrderIntent(idempotency_key=key, account_id="a1", symbol="XAUUSD",
                     side=side, intended_risk_pct=1.0,
                     strategy_version=make_version(type("S", (), {}), {}),
                     sl_price=sl, tp_price=2624.0, created_ms=1_000)
    return ApprovedOrder(intent=it, lot=lot, price_ref=2600.0,
                         risk_decision=RiskDecision("approved", (), lot, 0,
                                                    0, 0, "v1"),
                         compliance_reasons=(), mode="auto",
                         approval_token=None, approved_ms=1_000)


# ── Auth (fail closed) ───────────────────────────────────────────────────────

def test_requests_without_token_rejected(bridge):
    import requests
    r = requests.get(f"{bridge['base']}/health", timeout=5)
    assert r.status_code == 401


def test_requests_with_wrong_token_rejected(bridge):
    import requests
    r = requests.get(f"{bridge['base']}/health", timeout=5,
                     headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert TOKEN not in r.text          # token never echoed anywhere


def test_bridge_refuses_to_start_without_token():
    with pytest.raises(RuntimeError, match="without a token"):
        create_app(token="")


def test_bridge_refuses_public_exposure_without_tls():
    with pytest.raises(RuntimeError, match="TLS"):
        run(host="0.0.0.0", port=8732, token=TOKEN)


def test_client_refuses_plain_http_off_loopback():
    with pytest.raises(MT5BridgeError, match="TLS"):
        MT5BridgeAdapter("http://10.0.0.5:8001", token=TOKEN)


def test_client_refuses_missing_token():
    with pytest.raises(MT5BridgeError, match="token"):
        MT5BridgeAdapter("http://127.0.0.1:8731", token="")


# ── Server-side order_check enforcement (12.4) ───────────────────────────────

def test_send_without_prior_check_rejected(bridge):
    import requests
    req = {"symbol": "XAUUSD", "type": 0, "volume": 0.10,
           "sl": 2588.0, "comment": "unchecked-key"}
    r = requests.post(f"{bridge['base']}/order_send", json=req, timeout=5,
                      headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 422
    assert "order_check required first" in r.text


def test_send_after_check_succeeds(bridge):
    import requests
    h = {"Authorization": f"Bearer {TOKEN}"}
    req = {"symbol": "XAUUSD", "type": 0, "volume": 0.10,
           "sl": 2588.0, "comment": "checked-key"}
    r1 = requests.post(f"{bridge['base']}/order_check", json=req, timeout=5,
                       headers=h)
    assert r1.status_code == 200
    r2 = requests.post(f"{bridge['base']}/order_send", json=req, timeout=5,
                       headers=h)
    assert r2.status_code == 200
    assert r2.json()["retcode"] == 10009


def test_duplicate_key_rejected_by_backend(bridge):
    import requests
    h = {"Authorization": f"Bearer {TOKEN}"}
    req = {"symbol": "XAUUSD", "type": 0, "volume": 0.10,
           "sl": 2588.0, "comment": "checked-key"}     # already sent above
    r = requests.post(f"{bridge['base']}/order_send", json=req, timeout=5,
                      headers=h)
    assert r.status_code == 422
    assert "duplicate key" in r.text


# ── Client adapter round trip ────────────────────────────────────────────────

def test_adapter_round_trip(bridge):
    adapter = MT5BridgeAdapter(bridge["base"], token=TOKEN)
    adapter.connect()
    assert adapter.health()["backend"] == "fake"
    bid, ask = adapter.quote("XAUUSD")
    assert (bid, ask) == (2599.80, 2600.20)
    spec = adapter.symbol_spec("XAUUSD")
    assert spec["min_lot"] == 0.01 and spec["lot_step"] == 0.01

    approved = make_approved(key="rt-1", lot=0.10)
    check = adapter.order_check(approved)
    assert check["ok"] is True
    sent = adapter.place(approved)
    assert sent["ok"] is True
    assert sent["fill_price"] == pytest.approx(2600.20)   # ask for a buy
    positions = adapter.positions()
    assert positions[-1]["idempotency_key"] == "rt-1"

    # Timeout reconciliation finds the position by idempotency key.
    found = adapter.find_by_idempotency_key("rt-1")
    assert found is not None
    assert int(found["order_ref"]) == positions[-1]["ticket"]
    assert adapter.find_by_idempotency_key("nope") is None

    # Modify SL then close.
    ticket = int(sent["order_ref"])
    adapter.modify_position(str(ticket), sl_price=2595.0)
    assert adapter.positions()[-1]["sl"] == 2595.0
    closed = adapter.close_position(str(ticket), ts_ms=2_000)
    assert closed["pnl"] == pytest.approx((closed["exit_price"] - 2600.20)
                                          * 0.10 * 100)
    assert not [p for p in adapter.positions() if p["ticket"] == ticket]


def test_adapter_rejects_non_approved_objects(bridge):
    adapter = MT5BridgeAdapter(bridge["base"], token=TOKEN)
    with pytest.raises(TypeError):
        adapter.order_check({"symbol": "XAUUSD"})
    with pytest.raises(TypeError):
        adapter.place({"symbol": "XAUUSD"})


def test_rejected_send_reports_ok_false(bridge):
    """Regression: a broker rejection (duplicate key) must surface as
    ok=False on the adapter, never as a silent success."""
    adapter = MT5BridgeAdapter(bridge["base"], token=TOKEN)
    approved = make_approved(key="rej-1")
    assert adapter.order_check(approved)["ok"] is True   # check first
    assert adapter.place(approved)["ok"] is True
    again = adapter.place(make_approved(key="rej-1"))    # no re-check needed
    # ... but the backend still rejects the duplicate key.
    assert again["ok"] is False
    assert "duplicate" in str(again)


def test_real_backend_unavailable_off_windows():
    """On this macOS dev machine the Windows-only package cannot import;
    the failure must be explicit, not silent."""
    with pytest.raises(MT5Unavailable, match="Windows"):
        RealMT5Backend()


# ── BrokerAdapter contract parity (Part 9) ───────────────────────────────────

_CONTRACT_SEQ = iter(range(10_000))


def _contract(adapter: BrokerAdapter, connect: bool = True) -> dict:
    """The same contract exercised against any gateway."""
    assert_conforms(adapter)
    if connect:
        adapter.connect()
    health = adapter.health()
    assert "connected" in health
    spec = adapter.symbol_spec("XAUUSD")
    assert spec["lot_step"] > 0
    bid, ask = adapter.quote("XAUUSD")
    assert 0 < bid <= ask
    approved = make_approved(
        key=f"contract-{type(adapter).__name__}-{next(_CONTRACT_SEQ)}")
    assert adapter.order_check(approved)["ok"]
    sent = adapter.place(approved)
    assert sent["ok"]
    assert adapter.find_by_idempotency_key(approved.intent.idempotency_key)
    return {"fill": sent["fill_price"], "bid": bid, "ask": ask}


def test_paper_broker_satisfies_adapter_contract():
    from core.config import COST_MODELS
    broker = PaperBroker(cost_model=COST_MODELS["XAUUSD_ECN"])
    broker.on_tick("XAUUSD", 2599.80, 2600.20, ts_ms=1_000)
    out = _contract(broker)
    assert out["fill"] == pytest.approx(2600.20 + 0.05)  # slippage applied


def test_mt5_bridge_adapter_satisfies_adapter_contract(bridge):
    adapter = MT5BridgeAdapter(bridge["base"], token=TOKEN)
    out = _contract(adapter)
    assert out["fill"] == pytest.approx(2600.20)         # no synthetic slip


def test_only_gateway_differs_between_paper_and_mt5(bridge):
    """Part 9: same intent, same lot, same checks — the gateways differ
    only in fill mechanics (paper adds slippage; the bridge fills at the
    live ask)."""
    from core.config import COST_MODELS
    paper = PaperBroker(cost_model=COST_MODELS["XAUUSD_ECN"])
    paper.on_tick("XAUUSD", 2599.80, 2600.20, ts_ms=1_000)
    paper_out = _contract(paper)
    bridge_out = _contract(MT5BridgeAdapter(bridge["base"], token=TOKEN))
    assert paper_out["fill"] == pytest.approx(bridge_out["fill"] + 0.05)
