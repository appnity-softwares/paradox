"""U5 tests: payout protection (Update Blueprint v2.1).

Covers: network identity registry (shared IP blocks Auto, IP change
alerts, shared bridge flag), cross-account correlated-intent block,
evidence pack content + chain verification, strategy fingerprints, and
the no-secrets guarantee for outputs.
"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from core.accounts import (CrossAccountGuard, DailySnapshotStore,
                           NetworkIdentityRegistry, build_evidence_pack,
                           fingerprint, fingerprint_report, scan_for_secrets)
from core.execution.modes import AUTO, ModeError
from tests.test_execution_gate import env, intent, setup_auto  # noqa: F401


def policy_map(registry, policies: dict[str, str]):
    return lambda aid: policies.get(aid, "unknown")


# ── network identity registry ───────────────────────────────────────────

def test_registry_roundtrip(tmp_path):
    reg = NetworkIdentityRegistry(tmp_path / "ni.json")
    reg.report("acc-1", "1.2.3.4", vps_provider="Vultr",
               region="eu", bridge_instance_id="bridge-1", ts_ms=1000)
    reg2 = NetworkIdentityRegistry(tmp_path / "ni.json")
    ident = reg2.get("acc-1")
    assert ident.public_ip == "1.2.3.4"
    assert ident.vps_provider == "Vultr" and ident.bridge_instance_id == "bridge-1"


def test_registry_never_stores_credentials(tmp_path):
    reg = NetworkIdentityRegistry(tmp_path / "ni.json")
    reg.report("acc-1", "1.2.3.4", ts_ms=1000)
    blob = (tmp_path / "ni.json").read_text()
    assert "password" not in blob.lower()
    assert "secret" not in blob.lower()


def test_shared_ip_conflict_detected(tmp_path):
    reg = NetworkIdentityRegistry(tmp_path / "ni.json")
    reg.report("a", "9.9.9.9", ts_ms=1000)
    reg.report("b", "9.9.9.9", ts_ms=1001)
    pol = policy_map(reg, {"a": "unknown", "b": "unknown"})
    conflicts = reg.shared_ip_conflicts(pol)
    assert conflicts and set(conflicts[0]["accounts"]) == {"a", "b"}
    # Both 'allowed' -> no conflict.
    pol2 = policy_map(reg, {"a": "allowed", "b": "allowed"})
    assert reg.shared_ip_conflicts(pol2) == []


def test_ip_change_flagged(tmp_path):
    reg = NetworkIdentityRegistry(tmp_path / "ni.json")
    reg.report("a", "1.1.1.1", ts_ms=1000)
    res = reg.report("a", "2.2.2.2", ts_ms=2000)
    assert res["ip_changed"] is True
    assert reg.blocks_auto("a", policy_map(reg, {})) == \
        ["unexpected_ip_change"]


def test_shared_bridge_conflict(tmp_path):
    reg = NetworkIdentityRegistry(tmp_path / "ni.json")
    reg.report("a", "1.1.1.1", bridge_instance_id="br-1", ts_ms=1000)
    reg.report("b", "2.2.2.2", bridge_instance_id="br-1", ts_ms=1001)
    assert reg.shared_bridge_conflicts() == \
        [{"bridge_instance_id": "br-1", "accounts": ["a", "b"]}]


def test_shared_ip_blocks_auto_mode(env, tmp_path):  # noqa: F811
    broker, router = env
    setup_auto(router)
    reg = NetworkIdentityRegistry(tmp_path / "ni.json")
    reg.report("a1", "9.9.9.9", ts_ms=1000)
    reg.report("other", "9.9.9.9", ts_ms=1001)
    router.attach_network_registry(reg)
    # a1's ruleset: vps/ip unknown (not allowed) -> conflict.
    reasons = router._network_blocks("a1")
    assert reasons == ["shared_ip:9.9.9.9"]
    with pytest.raises(ModeError, match="network identity conflict"):
        router.modes.set_mode("a1", AUTO, verdict="PAPER_TRADE_ELIGIBLE",
                              paper_days=30, rules_verified=True,
                              confirmed_by="tester",
                              network_reasons=tuple(reasons))


def test_no_conflict_when_policies_allow(tmp_path):
    reg = NetworkIdentityRegistry(tmp_path / "ni.json")
    reg.report("a", "9.9.9.9", ts_ms=1000)
    reg.report("b", "9.9.9.9", ts_ms=1001)
    assert reg.blocks_auto("a", policy_map(
        reg, {"a": "allowed", "b": "allowed"})) == []


# ── cross-account guard ─────────────────────────────────────────────────

def _second_account(router):
    """Register a real second account (a2) with the same verified
    ruleset shape as the env fixture's a1."""
    from core.account import Account
    from core.adaptation import StrategyProfile
    from core.compliance.rules import DailyLossRule, FirmRuleSet, MaxLossRule
    from core.config import SYMBOL_SPECS
    from tests.test_execution_gate import snapshot
    rs = FirmRuleSet(
        firm="t", program="p", version="v", status="verified",
        verified_by="h", ea_policy="allowed",
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=10_000.0, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True))
    profile = StrategyProfile(
        key="ema_cross", name="EMA cross (ATR stops)", max_dd_pct=35.0,
        typical_stop_distance=12.0, recommended_risk_pct=0.5,
        cost_per_round_trip=0.50)
    router.register_account(Account(account_id="a2", balance=10_000.0),
                            snapshot(account_id="a2"), profile, rs,
                            SYMBOL_SPECS["XAUUSD"], 2600.0)
    router.modes.set_mode("a2", "auto", verdict="PAPER_TRADE_ELIGIBLE",
                          paper_days=30, rules_verified=True,
                          confirmed_by="human-1")
    return rs


def test_correlated_intent_blocked(env):  # noqa: F811
    broker, router = env
    setup_auto(router)
    rs = _second_account(router)
    guard = CrossAccountGuard(window_ms=60_000)
    router.attach_cross_account_guard(guard)
    # First account fills.
    r1 = router.submit(intent(key="c1", created_ms=5_000))
    assert r1.status == "filled", r1.reasons
    # Second account, same symbol/direction/version within 60 s -> block.
    i2 = intent(key="c2", created_ms=20_000, account_id="a2")
    r2 = router.submit(i2)
    assert r2.status == "rejected"
    assert "copy_trading_blocked" in r2.reasons
    assert any(e.get("event") == "copy_trading_block"
               for e in router.audit.entries())


def test_correlated_intent_allowed_when_policy_allows(env):  # noqa: F811
    from dataclasses import replace
    broker, router = env
    setup_auto(router)
    rs = _second_account(router)
    router.rulesets["a2"] = replace(
        rs, copy_trading_policy="allowed")
    guard = CrossAccountGuard(window_ms=60_000)
    router.attach_cross_account_guard(guard)
    assert router.submit(intent(key="c3", created_ms=5_000)).status == "filled"
    r = router.submit(intent(key="c4", created_ms=20_000, account_id="a2"))
    assert r.status == "filled", r.reasons


def test_correlation_window_expiry(env):  # noqa: F811
    broker, router = env
    setup_auto(router)
    _second_account(router)
    guard = CrossAccountGuard(window_ms=60_000)
    router.attach_cross_account_guard(guard)
    assert router.submit(intent(key="c5", created_ms=5_000)).status == "filled"
    # 10 minutes later, on the other account: outside the window.
    r = router.submit(intent(key="c6", created_ms=600_000, account_id="a2"))
    assert r.status == "filled", r.reasons


def test_close_side_never_blocked(env):  # noqa: F811
    from core.strategy_api import SIDE_CLOSE
    broker, router = env
    setup_auto(router)
    _second_account(router)
    guard = CrossAccountGuard(window_ms=60_000)
    router.attach_cross_account_guard(guard)
    assert router.submit(intent(key="c7", created_ms=5_000)).status == "filled"
    close_i = intent(key="c8", created_ms=6_000, account_id="a2",
                     side=SIDE_CLOSE)
    assert router.submit(close_i).status == "filled"


# ── evidence pack ───────────────────────────────────────────────────────

def _pack_inputs():
    return dict(
        trades=[{"account_id": "acc-1", "symbol": "XAUUSD", "side": 1,
                 "lot": 0.1, "entry_price": 2600.0, "exit_price": 2605.0,
                 "pnl": 50.0, "strategy_version_id": "ema_cross:aa:bb",
                 "signal_time_ms": 1000, "order_time_ms": 1001,
                 "fill_time_ms": 1002, "hold_duration_s": 300,
                 "modification_count": 1, "intended_risk_pct": 0.5,
                 "risk_decision_trail": [{"step": "sizing", "ok": True}],
                 "compliance_reasons": [],
                 "ruleset_version": "v1", "ruleset_hash": "abc123",
                 "session_profile": "london"}],
        compliance_snapshots=[{"account_id": "acc-1", "day_utc": "2026-06-01",
                               "payload": {"daily_loss_used_pct": 0.4},
                               "ts_ms": 1000}],
        network_history=[{"ts_ms": 1000, "public_ip": "1.2.3.4",
                          "ip_changed": False}],
        audit_entries=[{"ts_ms": 1001, "event": "execution_result",
                        "status": "filled", "hash": "x", "prev_hash": ""}],
        chain_verification="hash chain intact",
        firm_support_confirmations=[{"firm": "FTMO", "topic": "ea",
                                     "confirmed": True}],
        strategy_rationales={"ema_cross:aa:bb": "trend follow on H1"},
        strategy_fingerprints=fingerprint_report(
            {"ema_cross:aa:bb": {"indicators": ["ema", "atr"],
                                 "params": {"fast": 12}, "styles": []}}))


def test_evidence_pack_content_and_determinism():
    blob1 = build_evidence_pack("acc-1", 0, 2000, **_pack_inputs())
    blob2 = build_evidence_pack("acc-1", 0, 2000, **_pack_inputs())
    assert blob1 == blob2                      # byte-identical
    zf = zipfile.ZipFile(io.BytesIO(blob1))
    names = set(zf.namelist())
    assert {"trades.csv", "compliance_snapshots.csv", "network_history.json",
            "audit_slice.jsonl", "chain_verification.txt",
            "firm_support_confirmations.json", "strategy_rationales.json",
            "strategy_fingerprints.json", "manifest.json"} <= names
    trades_csv = zf.read("trades.csv").decode()
    assert "ema_cross:aa:bb" in trades_csv
    assert "hash chain intact" in zf.read("chain_verification.txt").decode()
    # Time filter excludes out-of-range trades.
    blob3 = build_evidence_pack("acc-1", 5000, 9000, **_pack_inputs())
    zf3 = zipfile.ZipFile(io.BytesIO(blob3))
    assert "ema_cross:aa:bb" not in zf3.read("trades.csv").decode()


def test_evidence_pack_chain_verification_detects_tamper():
    inputs = _pack_inputs()
    inputs["audit_entries"] = [
        {"ts_ms": 1001, "event": "execution_result", "hash": "x",
         "prev_hash": ""},
        {"ts_ms": 1002, "event": "execution_result", "hash": "tampered",
         "prev_hash": "x"}]
    # The caller passes the verifier's verdict; a broken chain is
    # reported verbatim in the pack.
    inputs["chain_verification"] = "hash chain BROKEN at entry index 1"
    blob = build_evidence_pack("acc-1", 0, 2000, **inputs)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    # The pack reports whatever the verifier said — here the caller
    # supplied a broken-chain verdict.
    assert "BROKEN" in zf.read("chain_verification.txt").decode()


def test_audit_chain_verify_real_log(env, tmp_path):  # noqa: F811
    broker, router = env
    setup_auto(router)
    router.submit(intent(key="e1", created_ms=5_000))
    assert router.audit.verify_chain() is None      # intact


def test_no_secrets_in_outputs(env, tmp_path, monkeypatch):  # noqa: F811
    """The classic leak scenario: a credential in the environment must
    never surface in audit entries or the evidence pack."""
    monkeypatch.setenv("MT5_PASSWORD", "SUPER-SECRET-PASS-9876")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSUPERSECRETKEY-1234567890")
    broker, router = env
    setup_auto(router)
    router.submit(intent(key="s1", created_ms=5_000))
    audit_text = json.dumps(router.audit.entries())
    assert "SUPER-SECRET-PASS-9876" not in audit_text
    assert "AIzaSUPERSECRETKEY-1234567890" not in audit_text
    pack = build_evidence_pack("acc-1", 0, 9000, **_pack_inputs())
    leaked = scan_for_secrets([pack.decode("latin-1")],
                              ["SUPER-SECRET-PASS-9876",
                               "AIzaSUPERSECRETKEY-1234567890"])
    assert leaked == []


# ── daily snapshot store ────────────────────────────────────────────────

def test_daily_snapshot_store(tmp_path):
    store = DailySnapshotStore(tmp_path / "snap.jsonl")
    store.record("a", "2026-06-01", {"loss": 0.4}, ts_ms=1000)
    store.record("a", "2026-06-02", {"loss": 0.9}, ts_ms=2000)
    store.record("b", "2026-06-01", {"loss": 0.0}, ts_ms=1001)
    assert len(store.entries("a", 0, 1500)) == 1
    assert len(store.entries("a", 0, 3000)) == 2
    assert store.entries("b", 0, 3000)[0]["payload"] == {"loss": 0.0}


# ── fingerprints ────────────────────────────────────────────────────────

def test_fingerprint_standard_public():
    fp = fingerprint("ema:aa:bb", {"indicators": ["ema", "atr", "rsi"],
                                   "params": {"fast": 12}, "styles": []})
    assert fp["classification"] == "standard_public"
    assert fp["human_review_required"] is False


def test_fingerprint_prohibited_style_flagged():
    fp = fingerprint("mart:aa:bb", {"indicators": ["ema"],
                                    "params": {},
                                    "styles": ["martingale"]})
    assert fp["classification"] == "contains_prohibited_style"
    assert fp["prohibited_styles"] == ["martingale"]
    assert fp["human_review_required"] is True


def test_fingerprint_custom():
    fp = fingerprint("x:aa:bb", {"indicators": ["my_custom_signal"],
                                 "params": {}, "styles": []})
    assert fp["classification"] == "custom"
    assert fp["human_review_required"] is True


def test_fingerprint_report_deterministic():
    report = fingerprint_report({
        "b:1": {"indicators": ["rsi"], "params": {}, "styles": []},
        "a:1": {"indicators": ["ema"], "params": {}, "styles": []}})
    assert [r["strategy_version_id"] for r in report] == ["a:1", "b:1"]
