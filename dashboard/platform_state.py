"""Platform state for the Part 14.1 dashboard: wires the Phase 3-5
engines (risk, compliance, execution, AI) into one place, independent of
the legacy prototype state. Paper mode only until an explicit Phase 4
sign-off is recorded; live controls stay disabled until then.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

PROTOTYPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROTOTYPE_ROOT)

from core.account import Account  # noqa: E402
from core.adaptation import StrategyProfile  # noqa: E402
from core.compliance import ComplianceEngine, FirmRuleSet, RuleStore  # noqa: E402
from core.compliance.rules import DailyLossRule, MaxLossRule  # noqa: E402
from core.config import COST_MODELS, SYMBOL_SPECS  # noqa: E402
from core.execution import (  # noqa: E402
    AuditLog,
    ExecutionRouter,
    ModeGate,
    Notifier,
    OrderIntent,
    PaperBroker,
)
from core.execution.monitoring import FeedMonitor, Reconciler  # noqa: E402
from core.accounts import (CrossAccountGuard, NetworkIdentityRegistry,
                           DailySnapshotStore, build_evidence_pack,
                           fingerprint_report)
from core.monitoring import DriftAlertStore, DriftMonitor
from core.session import SessionProfile, SessionRunStore, SessionScheduler  # noqa: E402
from core.risk import AccountSnapshot, DeterministicRiskEngine, KillSwitch, \
    RiskConfig  # noqa: E402

STATE_DIR = Path(os.environ.get(
    "DSH_STATE_DIR",
    str(Path(PROTOTYPE_ROOT) / "state" / "dashboard")))
PRICE = 2600.0


class PlatformState:
    """Singleton wiring for the console. Thread-safe-ish (GIL, simple ops)."""

    def __init__(self, state_dir: Path = STATE_DIR):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.broker = PaperBroker(cost_model=COST_MODELS["XAUUSD_ECN"])
        self.broker.connect()
        self.broker.on_tick("XAUUSD", bid=PRICE - 0.20, ask=PRICE + 0.20,
                            ts_ms=1_000)

        self.mode_gate = ModeGate()
        self.kill_switch = KillSwitch(self.state_dir / "kill_switch.json")
        self.risk_engine = DeterministicRiskEngine(
            RiskConfig(version="v1"))
        self.risk_engine.kill_switch = self.kill_switch
        self.compliance_engine = ComplianceEngine()
        self.audit = AuditLog(self.state_dir / "audit.jsonl")
        self.notifier = Notifier()
        self.router = ExecutionRouter(
            self.broker, self.risk_engine, self.compliance_engine,
            self.mode_gate, self.audit, self.notifier)
        self.reconciler = Reconciler(self.broker, self.audit, self.notifier)
        self.feed_monitor = FeedMonitor(self.broker, notifier=self.notifier,
                                        audit_log=self.audit)
        self.router.reconciler = self.reconciler
        self.router.feed_monitor = self.feed_monitor

        self.rule_store = RuleStore(self.state_dir / "rules")
        self.accounts: dict[str, Account] = {}
        self.snapshots: dict[str, AccountSnapshot] = {}
        self.profiles: dict[str, StrategyProfile] = {}
        self.rulesets: dict[str, FirmRuleSet | None] = {}

        # False until a real bridge feed is wired; paper mode tops up
        # its synthetic quote locally (labelled SYNTHETIC in the UI).
        self.live_feed_connected = False

        # Phase 4 sign-off: absent -> every live control is disabled.
        self._signoff_path = self.state_dir / "phase4_signoff.json"
        self._lock = threading.Lock()

        # ── U2: session profiles, scheduler and run ledger ──────────────
        self.session_profiles: dict[str, SessionProfile] = {}
        self._session_profiles_path = self.state_dir / "session_profiles.json"
        self._load_session_profiles()
        self.session_runs = SessionRunStore(self.state_dir / "session_runs.jsonl")
        self.scheduler = SessionScheduler(self.router, notifier=self.notifier,
                                          run_store=self.session_runs)

        # ── U5: payout protection ────────────────────────────────────────
        self.network_registry = NetworkIdentityRegistry(
            self.state_dir / "network_identity.json")
        self.router.attach_network_registry(self.network_registry)
        self.cross_guard = CrossAccountGuard()
        self.router.attach_cross_account_guard(self.cross_guard)
        self.compliance_snapshots = DailySnapshotStore(
            self.state_dir / "compliance_snapshots.jsonl")

        # ── U6: drift monitor -> auto-pause ──────────────────────────────
        self.drift_alerts = DriftAlertStore(
            self.state_dir / "drift_alerts.jsonl")
        self.drift_monitor = DriftMonitor(
            self.mode_gate, self.audit, self.notifier, self.drift_alerts)

    # ── accounts ─────────────────────────────────────────────────────────
    def register_account(self, account_id: str, balance: float,
                         ruleset: FirmRuleSet | None = None) -> None:
        account = Account(account_id=account_id, balance=balance, leverage=100)
        snapshot = AccountSnapshot(
            account_id=account_id, equity=balance, balance=balance,
            used_margin=0.0, open_positions=0, exposure_lots_by_symbol={},
            daily_realized_pnl=0.0, daily_floating_pnl=0.0,
            equity_high_watermark=balance,
            last_tick_ms={"XAUUSD": 1_000}, spread={"XAUUSD": 0.20},
            server_time_ms=1_000)
        profile = StrategyProfile(
            key="ema_cross", name="EMA cross (ATR stops)", max_dd_pct=35.0,
            typical_stop_distance=12.0, recommended_risk_pct=0.5,
            cost_per_round_trip=0.50)
        self.accounts[account_id] = account
        self.snapshots[account_id] = snapshot
        self.profiles[account_id] = profile
        self.rulesets[account_id] = ruleset
        self.router.register_account(account, snapshot, profile, ruleset,
                                     SYMBOL_SPECS["XAUUSD"], PRICE)
        # Paper mode by default; research until an operator changes it.
        if self.mode_gate.mode_of(account_id) == "research":
            self.mode_gate.set_mode(account_id, "paper")

    # ── sign-off gate ────────────────────────────────────────────────────
    @property
    def live_signoff(self) -> dict | None:
        if not self._signoff_path.exists():
            return None
        try:
            return json.loads(self._signoff_path.read_text())
        except (OSError, ValueError):
            return None

    def record_signoff(self, actor: str, notes: str) -> dict:
        record = {"actor": actor, "notes": notes,
                  "phase": "4A paper trading accepted"}
        self._signoff_path.write_text(json.dumps(record))
        self.audit.append({"event": "phase4_signoff", "actor": actor,
                           "notes": notes})
        return record

    # ── U2: session profiles ─────────────────────────────────────────────
    def _load_session_profiles(self) -> None:
        if not self._session_profiles_path.exists():
            return
        try:
            data = json.loads(self._session_profiles_path.read_text())
            for name, raw in data.items():
                self.session_profiles[name] = SessionProfile.from_dict(raw)
        except (OSError, ValueError, KeyError):
            self.session_profiles = {}

    def _save_session_profiles(self) -> None:
        payload = {name: p.to_dict()
                   for name, p in self.session_profiles.items()}
        self._session_profiles_path.write_text(json.dumps(payload, indent=1))

    def upsert_session_profile(self, profile: SessionProfile) -> dict:
        self.session_profiles[profile.name] = profile
        self._save_session_profiles()
        self.audit.append({"event": "session_profile_upserted",
                           "name": profile.name})
        return profile.to_dict()

    def bind_session_profile(self, account_id: str, name: str) -> dict:
        if name not in self.session_profiles:
            raise KeyError(f"unknown session profile {name!r}")
        if account_id not in self.router.accounts:
            raise KeyError(f"unknown account {account_id!r}")
        self.router.set_session_profile(account_id,
                                        self.session_profiles[name])
        self.audit.append({"event": "session_profile_bound",
                           "account_id": account_id, "profile": name})
        return {"account_id": account_id, "profile": name,
                "window_open": False}

    def unbind_session_profile(self, account_id: str) -> dict:
        self.router.clear_session(account_id)
        self.audit.append({"event": "session_profile_unbound",
                           "account_id": account_id})
        return {"account_id": account_id, "profile": None}

    def session_status(self, account_id: str) -> dict:
        """Status for the dashboard card: waiting/active/closed/skipped."""
        profile = self.router.sessions.get(account_id)
        win = self.router.session_windows.get(account_id)
        runs = [r for r in self.session_runs.entries()
                if r.get("account_id") == account_id]
        last = runs[-1] if runs else None
        if profile is None:
            state = "unbound"
        elif win is None:
            state = "waiting"
        else:
            now_ms = int(__import__("time").time() * 1000)
            if now_ms >= win["flatten_ms"]:
                state = "closed"
            else:
                state = "active"
        return {"account_id": account_id,
                "profile": profile.name if profile else None,
                "status": state,
                "window": win if win else None,
                "last_run": last}

    # ── U5: payout protection ────────────────────────────────────────────
    def _policy_of(self, account_id: str) -> str:
        rs = self.rulesets.get(account_id)
        if rs is None:
            return "unknown"
        if rs.vps_policy == "allowed" and rs.ip_policy == "allowed":
            return "allowed"
        return "not_allowed"

    def network_status(self) -> dict:
        return {"accounts": [a.to_dict() for a in
                             self.network_registry.entries()],
                "shared_ip_conflicts": self.network_registry.shared_ip_conflicts(
                    self._policy_of),
                "shared_bridge_conflicts":
                    self.network_registry.shared_bridge_conflicts()}

    def payout_evidence_pack(self, account_id: str, from_ms: int,
                             to_ms: int) -> bytes:
        """Deterministic ZIP; no LLM. Raises KeyError for unknown
        accounts."""
        if account_id not in self.accounts:
            raise KeyError(account_id)
        router = self.router
        chain = router.audit.verify_chain()
        chain_txt = ("hash chain intact" if chain is None
                     else f"hash chain BROKEN at entry index {chain}")
        trades = []
        for rec in getattr(self.broker, "order_history", []):
            appr = rec.approved
            if getattr(appr, "account_id", None) != account_id:
                continue
            intent = getattr(appr, "intent", None)
            sv = getattr(intent, "strategy_version", None)
            rs = self.rulesets.get(account_id)
            trades.append({
                "account_id": account_id,
                "symbol": getattr(intent, "symbol", "XAUUSD"),
                "side": getattr(intent, "side", ""),
                "lot": getattr(appr, "lot", ""),
                "entry_price": rec.fill_price,
                "exit_price": "",
                "pnl": "",
                "strategy_version_id": getattr(sv, "version_id", ""),
                "signal_time_ms": getattr(intent, "created_ms", ""),
                "order_time_ms": getattr(appr, "approved_ms", ""),
                "fill_time_ms": rec.fill_ms,
                "hold_duration_s": "",
                "modification_count": "",
                "intended_risk_pct":
                    getattr(intent, "intended_risk_pct", ""),
                "risk_decision_trail": list(
                    getattr(appr, "decision_trail", ()) or ()),
                "compliance_reasons": list(
                    getattr(appr, "compliance_reasons", ()) or ()),
                "ruleset_version": getattr(rs, "version", ""),
                "ruleset_hash": rs.content_hash() if rs is not None else "",
                "session_profile":
                    getattr(self.router.sessions.get(account_id),
                            "name", ""),
            })
        def _read_json(name, default):
            f = self.state_dir / name
            if f.exists():
                try:
                    return json.loads(f.read_text())
                except (OSError, ValueError):
                    return default
            return default
        specs = _read_json("strategy_specs.json", {})
        return build_evidence_pack(
            account_id, from_ms, to_ms,
            trades=trades,
            compliance_snapshots=self.compliance_snapshots.entries(
                account_id, from_ms, to_ms),
            network_history=self.network_registry.history(account_id),
            audit_entries=router.audit.entries(),
            chain_verification=chain_txt,
            firm_support_confirmations=_read_json(
                "firm_support_confirmations.json", []),
            strategy_rationales=_read_json("strategy_rationales.json", {}),
            strategy_fingerprints=fingerprint_report(specs))

    # ── data-source labels (Part 14.1: honest labels only) ───────────────
    def data_sources(self) -> list[dict]:
        return [
            {"source": "Dukascopy tick store (parquet)",
             "label": "DELAYED",
             "detail": "Real ingested ticks; historical by definition — "
                       "never labelled live"},
            {"source": "Paper broker quotes",
             "label": "SYNTHETIC",
             "detail": "Simulated fills for the paper gateway; explicitly "
                       "labelled synthetic"},
            {"source": "MT5 bridge feed",
             "label": "STALE",
             "detail": "No live terminal connected in this deployment; "
                       "last tick age exceeds the freshness window"},
        ]

    # ── live snapshot refresh ────────────────────────────────────────────
    def _refresh_snapshot(self, account_id: str, now_ms: int) -> None:
        """Rebuild the account snapshot from live broker state before
        every submit. Without this the risk engine and the compliance
        daily-loss rule would act on the frozen registration values —
        a real bug the replay soak exposed."""
        broker = self.broker
        account = self.accounts[account_id]
        old = self.snapshots[account_id]
        # Paper mode: the synthetic quote IS the feed — keep it fresh on
        # the real clock so the feed-staleness check stays meaningful.
        # (Labelled SYNTHETIC everywhere it is shown.) When a real bridge
        # feed is wired, live_feed_connected flips and this stops.
        if not self.live_feed_connected:
            broker.on_tick("XAUUSD", bid=PRICE - 0.20, ask=PRICE + 0.20,
                           ts_ms=now_ms)
        import datetime as _dt
        now_dt = _dt.datetime.fromtimestamp(now_ms / 1000,
                                            tz=_dt.timezone.utc)
        reset_today = now_dt.replace(hour=17, minute=0, second=0,
                                     microsecond=0)
        boundary_ms = int((reset_today if now_dt >= reset_today
                           else reset_today - _dt.timedelta(days=1)
                           ).timestamp() * 1000)
        realized = sum(float(c["pnl"]) for c in broker.closed
                       if c.get("account_id") == account_id
                       and c.get("ts_ms", 0) >= boundary_ms)
        balance = account.balance + sum(
            float(c["pnl"]) for c in broker.closed
            if c.get("account_id") == account_id)
        positions = broker.positions(account_id)
        floating = 0.0
        exposure: dict[str, float] = {}
        bid, ask = broker.quotes.get("XAUUSD", (0.0, 0.0))
        for pos in positions:
            if pos.get("symbol") != "XAUUSD":
                continue
            mark = ask if pos["side"] == 1 else bid
            if mark:
                floating += (mark - pos["entry_price"]) * pos["side"] \
                    * pos["lot"] * 100.0
            exposure["XAUUSD"] = exposure.get("XAUUSD", 0.0) + pos["lot"]
        equity = balance + floating
        spread = round(ask - bid, 5) if bid and ask else 0.20
        self.snapshots[account_id] = AccountSnapshot(
            account_id=account_id, equity=equity, balance=balance,
            used_margin=0.0, open_positions=len(positions),
            exposure_lots_by_symbol=exposure,
            daily_realized_pnl=realized, daily_floating_pnl=floating,
            equity_high_watermark=max(old.equity_high_watermark, equity),
            last_tick_ms=dict(broker.last_tick_ms),
            spread={"XAUUSD": spread}, server_time_ms=now_ms)
        # The router keeps its own snapshot dict — update BOTH or the
        # pipeline still sees the frozen registration values.
        self.router.snapshots[account_id] = self.snapshots[account_id]
        self.router.prices[account_id] = (bid + ask) / 2.0 if bid else PRICE

    # ── intent creation from the UI ──────────────────────────────────────
    def submit_ticket(self, account_id: str, symbol: str, side: str,
                      intended_risk_pct: float, sl_price: float,
                      tp_price: float | None, idempotency_key: str,
                      approval_token: str | None = None,
                      order_type: str = "market",
                      limit_price: float | None = None) -> dict:
        with self._lock:
            if account_id not in self.accounts:
                # Fail closed but gracefully: an unknown account is an
                # operator error, not a server fault.
                return {"status": "rejected", "reasons": ["unknown_account"],
                        "order_ref": None, "fill_price": None, "lot": 0.0,
                        "steps": []}
            now_ms = int(time.time() * 1000)
            self._refresh_snapshot(account_id, now_ms)
            intent = OrderIntent(
                idempotency_key=idempotency_key, account_id=account_id,
                symbol=symbol, side=1 if side == "long" else -1,
                intended_risk_pct=intended_risk_pct,
                strategy_version=__import__(
                    "core.execution.versions",
                    fromlist=["make_version"]).make_version(
                    type("ManualTicket", (), {}), {}),
                sl_price=sl_price, tp_price=tp_price, source="manual",
                created_ms=now_ms, order_type=order_type,
                limit_price=limit_price)
            result = self.router.submit(intent,
                                        approval_token=approval_token)
            if result.status == "filled":
                self.reconciler.track_fill(account_id, idempotency_key,
                                           result.order_ref)
            return {"status": result.status, "reasons": list(result.reasons),
                    "order_ref": result.order_ref,
                    "fill_price": result.fill_price,
                    "lot": result.filled_lot,
                    "steps": [dict(s) for s in result.steps]}


PLATFORM = PlatformState()
