"""Execution router (Part 12.1): the SINGLE order path.

    intent -> sizing -> risk -> compliance -> mode gate -> broker
             order_check -> send -> fill handling -> audit + notify

The broker adapter accepts only an `ApprovedOrder`, which only this
router can mint — an order that skips a step is impossible by
construction. Idempotency: retries with the same key return the recorded
outcome and never place a second order. A send timeout is reconciled
against the broker BEFORE any retry (Part 12.2).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.account import Account
from core.adaptation import StrategyProfile, adapt_config
from core.compliance import AccountState, ComplianceEngine, FirmRuleSet
from core.compliance.behaviour import BehaviourControls, BehaviourConfig
from core.compliance.engine import ProposedTrade
from core.config import SymbolSpec
from core.execution.intent import (
    ApprovedOrder,
    ExecutionResult,
    OrderIntent,
)
from core.execution.modes import AUTO, ModeGate
from core.strategy_api import SIDE_CLOSE
from typing import Any
from core.risk import AccountSnapshot, DeterministicRiskEngine
from core.risk.engine import OrderProposal
from core.sizing import size_fixed_risk


class AuditLog:
    """Append-only audit trail for every execution decision.

    Two modes:
    - ``path == ":memory:"`` keeps events in a process-local list (no file).
    - any other path is an append-only JSONL file.

    File mode is tamper-evident (U0): every entry written by this class
    carries ``prev_hash`` and ``hash`` where
    ``hash = sha256(prev_hash + canonical_json(event_without_hash_fields))``.
    Lines written before chaining was introduced (no ``hash`` field) are a
    legacy segment: they are returned by :meth:`entries` but cannot be
    verified; the chain starts fresh on the first chained write.
    """

    GENESIS_PREV_HASH = ""   # prev_hash of the first chained entry

    def __init__(self, path: str | Path):
        if str(path) == ":memory:":
            self.path: Path | None = None
            self._mem: list[dict] = []
            self._prev_hash: str | None = self.GENESIS_PREV_HASH
            return
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # None (no chained entries yet) resolves to the genesis prev so
        # the very first write starts the chain.
        self._prev_hash: str | None = \
            self._last_chain_hash() or self.GENESIS_PREV_HASH

    # ── writing ──────────────────────────────────────────────────────────
    def append(self, event: dict) -> None:
        event = {"ts_ms": int(time.time() * 1000), **event}
        if self.path is None:
            self._mem.append(event)
            return
        record = dict(event)
        # Chain every write; a pre-existing unchained (legacy) segment is
        # left untouched and the chain starts fresh here at genesis.
        record["prev_hash"] = self._prev_hash
        record["hash"] = self._entry_hash(record)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._prev_hash = record["hash"]

    @staticmethod
    def _entry_hash(record: dict) -> str:
        import hashlib
        body = {k: v for k, v in record.items()
                if k not in ("prev_hash", "hash")}
        canonical = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(
            (record["prev_hash"] + canonical).encode()).hexdigest()

    def _last_chain_hash(self) -> str | None:
        """Hash of the last chained entry already on disk, else None
        (nothing chained yet — the next write starts the chain)."""
        if not self.path.exists():
            return None
        last_hash = None
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "hash" in rec:
                last_hash = rec["hash"]
        return last_hash

    # ── reading / verification ───────────────────────────────────────────
    def entries(self) -> list[dict]:
        if self.path is None:
            return list(self._mem)
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def verify_chain(self) -> int | None:
        """Verify the hash chain. Returns the index of the FIRST broken
        entry, or None when the chain is intact. Legacy (unchained) lines
        before the first chained entry are skipped: they predate chaining
        and carry no verifiable commitment."""
        records = self.entries()
        chain_started = False
        prev = self.GENESIS_PREV_HASH
        for i, rec in enumerate(records):
            if "hash" not in rec:
                if chain_started:
                    return i   # chained segment followed by an unchained line
                continue       # legacy segment
            if rec.get("prev_hash") != prev:
                return i
            if self._entry_hash(rec) != rec["hash"]:
                return i
            prev = rec["hash"]
            chain_started = True
        return None


class Notifier:
    """In-app notification sink (Part 12.6); extra sinks can be attached."""

    def __init__(self):
        self.feed: list[dict] = []
        self._sinks = [self._in_app]

    def _in_app(self, event: dict) -> None:
        self.feed.append(event)

    def notify(self, kind: str, message: str, **details) -> None:
        for sink in self._sinks:
            sink({"kind": kind, "message": message, **details})

    def add_sink(self, fn) -> None:
        self._sinks.append(fn)


class ExecutionRouter:
    def __init__(self, broker, risk_engine: DeterministicRiskEngine,
                 compliance_engine: ComplianceEngine, mode_gate: ModeGate,
                 audit_log: AuditLog, notifier: Notifier | None = None,
                 behaviour: BehaviourControls | None = None):
        self.broker = broker
        self.risk = risk_engine
        self.compliance = compliance_engine
        self.modes = mode_gate
        self.audit = audit_log
        self.notifier = notifier or Notifier()
        # U1: deterministic behaviour controls (min hold, caps, prohibited
        # styles, hedging). Optional for backward compatibility.
        self.behaviour = behaviour or BehaviourControls()
        self._results: dict[str, ExecutionResult] = {}   # idempotency store
        # Per-account context used by the pipeline.
        self.accounts: dict[str, Account] = {}
        self.snapshots: dict[str, AccountSnapshot] = {}
        self.profiles: dict[str, StrategyProfile] = {}
        self.rulesets: dict[str, FirmRuleSet | None] = {}
        self.specs: dict[str, SymbolSpec] = {}
        self.prices: dict[str, float] = {}
        self._calendar: dict[int, str] | None = None
        # U2: session gating. A profile bound to an account plus the
        # window opened by the SessionScheduler. No window open ->
        # entries are refused (fail closed) while the profile is bound.
        self.sessions: dict[str, Any] = {}
        self.session_windows: dict[str, dict] = {}
        # Optional safety hooks set after construction: a Reconciler and a
        # FeedMonitor (Part 12.2). When either pauses an account, the
        # router refuses new entries BEFORE any pipeline step.
        self.reconciler = None
        self.feed_monitor = None

    def register_account(self, account: Account, snapshot: AccountSnapshot,
                         profile: StrategyProfile, ruleset: FirmRuleSet | None,
                         spec: SymbolSpec, price: float) -> None:
        self.accounts[account.account_id] = account
        self.snapshots[account.account_id] = snapshot
        self.profiles[account.account_id] = profile
        self.rulesets[account.account_id] = ruleset
        self.specs[account.account_id] = spec
        self.prices[account.account_id] = price

    # ── U2: session gating ───────────────────────────────────────────────
    def set_session_profile(self, account_id: str, profile) -> None:
        """Bind a SessionProfile to an account. Until a window is
        opened by the SessionScheduler, new entries are refused."""
        self.sessions[account_id] = profile
        self.session_windows.pop(account_id, None)

    def set_session_window(self, account_id: str, profile, day) -> dict:
        """Open (or re-open) the session window for a date; returns the
        window record used by the pipeline gate."""
        start_ms, end_ms = profile.window_utc(day)
        win = {
            "start_ms": start_ms, "end_ms": end_ms,
            "cutoff_ms": end_ms - profile.no_new_entries_after_minutes_before_end * 60_000,
            "flatten_ms": profile._flatten_instant(day),
            "max_trades": profile.max_trades_in_session,
            "entries": 0,
        }
        self.sessions[account_id] = profile
        self.session_windows[account_id] = win
        return win

    def clear_session(self, account_id: str) -> None:
        self.sessions.pop(account_id, None)
        self.session_windows.pop(account_id, None)

    # ── U3: economic calendar ────────────────────────────────────────────
    def attach_network_registry(self, registry) -> None:
        """U5: network identity registry for Auto-mode blocking."""
        self._network_registry = registry

    def attach_cross_account_guard(self, guard) -> None:
        """U5: correlated-intent guard across accounts."""
        self._cross_guard = guard

    def _network_blocks(self, account_id: str) -> list[str]:
        reg = getattr(self, "_network_registry", None)
        if reg is None:
            return []
        def policy_of(aid: str) -> str:
            rs = self.rulesets.get(aid)
            if rs is None:
                return "unknown"
            # 'allowed' only when BOTH vps and ip policies allow.
            if rs.vps_policy == "allowed" and rs.ip_policy == "allowed":
                return "allowed"
            return "not_allowed"
        return reg.blocks_auto(account_id, policy_of)

    def attach_calendar(self, service) -> None:
        """Wire a CalendarService. The compliance engine's fail-closed
        news check consumes it per intent; a missing/stale calendar
        blocks entries for accounts WITH a news restriction only."""
        self._calendar_service = service

    def _calendar_for(self, account_id: str, symbol: str,
                      ts_ms: int) -> dict[int, str] | None:
        """Filtered {event_start_ms: label} around the intent time, or
        None when unavailable (which the engine treats as fail-closed
        for news-restricted accounts)."""
        svc = getattr(self, "_calendar_service", None)
        ruleset = self.rulesets.get(account_id)
        if ruleset is None or not ruleset.news_restriction.enabled:
            return {} if svc is not None else None   # unrestricted accounts
                                                       # are not affected
        if svc is None:
            return None
        nr = ruleset.news_restriction
        return svc.calendar_map(nr.currencies, nr.impacts, ts_ms)

    def enforce_news_position_policy(self, account_id: str, ts_ms: int,
                                     horizon_s: int = 60) -> list[dict]:
        """When a news window STARTS, apply the rule set's position
        policy to open positions on affected symbols:
        keep -> no-op; tighten_sl -> risk-reducing SL move (protective);
        flatten -> protective close. Deterministic, audited."""
        svc = getattr(self, "_calendar_service", None)
        ruleset = self.rulesets.get(account_id)
        if svc is None or ruleset is None or \
                not ruleset.news_restriction.enabled:
            return []
        nr = ruleset.news_restriction
        policy = nr.position_policy
        if policy == "keep":
            return []
        # Symbols of this account whose quote currency matches the
        # event currencies (XAUUSD -> USD).
        syms = {getattr(self.specs.get(account_id), "symbol", "XAUUSD")}
        affected = {s for s in syms
                    if s[-3:] in set(nr.currencies)}
        if not affected:
            return []
        # Any window currently open or starting within the horizon?
        for sym in sorted(affected):
            events = svc.events_filtered(
                nr.currencies, nr.impacts,
                ts_ms - nr.window_before_min * 60_000,
                ts_ms + nr.window_after_min * 60_000 + horizon_s * 1000)
            if events:
                break
        else:
            return []
        actions: list[dict] = []
        try:
            open_positions = self.broker.positions_of(account_id)
        except Exception:
            open_positions = []
        for pos in open_positions:
            sym = getattr(pos, "symbol", "XAUUSD")
            if sym not in affected:
                continue
            pid = getattr(pos, "position_id", None) \
                or getattr(pos, "ticket", None)
            if pid is None:
                continue
            if policy == "flatten":
                rec = self.protective_close_all(account_id, ts_ms,
                                                "news_window_flatten")
                return rec
            if policy == "tighten_sl":
                # Risk-reducing move: SL to breakeven-or-better is the
                # deterministic maximum tightening applied here.
                entry = getattr(pos, "entry_price", None) \
                    or getattr(pos, "price_open", None)
                side = getattr(pos, "side", 1)
                if entry is None:
                    continue
                new_sl = entry if side == 1 else entry
                try:
                    self.behaviour.check_modify(account_id, str(pid),
                                                int(ts_ms), protective=True)
                except Exception:
                    pass
                try:
                    self.broker.modify_position(pid, sl_price=new_sl)
                    actions.append({"position_id": str(pid), "action":
                                    "tighten_sl", "sl_price": new_sl})
                    self.audit.append({"event": "news_tighten_sl",
                                       "account_id": account_id,
                                       "position_id": str(pid),
                                       "ts_ms": int(ts_ms)})
                except Exception:
                    continue
        return actions

    def protective_close_all(self, account_id: str, ts_ms: int,
                             reason: str = "session_flatten") -> list[dict]:
        """Close every open position of the account as a PROTECTIVE
        action (kill-switch class): bypasses entry gates and behaviour
        min-hold, always audited. Used by the SessionScheduler at
        flatten_by and available to the kill switch."""
        closed: list[dict] = []
        try:
            open_positions = self.broker.positions_of(account_id)
        except Exception:
            open_positions = []
        for pos in open_positions:
            pid = getattr(pos, "position_id", None) or getattr(pos, "ticket", None)
            if pid is None:
                continue
            # Log the protective exception in behaviour controls (U1).
            try:
                self.behaviour.check_close(account_id, str(pid), int(ts_ms),
                                           protective=True)
            except Exception:
                pass
            rec = self.broker.close_position(pid, ts_ms)
            rec["reason"] = reason
            rec["protective"] = True
            closed.append(rec)
            self.audit.append({"event": "protective_close",
                               "account_id": account_id,
                               "position_id": str(pid), "reason": reason,
                               "pnl": rec.get("pnl"), "ts_ms": int(ts_ms)})
        if closed:
            self.notifier.notify("protective_close",
                                 f"{account_id}: {reason} closed "
                                 f"{len(closed)} position(s)")
        return closed

    # ── the single path ──────────────────────────────────────────────────
    def submit(self, intent: OrderIntent, *,
               approval_token: str | None = None) -> ExecutionResult:
        # Safety pauses (Part 12.2): reconciliation mismatch or a stale
        # feed blocks NEW entries; open positions keep their SL/TP.
        if self.reconciler is not None and \
                not self.reconciler.entries_allowed(intent.account_id):
            self.audit.append({"event": "entry_blocked_paused",
                               "key": intent.idempotency_key,
                               "account": intent.account_id})
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   ("entries_paused_reconciliation",))
        if self.feed_monitor is not None and \
                not self.feed_monitor.check(intent.symbol, intent.created_ms):
            self.audit.append({"event": "entry_blocked_stale_feed",
                               "key": intent.idempotency_key,
                               "symbol": intent.symbol})
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   ("stale_feed",))

        # Idempotency FIRST: a retry never reaches the pipeline again.
        prior = self._results.get(intent.idempotency_key)
        if prior is not None:
            dup = ExecutionResult(
                intent.idempotency_key, "duplicate", prior.reasons,
                prior.order_ref, prior.fill_price, prior.filled_lot,
                prior.steps)
            self.audit.append({"event": "duplicate_suppressed",
                               "key": intent.idempotency_key})
            return dup

        steps: list[dict] = []
        # U1: every submit counts as a request for the per-minute cap —
        # including ones later rejected (that is what the firms count).
        self.behaviour.record_request(intent.account_id, intent.created_ms)
        result = self._pipeline(intent, approval_token, steps)
        self._results[intent.idempotency_key] = result
        # U5: record submitted (non-rejected) entries for correlation.
        guard = getattr(self, "_cross_guard", None)
        if guard is not None and result.status != "rejected" \
                and intent.side != SIDE_CLOSE:
            guard.record(intent.account_id, intent.symbol, intent.side,
                         getattr(intent.strategy_version, "version_id", ""),
                         intent.created_ms)
        self.audit.append({"event": "execution_result",
                           "key": intent.idempotency_key,
                           "status": result.status,
                           "reasons": list(result.reasons),
                           "steps": steps})
        return result

    def _pipeline(self, intent: OrderIntent, approval_token: str | None,
                  steps: list[dict]) -> ExecutionResult:
        account = self.accounts.get(intent.account_id)
        spec = self.specs.get(intent.account_id)
        profile = self.profiles.get(intent.account_id)
        snapshot = self.snapshots.get(intent.account_id)
        ruleset = self.rulesets.get(intent.account_id, None)
        price = self.prices.get(intent.account_id)

        # Step 0: mode gate (research/backtest place no orders; manual and
        # semi-auto need a human token).
        try:
            self.modes.check_order_permission(intent.account_id,
                                              approval_token=approval_token)
            mode = self.modes.mode_of(intent.account_id)
            steps.append({"step": "mode_gate", "ok": True, "mode": mode})
        except Exception as exc:                      # ModeError
            steps.append({"step": "mode_gate", "ok": False, "error": str(exc)})
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   (f"mode_gate: {exc}",), steps=tuple(steps))

        # U1 RULE: ea_policy `unknown`/`forbidden` blocks Auto at order
        # time too (the gate check at set_mode is not enough — the
        # ruleset may have changed since).
        if mode == AUTO and ruleset is not None and not ruleset.allows_auto():
            steps.append({"step": "mode_gate", "ok": False,
                          "error": "ea_policy_blocks_auto"})
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   ("ea_policy_blocks_auto",),
                                   steps=tuple(steps))

        # U5: network identity re-check at order time (the ruleset or
        # IP picture may have changed since set_mode).
        if mode == AUTO:
            net_blocks = self._network_blocks(intent.account_id)
            if net_blocks:
                steps.append({"step": "network_identity", "ok": False,
                              "error": "network_identity_conflict"})
                self.audit.append({"event": "network_block",
                                   "account_id": intent.account_id,
                                   "reasons": net_blocks})
                self.notifier.notify("network_conflict",
                                     f"auto order blocked: {net_blocks}")
                return ExecutionResult(intent.idempotency_key, "rejected",
                                       ("network_identity_conflict",),
                                       steps=tuple(steps))

        # U5: cross-account correlation guard for NEW entries. Closing
        # an existing position is risk-REDUCING and never blocked here.
        guard = getattr(self, "_cross_guard", None)
        if guard is not None and intent.side != SIDE_CLOSE:
            decision = guard.check(
                intent.account_id, intent.symbol, intent.side,
                getattr(intent.strategy_version, "version_id", ""),
                intent.created_ms,
                getattr(ruleset, "copy_trading_policy", "unknown")
                if ruleset is not None else "unknown")
            if decision["blocked"]:
                steps.append({"step": "cross_account", "ok": False,
                              "error": "copy_trading_blocked"})
                self.audit.append({"event": "copy_trading_block",
                                   "account_id": intent.account_id,
                                   "matches": decision["matches"]})
                self.notifier.notify(
                    "copy_trading_block",
                    f"correlated intent blocked on {intent.account_id}")
                return ExecutionResult(intent.idempotency_key, "rejected",
                                       ("copy_trading_blocked",),
                                       steps=tuple(steps))
            steps.append({"step": "cross_account", "ok": True})

        # U2: session gate. The boundary is enforced HERE, in the gate,
        # not only in the strategy. Management of open trades is NOT
        # gated; only new entries are.
        if intent.side != SIDE_CLOSE:
            sess_profile = self.sessions.get(intent.account_id)
            if sess_profile is not None:
                win = self.session_windows.get(intent.account_id)
                if win is None or not (win["start_ms"] <= intent.created_ms < win["end_ms"]):
                    steps.append({"step": "session", "ok": False,
                                  "error": "outside_session_window"})
                    return ExecutionResult(intent.idempotency_key, "rejected",
                                           ("outside_session_window",),
                                           steps=tuple(steps))
                if intent.created_ms >= win["cutoff_ms"]:
                    steps.append({"step": "session", "ok": False,
                                  "error": "session_no_new_entries"})
                    return ExecutionResult(intent.idempotency_key, "rejected",
                                           ("session_no_new_entries",),
                                           steps=tuple(steps))
                if win["max_trades"] is not None and                         win["entries"] >= win["max_trades"]:
                    steps.append({"step": "session", "ok": False,
                                  "error": "session_max_trades"})
                    return ExecutionResult(intent.idempotency_key, "rejected",
                                           ("session_max_trades",),
                                           steps=tuple(steps))
                steps.append({"step": "session", "ok": True})

        # Step 1: deterministic position sizing for this account.
        if account is None or spec is None or profile is None or price is None:
            steps.append({"step": "sizing", "ok": False,
                          "error": "account context not registered"})
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   ("account_context_missing",),
                                   steps=tuple(steps))
        adapted = adapt_config(account, profile, spec, price)
        sizing = size_fixed_risk(account.equity, adapted.risk_per_trade_pct,
                                 profile.typical_stop_distance, spec, price,
                                 account.leverage)
        if not sizing.passes:
            steps.append({"step": "sizing", "ok": False,
                          "reasons": sizing.reasons})
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   tuple(sizing.reasons), steps=tuple(steps))
        steps.append({"step": "sizing", "ok": True, "lot": sizing.lot,
                      "risk_money": sizing.risk_money})

        # Step 2: risk engine.
        proposal_lot = sizing.lot
        risk_decision = self.risk.evaluate(
            OrderProposal(
                account_id=intent.account_id, symbol=intent.symbol,
                side=intent.side, lot=proposal_lot, price=price,
                sl_price=intent.sl_price, tp_price=intent.tp_price,
                time_ms=intent.created_ms),
            snapshot, spec=spec)
        steps.append({"step": "risk", "ok": risk_decision.outcome != "rejected",
                      "outcome": risk_decision.outcome,
                      "reasons": list(risk_decision.reasons)})
        if risk_decision.outcome == "rejected":
            self.notifier.notify("risk_limit",
                                 f"order rejected by risk engine: "
                                 f"{risk_decision.reasons}")
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   risk_decision.reasons, steps=tuple(steps))
        lot = risk_decision.lot  # approved or reduced size

        # Step 3: compliance engine (verified rules only).
        compliance = self.compliance.pre_trade_check(
            ruleset,
            self._compliance_state(intent.account_id),
            ProposedTrade(symbol=intent.symbol, lot=lot, side=intent.side,
                           price=price, sl_price=intent.sl_price,
                           time_ms=intent.created_ms),
            calendar_events_ms=self._calendar_for(intent.account_id,
                                                  intent.symbol,
                                                  intent.created_ms),
            spec_max_lot=spec.max_lot)
        steps.append({"step": "compliance", "ok": compliance.allowed,
                      "reasons": list(compliance.reasons)})
        if not compliance.allowed:
            self.notifier.notify("compliance_breach",
                                 f"order blocked by compliance: "
                                 f"{compliance.reasons}")
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   compliance.reasons, steps=tuple(steps))

        # Step 3.5 (U1): behaviour controls — min hold, per-day/per-minute
        # caps, prohibited styles, hedging. Fail closed.
        bconf = self.behaviour.config_for(ruleset)
        breasons: list[str] = []
        bdetails: dict = {}
        hed = self.behaviour.check_hedging(
            intent.account_id, intent.symbol, intent.side)
        if not hed.allowed:
            breasons.extend(hed.reasons)
            bdetails.update(hed.details)
        ent = self.behaviour.check_entry(
            intent.account_id, intent.symbol, intent.side, lot, price,
            intent.created_ms) if bconf.prohibited_styles or \
            bconf.max_orders_per_day is not None or \
            bconf.max_requests_per_minute is not None else None
        if ent is not None and not ent.allowed:
            breasons.extend(ent.reasons)
            bdetails.update(ent.details)
        steps.append({"step": "behaviour", "ok": not breasons,
                      "reasons": breasons, **bdetails})
        if breasons:
            self.notifier.notify("compliance_breach",
                                 f"order blocked by behaviour controls: "
                                 f"{breasons}")
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   tuple(breasons), steps=tuple(steps))

        # Step 4: mint the ApprovedOrder — the ONLY object the broker accepts.
        approved = ApprovedOrder(
            intent=intent, lot=lot, price_ref=price,
            risk_decision=risk_decision,
            compliance_reasons=compliance.reasons, mode=mode,
            approval_token=approval_token,
            approved_ms=intent.created_ms, decision_trail=tuple(steps))

        # Step 5: broker order_check (dry-run) then send.
        check = self.broker.order_check(approved) \
            if hasattr(self.broker, "order_check") else {"ok": True}
        steps.append({"step": "order_check", "ok": bool(check.get("ok")),
                      "detail": check.get("detail", "")})
        if not check.get("ok"):
            return ExecutionResult(intent.idempotency_key, "rejected",
                                   ("order_check_failed",), steps=tuple(steps))

        try:
            rec = self.broker.place(approved)
        except Exception as exc:
            # Timeout reconciliation BEFORE any retry (Part 12.2): ask the
            # broker what actually happened to this idempotency key.
            known = self.broker.find_by_idempotency_key(intent.idempotency_key)
            if known is not None:
                steps.append({"step": "send", "ok": True,
                              "reconciled": True, "error": str(exc)})
                self.notifier.notify("execution_reconciled",
                                     f"timeout reconciled: {intent.idempotency_key}")
                return ExecutionResult(intent.idempotency_key, "filled",
                                       (), known["order_ref"],
                                       known["fill_price"], approved.lot,
                                       tuple(steps))
            steps.append({"step": "send", "ok": False, "error": str(exc)})
            self.notifier.notify("execution_failure",
                                 f"send failed: {exc}")
            return ExecutionResult(intent.idempotency_key, "pending_unknown",
                                   (f"send_error: {exc}",), steps=tuple(steps))

        steps.append({"step": "send", "ok": True, "order_ref": rec["order_ref"],
                      "fill_price": rec["fill_price"],
                      "pending": bool(rec.get("pending"))})
        if not rec.get("pending") and rec.get("fill_price") is not None:
            # U1: feed the behaviour controls' per-account history.
            self.behaviour.record_entry(
                intent.account_id, rec["order_ref"], intent.symbol,
                intent.side, approved.lot, rec["fill_price"],
                intent.created_ms)
            # U2: count the accepted entry against the session's
            # max_trades_in_session cap.
            win = self.session_windows.get(intent.account_id)
            if win is not None and win["max_trades"] is not None:
                win["entries"] += 1
        if rec.get("pending"):
            # Stop/limit accepted at the broker; it fills on a later tick.
            self.notifier.notify("order_pending",
                                 f"{intent.symbol} {intent.order_type} "
                                 f"{approved.lot} lots @ {intent.limit_price} "
                                 f"queued",
                                 order_ref=rec["order_ref"])
            return ExecutionResult(intent.idempotency_key, "pending_accepted",
                                   (), rec["order_ref"], None, 0.0,
                                   tuple(steps))
        self.notifier.notify("trade_opened",
                             f"{intent.symbol} {'long' if intent.side == 1 else 'short'} "
                             f"{approved.lot} lots @ {rec['fill_price']}",
                             order_ref=rec["order_ref"])
        return ExecutionResult(intent.idempotency_key, "filled", (),
                               rec["order_ref"], rec["fill_price"],
                               approved.lot, tuple(steps))

    def _compliance_state(self, account_id: str) -> AccountState:
        snap = self.snapshots[account_id]
        return AccountState(
            account_id=account_id, balance=snap.balance, equity=snap.equity,
            balance_at_reset=snap.balance, equity_at_reset=snap.equity,
            equity_high_watermark=snap.equity_high_watermark,
            open_positions=snap.open_positions,
            exposure_lots=sum(snap.exposure_lots_by_symbol.values()),
            daily_realized_pnl=snap.daily_realized_pnl,
            daily_floating_pnl=snap.daily_floating_pnl)
