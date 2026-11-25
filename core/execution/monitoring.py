"""Reconciliation loop and connection monitoring (Part 12.2).

- Reconciler: periodically compares the broker's open positions with the
  platform's tracked state. Any mismatch raises an alert and pauses new
  entries until a human resumes.
- FeedMonitor: on a stale tick feed, pause new entries and alert; open
  positions keep their protective SL/TP at the broker (define-and-log).
"""
from __future__ import annotations

from core.execution.router import Notifier


class Reconciler:
    def __init__(self, broker, audit_log, notifier: Notifier | None = None):
        self.broker = broker
        self.audit = audit_log
        self.notifier = notifier or Notifier()
        self.paused: dict[str, bool] = {}     # account -> new entries paused
        # Platform-side truth: idempotency_key -> (account_id, position ref)
        self.tracked: dict[str, tuple[str, str]] = {}

    def track_fill(self, account_id: str, idempotency_key: str,
                   order_ref: str) -> None:
        self.tracked[idempotency_key] = (account_id, order_ref)

    def reconcile(self) -> dict:
        broker_keys = {r.idempotency_key for r in self.broker.order_history}
        tracked_keys = set(self.tracked)
        missing_at_broker = sorted(tracked_keys - broker_keys)
        unknown_at_broker = sorted(broker_keys - tracked_keys)
        mismatch = bool(missing_at_broker or unknown_at_broker)
        if mismatch:
            for key in missing_at_broker:
                account = self.tracked[key][0]
                self.paused[account] = True
                self.notifier.notify("reconciliation_mismatch",
                                     f"platform order {key} missing at broker; "
                                     f"new entries paused for {account}")
            for key in unknown_at_broker:
                # Pause the account that owns the unknown broker position.
                rec = next((r for r in self.broker.order_history
                            if r.idempotency_key == key), None)
                if rec is not None:
                    self.paused[rec.approved.intent.account_id] = True
                self.notifier.notify("reconciliation_mismatch",
                                     f"broker position {key} unknown to platform; "
                                     "new entries paused")
            self.audit.append({"event": "reconciliation_mismatch",
                               "missing_at_broker": missing_at_broker,
                               "unknown_at_broker": unknown_at_broker})
        else:
            self.audit.append({"event": "reconciliation_ok",
                               "open_orders": len(broker_keys)})
        return {"mismatch": mismatch,
                "missing_at_broker": missing_at_broker,
                "unknown_at_broker": unknown_at_broker}

    def resume(self, account_id: str, actor: str, reason: str) -> None:
        self.paused[account_id] = False
        self.audit.append({"event": "reconciliation_resume",
                           "account": account_id, "actor": actor,
                           "reason": reason})

    def entries_allowed(self, account_id: str) -> bool:
        return not self.paused.get(account_id, False)


class FeedMonitor:
    def __init__(self, broker, max_staleness_ms: int = 10_000,
                 notifier: Notifier | None = None, audit_log=None):
        self.broker = broker
        self.max_staleness_ms = max_staleness_ms
        self.notifier = notifier or Notifier()
        self.audit = audit_log
        self.paused: dict[str, bool] = {}

    def check(self, symbol: str, now_ms: int) -> bool:
        """True = feed healthy. Stale feed pauses new entries; open
        positions keep their broker-side SL/TP protection."""
        last = self.broker.last_tick_ms.get(symbol)
        if last is None:
            healthy = False
        else:
            healthy = (now_ms - last) <= self.max_staleness_ms
        if not healthy and not self.paused.get(symbol, False):
            self.paused[symbol] = True
            self.notifier.notify("broker_disconnect",
                                 f"stale feed for {symbol}; new entries paused; "
                                 "open positions keep broker-side SL/TP")
            if self.audit is not None:
                self.audit.append({"event": "feed_stale", "symbol": symbol,
                                   "last_tick_ms": last, "now_ms": now_ms})
        elif healthy and self.paused.get(symbol, False):
            self.paused[symbol] = False
            self.notifier.notify("feed_recovered", f"feed healthy for {symbol}")
        return healthy
