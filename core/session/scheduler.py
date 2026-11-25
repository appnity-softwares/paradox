"""Session scheduling and run records (Update Blueprint v2.1, U2).

The scheduler is fully deterministic: pre-session health checks, the
flatten action and the run ledger never call the LLM layer. LLM use is
allowed only in explicit user-requested actions and optional
pre/post-session reports guarded by the daily budget cap (core/llm.py).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from core.session.profile import SessionProfile


class SessionRunStore:
    """Append-only JSONL ledger of session runs (state file/table)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict) -> dict:
        entry = dict(entry)
        entry.setdefault("recorded_ms", int(
            datetime.now(timezone.utc).timestamp() * 1000))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def for_deployment(self, deployment_id: str) -> list[dict]:
        return [e for e in self.entries()
                if e.get("deployment_id") == deployment_id]


class SessionScheduler:
    """Runs one bounded session for one account.

    Lifecycle: WAITING -> (pre-checks) -> ACTIVE -> FLATTEN -> CLOSED,
    or SKIPPED / ABORTED. All decisions are deterministic.
    """

    def __init__(self, router, notifier=None, run_store: SessionRunStore | None = None):
        self.router = router
        self.notifier = notifier or getattr(router, "notifier", None)
        self.runs = run_store or SessionRunStore("state/session_runs.jsonl")

    # ── pre-session health checks ────────────────────────────────────────

    def pre_checks(self, account_id: str, profile: SessionProfile,
                   ts_ms: int) -> tuple[bool, list[str]]:
        """(ok, failure_reasons). Any failure means the session is
        SKIPPED — fail closed."""
        problems: list[str] = []
        router = self.router

        # 1. Account snapshot present.
        if account_id not in getattr(router, "accounts", {}):
            problems.append("account_snapshot_missing")

        # 2. Feed fresh for every symbol the account may trade.
        feed = getattr(router, "feed_monitor", None)
        if feed is None:
            problems.append("feed_monitor_missing")
        else:
            symbols = set()
            for acc in getattr(router, "accounts", {}).values():
                symbols.update(getattr(acc, "symbols", ()) or ())
            if not symbols:
                symbols = {"XAUUSD"}
            for sym in sorted(symbols):
                if not feed.check(sym, ts_ms):
                    problems.append(f"feed_stale_{sym}")

        # 3. Kill switch off.
        ks = getattr(getattr(router, "risk_engine", None), "kill_switch",
                     None)
        if ks is not None and ks.is_blocked(account_id):
            problems.append("kill_switch_engaged")

        # 4. Compliance status not breached: the ruleset must exist and
        #    not be in a breach state (verified accounts only trade).
        ruleset = getattr(router, "rulesets", {}).get(account_id)
        if ruleset is None:
            problems.append("ruleset_missing")
        elif getattr(ruleset, "status", None) != "verified":
            problems.append("ruleset_not_verified")

        # 5. Calendar loaded AND fresh if the news restriction is
        #    enabled (U3). Fail closed: missing or stale -> skip.
        if ruleset is not None and \
                getattr(getattr(ruleset, "news_restriction", None),
                        "enabled", False):
            svc = getattr(router, "_calendar_service", None)
            if svc is None:
                problems.append("calendar_not_loaded")
            else:
                st = svc.status()
                if st == "missing":
                    problems.append("calendar_not_loaded")
                elif st == "stale":
                    problems.append("calendar_stale")

        return (len(problems) == 0, problems)

    # ── flatten ──────────────────────────────────────────────────────────

    def flatten_account(self, account_id: str, ts_ms: int,
                        reason: str = "session_flatten") -> list[dict]:
        """Close remaining open positions THROUGH the router as a
        protective action; each close is audited."""
        closed = self.router.protective_close_all(account_id, ts_ms, reason)
        if closed and self.notifier is not None:
            self.notifier.notify(
                "session_flatten",
                f"{account_id}: flattened {len(closed)} position(s) ({reason})")
        return closed

    # ── one session run ──────────────────────────────────────────────────

    def run_session(self, account_id: str, profile: SessionProfile,
                    day: date, deployment_id: str = "default",
                    now_ms: int | None = None) -> dict:
        """Open the session window for the account after pre-checks and
        record the run. The strategy loop itself runs elsewhere (the
        engine/bridge calls entries through the router, which enforces
        the window); this method owns gating, flatten and the ledger."""
        start_ms, end_ms = profile.window_utc(day)
        ts = now_ms if now_ms is not None else start_ms
        ok, problems = self.pre_checks(account_id, profile, ts)
        if not ok:
            rec = self.runs.record({
                "date": day.isoformat(), "deployment_id": deployment_id,
                "account_id": account_id, "profile": profile.name,
                "status": "skipped", "reason": ";".join(problems),
                "trades": 0, "pnl": 0.0, "drawdown_pct": 0.0,
            })
            if self.notifier is not None:
                self.notifier.notify(
                    "session_skipped",
                    f"{account_id} {profile.name} skipped: "
                    f"{';'.join(problems)}")
            return rec

        # Open the gate for this window.
        self.router.set_session_window(account_id, profile, day)
        rec = self.runs.record({
            "date": day.isoformat(), "deployment_id": deployment_id,
            "account_id": account_id, "profile": profile.name,
            "status": "active", "reason": "", "trades": 0, "pnl": 0.0,
            "drawdown_pct": 0.0, "window_start_ms": start_ms,
            "window_end_ms": end_ms,
        })
        return rec

    def close_session(self, account_id: str, profile: SessionProfile,
                      day: date, deployment_id: str = "default",
                      now_ms: int | None = None) -> dict:
        """At flatten_by: flatten, then mark the run completed."""
        ts = now_ms if now_ms is not None else profile.window_utc(day)[1]
        closed = self.flatten_account(account_id, ts)
        snapshot = self.router.snapshots.get(account_id)
        rec = self.runs.record({
            "date": day.isoformat(), "deployment_id": deployment_id,
            "account_id": account_id, "profile": profile.name,
            "status": "completed",
            "reason": f"flattened_{len(closed)}",
            "trades": len(closed),
            "pnl": float(getattr(snapshot, "realized_pnl", 0.0) or 0.0)
                if snapshot is not None else 0.0,
            "drawdown_pct": 0.0,
        })
        return rec
