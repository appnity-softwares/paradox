"""Kill switch (Part 8.3): global and per-account trading halt.

State persists to disk so an engaged switch survives restarts. Only an
explicit human action (authenticated actor id + reason) may disengage;
every engage/disengage is appended to an audit log. Fail closed: if the
state file is unreadable, the switch reports BLOCKED.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class KillSwitch:
    def __init__(self, state_path: str | Path):
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            self._state = {"global": False, "accounts": {}}
            self._audit: list[dict] = []
            return
        try:
            raw = json.loads(self.path.read_text())
            self._state = {"global": bool(raw.get("global", False)),
                           "accounts": {k: bool(v)
                                        for k, v in raw.get("accounts", {}).items()}}
            self._audit = list(raw.get("audit", []))
        except (json.JSONDecodeError, OSError, ValueError):
            # Fail closed: unreadable state = blocked.
            self._state = {"global": True, "accounts": {}}
            self._audit = [{"ts_ms": int(time.time() * 1000),
                            "action": "fail_closed_load_error"}]

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {"global": self._state["global"],
             "accounts": self._state["accounts"],
             "audit": self._audit[-500:]},
            indent=1))

    # ── queries ──────────────────────────────────────────────────────────
    def public_state(self) -> dict:
        """Read-only view for dashboards/APIs."""
        return {"global": bool(self._state.get("global", False)),
                "accounts": dict(self._state.get("accounts", {}))}

    def is_blocked(self, account_id: str | None = None) -> bool:
        if self._state["global"]:
            return True
        if account_id is not None:
            return bool(self._state["accounts"].get(account_id, False))
        return False

    @property
    def audit_log(self) -> tuple[dict, ...]:
        return tuple(self._audit)

    # ── actions (human-initiated; every action is audit-logged) ─────────
    def _log(self, action: str, scope: str, actor: str, reason: str) -> None:
        self._audit.append({
            "ts_ms": int(time.time() * 1000),
            "action": action, "scope": scope,
            "actor": actor, "reason": reason,
        })

    def engage_global(self, actor: str, reason: str) -> None:
        self._state["global"] = True
        self._log("engage_global", "global", actor, reason)
        self._save()

    def engage_account(self, account_id: str, actor: str, reason: str) -> None:
        self._state["accounts"][account_id] = True
        self._log("engage_account", account_id, actor, reason)
        self._save()

    def disengage_global(self, actor: str, reason: str) -> None:
        self._state["global"] = False
        self._log("disengage_global", "global", actor, reason)
        self._save()

    def disengage_account(self, account_id: str, actor: str, reason: str) -> None:
        self._state["accounts"][account_id] = False
        self._log("disengage_account", account_id, actor, reason)
        self._save()
