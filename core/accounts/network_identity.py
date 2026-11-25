"""Network identity registry (Update Blueprint v2.1, U5).

Per prop account: vps_provider, region, public_ip, first/last seen,
bridge instance id. The bridge reports its public IP on /health.

BLOCKING RULES (fail closed):
- Two prop accounts reporting the same public IP while either firm's
  vps_policy/ip_policy is not 'allowed' -> alert + Auto mode blocked.
- An account's public IP changes unexpectedly -> alert.
Credentials are NEVER stored here (IPs and provider names only).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED = "allowed"


@dataclass
class NetworkIdentity:
    account_id: str
    vps_provider: str = "unknown"
    region: str = "unknown"
    public_ip: str = "unknown"
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    bridge_instance_id: str = "unknown"
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"account_id": self.account_id,
                "vps_provider": self.vps_provider, "region": self.region,
                "public_ip": self.public_ip,
                "first_seen_ms": self.first_seen_ms,
                "last_seen_ms": self.last_seen_ms,
                "bridge_instance_id": self.bridge_instance_id,
                "history": list(self.history)}

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkIdentity":
        return cls(**{k: v for k, v in d.items() if k in
                      {"account_id", "vps_provider", "region", "public_ip",
                       "first_seen_ms", "last_seen_ms",
                       "bridge_instance_id", "history"}})


class NetworkIdentityRegistry:
    """Persists to a JSON store. No credentials, ever."""

    def __init__(self, store_path: str | Path = "state/network_identity.json"):
        self.store_path = Path(store_path)
        self._accounts: dict[str, NetworkIdentity] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            blob = json.loads(self.store_path.read_text())
            self._accounts = {a["account_id"]: NetworkIdentity.from_dict(a)
                              for a in blob.get("accounts", [])}
        except (OSError, ValueError, KeyError, TypeError):
            self._accounts = {}

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(
            {"accounts": [a.to_dict() for a in
                          sorted(self._accounts.values(),
                                 key=lambda a: a.account_id)]}, indent=1))

    # ── reporting (called by the bridge /health path) ────────────────────

    def report(self, account_id: str, public_ip: str, *,
               vps_provider: str | None = None, region: str | None = None,
               bridge_instance_id: str | None = None,
               ts_ms: int | None = None) -> dict:
        """Upsert the identity; an IP change is recorded in history and
        flagged so the caller can alert."""
        now = ts_ms if ts_ms is not None else int(time.time() * 1000)
        ident = self._accounts.get(account_id)
        changed = False
        if ident is None:
            ident = NetworkIdentity(account_id=account_id, public_ip=public_ip,
                                    first_seen_ms=now, last_seen_ms=now)
            self._accounts[account_id] = ident
        else:
            changed = ident.public_ip != public_ip
            ident.last_seen_ms = now
            ident.public_ip = public_ip
        for k, v in (("vps_provider", vps_provider),
                     ("region", region),
                     ("bridge_instance_id", bridge_instance_id)):
            if v is not None:
                setattr(ident, k, v)
        ident.history.append({"ts_ms": now, "public_ip": public_ip,
                              "ip_changed": changed})
        self._save()
        return {"account_id": account_id, "ip_changed": changed,
                "public_ip": public_ip}

    def entries(self) -> list[NetworkIdentity]:
        return sorted(self._accounts.values(), key=lambda a: a.account_id)

    def get(self, account_id: str) -> NetworkIdentity | None:
        return self._accounts.get(account_id)

    def history(self, account_id: str) -> list[dict]:
        ident = self._accounts.get(account_id)
        return list(ident.history) if ident else []

    # ── checks ───────────────────────────────────────────────────────────

    def shared_ip_conflicts(self, policy_of) -> list[dict]:
        """Groups of accounts sharing a public IP where the policy getter
        (account_id -> 'allowed' | other) says it is NOT allowed for at
        least one of them. `policy_of` may raise KeyError for unknown
        accounts -> treated as not allowed (fail closed)."""
        by_ip: dict[str, list[str]] = {}
        for ident in self._accounts.values():
            by_ip.setdefault(ident.public_ip, []).append(ident.account_id)
        out = []
        for ip, ids in sorted(by_ip.items()):
            if ip == "unknown" or len(ids) < 2:
                continue
            offenders = [a for a in ids
                         if _policy(policy_of, a) != ALLOWED]
            if offenders:
                out.append({"public_ip": ip, "accounts": sorted(ids),
                            "offenders": sorted(offenders)})
        return out

    def shared_bridge_conflicts(self) -> list[dict]:
        """Two prop accounts on the SAME bridge instance (one bridge per
        account is the documented requirement)."""
        by_bridge: dict[str, list[str]] = {}
        for ident in self._accounts.values():
            by_bridge.setdefault(ident.bridge_instance_id,
                                 []).append(ident.account_id)
        return [{"bridge_instance_id": b, "accounts": sorted(ids)}
                for b, ids in sorted(by_bridge.items())
                if b != "unknown" and len(ids) > 1]

    def blocks_auto(self, account_id: str, policy_of) -> list[str]:
        """Reasons this account must not run Auto right now."""
        reasons: list[str] = []
        for conflict in self.shared_ip_conflicts(policy_of):
            if account_id in conflict["accounts"]:
                reasons.append(f"shared_ip:{conflict['public_ip']}")
        ident = self._accounts.get(account_id)
        if ident is not None:
            changes = [h for h in ident.history if h.get("ip_changed")]
            if changes:
                reasons.append("unexpected_ip_change")
        return reasons


def _policy(policy_of, account_id: str) -> str:
    try:
        return policy_of(account_id)
    except Exception:
        return "unknown"
