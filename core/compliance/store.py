"""Rule store (Part 10.2): versioned FirmRuleSets with a draft/verify
lifecycle, persisted as JSON. Every change creates a new version with a
recorded diff. Enforcement reads ONLY verified versions.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.compliance.rules import (
    STATUS_VERIFIED,
    FirmRuleSet,
    diff_rulesets,
)


class RuleStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, firm: str, program: str) -> Path:
        safe = f"{firm}__{program}".replace("/", "_")
        return self.root / f"{safe}.json"

    def _load_all(self, firm: str, program: str) -> list[FirmRuleSet]:
        p = self._path(firm, program)
        if not p.exists():
            return []
        return [FirmRuleSet.from_dict(d) for d in json.loads(p.read_text())]

    def _save_all(self, firm: str, program: str,
                  versions: list[FirmRuleSet]) -> None:
        payload = json.dumps([v.to_dict() for v in versions], indent=1)
        self._path(firm, program).write_text(payload)

    def all(self) -> list[FirmRuleSet]:
        """Every stored version of every firm/program (latest first)."""
        out: list[FirmRuleSet] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                for d in json.loads(path.read_text()):
                    out.append(FirmRuleSet.from_dict(d))
            except (OSError, ValueError, KeyError):
                continue        # corrupt file: skip, never crash the reader
        return out

    # ── lifecycle ────────────────────────────────────────────────────────
    def add_draft(self, rs: FirmRuleSet) -> FirmRuleSet:
        """Ingest/extract step: always stored as a draft."""
        if rs.status != "draft":
            rs = type(rs)(**{**rs.to_dict(), "status": "draft",
                             "verified_by": None})
        versions = self._load_all(rs.firm, rs.program)
        if any(v.version == rs.version for v in versions):
            raise ValueError(f"version {rs.version} already exists")
        versions.append(rs)
        self._save_all(rs.firm, rs.program, versions)
        return rs

    def verify(self, firm: str, program: str, version: str,
               verifier: str) -> FirmRuleSet:
        """Human verification: draft -> verified, after schema validation."""
        versions = self._load_all(firm, program)
        for i, v in enumerate(versions):
            if v.version == version:
                problems = v.validate()
                if problems:
                    raise ValueError(f"cannot verify: {problems}")
                if not v.source_clause_refs:
                    raise ValueError("cannot verify without clause citations")
                verified = type(v)(**{**v.to_dict(), "status": "verified",
                                      "verified_by": verifier})
                versions[i] = verified
                self._save_all(firm, program, versions)
                return verified
        raise KeyError(f"unknown version {version}")

    def latest(self, firm: str, program: str) -> FirmRuleSet | None:
        versions = self._load_all(firm, program)
        return versions[-1] if versions else None

    def get(self, firm: str, program: str, version: str) -> FirmRuleSet:
        for v in self._load_all(firm, program):
            if v.version == version:
                return v
        raise KeyError(f"unknown version {version}")

    def verified(self, firm: str, program: str) -> FirmRuleSet | None:
        """The verified version that enforcement is allowed to read."""
        versions = self._load_all(firm, program)
        for v in reversed(versions):
            if v.status == STATUS_VERIFIED:
                return v
        return None

    def diff(self, firm: str, program: str, version_a: str,
             version_b: str) -> dict[str, tuple]:
        return diff_rulesets(self.get(firm, program, version_a),
                             self.get(firm, program, version_b))

    def needs_reverification(self, firm: str, program: str,
                             account_version: str) -> bool:
        """True when the firm published a newer version than the account
        runs on — the account must be re-verified against it."""
        latest = self.latest(firm, program)
        return latest is not None and latest.version != account_version
