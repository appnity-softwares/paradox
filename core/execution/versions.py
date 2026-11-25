"""Immutable strategy versions (Part 9).

A deployment always references a specific version: the strategy class
source hash plus the parameter hash. Any change — code or params —
creates a NEW version that must be re-validated before deployment.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyVersion:
    name: str
    code_hash: str
    params_hash: str
    version_id: str

    def to_dict(self) -> dict:
        return {"name": self.name, "code_hash": self.code_hash,
                "params_hash": self.params_hash,
                "version_id": self.version_id}


def make_version(cls, params: dict) -> StrategyVersion:
    """Hash the class source and the sorted params. Deterministic."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        source = cls.__name__
    code_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
    params_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    name = cls.__name__
    version_id = f"{name}:{code_hash}:{params_hash}"
    return StrategyVersion(name, code_hash, params_hash, version_id)
