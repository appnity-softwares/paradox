"""U0 tests: AuditLog in-memory mode and hash-chained file mode.

Blueprint U0.2/U0.3:
- AuditLog(":memory:") must NOT create a file named ':memory:'.
- File mode entries are hash-chained (prev_hash + sha256) and
  verify_chain() returns the first broken index or None.
- A pre-existing unchained (legacy) segment is preserved; the chain
  starts on the first chained write.
"""
from __future__ import annotations

import json

import pytest

from core.execution.router import AuditLog


def test_memory_mode_creates_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = AuditLog(":memory:")
    log.append({"event": "x", "n": 1})
    log.append({"event": "x", "n": 2})
    got = [{k: v for k, v in e.items() if k != "ts_ms"}
           for e in log.entries()]
    assert got == [{"event": "x", "n": 1}, {"event": "x", "n": 2}]
    # Nothing may appear on disk under the literal name ':memory:'.
    assert not (tmp_path / ":memory:").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_file_mode_chain_is_verifiable(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append({"event": "e", "i": i})
    records = log.entries()
    assert all("prev_hash" in r and "hash" in r for r in records)
    # Each entry links to its predecessor.
    for prev, cur in zip(records, records[1:]):
        assert cur["prev_hash"] == prev["hash"]
    assert log.verify_chain() is None


def test_tampered_entry_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.append({"event": "e", "i": i})
    lines = path.read_text().splitlines()
    tampered = json.loads(lines[2])
    tampered["i"] = 999                      # rewrite history
    lines[2] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n")
    assert AuditLog(path).verify_chain() == 2   # first broken index


def test_legacy_unchained_segment_is_preserved_and_skipped(tmp_path):
    path = tmp_path / "audit.jsonl"
    legacy = [{"ts_ms": 1, "event": "old-1"}, {"ts_ms": 2, "event": "old-2"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in legacy))
    log = AuditLog(path)                     # opens the legacy file
    log.append({"event": "new-1"})           # chain starts here
    log.append({"event": "new-2"})
    records = log.entries()
    assert records[:2] == legacy             # untouched
    assert "hash" not in records[0] and "hash" in records[2]
    assert records[2]["prev_hash"] == AuditLog.GENESIS_PREV_HASH
    assert log.verify_chain() is None


def test_chain_continues_across_reopened_logs(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append({"event": "a"})
    AuditLog(path).append({"event": "b"})    # new instance, same file
    records = AuditLog(path).entries()
    assert records[1]["prev_hash"] == records[0]["hash"]
    assert AuditLog(path).verify_chain() is None


def test_verify_chain_flags_unchained_line_after_chained_segment(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append({"event": "a"})
    log.append({"event": "b"})
    lines = path.read_text().splitlines()
    # A writer that does not chain appends a plain line after the chain.
    with path.open("a") as f:
        f.write(json.dumps({"ts_ms": 3, "event": "rogue"}) + "\n")
    assert AuditLog(path).verify_chain() == 2
