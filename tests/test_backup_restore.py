"""Backup/restore tests (Part 14.2): round trip integrity, tamper
detection, and fail-closed restore onto changed data.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.backup_state import backup, restore, verify


@pytest.fixture()
def state(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "audit.jsonl").write_text('{"event": "one"}\n{"event": "two"}\n')
    sub = d / "rules"
    sub.mkdir()
    (sub / "rules.json").write_text(json.dumps([{"firm": "ftmo_style"}]))
    (d / "kill_switch.json").write_text(json.dumps({"global": False,
                                                    "accounts": {}}))
    return d


def test_backup_round_trip_restores_identical_state(state, tmp_path):
    out = tmp_path / "backup.zip"
    backup(state, out)
    # Mutate the live state after the backup.
    (state / "audit.jsonl").write_text("corrupted\n")
    (state / "extra.txt").write_text("junk")
    target = tmp_path / "restored"
    restore(out, target)
    assert (target / "audit.jsonl").read_text() == \
        '{"event": "one"}\n{"event": "two"}\n'
    assert json.loads((target / "kill_switch.json").read_text()) == \
        {"global": False, "accounts": {}}
    assert (target / "rules" / "rules.json").exists()
    assert not (target / "extra.txt").exists()   # junk not carried over


def test_verify_detects_tampered_content(state, tmp_path):
    out = tmp_path / "backup.zip"
    backup(state, out)
    # Tamper: rewrite one archived file in place.
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        payload = {n: z.read(n) for n in names}
    payload["audit.jsonl"] = b'tampered\n'
    with zipfile.ZipFile(out, "w") as z:
        for n, data in payload.items():
            z.writestr(n, data)
    report = verify(out)
    assert report["ok"] is False
    assert any("audit.jsonl" in p for p in report["problems"])


def test_verify_detects_missing_file(state, tmp_path):
    out = tmp_path / "backup.zip"
    backup(state, out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        payload = {n: z.read(n) for n in names if n != "audit.jsonl"}
    with zipfile.ZipFile(out, "w") as z:
        for n, data in payload.items():
            z.writestr(n, data)
    report = verify(out)
    assert report["ok"] is False
    assert any("missing from zip" in p for p in report["problems"])


def test_restore_refuses_tampered_backup_fail_closed(state, tmp_path,
                                                     capsys):
    out = tmp_path / "backup.zip"
    backup(state, out)
    with zipfile.ZipFile(out) as z:
        payload = {n: z.read(n) for n in z.namelist()}
    payload["kill_switch.json"] = b'{"global": true}'
    with zipfile.ZipFile(out, "w") as z:
        for n, data in payload.items():
            z.writestr(n, data)
    target = tmp_path / "should_not_exist"
    with pytest.raises(SystemExit):
        restore(out, target)
    assert not target.exists()          # live state untouched


def test_restore_refuses_when_dataset_changed(state, tmp_path, monkeypatch):
    out = tmp_path / "backup.zip"
    backup(state, out)
    import scripts.backup_state as bs
    monkeypatch.setattr(bs, "_dataset_hash_snapshot",
                        lambda: {"XAUUSD": "DIFFERENT"})
    target = tmp_path / "no_restore"
    with pytest.raises(SystemExit):
        restore(out, target)
    assert not target.exists()
