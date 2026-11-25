"""State backup and tested restore (Part 14.2).

Backs up the dashboard state directory (audit log, rule store, kill
switch, sign-off record) into a versioned zip with a SHA-256 manifest,
plus the dataset hash so a restore can prove it still matches the
ingested data. Restore FAILS CLOSED: a manifest mismatch, a missing
file or a corrupt zip refuses to touch the live state.

Usage:
    python -m scripts.backup_state backup  [--state-dir DIR] [--out FILE]
    python -m scripts.backup_state restore BACKUP.zip [--state-dir DIR]
    python -m scripts.backup_state verify  BACKUP.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dataset_hash_snapshot() -> dict:
    """Record the dataset fingerprint so a restore can be checked
    against the data it will trade on."""
    try:
        from core.marketdata.store import dataset_hash
        return {"XAUUSD": dataset_hash("XAUUSD")}
    except Exception as exc:                 # store unavailable: record why
        return {"error": f"dataset hash unavailable: {exc}"}


def backup(state_dir: Path, out_path: Path | None = None) -> Path:
    state_dir = Path(state_dir)
    if not state_dir.exists():
        raise SystemExit(f"state dir does not exist: {state_dir}")
    files = sorted(p for p in state_dir.rglob("*") if p.is_file())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": _dataset_hash_snapshot(),
        "files": {str(p.relative_to(state_dir)): _sha256(p.read_bytes())
                  for p in files},
    }
    out_path = out_path or Path(tempfile.gettempdir()) / (
        f"dsh_state_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=str(p.relative_to(state_dir)))
        z.writestr(MANIFEST_NAME, json.dumps(manifest, indent=1))
    return out_path


def verify(backup_path: Path) -> dict:
    """Check every file against the manifest WITHOUT touching state."""
    backup_path = Path(backup_path)
    with zipfile.ZipFile(backup_path) as z:
        manifest = json.loads(z.read(MANIFEST_NAME))
        problems = []
        for rel, expected in manifest["files"].items():
            try:
                if _sha256(z.read(rel)) != expected:
                    problems.append(f"hash mismatch: {rel}")
            except KeyError:
                problems.append(f"missing from zip: {rel}")
        extra = [n for n in z.namelist()
                 if n != MANIFEST_NAME and n not in manifest["files"]]
        problems.extend(f"unexpected file: {n}" for n in extra)
    return {"ok": not problems, "problems": problems,
            "n_files": len(manifest["files"]),
            "dataset_hash": manifest["dataset_hash"],
            "created_utc": manifest["created_utc"]}


def restore(backup_path: Path, state_dir: Path) -> None:
    """Verify first, then replace the state directory. Any problem
    aborts BEFORE the live state is touched (fail closed)."""
    report = verify(backup_path)
    if not report["ok"]:
        raise SystemExit("backup failed verification — refusing to "
                         f"restore: {report['problems']}")
    live_hash = _dataset_hash_snapshot()
    if "XAUUSD" in report["dataset_hash"] and \
            live_hash.get("XAUUSD") != report["dataset_hash"]["XAUUSD"]:
        raise SystemExit("dataset hash changed since the backup — "
                         "refusing to restore state onto different data")
    state_dir = Path(state_dir)
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)
    with zipfile.ZipFile(backup_path) as z:
        for rel in manifest_files(z):
            target = state_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(rel))
    print(f"restored {report['n_files']} files into {state_dir}")


def manifest_files(z: zipfile.ZipFile) -> list[str]:
    return [n for n in z.namelist() if n != MANIFEST_NAME]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["backup", "restore", "verify"])
    parser.add_argument("path", nargs="?", help="backup zip (restore/verify)")
    parser.add_argument("--state-dir",
                        default=str(Path(__file__).resolve().parent.parent
                                    / "state" / "dashboard"))
    parser.add_argument("--out", default=None, help="backup output path")
    args = parser.parse_args(argv)
    if args.command == "backup":
        out = backup(Path(args.state_dir),
                     Path(args.out) if args.out else None)
        print(f"backup written: {out}")
    elif args.command == "verify":
        if not args.path:
            parser.error("verify needs a backup path")
        print(json.dumps(verify(Path(args.path)), indent=1))
    else:
        if not args.path:
            parser.error("restore needs a backup path")
        restore(Path(args.path), Path(args.state_dir))


if __name__ == "__main__":
    main()
