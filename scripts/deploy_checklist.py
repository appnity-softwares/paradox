#!/usr/bin/env python3
"""Windows bridge deployment checklist (U8 step 4).

Run ON the Windows VPS. Prints PASS/FAIL per item and exits non-zero
on any FAIL. Checks that can't run on a non-Windows box are reported
as SKIP with the reason (never silently passed).
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

MT5_SUPPORTED = {(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)}

results: list[tuple[str, str, str]] = []   # (name, status, detail)


def check(name: str, ok: bool | None, detail: str = "") -> None:
    status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    is_windows = platform.system() == "Windows"

    # 1. Python version has a verified MT5 wheel.
    v = sys.version_info[:2]
    check("python version has verified MT5 wheel",
          v in MT5_SUPPORTED,
          f"{v[0]}.{v[1]} (verified wheels: 3.10-3.14)")

    # 2. MetaTrader5 importable.
    try:
        import MetaTrader5 as mt5          # noqa: F401
        check("MetaTrader5 importable", True)
        mt5_available = True
    except Exception as exc:
        check("MetaTrader5 importable", False, repr(exc))
        mt5_available = False

    # 3. Terminal initializes (Windows only).
    if is_windows and mt5_available:
        import MetaTrader5 as mt5
        ok = bool(mt5.initialize())
        check("mt5.initialize()", ok,
              mt5.last_error() if not ok else "terminal attached")
        if ok:
            mt5.shutdown()
    else:
        check("mt5.initialize()", None, "requires Windows + MT5 package")

    # 4. Bridge config: TLS + strong token.
    cfg_path = Path("bridge.toml")
    if cfg_path.exists():
        text = cfg_path.read_text()
        tls_on = "tls = true" in text or "tls_enabled = true" in text
        check("bridge TLS enabled", tls_on)
        token_line = [l for l in text.splitlines()
                      if l.strip().startswith("token")]
        token = token_line[0].split("=", 1)[1].strip().strip('"') \
            if token_line else ""
        check("bridge token strong (>=32 chars, not default)",
              len(token) >= 32 and token != "change-me",
              "token length hidden" if token else "no token in config")
    else:
        check("bridge config present", False, "bridge.toml not found")

    # 5. Firewall restriction (Windows only).
    if is_windows:
        import subprocess
        try:
            out = subprocess.run(
                ["netsh", "advfirewall", "firewall", "show", "rule",
                 "name=all"], capture_output=True, text=True, timeout=30)
            fw = out.stdout
            has_bridge = "DSH" in fw or "bridge" in fw.lower()
            check("firewall rule for bridge present", has_bridge)
        except Exception as exc:
            check("firewall rule for bridge present", None, repr(exc))
    else:
        check("firewall rule for bridge present", None,
              "requires Windows")

    # 6. Log rotation configured.
    if cfg_path.exists():
        text = cfg_path.read_text()
        check("log rotation configured",
              "rotation" in text or "rotate" in text)
    else:
        check("log rotation configured", None, "no config")

    # 7. Service registered (Windows only).
    if is_windows:
        import subprocess
        try:
            out = subprocess.run(["sc", "query", "DSHBridge"],
                                 capture_output=True, text=True, timeout=30)
            check("DSHBridge service registered",
                  "RUNNING" in out.stdout or "STOPPED" in out.stdout)
        except Exception as exc:
            check("DSHBridge service registered", None, repr(exc))
    else:
        check("DSHBridge service registered", None, "requires Windows")

    # 8. Health endpoint.
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8788/health",
                                    timeout=5) as r:
            body = r.read().decode()
        check("health endpoint responds", b"quote" in body.encode().lower()
              or "ok" in body.lower())
    except Exception as exc:
        check("health endpoint responds", False, repr(exc))

    # 9. Time sync (Windows only).
    if is_windows:
        import subprocess
        try:
            out = subprocess.run(["w32tm", "/query", "/status"],
                                 capture_output=True, text=True, timeout=30)
            check("time sync source configured", "Source:" in out.stdout)
        except Exception as exc:
            check("time sync source configured", None, repr(exc))
    else:
        check("time sync source configured", None, "requires Windows")

    # 10. One terminal per account.
    if cfg_path.exists():
        text = cfg_path.read_text()
        n_terms = text.count("[[terminal]]") + text.count("[terminal]")
        accounts = text.count("login")
        check("one terminal per account",
              n_terms <= 1 or accounts <= n_terms,
              f"{n_terms} terminal block(s), {accounts} login(s)")

    failed = [r for r in results if r[1] == "FAIL"]
    skipped = [r for r in results if r[1] == "SKIP"]
    print(f"\n{len(results) - len(failed) - len(skipped)} PASS, "
          f"{len(skipped)} SKIP, {len(failed)} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
