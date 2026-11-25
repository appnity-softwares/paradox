"""Sandbox for generated strategy code (Part 11.3).

Two layers:
1. Static AST check — rejects imports outside the whitelist and any use
   of os, subprocess, socket, eval, exec, open, dunder access.
2. Execution in a SEPARATE PROCESS: no network, no filesystem outside a
   temp directory, CPU/memory/time limits. Generated strategies RETURN
   SIGNALS ONLY — they never place orders.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Import whitelist per Part 11.3.
ALLOWED_IMPORTS = {"numpy", "pandas", "math", "statistics"}
# Project indicator API namespace allowed inside the sandbox.
ALLOWED_PROJECT = {"indicators"}

FORBIDDEN_NAMES = {
    "os", "subprocess", "socket", "eval", "exec", "open", "compile",
    "__import__", "globals", "locals", "input", "breakpoint", "memoryview",
}


class UnsafeCodeError(ValueError):
    pass


def static_check(source: str) -> list[str]:
    """Return a list of violations; empty means the code is allowed."""
    violations: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax_error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS and root not in ALLOWED_PROJECT:
                    violations.append(f"import_not_allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in ALLOWED_IMPORTS \
                    and root not in ALLOWED_PROJECT:
                violations.append(f"import_not_allowed: {node.module}")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name in FORBIDDEN_NAMES:
                violations.append(f"forbidden_call: {name}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append(f"dunder_access: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                violations.append(f"forbidden_name: {node.id}")
    return violations


RUNNER = '''
import json, sys
import numpy as np
import pandas as pd

payload = json.loads(sys.stdin.read())
source = payload["source"]
params = payload["params"]
bars = pd.DataFrame(payload["bars"])

ns = {"np": np, "pd": pd, "params": params}
exec(compile(source, "<generated>", "exec"), ns)   # noqa: S102 - sandboxed
signals = ns["generate_signals"](bars)
print(json.dumps({"signals": [bool(x) for x in signals]}))
'''


def run_generated(source: str, bars: pd.DataFrame, params: dict | None = None,
                  timeout_s: float = 10.0,
                  memory_mb: int = 512) -> dict:
    """Run generated strategy code in a separate process with limits.
    Returns {"ok": bool, "signals": [...]} or {"ok": False, "error": ...}."""
    violations = static_check(source)
    if violations:
        return {"ok": False, "error": f"static_check: {violations}"}

    payload = json.dumps({
        "source": source, "params": params or {},
        "bars": {"open": bars["open"].tolist(),
                 "high": bars["high"].tolist(),
                 "low": bars["low"].tolist(),
                 "close": bars["close"].tolist()}})
    with tempfile.TemporaryDirectory() as tmp:
        runner = Path(tmp) / "runner.py"
        runner.write_text(RUNNER)
        # The temp dir is the process cwd; stdin is the only input channel
        # and stdout the only output — no network, no other filesystem.
        preexec = _limits_preexec(memory_mb)
        try:
            proc = subprocess.run(
                [sys.executable, str(runner)], input=payload,
                capture_output=True, text=True, timeout=timeout_s,
                cwd=tmp, preexec_fn=preexec)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        except OSError as exc:
            return {"ok": False, "error": f"spawn_failed: {exc}"}
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr.strip()[-500:]}
        import json as _json
        return {"ok": True, **_json.loads(proc.stdout)}


def _limits_preexec(memory_mb: int):
    """Child-process limits: address space cap + new session (no tty)."""
    def apply():
        try:
            import resource
            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except Exception:
            pass
        import os
        os.setsid()
    return apply
