"""Minimal MCP client for the local MetaTrader 5 / MetaEditor servers.

MetaQuotes ships an MCP endpoint inside the terminal and the editor. On macOS
the terminal runs under Wine, and the official `MetaTrader5` Python package is
Windows-only — this client is the supported way to drive the Mac terminal.

Credentials are read from the terminal's own `config/assistant.ini` at call
time. They are never logged, never returned, and never sent to an LLM.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

MT5_DIR = Path(os.path.expanduser(
    "~/Library/Application Support/net.metaquotes.wine.metatrader5"
    "/drive_c/Program Files/MetaTrader 5"
))
ASSISTANT_INI = MT5_DIR / "config" / "assistant.ini"


def _read_ini() -> str:
    raw = ASSISTANT_INI.read_bytes()
    for encoding in ("utf-16-le", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Cannot decode assistant.ini")


def endpoint(section: str) -> tuple[str, str]:
    """Return (url, api_key) for 'MCP.MetaEditor' or 'MCP.MetaTrader'."""
    text = _read_ini()
    block = re.search(rf"\[{re.escape(section)}\](.*?)(?:\n\[|\Z)", text, re.S)
    if not block:
        raise RuntimeError(f"{section} not configured in assistant.ini")
    body = block.group(1)
    url = re.search(r"Endpoint\s*=\s*(\S+)", body)
    key = re.search(r"ApiKey\s*=\s*(\S+)", body)
    if not url or not key:
        raise RuntimeError(f"{section} is missing Endpoint or ApiKey")
    return url.group(1), key.group(1)


class MCPClient:
    """Streamable-HTTP MCP client (initialize -> tools/list -> tools/call)."""

    def __init__(self, section: str = "MCP.MetaTrader", timeout: float = 120.0):
        self.url, self._key = endpoint(section)
        self.timeout = timeout
        self.session: str | None = None
        self._id = 0

    def _post(self, payload: dict, notify: bool = False):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._key}",
        }
        if self.session:
            headers["mcp-session-id"] = self.session
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            sid = response.headers.get("mcp-session-id")
            if sid:
                self.session = sid
            body = response.read().decode()
        if notify or not body.strip():
            return None
        # Server may answer as SSE frames.
        if body.lstrip().startswith("event:") or "\ndata:" in body or body.startswith("data:"):
            for line in body.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
            return None
        return json.loads(body)

    def _call(self, method: str, params: dict | None = None):
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            payload["params"] = params
        result = self._post(payload)
        if result and "error" in result:
            raise RuntimeError(f"MCP {method} failed: {result['error']}")
        return (result or {}).get("result")

    def connect(self):
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "quant-platform", "version": "0.4"},
        })
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, notify=True)
        return self

    def tools(self) -> list[dict]:
        return (self._call("tools/list") or {}).get("tools", [])

    def call(self, name: str, arguments: dict | None = None):
        return self._call("tools/call", {"name": name, "arguments": arguments or {}})


def available(section: str = "MCP.MetaTrader") -> bool:
    try:
        client = MCPClient(section, timeout=10.0).connect()
        return bool(client.tools())
    except Exception:
        return False
