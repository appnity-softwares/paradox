"""Admin panel configuration: UI routing and AI provider settings.

Stored as a small JSON file next to the dashboard package so the running
server can re-read it on every request without a restart. Env vars always
win over the file for secrets (fail closed, explicit beats implicit).
"""
from __future__ import annotations

import json
import os
from typing import Any

CONFIG_FILENAME = "admin_config.json"

VALID_PAGES = ("terminal", "client", "legacy")

DEFAULTS: dict[str, Any] = {
    "default_page": "terminal",
    "gemini_api_key": "",
}


def config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    try:
        with open(config_path(), "r", encoding="utf-8") as fh:
            stored = json.load(fh)
        if isinstance(stored, dict):
            cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    if cfg["default_page"] not in VALID_PAGES:
        cfg["default_page"] = "terminal"
    return cfg


def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    for key, value in patch.items():
        if key in DEFAULTS:
            cfg[key] = value
    if cfg["default_page"] not in VALID_PAGES:
        raise ValueError("invalid default_page")
    tmp = config_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    os.replace(tmp, config_path())
    return cfg
