"""Strategy catalog: metadata, parameter schema, version, code hash (6.5).

Part 6.5 requires every strategy to carry metadata, a parameter schema with
ranges, a version and a code hash, so any result can be tied to the exact
code that produced it. The hash is sha256 over the strategy class source.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Any

from strategies import library

CATEGORY = {
    "ema_cross": "trend",
    "macd_cross": "trend",
    "supertrend": "trend",
    "psar_trend": "trend",
    "adx_trend": "trend",
    "rsi_reversal": "momentum",
    "stoch_cross": "momentum",
    "bollinger": "mean_reversion",
    "zscore_reversion": "mean_reversion",
    "vwap_ema": "trend",
    "session_breakout": "session",
    "smc_fvg": "smc",
    "smc_order_block": "smc",
    "smc_liquidity_sweep": "smc",
    "smc_bos": "smc",
}


def code_hash(cls: type) -> str:
    """sha256 over the class source — changes whenever the logic changes."""
    src = inspect.getsource(cls)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def catalog() -> dict[str, dict[str, Any]]:
    """Metadata for every strategy in the library, keyed by ``key``."""
    out: dict[str, dict[str, Any]] = {}
    for name in dir(library):
        obj = getattr(library, name)
        if not inspect.isclass(obj) or not hasattr(obj, "generate_signals"):
            continue
        key = getattr(obj, "key", None)
        if key is None:
            continue
        schema = getattr(obj, "PARAM_SCHEMA", {})
        out[key] = {
            "class": obj.__name__,
            "category": CATEGORY.get(key, "uncategorised"),
            "params": {p: {"min": lo, "max": hi} for p, (lo, hi) in schema.items()},
            "code_hash": code_hash(obj),
        }
    return out
