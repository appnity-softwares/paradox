"""Shared test fixtures and markers.

`requires_dataset` marks tests that read the REAL ingested XAUUSD tick
store. On a machine (or CI runner) without the dataset those tests are
SKIPPED — reported as skipped, never silently passed — because
inventing data for them would violate the no-synthetic-data rule.
"""
from __future__ import annotations

import pytest


def dataset_available() -> bool:
    try:
        from core.marketdata.store import TICK_ROOT
        root = TICK_ROOT / "XAUUSD"
        return root.exists() and any(root.glob("*/*/ticks.parquet"))
    except Exception:
        return False


requires_dataset = pytest.mark.skipif(
    not dataset_available(),
    reason="real XAUUSD tick dataset not present on this machine")
