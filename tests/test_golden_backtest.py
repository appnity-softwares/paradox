"""Golden test for the bar backtest engine (Part 14.3).

Fixed REAL bars (an immutable slice of the ingested XAUUSD dataset) plus
a fixed config must produce the exact stored trade log. Any change in
engine behaviour — fills, costs, exits — shifts a price or P&L and fails.
Regenerate deliberately via REGEN_GOLDEN=1 after reviewing the diff.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from core.backtest import run_backtest
from core.config import COST_MODELS
from core.marketdata.resampler import resample_ticks_to_bars
from core.marketdata.store import read_range
from strategies.base import StrategyConfig
from strategies.library import EMACross


from tests.conftest import requires_dataset

pytestmark = requires_dataset

GOLDEN = Path(__file__).parent / "golden" / "bar_engine_golden.json"

# Fixed real-data window: 2025-01-02 to 2025-01-08 (UTC ms).
START_MS = 1735776000000
END_MS = 1736294400000


def load_bars() -> pd.DataFrame:
    ticks = read_range("XAUUSD", START_MS, END_MS)
    return resample_ticks_to_bars(ticks, "15m")


def trade_log(bars: pd.DataFrame) -> list[dict]:
    cfg = StrategyConfig(sl_method="atr", sl_atr_mult=1.5, rr=1.5,
                         session="london_ny", risk_pct=0.5)
    equity, trades = run_backtest(
        bars, EMACross(cfg, fast=20, slow=60), cfg,
        start_equity=10_000.0, symbol="XAUUSD")
    out = []
    for _, t in trades.iterrows():
        out.append({k: (round(float(v), 6) if isinstance(v, (int, float))
                        else str(v))
                    for k, v in t.items()})
    return out


def test_golden_trade_log_reproducible():
    bars = load_bars()
    assert len(bars) > 100                       # real data present
    current = trade_log(bars)
    assert current, "expected trades in the fixed window"
    if os.environ.get("REGEN_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(current, indent=1))
    golden = json.loads(GOLDEN.read_text())
    assert current == golden, (
        "bar-engine trade log changed — review the diff and regenerate "
        "the golden file deliberately if the change is intended")


def test_golden_is_from_real_data():
    """The golden window must be real ingested ticks, not synthetic."""
    ticks = read_range("XAUUSD", START_MS, END_MS)
    assert len(ticks) > 10_000
    assert ticks["bid"].min() > 0 and ticks["ask"].max() < 10_000
