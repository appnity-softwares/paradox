"""SMC strategies pinned to docs/smc_definitions.md on hand-labelled bars.

Each test builds a synthetic bar sequence whose correct signal is known by
construction from the written definition, then asserts the implementation
reproduces it. The document is the specification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.base import StrategyConfig
from strategies.library import BreakOfStructure, FairValueGap, LiquiditySweep, SMCOrderBlock


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """rows of (open, high, low, close) -> 1-min OHLC frame."""
    n = len(rows)
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 100.0,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"),
    )


def flat(price: float, count: int) -> list[tuple[float, float, float, float]]:
    return [(price, price + 0.1, price - 0.1, price) for _ in range(count)]


# ---------------------------------------------------------------------------
# Fair Value Gap
# ---------------------------------------------------------------------------


def test_fvg_bullish_gap_detected_and_fires_on_mitigation():
    # 40 flat bars at 2000 (past the 30-bar warmup; ATR ~0.2, min gap
    # = 0.4*ATR ~ 0.08), then a 3-bar gap: bar i-2 high 2000.1, middle up
    # candle, bar i low 2001.5.
    rows = flat(2000.0, 40)
    rows += [(2000.0, 2000.1, 1999.9, 2000.0),   # i-2
             (2000.0, 2001.0, 2000.0, 2001.0),   # i-1 up candle
             (2001.0, 2001.6, 2001.5, 2001.5)]   # i: low 2001.5 > high[i-2]+gap
    rows += flat(2001.5, 3)
    # Mitigation: a bar whose low dips into the zone [2000.1, 2001.5]
    rows += [(2001.5, 2001.6, 2000.8, 2001.4)]
    df = bars(rows)
    sig = FairValueGap(StrategyConfig(), min_gap_atr=0.4, max_age=25).generate_signals(df)
    # Warmup suppresses the first 30 bars; the gap fires at/after mitigation.
    assert sig.iloc[-1] == 1.0
    assert (sig.iloc[:30] == 0.0).all()


def test_fvg_invalidated_when_close_trades_through_zone():
    rows = flat(2000.0, 40)
    rows += [(2000.0, 2000.1, 1999.9, 2000.0),
             (2000.0, 2001.0, 2000.0, 2001.0),
             (2001.0, 2001.6, 2001.5, 2001.5)]
    # Close below the gap bottom -> invalidated, never fires.
    rows += flat(1999.5, 10)
    df = bars(rows)
    sig = FairValueGap(StrategyConfig(), min_gap_atr=0.4, max_age=25).generate_signals(df)
    assert (sig == 0.0).all()


# ---------------------------------------------------------------------------
# Order Block
# ---------------------------------------------------------------------------


def test_order_block_bullish_retest_fires_long():
    # Sideways base, then an impulse that closes above the swing high with a
    # move > 1.5 ATR, leaving a down candle inside bars i-1..i-3.
    rows = flat(2000.0, 25)
    rows += [(2000.0, 2000.2, 1999.0, 1999.2),   # down candle (the future OB)
             (1999.2, 2000.0, 1999.0, 1999.8),
             (1999.8, 2003.5, 1999.8, 2003.4)]   # impulse: close > swing high
    rows += flat(2003.4, 3)
    # Retest: price dips back into the OB zone [1999.0, 2000.0]
    rows += [(2003.4, 2003.5, 1999.8, 2000.5)]
    df = bars(rows)
    sig = SMCOrderBlock(StrategyConfig(), swing_len=5, impulse_mult=1.5).generate_signals(df)
    assert sig.iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# Liquidity Sweep
# ---------------------------------------------------------------------------


def test_liquidity_sweep_bearish_stop_hunt_fires_short_then_expires():
    rows = flat(2000.0, 25)
    # Sweep bar: high pierces the 20-bar high by the buffer, closes back below.
    rows += [(2000.0, 2000.6, 1999.9, 1999.9)]
    df = bars(rows)
    strat = LiquiditySweep(StrategyConfig(), lookback=20, sweep_buffer=0.2, hold_bars=4)
    sig = strat.generate_signals(df)
    assert sig.iloc[-1] == -1.0
    # After hold_bars the state expires back to 0.
    df2 = bars(rows + flat(1999.9, 5))
    sig2 = strat.generate_signals(df2)
    assert sig2.iloc[-1] == 0.0


def test_liquidity_sweep_bullish_sweep_fires_long():
    rows = flat(2000.0, 25)
    rows += [(2000.0, 2000.1, 1999.4, 2000.1)]  # low pierces, close back above
    df = bars(rows)
    sig = LiquiditySweep(StrategyConfig(), lookback=20, sweep_buffer=0.2, hold_bars=4).generate_signals(df)
    assert sig.iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# Break of Structure
# ---------------------------------------------------------------------------


def test_bos_breaks_confirmed_swing_high_upward():
    # Build a clear swing high at bar 10 (peak 2010), flat either side, then a
    # close above 2010 after the swing is confirmed (bar 10 + swing_len).
    rows = flat(2000.0, 10)
    rows += [(2000.0, 2010.0, 1999.9, 2009.0)]   # swing high bar (peak)
    rows += [(2009.0, 2009.5, 2000.0, 2005.0) for _ in range(4)]  # right side lower
    rows += flat(2005.0, 5)
    rows += [(2005.0, 2011.0, 2004.9, 2010.5)]   # close breaks the swing high
    df = bars(rows)
    sig = BreakOfStructure(StrategyConfig(), swing_len=5).generate_signals(df)
    assert sig.iloc[-1] == 1.0
    # Before the break the state is flat/0 (no false early signal).
    assert (sig.iloc[:15] == 0.0).all()


def test_bos_breaks_confirmed_swing_low_downward():
    rows = flat(2000.0, 10)
    rows += [(2000.0, 2000.1, 1990.0, 1991.0)]   # swing low bar
    rows += [(1991.0, 1995.0, 1990.5, 1994.0) for _ in range(4)]
    rows += flat(1994.0, 5)
    rows += [(1994.0, 1994.5, 1989.0, 1989.5)]   # close breaks the swing low
    df = bars(rows)
    sig = BreakOfStructure(StrategyConfig(), swing_len=5).generate_signals(df)
    assert sig.iloc[-1] == -1.0


def test_bos_swing_level_consumed_after_single_break():
    rows = flat(2000.0, 10)
    rows += [(2000.0, 2010.0, 1999.9, 2009.0)]
    rows += [(2009.0, 2009.5, 2000.0, 2005.0) for _ in range(4)]
    rows += flat(2005.0, 5)
    rows += [(2005.0, 2011.0, 2004.9, 2010.5)]   # first break -> +1
    rows += flat(2010.5, 3)
    rows += [(2010.5, 2011.2, 2010.0, 2011.0)]   # higher close, no new swing: no change
    df = bars(rows)
    sig = BreakOfStructure(StrategyConfig(), swing_len=5).generate_signals(df)
    assert sig.iloc[-1] == 1.0  # state persists; level was consumed, not re-fired


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"min_gap_atr": 0.0}, {"max_age": 2},
])
def test_fvg_rejects_invalid_params(kwargs):
    with pytest.raises(ValueError):
        FairValueGap(StrategyConfig(), **kwargs)


def test_bos_rejects_tiny_swing():
    with pytest.raises(ValueError):
        BreakOfStructure(StrategyConfig(), swing_len=1)
