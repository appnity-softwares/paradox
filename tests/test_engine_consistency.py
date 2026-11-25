"""Tick engine vs bar engine consistency (Master Prompt 6.4 / 6.8.3).

A trivial EMA-cross strategy runs through both engines on the same tick
series (the bar engine receives the mid bars resampled from those ticks).
The engines use different fill models (next-bar open vs first tick after
latency), so exact equality is impossible; the Master Prompt requires
agreement within a documented tolerance:

- trade count within +/-1 (end-of-data handling differs by construction),
- identical side sequence for the matched trades,
- entry prices within $1.00 (spread + slippage + intra-minute drift).

Synthetic ticks are used here only; unit tests may synthesize, the validated
path may not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest import run_backtest
from core.marketdata.resampler import resample_ticks_to_bars
from core.strategy_api import (
    SIDE_CLOSE,
    SIDE_LONG,
    SIDE_SHORT,
    Context,
    Signal,
    Strategy,
)
from core.tick_engine import BacktestConfig, TickEngine


MIN = 60_000


def build_tick_series() -> pd.DataFrame:
    """Deterministic 3-hour tick series: trend up, chop, trend down."""
    rng = np.random.default_rng(7)
    rows = []
    price = 2000.0
    t = 0
    for minute in range(180):
        drift = 0.30 if minute < 60 else (-0.30 if minute >= 120 else 0.0)
        for _ in range(20):  # 20 ticks per minute
            price += drift / 20 + rng.normal(0, 0.05)
            spread = 0.40
            rows.append((t, round(price - spread / 2, 3), round(price + spread / 2, 3)))
            t += 3_000
    return pd.DataFrame({
        "time_utc_ms": [r[0] for r in rows],
        "bid": [r[1] for r in rows],
        "ask": [r[2] for r in rows],
        "bid_volume": 1.0,
        "ask_volume": 1.0,
        "source": "synthetic-test",
    })


def ema(values: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def compute_signals(bars: pd.DataFrame, fast: int = 10, slow: int = 30, warmup: int = 40) -> np.ndarray:
    sig = np.sign(ema(bars["close"].to_numpy(float), fast) - ema(bars["close"].to_numpy(float), slow))
    sig[:warmup] = 0.0
    return sig


class BarEngineStrategy:
    """Legacy bar-engine interface over the precomputed signal series."""

    def __init__(self, signals: np.ndarray):
        self.signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.signals[: len(df)], index=df.index)


class TickEngineAdapter(Strategy):
    """Same signal series emitted at bar closes through the new interface."""

    name = "ema_cross_consistency"
    warmup_bars = 0

    def __init__(self, signals: np.ndarray, sl_dist: float = 2.0, rr: float = 2.0):
        self.signals = signals
        self.sl_dist = sl_dist
        self.rr = rr
        self.bar_index = -1

    def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
        self.bar_index += 1
        if self.bar_index >= len(self.signals):
            return []
        sig = self.signals[self.bar_index]
        close = bar["close"]
        out: list[Signal] = []
        pos = ctx.position
        if pos.side == SIDE_LONG and sig <= 0:
            out.append(Signal(SIDE_CLOSE))
        elif pos.side == SIDE_SHORT and sig >= 0:
            out.append(Signal(SIDE_CLOSE))
        if pos.side == 0 and sig != 0:
            direction = 1 if sig > 0 else -1
            out.append(Signal(
                SIDE_LONG if direction == 1 else SIDE_SHORT,
                sl=close - direction * self.sl_dist,
                tp=close + direction * self.sl_dist * self.rr,
            ))
        return out


def test_tick_and_bar_engines_agree_on_trivial_strategy():
    ticks = build_tick_series()
    bars = resample_ticks_to_bars(ticks, "1m", price_col="mid")
    assert len(bars) >= 150

    signals = compute_signals(bars)

    # Bar engine: legacy interface, fixed SL, RR targets, 1% risk.
    from core.config import get_cost_model

    class Cfg:
        risk_pct = 1.0
        sl_method = "fixed"
        sl_atr_mult = 1.5
        sl_fixed = 2.0
        rr = 2.0
        session = "all"
        session_hours = None
        max_trades_per_day = 100
        daily_loss_limit_pct = 100.0

    _, bar_trades_df = run_backtest(
        bars, BarEngineStrategy(signals), Cfg(),
        start_equity=100_000.0,
        symbol="XAUUSD",
    )
    bar_trades = bar_trades_df.to_dict("records")

    # Tick engine: same signals through the shared interface.
    cfg = BacktestConfig(
        symbol="XAUUSD",
        initial_balance=100_000.0,
        risk_pct=1.0,
        dataset_hash="consistency-test",
    )
    result = TickEngine().run(TickEngineAdapter(signals), ticks, cfg)
    assert result["valid"], result

    tick_trades = result["trades"]
    # Tolerance 1: trade count within one (end-of-data liquidation differs).
    assert abs(len(tick_trades) - len(bar_trades)) <= 1, (
        f"tick={len(tick_trades)} bar={len(bar_trades)}"
    )

    # Tolerance 2: identical side sequence for the matched trades.
    n = min(len(tick_trades), len(bar_trades))
    tick_sides = [tr["side"] for tr in tick_trades[:n]]
    bar_sides = [tr["side"] for tr in bar_trades[:n]]
    assert tick_sides == bar_sides, (tick_sides, bar_sides)

    # Tolerance 3: entry prices within $1.00.
    for tt, bt in zip(tick_trades[:n], bar_trades[:n]):
        assert abs(tt["entry_price"] - bt["entry"]) <= 1.00, (
            f"entry mismatch: tick={tt['entry_price']} bar={bt['entry']}"
        )
