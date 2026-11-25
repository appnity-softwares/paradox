"""Phase 1C: new strategies with known expected signals on fixed datasets,
plus the legacy adapter bridging onto the shared interface (Part 6.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.tick_engine import BacktestConfig, TickEngine
from strategies.adapter import LegacyAdapter
from strategies.base import StrategyConfig
from strategies.catalog import catalog
from strategies.library import (
    ADXTrend,
    EMACross,
    PSARTrend,
    StochasticCross,
    ZScoreReversion,
)


def ohlc(closes: list[float], spread: float = 0.5) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + spread / 2,
            "low": closes - spread / 2,
            "close": closes,
            "volume": 100.0,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"),
    )


# ---------------------------------------------------------------------------
# PSARTrend
# ---------------------------------------------------------------------------


def test_psar_long_in_uptrend_short_in_downtrend():
    up = list(np.linspace(2000, 2020, 60))
    down = list(np.linspace(2020, 1995, 60))
    sig = PSARTrend(StrategyConfig()).generate_signals(ohlc(up + down))
    assert sig.iloc[55] == 1.0    # established uptrend
    assert sig.iloc[-1] == -1.0   # established downtrend


def test_psar_rejects_bad_acceleration_params():
    with pytest.raises(ValueError):
        PSARTrend(StrategyConfig(), af_start=0.3, af_step=0.02)


# ---------------------------------------------------------------------------
# StochasticCross
# ---------------------------------------------------------------------------


def test_stochastic_fires_long_on_oversold_bullish_cross():
    # Sell-off into oversold, then a turn up: %K crosses %D below 25.
    # Persistent-state semantics: the first non-zero signal after the turn
    # must be the long; the later overbought cross-down legitimately flips
    # the state short again.
    closes = list(np.linspace(2000, 1980, 30))          # down leg
    closes += list(np.linspace(1980, 1984, 8))          # turn up
    closes += [1984.0] * 20
    strat = StochasticCross(StrategyConfig(), length=14, oversold=25.0, overbought=75.0)
    out = strat.generate_signals(ohlc(closes)).to_numpy()
    nonzero = np.nonzero(out)[0]
    assert len(nonzero) and out[nonzero[0]] == 1.0
    assert out[-1] == -1.0  # overbought %K/%D cross-down after the recovery


def test_stochastic_stays_flat_in_mid_range_chop():
    rng = np.random.default_rng(3)
    closes = list(2000 + rng.normal(0, 0.2, 80))
    sig = StochasticCross(StrategyConfig()).generate_signals(ohlc(closes))
    assert (sig.iloc[sig.warmup if hasattr(sig, "warmup") else 0:] == 0.0).all() or sig.iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# ZScoreReversion
# ---------------------------------------------------------------------------


def test_zscore_longs_the_stretch_down_and_exits_at_the_mean():
    base = [2000.0] * 40
    dip = list(np.linspace(2000, 1980, 6))    # z << -entry_z
    back = list(np.linspace(1980, 2000, 30))  # mean reversion
    sig = ZScoreReversion(StrategyConfig(), length=30, entry_z=2.0, exit_z=0.5)
    out = sig.generate_signals(ohlc(base + dip + back))
    series = out.to_numpy()
    assert 1.0 in series            # went long on the dip
    assert series[-1] == 0.0        # back at the mean -> flat


def test_zscore_shorts_the_stretch_up():
    base = [2000.0] * 40
    spike = list(np.linspace(2000, 2025, 6))
    sig = ZScoreReversion(StrategyConfig(), length=30, entry_z=2.0, exit_z=0.5)
    out = sig.generate_signals(ohlc(base + spike))
    assert out.iloc[-1] == -1.0


# ---------------------------------------------------------------------------
# ADXTrend
# ---------------------------------------------------------------------------


def test_adx_trend_follows_strong_trend_and_goes_flat_in_chop():
    strong = list(np.linspace(2000, 2060, 120))
    sig = ADXTrend(StrategyConfig(), fast=10, slow=40, adx_threshold=20.0)
    out = sig.generate_signals(ohlc(strong))
    assert out.iloc[-1] == 1.0

    rng = np.random.default_rng(5)
    chop = list(2000 + rng.normal(0, 0.15, 160))
    out2 = sig.generate_signals(ohlc(chop))
    assert out2.iloc[-1] == 0.0     # weak trend filtered out


# ---------------------------------------------------------------------------
# Catalog (Part 6.5 metadata contract)
# ---------------------------------------------------------------------------


def test_catalog_covers_fifteen_strategies_with_metadata():
    cat = catalog()
    assert len(cat) >= 15
    for key, meta in cat.items():
        assert meta["category"], key
        assert meta["code_hash"], key
        assert isinstance(meta["params"], dict), key


def test_param_grid_keys_all_exist_in_catalog():
    from core.optimize import PARAM_GRID

    cat = catalog()
    for key in PARAM_GRID:
        assert key in cat, f"PARAM_GRID key {key} has no strategy"


# ---------------------------------------------------------------------------
# Legacy adapter on the tick engine
# ---------------------------------------------------------------------------


def _tick_frame(closes: np.ndarray) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "time_utc_ms": np.arange(n) * 60_000,
        "bid": closes - 0.2,
        "ask": closes + 0.2,
        "bid_volume": 1.0,
        "ask_volume": 1.0,
        "source": "synthetic-test",
    })


def test_legacy_adapter_runs_ema_cross_on_tick_engine():
    rng = np.random.default_rng(11)
    steps = rng.normal(0, 0.4, 3000).cumsum()
    closes = 2000 + steps
    ticks = _tick_frame(closes)

    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0, risk_pct=1.0)
    adapter = LegacyAdapter(EMACross(cfg, fast=20, slow=80), cfg)

    # Record exactly what the legacy strategy emitted per bar close.
    recorded: list[float] = []
    orig = adapter.legacy.generate_signals

    def wrapped(df):
        s = orig(df)
        recorded.append(float(s.iloc[-1]))
        return s

    adapter.legacy.generate_signals = wrapped

    bt_cfg = BacktestConfig(symbol="XAUUSD", dataset_hash="adapter-test")
    result = TickEngine().run(adapter, ticks, bt_cfg)
    assert result["valid"], result
    assert result["total_trades"] > 0
    assert recorded

    # The recorded signals must match an independent full-history EMA-sign
    # computation. The adapter recomputes on a bounded 1500-bar window, so
    # crossover timing can jitter by a bar or two; allow < 2% mismatches.
    from core.indicators import ema

    close_s = pd.Series(closes)
    expected = np.sign(ema(close_s, 20) - ema(close_s, 80)).to_numpy()
    first_bar = adapter.warmup_bars - 1
    exp = expected[first_bar: first_bar + len(recorded)]
    mismatches = int(np.sum(np.sign(np.asarray(recorded)) != np.sign(exp)))
    assert mismatches / len(recorded) < 0.02, (
        f"{mismatches}/{len(recorded)} legacy signals diverge from full-history EMA"
    )


def test_legacy_adapter_respects_warmup():
    cfg = StrategyConfig()
    adapter = LegacyAdapter(EMACross(cfg, fast=20, slow=50), cfg)
    assert adapter.warmup_bars == 61  # legacy warmup 50 + 1
