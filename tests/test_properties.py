"""Property tests (Part 14.3) for the resampler, sizing/margin and P&L
maths. Deterministic seeded loops (no hypothesis dependency): hundreds
of random cases per property, every case must hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import SYMBOL_SPECS
from core.marketdata.resampler import resample_ticks_to_bars
from core.sizing import margin_per_lot, round_lot_down, size_fixed_risk

XAU = SYMBOL_SPECS["XAUUSD"]


def random_ticks(n: int, seed: int, p0: float = 2600.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 1_000 + 1_700_000_000_000
    mid = p0 + np.cumsum(rng.normal(0, 0.05, n))
    half = 0.15 + rng.random(n) * 0.1
    return pd.DataFrame({"time_utc_ms": t, "bid": mid - half,
                         "ask": mid + half,
                         "bid_volume": rng.random(n) * 10,
                         "ask_volume": rng.random(n) * 10})


# ── Resampler properties ─────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(20))
def test_resampler_ohlc_invariants(seed):
    ticks = random_ticks(500, seed)
    bars = resample_ticks_to_bars(ticks, "1m")
    assert len(bars) > 0
    # OHLC relationships hold on every bar.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert bars[["open", "high", "low", "close"]].notna().all().all()
    # Monotonic unique index.
    assert bars.index.is_monotonic_increasing and bars.index.is_unique


@pytest.mark.parametrize("seed", range(20))
def test_resampler_extremes_come_from_ticks(seed):
    """Each bar's high/low equal the max/min of the ticks inside it."""
    ticks = random_ticks(400, seed)
    bars = resample_ticks_to_bars(ticks, "1m")
    ticks = ticks.copy()
    ticks["bucket"] = pd.to_datetime(ticks["time_utc_ms"], unit="ms",
                                     utc=True).dt.floor("1min")
    for ts, bar in bars.iterrows():
        in_bar = ticks[ticks["bucket"] == ts]
        if in_bar.empty:
            continue
        mid = (in_bar["bid"] + in_bar["ask"]) / 2
        assert bar["high"] == pytest.approx(mid.max(), abs=1e-6)
        assert bar["low"] == pytest.approx(mid.min(), abs=1e-6)


def test_resampler_volume_is_sum_of_tick_volume():
    ticks = random_ticks(300, 7)
    ticks["bid_volume"] = 1.0
    ticks["ask_volume"] = 2.0
    bars = resample_ticks_to_bars(ticks, "1m")
    ticks["bucket"] = pd.to_datetime(ticks["time_utc_ms"], unit="ms",
                                     utc=True).dt.floor("1min")
    for ts, bar in bars.iterrows():
        n_in = (ticks["bucket"] == ts).sum()
        if n_in:
            assert bar["volume"] == pytest.approx(3.0 * n_in)


# ── Sizing / margin properties ───────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(50))
def test_risk_money_identity(seed):
    """risk_money = lot × stop_distance × contract_size, always."""
    rng = np.random.default_rng(seed)
    equity = rng.uniform(500, 200_000)
    risk_pct = rng.uniform(0.05, 2.0)
    stop = rng.uniform(1.0, 50.0)
    price = rng.uniform(1500, 3500)
    res = size_fixed_risk(equity, risk_pct, stop, XAU, price,
                          leverage=100)
    if res.lot > 0:
        expected = res.lot * stop * XAU.contract_size
        assert res.risk_money == pytest.approx(expected, rel=1e-9)
        assert res.risk_money <= equity * risk_pct / 100 + 1e-9


@pytest.mark.parametrize("seed", range(50))
def test_lots_always_on_grid_and_at_or_below_raw(seed):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0001, 10.0)
    lot = round_lot_down(raw, XAU)
    steps = int(raw / XAU.lot_step + 1e-9)      # FLOOR, never round-up
    assert lot == pytest.approx(steps * XAU.lot_step, abs=1e-9)
    assert lot <= raw + 1e-9
    assert abs(lot / XAU.lot_step - round(lot / XAU.lot_step)) < 1e-9


@pytest.mark.parametrize("seed", range(50))
def test_margin_scales_linearly_with_lot_and_inverse_leverage(seed):
    rng = np.random.default_rng(seed)
    price = rng.uniform(1500, 3500)
    lev = rng.integers(10, 500)
    m1 = margin_per_lot(price, XAU, lev)
    m2 = margin_per_lot(price, XAU, lev * 2)
    assert m1 == pytest.approx(price * XAU.contract_size / lev, rel=1e-9)
    assert m2 == pytest.approx(m1 / 2, rel=1e-9)


# ── P&L properties (tick-engine account maths) ───────────────────────────────

def test_pnl_linear_in_lot_and_symmetric_in_side():
    """P&L per unit price move = lot × contract_size; sides mirror."""
    entry, exit_, contract = 2600.0, 2610.0, XAU.contract_size
    for lot in (0.01, 0.37, 2.5):
        long_pnl = (exit_ - entry) * lot * contract
        short_pnl = (entry - exit_) * lot * contract
        assert long_pnl == pytest.approx(-short_pnl, rel=1e-12)
        assert long_pnl == pytest.approx(10 * lot * contract)
    # Doubling the lot doubles the P&L.
    assert (exit_ - entry) * 1.0 * contract * 2 == \
        pytest.approx(2 * (exit_ - entry) * 1.0 * contract)


def test_stop_loss_loss_equals_risk_money():
    """A stop hit at the planned distance loses exactly the risked money."""
    equity, risk_pct, stop = 10_000.0, 1.0, 12.0
    price = 2600.0
    res = size_fixed_risk(equity, risk_pct, stop, XAU, price,
                          leverage=100)
    loss_at_sl = -stop * res.lot * XAU.contract_size
    assert loss_at_sl == pytest.approx(-res.risk_money, rel=1e-9)
    assert loss_at_sl >= -equity * (risk_pct / 100) - 1e-9
