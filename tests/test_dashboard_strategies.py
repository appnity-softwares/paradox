"""The three dashboard-visible strategies added with the multi-symbol UI:
DonchianBreakout, TripleMAAlignment, KeltnerTrend. Known expected signals
on fixed synthetic datasets + validation guards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.base import StrategyConfig
from strategies.library import DonchianBreakout, KeltnerTrend, TripleMAAlignment


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


def test_donchian_long_on_breakout_short_on_breakdown():
    flat = [100.0] * 30
    up = list(np.linspace(100, 110, 20))     # breaks the 20-bar high
    down = list(np.linspace(110, 95, 25))    # then breaks the low
    sig = DonchianBreakout(StrategyConfig(), entry_len=20).generate_signals(
        ohlc(flat + up + down)
    )
    sig = sig.to_numpy(dtype=float)
    assert sig[-1] == -1.0                    # end of the breakdown leg
    assert (sig == 1.0).any()                 # went long during the breakout


def test_donchian_flat_market_stays_flat():
    sig = DonchianBreakout(StrategyConfig(), entry_len=20).generate_signals(
        ohlc([100.0 + 0.01 * (i % 3) for i in range(120)])
    )
    assert (sig.to_numpy(dtype=float) == 0.0).all()


def test_donchian_rejects_bad_params():
    with pytest.raises(ValueError):
        DonchianBreakout(StrategyConfig(), entry_len=2)
    with pytest.raises(ValueError):
        DonchianBreakout(StrategyConfig(), buffer_atr=-0.1)


def test_triple_ma_long_in_uptrend_short_in_downtrend():
    up = list(np.linspace(100, 130, 260))
    down = list(np.linspace(130, 100, 260))
    sig = TripleMAAlignment(StrategyConfig(), fast=9, mid=21, slow=55).generate_signals(
        ohlc(up + down)
    )
    sig = sig.to_numpy(dtype=float)
    assert sig[200] == 1.0                    # stacked alignment mid-uptrend
    assert sig[-1] == -1.0                    # stacked reverse mid-downtrend


def test_triple_ma_rejects_inverted_spans():
    with pytest.raises(ValueError):
        TripleMAAlignment(StrategyConfig(), fast=50, mid=21, slow=55)


def test_keltner_regime_follows_band_breaks():
    flat = [100.0] * 40
    up = list(np.linspace(100, 115, 40))      # blows through the upper band
    down = list(np.linspace(115, 90, 40))     # then the lower band
    sig = KeltnerTrend(StrategyConfig(), length=20, mult=2.0).generate_signals(
        ohlc(flat + up + down)
    )
    sig = sig.to_numpy(dtype=float)
    assert sig[-1] == -1.0
    assert (sig == 1.0).any()


def test_keltner_rejects_bad_params():
    with pytest.raises(ValueError):
        KeltnerTrend(StrategyConfig(), length=2)
    with pytest.raises(ValueError):
        KeltnerTrend(StrategyConfig(), mult=0)


def test_new_keys_registered():
    from strategies import ALL_STRATEGIES
    for key in ("donchian_breakout", "triple_ma", "keltner_trend"):
        assert key in ALL_STRATEGIES
        assert ALL_STRATEGIES[key].key == key
