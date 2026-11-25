import numpy as np
import pandas as pd

from core.data import fetch_market_data
from core.backtest import run_backtest
from core.prop_firm_simulator import (
    PROP_FIRM_PROFILES,
    PropFirmChallengeSimulator,
    PropFirmProfile,
)
from strategies import (
    ALL_STRATEGIES,
    StrategyConfig,
    FairValueGap,
    SMCOrderBlock,
    LiquiditySweep,
)


def _sample_df():
    # Fetch real or fallback market data
    res = fetch_market_data("15m", 300)
    return res.data


def test_smc_strategies_in_registry():
    assert "smc_fvg" in ALL_STRATEGIES
    assert "smc_order_block" in ALL_STRATEGIES
    assert "smc_liquidity_sweep" in ALL_STRATEGIES


def test_fair_value_gap_signals():
    df = _sample_df()
    cfg = StrategyConfig(name="FVG_Test")
    strat = FairValueGap(cfg, min_gap_atr=0.3, max_age=20)
    sig = strat.generate_signals(df)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})
    assert (sig.iloc[: strat.warmup] == 0.0).all()


def test_smc_order_block_signals():
    df = _sample_df()
    cfg = StrategyConfig(name="OB_Test")
    strat = SMCOrderBlock(cfg, swing_len=5, impulse_mult=1.2, max_bars=25)
    sig = strat.generate_signals(df)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})
    assert (sig.iloc[: strat.warmup] == 0.0).all()


def test_liquidity_sweep_signals():
    df = _sample_df()
    cfg = StrategyConfig(name="Sweep_Test")
    strat = LiquiditySweep(cfg, lookback=15, sweep_buffer=0.1, hold_bars=3)
    sig = strat.generate_signals(df)
    assert len(sig) == len(df)
    assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})
    assert (sig.iloc[: strat.warmup] == 0.0).all()


def test_backtest_with_smc():
    df = _sample_df()
    cfg = StrategyConfig(name="FVG_BT", risk_pct=1.0, sl_method="atr", sl_atr_mult=1.5, rr=2.0)
    strat = FairValueGap(cfg)
    equity, trades = run_backtest(df, strat, cfg)
    assert len(equity) == len(df)
    assert isinstance(trades, pd.DataFrame)


def test_prop_firm_simulator():
    profile = PROP_FIRM_PROFILES["ftmo_100k"]
    simulator = PropFirmChallengeSimulator(profile)

    # Synthetic trades with positive expectancy
    trades = pd.DataFrame({
        "pnl": [150.0, -100.0, 250.0, -80.0, 300.0, -90.0, 180.0, 220.0, -110.0, 160.0] * 5,
        "entry_time": pd.date_range("2026-01-01", periods=50, freq="4h"),
    })

    res = simulator.simulate_from_trades(trades, num_simulations=100)
    assert res["simulations"] == 100
    assert 0.0 <= res["pass_rate_pct"] <= 100.0
    assert 0.0 <= res["daily_breach_rate_pct"] <= 100.0
    assert 0.0 <= res["max_dd_breach_rate_pct"] <= 100.0
    assert len(res["sample_paths"]) > 0
    assert "grade" in res
    assert "status" in res
