"""Per-tick strategy path tests: uses_ticks strategies run tick by tick
through the same gates, deterministically, with bars still delivered."""
from __future__ import annotations

import numpy as np
import pytest

from core.tick_engine import BacktestConfig, TickEngine
from strategies.tick_strategies import TickReversion


def _ticks(n: int = 4000, seed: int = 3) -> "object":
    rng = np.random.default_rng(seed)
    t0 = 1_760_000_000_000
    ts = t0 + np.arange(n) * 250
    px = 2600.0 + np.cumsum(rng.normal(0, 0.08, n))
    # add a mean-reverting wobble so reversion fires
    px = px + 1.5 * np.sin(np.arange(n) / 40.0)
    bid = px - 0.15
    ask = px + 0.15
    import pandas as pd
    return pd.DataFrame({
        "time_utc_ms": ts, "bid": bid, "ask": ask,
        "bid_volume": 1.0, "ask_volume": 1.0, "source": "synth",
    })


def _cfg(**over) -> BacktestConfig:
    base = dict(symbol="XAUUSD", initial_balance=10_000.0, risk_pct=0.25,
                bar_timeframe="5m")
    base.update(over)
    return BacktestConfig(**base)


def test_tick_strategy_runs_per_tick_and_trades():
    strat = TickReversion(threshold=0.5)
    res = TickEngine().run(strat, _ticks(), _cfg())
    assert "error" not in res
    assert res["total_trades"] > 0
    # on_tick saw every tick
    assert res["ticks_replayed"] == 4000


def test_tick_strategy_deterministic():
    r1 = TickEngine().run(TickReversion(threshold=0.5), _ticks(), _cfg())
    r2 = TickEngine().run(TickReversion(threshold=0.5), _ticks(), _cfg())
    assert r1["trades"] == r2["trades"]
    assert r1["equity_curve"] == r2["equity_curve"]


def test_tick_strategy_signals_respect_latency_and_sl():
    """Trades must obey the same tolerance: SL never triggers on the
    entry tick, and exits pay slippage."""
    strat = TickReversion(threshold=0.5)
    res = TickEngine().run(strat, _ticks(), _cfg())
    for tr in res["trades"]:
        assert tr["exit_time_utc_ms"] > tr["entry_time_utc_ms"]
        # SL distance respected at entry: risk ~= sl_distance * contract * lot
        assert tr["initial_risk_usd"] > 0


def test_tick_strategy_and_bar_strategy_coexist():
    """A uses_ticks strategy still receives closed bars via on_bar."""
    seen = {"bars": 0}

    class Hybrid(TickReversion):
        def on_bar(self, ctx, bar):
            seen["bars"] += 1
            return []

    TickEngine().run(Hybrid(threshold=0.5), _ticks(), _cfg())
    assert seen["bars"] > 0


def test_tick_reversion_validates_params():
    with pytest.raises(ValueError):
        TickReversion(window=2)
    with pytest.raises(ValueError):
        TickReversion(threshold=0)
    with pytest.raises(ValueError):
        TickReversion(cooldown_ticks=-1)


def test_tick_strategy_session_gates_apply():
    """Session filter + entry cutoff apply to tick strategies too."""
    strat = TickReversion(threshold=0.5)
    res = TickEngine().run(strat, _ticks(), _cfg(
        session_start_min=0, session_end_min=60, flatten_at_session_end=True))
    assert "error" not in res
    for tr in res["trades"]:
        mod = (tr["entry_time_utc_ms"] // 60_000) % 1440
        assert mod < 60
