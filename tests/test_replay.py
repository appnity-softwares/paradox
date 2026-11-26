"""Tests for the Time Machine replay engine and exit-planner suggestor."""

import numpy as np
import pandas as pd
import pytest

from core.replay import replay_trade, suggest_stops


def make_df(days=70, bars_per_day=96, drift=0.0, seed=7):
    """Synthetic 15m bars: one bar per 15 min, bars_per_day per day."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-01", periods=days * bars_per_day, freq="15min", tz="UTC")
    price = 3500.0
    rows = []
    for ts in idx:
        step = rng.normal(drift, 1.0)
        o = price
        c = price + step
        h = max(o, c) + abs(rng.normal(0, 0.4))
        l = min(o, c) - abs(rng.normal(0, 0.4))
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 100})
        price = c
    return pd.DataFrame(rows, index=idx)


def test_replay_updrift_favours_target():
    df = make_df(drift=0.35)
    r = replay_trade(df, entry=3500, stop=3494, target=3512,
                     side="buy", horizon_bars=16, replays=500, seed=19)
    assert r.target_first > r.stop_first
    d = r.to_dict()
    assert d["odds_target"] + d["odds_stop"] + d["odds_neither"] == pytest.approx(1.0, abs=1e-6)
    assert d["avg_r"] > 0


def test_replay_downdrift_favours_stop_for_long():
    df = make_df(drift=-0.35)
    r = replay_trade(df, entry=3500, stop=3494, target=3512,
                     side="buy", horizon_bars=16, replays=500, seed=19)
    assert r.stop_first > r.target_first
    assert r.avg_r < 0


def test_replay_sell_side_symmetry():
    df = make_df(drift=-0.35)
    r = replay_trade(df, entry=3500, stop=3506, target=3488,
                     side="sell", horizon_bars=16, replays=500, seed=19)
    assert r.target_first > r.stop_first


def test_replay_counts_are_exact_and_deterministic():
    df = make_df()
    a = replay_trade(df, entry=3500, stop=3494, target=3512, replays=300, seed=5)
    b = replay_trade(df, entry=3500, stop=3494, target=3512, replays=300, seed=5)
    assert (a.target_first, a.stop_first, a.neither) == (b.target_first, b.stop_first, b.neither)
    assert a.target_first + a.stop_first + a.neither == 300


def test_replay_rejects_bad_plans():
    df = make_df()
    with pytest.raises(ValueError):
        replay_trade(df, entry=3500, stop=3510, target=3520)  # stop above long entry
    with pytest.raises(ValueError):
        replay_trade(df, entry=3500, stop=3490, target=3480)  # target below long entry
    with pytest.raises(ValueError):
        replay_trade(df.head(10), entry=3500, stop=3494, target=3512)  # too few bars


def test_replay_uses_same_time_of_day_slots():
    """Late-night slots must stay quieter than London open slots."""
    df = make_df()
    # stamp real intraday volatility: bigger ranges during London/NY hours
    hour = df.index.hour
    scale = np.where((hour >= 7) & (hour < 17), 3.0, 0.5)
    df["high"] = df["open"] + (df["high"] - df["open"]) * scale
    df["low"] = df["open"] - (df["open"] - df["low"]) * scale
    df["close"] = df["open"] + (df["close"] - df["open"]) * scale
    r_quiet = replay_trade(df[df.index.hour < 5], entry=3500, stop=3495, target=3505,
                           horizon_bars=4, replays=400, seed=3)
    r_loud = replay_trade(df[df.index.hour >= 10], entry=3500, stop=3495, target=3505,
                          horizon_bars=4, replays=400, seed=3)
    # a 10-point target/stop is far more reachable in the loud pool
    assert r_loud.target_first + r_loud.stop_first > r_quiet.target_first + r_quiet.stop_first


def test_suggest_stops_basic():
    df = make_df(days=5)
    s = suggest_stops(df, entry=3500, side="buy", atr_mult=1.5, r_multiple=2.0)
    assert s["stop"] < 3500 < s["target"]
    assert s["risk_pct"] > 0
    # 2R: target distance is twice the stop distance
    assert (s["target"] - s["entry"]) == pytest.approx(2 * (s["entry"] - s["stop"]), abs=0.02)


def test_suggest_stops_sell_side():
    df = make_df(days=5)
    s = suggest_stops(df, entry=3500, side="sell")
    assert s["stop"] > 3500 > s["target"]
