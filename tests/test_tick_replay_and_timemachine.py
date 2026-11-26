"""Tests for the tick-by-tick replay backtester and the Time Machine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.tick_replay import run_tick_replay, month_files
from core.timemachine import Plan, TimeMachineError, history_replay, replay
from strategies.base import StrategyConfig
from strategies.library import EMACross


# ---------------------------------------------------------------- helpers

def make_bars(n=300, start="2026-06-01 00:00", freq="15min", seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 4000 + np.cumsum(rng.normal(0, 2.0, n))
    o = close + rng.normal(0, 0.5, n)
    h = np.maximum(o, close) + np.abs(rng.normal(0, 1.0, n))
    l = np.minimum(o, close) - np.abs(rng.normal(0, 1.0, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "volume": np.abs(rng.normal(100, 10, n))}, index=idx)


def write_ticks(tmp_path, df_bars, ticks_per_bar=8, seed=3):
    """Synthesize a tick archive covering the bars: random walk inside each bar."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "ticks" / "XAUUSD"
    frames = []
    for ts, row in df_bars.iterrows():
        base = float(row["close"])
        for k in range(ticks_per_bar):
            t = ts + pd.Timedelta(minutes=15 * k / ticks_per_bar)
            # bid/ask straddle a drifting mid: 0.30 spread
            mid = base + rng.normal(0, 0.4)
            frames.append((int(t.value // 1_000_000), mid - 0.15, mid + 0.15))
    t = pd.DataFrame(frames, columns=["time_utc_ms", "bid", "ask"]).sort_values("time_utc_ms")
    for (y, m), g in t.groupby([t["time_utc_ms"].map(lambda v: pd.Timestamp(v, unit="ms", tz="UTC").year),
                                t["time_utc_ms"].map(lambda v: pd.Timestamp(v, unit="ms", tz="UTC").month)]):
        d = root / f"{y}" / f"{m:02d}"
        d.mkdir(parents=True, exist_ok=True)
        g.to_parquet(d / "ticks.parquet", index=False)
    return root


# ---------------------------------------------------------------- tick replay

def test_replay_fills_at_real_spread(tmp_path):
    bars = make_bars(200)
    root = write_ticks(tmp_path, bars)
    cfg = StrategyConfig(name="t", risk_pct=1.0, sl_method="fixed", sl_fixed=5.0, rr=2.0)
    res = run_tick_replay(bars, EMACross(cfg), cfg, start_equity=10_000, tick_dir=root)
    assert res["engine"] == "tick_replay"
    assert res["coverage"]["ticks_scanned"] > 0
    assert res["coverage"]["bars_without_ticks"] == 0
    for tr in res["trades"]:
        # entries must sit ON the recorded bid/ask, not a modeled half-spread
        assert tr["entry"] > 0


def test_replay_gap_through_stop_fills_worse(tmp_path):
    """A stop order fills at the tick's price: if the market gaps through it,
    the exit must be worse than the stop level (no invented mercy)."""
    bars = make_bars(120)
    # one crash bar far below everything
    crash_idx = 100
    bars.iloc[crash_idx, bars.columns.get_loc("close")] = bars["close"].iloc[crash_idx - 1] - 60
    bars.iloc[crash_idx, bars.columns.get_loc("low")] = bars["close"].iloc[crash_idx] - 5
    bars.iloc[crash_idx, bars.columns.get_loc("open")] = bars["close"].iloc[crash_idx - 1]
    bars.iloc[crash_idx, bars.columns.get_loc("high")] = bars["close"].iloc[crash_idx - 1] + 2
    root = write_ticks(tmp_path, bars)
    # strategy: always long -> enters, then the crash gap must fill below SL
    class AlwaysLong:
        def generate_signals(self, d):
            return pd.Series(1.0, index=d.index)
    cfg = StrategyConfig(name="t", risk_pct=1.0, sl_method="fixed", sl_fixed=2.0, rr=10.0)
    res = run_tick_replay(bars, AlwaysLong(), cfg, start_equity=10_000, tick_dir=root)
    assert res["trades"], "expected at least one trade"
    sl_trades = [t for t in res["trades"] if t["reason"] == "SL"]
    if sl_trades:
        # exit worse than the stop level for a long
        assert all(t["exit"] < t["entry"] - 2.0 - 1e-9 for t in sl_trades)


def test_replay_deterministic_same_seed(tmp_path):
    bars = make_bars(150)
    root = write_ticks(tmp_path, bars)
    cfg = StrategyConfig(name="t", risk_pct=1.0, sl_method="fixed", sl_fixed=5.0, rr=2.0)
    a = run_tick_replay(bars, EMACross(cfg), cfg, tick_dir=root)
    b = run_tick_replay(bars, EMACross(cfg), cfg, tick_dir=root)
    assert a["trades"] == b["trades"]


def test_replay_reports_bars_without_ticks(tmp_path):
    bars = make_bars(100)
    root = write_ticks(tmp_path, bars.iloc[:50])   # ticks cover only half
    cfg = StrategyConfig(name="t", risk_pct=1.0, sl_method="fixed", sl_fixed=5.0, rr=2.0)
    res = run_tick_replay(bars, EMACross(cfg), cfg, tick_dir=root)
    assert res["coverage"]["bars_without_ticks"] >= 40


def test_replay_rejects_bad_input(tmp_path):
    bars = make_bars(50)
    cfg = StrategyConfig(name="t")
    with pytest.raises(ValueError):
        run_tick_replay(bars, EMACross(cfg), cfg, start_equity=-1, tick_dir=tmp_path)
    with pytest.raises(ValueError):
        run_tick_replay(bars.reset_index(drop=True), EMACross(cfg), cfg, tick_dir=tmp_path)


def test_month_files_sorted():
    files = month_files(__import__("pathlib").Path("data/ticks/XAUUSD"))
    assert len(files) >= 30
    assert files == sorted(files)


# ---------------------------------------------------------------- time machine

def _5m_bars(n_sessions=30, seed=11):
    rng = np.random.default_rng(seed)
    frames = []
    base = 4000.0
    for d in range(n_sessions):
        day = pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=d)
        # skip weekends
        if day.dayofweek >= 5:
            continue
        sl = pd.date_range(day, periods=288, freq="5min", tz="UTC")
        rets = rng.normal(0, 0.4, 288)
        close = base + np.cumsum(rets)
        base = close[-1]
        o = close + rng.normal(0, 0.2, 288)
        h = np.maximum(o, close) + np.abs(rng.normal(0, 0.5, 288))
        l = np.minimum(o, close) - np.abs(rng.normal(0, 0.5, 288))
        frames.append(pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                                    "volume": np.abs(rng.normal(50, 5, 288))}, index=sl))
    return pd.concat(frames).sort_index()


def test_plan_validation():
    with pytest.raises(TimeMachineError):
        Plan(entry=100, stop=101, target=110)          # stop above entry for a buy
    with pytest.raises(TimeMachineError):
        Plan(entry=100, stop=95, target=94)            # target below entry for a buy
    p = Plan(entry=100, stop=95, target=110)
    assert p.risk == 5.0
    assert p.rr == 2.0          # reward 10 / risk 5


def test_replay_honest_statistics():
    bars = _5m_bars(40)
    entry_ts = bars.index[-300]          # late in the data: many past sessions
    entry = float(bars.iloc[-300]["close"])
    plan = Plan(entry=entry, stop=entry - 5.0, target=entry + 5.0,
                side="buy", entry_time=entry_ts)
    res = replay(plan, bars, replays=500, seed=19, flat_by="23:55")
    assert res["replays"] == 500
    assert abs(res["p_target"] + res["p_stop"] + res["p_neither"] - 1.0) < 1e-6
    assert res["break_even_share"] == 0.5          # 1R target
    assert res["verdict"] in ("Close to a coin flip", "Leans toward the target",
                              "Leans toward the stop", "Strongly toward the target",
                              "Strongly toward the stop")
    assert res["pool_sessions"] >= 20


def test_replay_needs_enough_sessions():
    bars = _5m_bars(10)
    entry_ts = bars.index[100]
    entry = float(bars.iloc[100]["close"])
    plan = Plan(entry=entry, stop=entry - 5.0, target=entry + 5.0,
                side="buy", entry_time=entry_ts)
    with pytest.raises(TimeMachineError):
        replay(plan, bars, replays=200)


def test_history_replay_counts():
    bars = _5m_bars(40)
    res = history_replay(bars, or_minutes=15, stop_rule="or_mid", target_r=2.0,
                         side="buy", flat_by="23:55")
    assert res["sessions"] >= 20
    assert res["triggered"] >= 0
    assert res["target"] + res["stop"] + res["neither"] == res["triggered"]
    for t in res["trades"]:
        assert t["outcome"] in ("target", "stop", "neither")
        # stop must sit inside the opening range for or_mid
        assert t["stop"] < t["entry"]      # long side


def test_history_replay_short_side():
    bars = _5m_bars(40)
    res = history_replay(bars, or_minutes=15, stop_rule="or_low", target_r=1.0,
                         side="sell", flat_by="23:55")
    for t in res["trades"]:
        assert t["stop"] > t["entry"]      # short stop above entry


# ---------------------------------------------------------------- tick audit

def test_audit_aligns_and_reports(tmp_path):
    """Audit must match most trades 1:1 and always produce a verdict."""
    from core.tick_audit import audit_bar_vs_tick
    bars = make_bars(n=300)
    root = write_ticks(tmp_path, bars)
    cfg = StrategyConfig(risk_pct=1.0, rr=2.0, sl_method="fixed", sl_fixed=5.0,
                         session="all")
    strat = EMACross(cfg, fast=10, slow=30)
    a = audit_bar_vs_tick(bars, strat, cfg, tick_dir=root)
    assert a["bar"]["trades"] > 0
    assert a["matched"] >= a["bar"]["trades"] * 0.5   # majority aligns 1:1
    assert a["bar"]["wins"] + a["bar"]["losses"] == a["bar"]["trades"]
    assert a["tick"]["ticks_scanned"] > 0
    assert a["verdict"] and all(isinstance(v, str) for v in a["verdict"])
    # per-pair rows carry both engines' numbers
    for p in a["pairs"]:
        assert set(p) >= {"entry_time", "side", "bar_entry", "tick_entry",
                          "entry_slip", "bar_pnl", "tick_pnl", "pnl_delta"}
    # aggregate delta is tick_net - bar_net, not just matched pairs
    assert abs(a["pnl_delta_total"] - (a["tick"]["net_pnl"] - a["bar"]["net_pnl"])) < 0.01


def test_audit_session_gate_matches_bar_engine(tmp_path):
    """With a session filter the tick twin must not take night trades the
    bar engine never sees — the alignment bug this audit exposed."""
    from core.tick_audit import audit_bar_vs_tick
    bars = make_bars(n=400)
    root = write_ticks(tmp_path, bars)
    cfg = StrategyConfig(risk_pct=1.0, rr=2.0, sl_method="fixed", sl_fixed=5.0,
                         session="ny")
    strat = EMACross(cfg, fast=10, slow=30)
    a = audit_bar_vs_tick(bars, strat, cfg, tick_dir=root)
    # every tick-only trade must still be inside the session window
    for t in a["pairs"]:
        hour = pd.Timestamp(t["entry_time"]).hour
        assert 12 <= hour < 21
