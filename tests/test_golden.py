"""Golden-file regression for the tick engine (Master Prompt 6.4).

Replays a fixed synthetic tick series with a fixed strategy and compares the
full trade log against tests/golden/tick_engine_golden.json. Any change in
engine behaviour -- order handling, cost application, exit logic -- shifts a
price or a P&L and fails this test. Regenerate deliberately via
REGEN_GOLDEN=1 after reviewing the diff.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.tick_engine import BacktestConfig, TickEngine
from core.strategy_api import Context, Signal, Strategy, SIDE_CLOSE, SIDE_LONG, SIDE_SHORT

GOLDEN = Path(__file__).parent / "golden" / "tick_engine_golden.json"
MIN = 60_000


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    price = 2000.0
    t = 0
    for minute in range(240):
        drift = 0.25 if minute < 80 else (-0.35 if minute >= 160 else 0.0)
        for _ in range(20):
            price += drift / 20 + rng.normal(0, 0.06)
            rows.append((t, round(price - 0.2, 3), round(price + 0.2, 3)))
            t += 3_000
    return pd.DataFrame({
        "time_utc_ms": [r[0] for r in rows],
        "bid": [r[1] for r in rows],
        "ask": [r[2] for r in rows],
        "bid_volume": 1.0,
        "ask_volume": 1.0,
        "source": "synthetic-test",
    })


class GoldenStrategy(Strategy):
    name = "golden_ema_cross"
    version = "1"
    warmup_bars = 40

    def __init__(self):
        self.ef: float | None = None
        self.es: float | None = None
        self.af = 2 / 11
        self.as_ = 2 / 31

    def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
        c = bar["close"]
        self.ef = c if self.ef is None else self.af * c + (1 - self.af) * self.ef
        self.es = c if self.es is None else self.as_ * c + (1 - self.as_) * self.es
        if len(ctx.bars) < self.warmup_bars:
            return []
        out: list[Signal] = []
        if ctx.position.side > 0 and self.ef < self.es:
            out.append(Signal(SIDE_CLOSE))
        elif ctx.position.side < 0 and self.ef > self.es:
            out.append(Signal(SIDE_CLOSE))
        if ctx.position.side == 0:
            if self.ef > self.es:
                out.append(Signal(SIDE_LONG, sl=c - 2.0, tp=c + 4.0))
            elif self.ef < self.es:
                out.append(Signal(SIDE_SHORT, sl=c + 2.0, tp=c - 4.0))
        return out


def run_reference() -> dict:
    ticks = build_dataset()
    cfg = BacktestConfig(
        symbol="XAUUSD",
        initial_balance=100_000.0,
        risk_pct=1.0,
        dataset_hash="golden-test-dataset",
        breakeven_rr=1.0,
        breakeven_offset=0.10,
    )
    result = TickEngine().run(GoldenStrategy(), ticks, cfg)
    assert result["valid"], result
    return {
        "config_hash": result["config_hash"],
        "engine_version": result["engine_version"],
        "ticks_replayed": result["ticks_replayed"],
        "total_trades": result["total_trades"],
        "net_profit_usd": result["net_profit_usd"],
        "max_drawdown_usd": result["max_drawdown_usd"],
        "win_rate_pct": result["win_rate_pct"],
        "profit_factor": result["profit_factor"],
        "trades": [
            {k: tr[k] for k in (
                "side", "entry_time_utc_ms", "exit_time_utc_ms",
                "entry_price", "exit_price", "volume", "pnl", "exit_reason")}
            for tr in result["trades"]
        ],
    }


def _round_floats(obj, ndigits: int = 6):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def test_tick_engine_matches_golden_trade_log():
    if os.environ.get("REGEN_GOLDEN") == "1":
        golden = _round_floats(run_reference())
        GOLDEN.write_text(json.dumps(golden, indent=1))
        pytest.skip("golden file regenerated")

    assert GOLDEN.exists(), "golden file missing; run REGEN_GOLDEN=1 pytest tests/test_golden.py"
    expected = json.loads(GOLDEN.read_text())
    actual = _round_floats(run_reference())
    assert actual == expected, "tick engine behaviour drifted from the golden trade log"
