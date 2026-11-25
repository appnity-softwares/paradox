"""Phase 1B tests: production tick engine (Master Prompt 6.2-6.4).

Synthetic ticks are used HERE ONLY, as the Master Prompt allows: synthetic
data is permitted in unit tests and must stay out of the validated path.
Every test builds explicit tick sequences so expected fills and P&L are
hand-calculable.

Event timing used throughout (matches the engine):
- A bar covering minute m is delivered when the first tick of minute m+1
  arrives; signals from it are acted on at the first tick at/after
  (m+1)*minute + fill_latency_ms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import CostModel, register_cost_model
from core.tick_engine import BacktestConfig, TickEngine
from core.strategy_api import (
    KIND_LIMIT,
    KIND_STOP,
    SIDE_CLOSE,
    SIDE_LONG,
    SIDE_SHORT,
    Context,
    Fill,
    LookAheadError,
    Signal,
    Strategy,
)


MIN = 60_000


def make_ticks(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    """rows of (time_utc_ms, bid, ask) -> normalized tick frame."""
    return pd.DataFrame({
        "time_utc_ms": [r[0] for r in rows],
        "bid": [r[1] for r in rows],
        "ask": [r[2] for r in rows],
        "bid_volume": [1.0] * len(rows),
        "ask_volume": [1.0] * len(rows),
        "source": ["synthetic-test"] * len(rows),
    })


def base_config(**overrides) -> BacktestConfig:
    defaults = dict(
        symbol="XAUUSD",
        initial_balance=100_000.0,
        risk_pct=1.0,
        cost_key="XAUUSD_ECN",
        broker_key="mt5_bridge",
        dataset_hash="test-hash",
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


class FixedSignalStrategy(Strategy):
    """Emits given signals at given bar closes (1-indexed)."""

    name = "fixed_signal"

    def __init__(self, script: dict[int, list[Signal]] | None = None, warmup_bars: int = 0):
        self.script = script or {}
        self.warmup_bars = warmup_bars
        self.bar_count = 0
        self.fills: list[Fill] = []

    def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
        self.bar_count += 1
        return self.script.get(self.bar_count, [])

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        self.fills.append(fill)


# ---------------------------------------------------------------------------
# Correct sides of the quote
# ---------------------------------------------------------------------------


def test_long_entry_fills_at_ask_and_exit_at_bid():
    # Bar 1 (minute 0) closes -> long signal; decision t=60000, target 60100.
    # First tick >= 60100 is the 2-minute tick: entry at ask 2002.50 + 0.05.
    # Bar 3 (minute 2) closes -> close signal; fill at 4-minute bid 2004.00 - 0.05.
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2004.00, 2004.50),
        (5 * MIN, 2005.00, 2005.50),
    ])
    strat = FixedSignalStrategy({
        1: [Signal(SIDE_LONG, sl=1990.0)],
        3: [Signal(SIDE_CLOSE)],
    })
    result = TickEngine().run(strat, ticks, base_config())
    assert result["valid"], result
    assert len(result["trades"]) == 1
    tr = result["trades"][0]
    assert tr["entry_price"] == pytest.approx(2002.50 + 0.05)   # ask side
    assert tr["exit_price"] == pytest.approx(2004.00 - 0.05)    # bid side


def test_short_entry_fills_at_bid_and_exit_at_ask():
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),   # entry fill: bid 2002.00 - 0.05
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2004.00, 2004.50),   # exit fill: ask 2004.50 + 0.05
        (5 * MIN, 2004.00, 2004.50),
    ])
    strat = FixedSignalStrategy({
        1: [Signal(SIDE_SHORT, sl=2010.0)],
        3: [Signal(SIDE_CLOSE)],
    })
    result = TickEngine().run(strat, ticks, base_config())
    assert result["valid"], result
    tr = result["trades"][0]
    assert tr["side"] == "SHORT"
    assert tr["entry_price"] == pytest.approx(2002.00 - 0.05)   # bid side
    assert tr["exit_price"] == pytest.approx(2004.50 + 0.05)    # ask side


# ---------------------------------------------------------------------------
# SL / TP evaluated per tick, tick order decides
# ---------------------------------------------------------------------------


def test_sl_and_tp_inside_one_bar_follow_tick_order_not_sl_first():
    """The bar engine must assume SL first; the tick engine follows the real
    tick order: TP hits first here."""
    # Entry at the 2-minute tick: ask 2000.50 + 0.05 = 2000.55.
    sl = 1999.55
    tp = 2002.55
    ticks = make_ticks([
        (0 * MIN, 1999.00, 1999.50),
        (1 * MIN, 1999.50, 2000.00),
        (2 * MIN, 2000.00, 2000.50),      # entry
        (3 * MIN, 2003.00, 2003.50),      # bid 2003 >= tp -> TP first
        (3 * MIN + 1, 1999.00, 1999.50),  # would be SL, but already flat
        (4 * MIN, 1999.00, 1999.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=sl, tp=tp)]})
    result = TickEngine().run(strat, ticks, base_config())
    assert result["valid"], result
    assert len(result["trades"]) == 1
    assert result["trades"][0]["exit_reason"] == "TP"


def test_stop_loss_triggers_on_correct_side_per_tick():
    ticks = make_ticks([
        (0 * MIN, 1999.00, 1999.50),
        (1 * MIN, 1999.50, 2000.00),
        (2 * MIN, 2000.00, 2000.50),      # entry at 2000.55, sl 1999.55
        (3 * MIN, 2000.40, 2000.90),      # bid 2000.40 > sl: hold
        (3 * MIN + 1, 1999.40, 1999.90),  # bid 1999.40 <= sl -> SL
        (4 * MIN, 1999.00, 1999.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=1999.55)]})
    result = TickEngine().run(strat, ticks, base_config())
    tr = result["trades"][0]
    assert tr["exit_reason"] == "SL"
    # SL fills through: bid at the triggering tick minus stop slippage
    assert tr["exit_price"] == pytest.approx(1999.40 - 0.10)


# ---------------------------------------------------------------------------
# Pending orders: limit and stop
# ---------------------------------------------------------------------------


def test_limit_buy_fills_at_limit_price_or_better():
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),   # signal bar: buy limit at 2000.20
        (2 * MIN, 2001.50, 2002.00),   # ask 2002 > limit: no fill
        (3 * MIN, 2000.00, 2000.10),   # ask 2000.10 <= 2000.20 -> fill at ask
        (4 * MIN, 2000.50, 2001.00),
        (5 * MIN, 2000.50, 2001.00),
        (6 * MIN, 2000.50, 2001.00),
        (7 * MIN, 2000.50, 2001.00),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, kind=KIND_LIMIT, price=2000.20, sl=1995.0)]})
    result = TickEngine().run(strat, ticks, base_config())
    assert result["valid"], result
    tr = result["trades"][0]
    assert tr["entry_price"] == pytest.approx(2000.10)  # better than the limit


def test_stop_buy_fills_through_with_stop_slippage():
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),   # signal: buy stop at 2002.00
        (2 * MIN, 2001.00, 2001.50),   # no trigger
        (3 * MIN, 2002.00, 2002.30),   # ask 2002.30 >= 2002 -> fill 2002.30+0.10
        (4 * MIN, 2002.50, 2003.00),
        (5 * MIN, 2002.50, 2003.00),
        (6 * MIN, 2002.50, 2003.00),
        (7 * MIN, 2002.50, 2003.00),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, kind=KIND_STOP, price=2002.00, sl=1998.0)]})
    result = TickEngine().run(strat, ticks, base_config())
    tr = result["trades"][0]
    assert tr["entry_price"] == pytest.approx(2002.30 + 0.10)


# ---------------------------------------------------------------------------
# Trade management: partial close, breakeven, trailing, time exit
# ---------------------------------------------------------------------------


def test_partial_close_books_two_trades_and_keeps_remainder():
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2000.50, 2001.00),
        (2 * MIN, 2001.00, 2001.50),   # entry fill: ask 2001.50 + 0.05
        (3 * MIN, 2002.00, 2002.50),
        (4 * MIN, 2003.00, 2003.50),   # close 50% at bid 2003.00
        (5 * MIN, 2004.00, 2004.50),
        (6 * MIN, 2004.00, 2004.50),   # close rest at bid 2004.00
        (7 * MIN, 2004.00, 2004.50),
    ])
    strat = FixedSignalStrategy({
        1: [Signal(SIDE_LONG, sl=2000.00)],
        3: [Signal(SIDE_CLOSE, fraction=0.5)],   # bar 3 closes at t=240000
        4: [Signal(SIDE_CLOSE)],                 # bar 4 closes at t=300000
    })
    result = TickEngine().run(strat, ticks, base_config())
    assert result["valid"], result
    assert len(result["trades"]) == 2
    vols = [tr["volume"] for tr in result["trades"]]
    assert vols[0] == pytest.approx(vols[1])


def test_breakeven_moves_stop_to_entry_plus_offset():
    # Entry at 2-minute tick: 2002.50 + 0.05 = 2002.55. Bar-1 ref mid is
    # 2000.25, sl 2000.00 -> R = 0.25 -> lots = 1000 / 25 = 40.
    # BE armed when bid >= 2002.55 + 2.55 = 2005.10 (4-minute tick: 2006).
    # BE stop = entry + 0.10 = 2002.65; the 5-minute bid 2002.50 hits it.
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2006.00, 2006.50),   # arms breakeven (stop -> 2002.65)
        (5 * MIN, 2002.50, 2003.00),   # bid 2002.50 <= 2002.65 -> BE exit
        (6 * MIN, 2002.50, 2003.00),
        (7 * MIN, 2002.50, 2003.00),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=2000.00)]})
    result = TickEngine().run(strat, ticks, base_config(breakeven_rr=1.0, breakeven_offset=0.10))
    assert result["valid"], result
    tr = result["trades"][0]
    assert tr["exit_reason"] == "SL"
    assert tr["exit_price"] == pytest.approx(2002.50 - 0.10)
    # Roughly breakeven: -600 vs -10,200 for a full stop-out at the original SL.
    assert tr["pnl"] > -1_000


def test_trailing_stop_follows_price_up():
    # Entry 2002.55; trail 1.0 behind the highest bid.
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2006.00, 2006.50),   # trail stop -> 2006.00 - 1.00 = 2005.00
        (5 * MIN, 2004.90, 2005.40),   # bid 2004.90 <= 2005.00 -> trail exit
        (6 * MIN, 2004.50, 2005.00),
        (7 * MIN, 2004.50, 2005.00),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=1999.00)]})
    result = TickEngine().run(strat, ticks, base_config(trail_distance=1.0))
    assert result["valid"], result
    tr = result["trades"][0]
    assert tr["exit_reason"] == "SL"
    assert tr["exit_price"] == pytest.approx(2004.90 - 0.10)


def test_time_exit_closes_after_configured_minutes():
    # Entry at the 2-minute tick (t=120000); time exit 2 minutes later.
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),   # entry t=120000
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2003.00, 2003.50),   # t=240000 >= entry + 120000 -> exit
        (5 * MIN, 2003.00, 2003.50),
        (6 * MIN, 2003.00, 2003.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=1990.0)]})
    result = TickEngine().run(strat, ticks, base_config(time_exit_minutes=2.0))
    assert result["valid"], result
    assert result["trades"][0]["exit_reason"] == "time_exit"


# ---------------------------------------------------------------------------
# Costs: commission, news slippage
# ---------------------------------------------------------------------------


def test_commission_charged_per_side():
    register_cost_model("TEST_COMM", CostModel(
        symbol="XAUUSD", commission_per_lot=7.0, spread_model="recorded",
    ))
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2000.50, 2001.00),
        (2 * MIN, 2001.00, 2001.50),   # entry: ref mid 2000.75, sl 2000 -> R=0.75
        (3 * MIN, 2002.00, 2002.50),
        (4 * MIN, 2002.00, 2002.50),   # close at bid 2002.00
        (5 * MIN, 2002.00, 2002.50),
        (6 * MIN, 2002.00, 2002.50),
    ])
    strat = FixedSignalStrategy({
        1: [Signal(SIDE_LONG, sl=2000.0)],
        3: [Signal(SIDE_CLOSE)],
    })
    result = TickEngine().run(strat, ticks, base_config(cost_key="TEST_COMM"))
    assert result["valid"], result
    tr = result["trades"][0]
    lots = tr["volume"]
    # entry fee + exit fee = 2 * 7.0 * lots
    assert tr["entry_fee"] == pytest.approx(7.0 * lots, abs=0.01)
    assert tr["exit_fee"] == pytest.approx(7.0 * lots, abs=0.01)
    assert result["costs_paid_usd"]["commission"] == pytest.approx(14.0 * lots, abs=0.02)


def test_news_window_multiplies_slippage():
    # Entry tick (2-minute) falls inside the news window: slip 0.05 * 3.
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2004.00, 2004.50),
        (5 * MIN, 2004.00, 2004.50),
        (6 * MIN, 2004.00, 2004.50),
        (7 * MIN, 2004.00, 2004.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=1990.0)]})
    news = ((1 * MIN, 10 * MIN),)
    result = TickEngine().run(strat, ticks, base_config(news_windows=news))
    tr = result["trades"][0]
    assert tr["entry_price"] == pytest.approx(2002.50 + 0.05 * 3.0)


# ---------------------------------------------------------------------------
# Latency: fills use the first tick AFTER the delay
# ---------------------------------------------------------------------------


def test_market_order_fills_after_latency_on_later_tick():
    # Bar 1 closes at t=60000; target = 60100. Ticks at 60000 and 60050 are
    # too early; 60150 is the first eligible tick.
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (60_000, 2001.00, 2001.50),    # bar 1 close tick; too early
        (60_050, 2001.00, 2001.50),    # still before 60100
        (60_150, 2002.00, 2002.50),    # first tick after latency -> fill
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2002.00, 2002.50),
        (4 * MIN, 2002.00, 2002.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=1990.0)]})
    result = TickEngine().run(strat, ticks, base_config())
    tr = result["trades"][0]
    assert tr["entry_time_utc_ms"] == 60_150
    assert tr["entry_price"] == pytest.approx(2002.50 + 0.05)


# ---------------------------------------------------------------------------
# Margin and stop-out
# ---------------------------------------------------------------------------


def test_stop_out_closes_position_when_margin_level_hits_broker_level():
    # No SL, 100% risk, default sl distance 1.0 -> max_lot 500 caps the size.
    # Notional ~500 * 100 * 2002 = ~10M; margin at 1:100 ~ 100k = the balance.
    # A $1 adverse move halves the margin level -> stop-out at 50%.
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),   # entry ~2002.55
        (3 * MIN, 2002.00, 2002.50),
        (4 * MIN, 2001.00, 2001.50),   # floating ~ -77k -> margin level < 50%
        (5 * MIN, 2001.00, 2001.50),
        (6 * MIN, 2001.00, 2001.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG)]})
    result = TickEngine().run(
        strat, ticks, base_config(risk_pct=100.0, require_sl=False, default_sl_distance=1.0))
    assert result["valid"], result
    assert result["trades"][0]["exit_reason"] == "stop_out"


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------


def test_same_config_and_data_reproduce_identical_trade_log():
    rng = np.random.default_rng(42)
    n = 20_000
    steps = rng.normal(0, 0.3, n).cumsum()
    bids = 2000.0 + steps
    asks = bids + 0.4
    times = np.arange(n) * 250
    ticks = pd.DataFrame({
        "time_utc_ms": times, "bid": bids, "ask": asks,
        "bid_volume": 1.0, "ask_volume": 1.0, "source": "synthetic-test",
    })

    class NoiseStrategy(Strategy):
        name = "noise"
        warmup_bars = 2

        def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
            if len(ctx.bars) < 3:
                return []
            c = ctx.bars.closes(3)
            if c[-2] < 2000 <= c[-1]:
                return [Signal(SIDE_LONG, sl=bar["close"] - 2.0)]
            if c[-2] > 2000 >= c[-1]:
                return [Signal(SIDE_SHORT, sl=bar["close"] + 2.0)]
            return []

    cfg = base_config(risk_pct=0.5)
    r1 = TickEngine().run(NoiseStrategy(), ticks.copy(), cfg)
    r2 = TickEngine().run(NoiseStrategy(), ticks.copy(), cfg)
    assert r1["valid"] and r2["valid"]
    assert r1["trades"] == r2["trades"]
    assert r1["equity_curve"] == r2["equity_curve"]
    assert r1["config_hash"] == r2["config_hash"]


# ---------------------------------------------------------------------------
# Look-ahead detector and safety defaults
# ---------------------------------------------------------------------------


def test_cheating_strategy_is_detected_and_run_marked_invalid():
    class Cheater(Strategy):
        name = "cheater"
        warmup_bars = 1

        def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
            ctx.bars.tail(10_000_000)  # far beyond the lookback cap
            return []

    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2003.00, 2003.50),
    ])
    result = TickEngine().run(Cheater(), ticks, base_config())
    assert result["valid"] is False
    assert result["lookahead_violations"]


def test_entry_without_stop_loss_is_rejected_by_default():
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2003.00, 2003.50),
        (5 * MIN, 2003.00, 2003.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG)]})  # no SL
    result = TickEngine().run(strat, ticks, base_config())
    assert result["total_trades"] == 0  # mandatory-SL rule blocks the entry


def test_lot_rounding_respects_min_lot_and_step():
    # risk 0.1% of 100k = 100 USD; ref mid 2000.25, sl 1953 -> R = 47.25
    # lots = 100 / (47.25 * 100) = 0.0212 -> rounds DOWN to 0.02
    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (1 * MIN, 2001.00, 2001.50),
        (2 * MIN, 2002.00, 2002.50),
        (3 * MIN, 2003.00, 2003.50),
        (4 * MIN, 2003.00, 2003.50),
        (5 * MIN, 2003.00, 2003.50),
        (6 * MIN, 2003.00, 2003.50),
        (7 * MIN, 2003.00, 2003.50),
    ])
    strat = FixedSignalStrategy({1: [Signal(SIDE_LONG, sl=1953.0)]})
    result = TickEngine().run(strat, ticks, base_config(risk_pct=0.1))
    assert result["valid"], result
    assert result["trades"][0]["volume"] == pytest.approx(0.02)


def test_final_partial_bar_is_never_delivered():
    """Ticks ending mid-minute must not produce a bar-close signal."""

    class BarCounter(Strategy):
        name = "counter"
        warmup_bars = 0
        count = 0

        def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
            self.count += 1
            return []

    ticks = make_ticks([
        (0 * MIN, 2000.00, 2000.50),
        (30_000, 2000.10, 2000.60),
        (1 * MIN, 2001.00, 2001.50),            # bar 1 closes here
        (1 * MIN + 30_000, 2001.10, 2001.60),   # partial bar 2: never closes
    ])
    strat = BarCounter()
    result = TickEngine().run(strat, ticks, base_config())
    assert strat.count == 1  # only the first fully-closed bar
