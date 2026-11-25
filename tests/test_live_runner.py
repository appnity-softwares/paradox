"""Live runtime tests: ReplaySource, BarAggregator (engine-identical
convention), LiveRunner through the real ExecutionRouter, and the
replay-vs-backtest determinism cross-check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.execution.live import (BarAggregator, LiveRunner, ReplaySource,
                                 Tick)
from core.tick_engine import BAR_MS, BacktestConfig, TickEngine

DAY_MS = 86_400_000
T0 = (1_760_000_000_000 // DAY_MS) * DAY_MS


def synth_ticks(days: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    px = 2600.0
    for d in range(days):
        base = T0 + d * DAY_MS
        for s in range(25_200, 39_600, 5):      # London+NY hours only
            px += rng.normal(0, 0.15)
            rows.append((base + s * 1000, round(px - 0.15, 3),
                         round(px + 0.15, 3), 1.0, 1.0, "synth"))
    return pd.DataFrame(rows, columns=["time_utc_ms", "bid", "ask",
                                       "bid_volume", "ask_volume", "source"])


# ── ReplaySource ─────────────────────────────────────────────────────────

def test_replay_source_yields_in_order():
    df = synth_ticks(days=2)
    src = ReplaySource(df)
    ticks = list(src)
    assert len(ticks) == len(df)
    ts = [t.time_utc_ms for t in ticks]
    assert ts == sorted(ts)
    assert src.exhausted


def test_replay_source_validates_columns():
    with pytest.raises(ValueError):
        ReplaySource(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError):
        ReplaySource(synth_ticks(), speed=0)


# ── BarAggregator: engine-identical convention ──────────────────────────

def test_aggregator_matches_engine_bars():
    df = synth_ticks(days=3)
    # Engine bars via a trivial strategy that records bar closes.
    from core.strategy_api import Strategy

    class Recorder(Strategy):
        key = "recorder"
        version = "1"
        warmup_bars = 1

        def __init__(self):
            self.seen = []

        def on_bar(self, ctx, bar):
            self.seen.append((int(bar.name) if hasattr(bar, "name")
                              else int(bar["time_utc_ms"]),
                              round(float(bar["open"]), 6),
                              round(float(bar["close"]), 6)))
            return []

    rec = Recorder()
    cfg = BacktestConfig(symbol="XAUUSD", initial_balance=10_000.0,
                         risk_pct=0.25, bar_timeframe="5m")
    TickEngine().run(rec, df, cfg)

    agg = BarAggregator("5m")
    live_bars = []
    for t in ReplaySource(df):
        closed = agg.feed(Tick(t.time_utc_ms, t.bid, t.ask))
        if closed:
            live_bars.append((closed["time_utc_ms"],
                              round(closed["open"], 6),
                              round(closed["close"], 6)))
    # The engine delivers the final bar only if a later chunk exists;
    # the aggregator does the same (last bucket never closes). Compare
    # the overlapping prefix.
    n = min(len(rec.seen), len(live_bars))
    assert n > 100
    assert rec.seen[:n] == live_bars[:n]


def test_aggregator_rejects_bad_timeframe():
    with pytest.raises(ValueError):
        BarAggregator("7m")


# ── LiveRunner through the real router ──────────────────────────────────

def _snapshot():
    from core.risk import AccountSnapshot
    return AccountSnapshot(account_id="a1", balance=10_000.0,
                           equity=10_000.0, used_margin=0.0,
                           open_positions=0,
                           exposure_lots_by_symbol={},
                           daily_realized_pnl=0.0,
                           daily_floating_pnl=0.0,
                           equity_high_watermark=10_000.0,
                           last_tick_ms={"XAUUSD": 1_000},
                           spread={"XAUUSD": 0.4})


def _rules():
    from core.compliance import FirmRuleSet
    from core.compliance.rules import DailyLossRule, MaxLossRule
    return FirmRuleSet(
        firm="test_firm", program="100k", version="v1", status="verified",
        ea_policy="allowed", verified_by="human-1",
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=10_000.0, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
        min_trading_days=0, weekend_holding="allowed")


@pytest.fixture()
def gate_env(tmp_path):
    """Broker + router wired like the execution-gate tests."""
    from core.account import Account
    from core.adaptation import StrategyProfile
    from core.compliance import ComplianceEngine
    from core.config import COST_MODELS, SYMBOL_SPECS
    from core.execution import (AuditLog, ExecutionRouter, ModeGate,
                                Notifier, PaperBroker)
    from core.risk import DeterministicRiskEngine, RiskConfig
    broker = PaperBroker(cost_model=COST_MODELS["XAUUSD_ECN"])
    broker.connect()
    broker.on_tick("XAUUSD", bid=2599.80, ask=2600.20, ts_ms=1_000)
    router = ExecutionRouter(
        broker, DeterministicRiskEngine(RiskConfig(version="v1")),
        ComplianceEngine(), ModeGate(),
        AuditLog(tmp_path / "audit" / "log.jsonl"), Notifier())
    profile = StrategyProfile(
        key="trend_pullback", name="Trend pullback", max_dd_pct=35.0,
        typical_stop_distance=12.0, recommended_risk_pct=1.0,
        cost_per_round_trip=0.50)
    router.register_account(Account(account_id="a1", balance=10_000.0),
                            _snapshot(), profile, _rules(),
                            SYMBOL_SPECS["XAUUSD"], 2600.0)
    router.modes.set_mode("a1", "auto", verdict="PAPER_TRADE_ELIGIBLE",
                          paper_days=30, rules_verified=True,
                          confirmed_by="human-1")
    return broker, router


def test_live_runner_paper_mode_routes_through_router(gate_env):
    broker, router = gate_env
    df = synth_ticks(days=6)
    from strategies.variants import TrendPullback
    runner = LiveRunner(
        source=ReplaySource(df), strategy=TrendPullback(atr_mult=1.0,
                                                        rr=1.5),
        symbol="XAUUSD", account_id="a1", mode="paper", router=router,
        paper_broker=broker, bar_timeframe="5m", risk_pct=0.25,
        session_start_min=420, session_end_min=660)
    out = runner.run()
    assert out["bars_closed"] > 100
    assert out["ticks_seen"] == len(df)
    # Every emitted signal passed the full compliance pipeline.
    assert all(s["status"] in ("filled", "pending_accepted")
               for s in runner.signals_sent)


def test_live_runner_semi_auto_emits_alerts(gate_env):
    broker, router = gate_env
    df = synth_ticks(days=6)

    class FakeNotifier:
        def __init__(self):
            self.sent = []

        def send(self, msg: str):
            self.sent.append(msg)

    notifier = FakeNotifier()
    from strategies.variants import TrendPullback
    runner = LiveRunner(
        source=ReplaySource(df), strategy=TrendPullback(atr_mult=1.0,
                                                        rr=1.5),
        symbol="XAUUSD", account_id="a1", mode="semi_auto",
        router=router, notifier=notifier, bar_timeframe="5m",
        risk_pct=0.25, session_start_min=420, session_end_min=660)
    runner.run()
    for msg in notifier.sent:
        assert "XAUUSD" in msg and "SL=" in msg


def test_live_runner_rejects_bad_mode(gate_env):
    broker, router = gate_env
    with pytest.raises(ValueError):
        LiveRunner(source=None, strategy=None, symbol="XAUUSD",
                   account_id="a1", mode="auto", router=router)
    with pytest.raises(ValueError):
        LiveRunner(source=None, strategy=None, symbol="XAUUSD",
                   account_id="a1", mode="paper", router=router,
                   paper_broker=None)


def test_live_runner_strategy_error_does_not_kill_feed(gate_env):
    broker, router = gate_env
    df = synth_ticks(days=3)

    class Broken:
        key = "broken"
        version = "1"
        warmup_bars = 1

        def on_bar(self, ctx, bar):
            raise RuntimeError("boom")

    from strategies.variants import TrendPullback
    runner = LiveRunner(
        source=ReplaySource(df), strategy=Broken(), symbol="XAUUSD",
        account_id="a1", mode="paper", router=router,
        paper_broker=broker, bar_timeframe="5m", risk_pct=0.25)
    out = runner.run()
    assert out["bars_closed"] > 50          # feed kept running
    assert any("error" in s for s in runner.signals_sent)
