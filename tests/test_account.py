"""Part 7 tests: account model, adaptation, suitability report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.account import Account, AccountLimits, Position
from core.adaptation import (
    MARGIN_UTILIZATION_CAP,
    StrategyProfile,
    adapt_config,
    suitability_report,
)
from core.config import SYMBOL_SPECS
from core.backtest import run_backtest
from core.sizing import CurrencyRate
from strategies.base import StrategyConfig
from strategies.library import EMACross

XAU = SYMBOL_SPECS["XAUUSD"]

# Profile measured by the Phase 1 stack on real data (ema_cross ATR cell):
# worst DD ~35% on H1-2025, typical ATR(1.5) stop ~ $12 at 2025 volatility,
# round-trip cost ~ spread 0.30 + 2x slippage 0.10 = 0.50 price units.
PROFILE = StrategyProfile(
    key="ema_cross", name="EMA cross (ATR stops)",
    max_dd_pct=35.0, typical_stop_distance=12.0,
    recommended_risk_pct=1.0, cost_per_round_trip=0.50,
)


def make_bars(n: int = 1600, seed: int = 8, drift: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(drift, 0.6, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.4, "low": close - 0.4,
         "close": close, "volume": 100.0},
        index=pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"),
    )


# ── Account model ────────────────────────────────────────────────────────────

def test_account_derived_properties():
    acct = Account(
        account_id="10k", balance=10_000.0, leverage=100,
        positions=(
            Position("XAUUSD", +1, 0.20, 2000.0, margin_used=400.0),
            Position("XAUUSD", -1, 0.10, 2005.0, margin_used=200.0),
        ),
        floating_pnl=-150.0,
    )
    assert acct.equity == 9_850.0
    assert acct.used_margin == 600.0
    assert acct.free_margin == 9_250.0
    assert acct.margin_level_pct == pytest.approx(9850 / 600 * 100)
    assert acct.open_positions == 2
    assert acct.total_exposure_lots == pytest.approx(0.30)


def test_account_margin_level_infinite_without_positions():
    acct = Account(account_id="flat", balance=5_000.0)
    assert acct.margin_level_pct == float("inf")
    assert acct.free_margin == 5_000.0


def test_account_multiple_accounts_independent():
    a = Account(account_id="10k", balance=10_000.0,
                limits=AccountLimits(risk_per_trade_pct=1.0))
    b = Account(account_id="100k", balance=100_000.0,
                limits=AccountLimits(risk_per_trade_pct=0.5))
    assert a.limits.risk_per_trade_pct != b.limits.risk_per_trade_pct
    assert a.equity != b.equity


def test_account_validation():
    with pytest.raises(ValueError):
        Account(account_id="bad", balance=-1)
    with pytest.raises(ValueError):
        AccountLimits(risk_per_trade_pct=0.0)
    with pytest.raises(ValueError):
        AccountLimits(max_open_positions=0)


# ── Adaptation: 10k vs 100k produce different, sensible configs ──────────────

def test_adapt_10k_vs_100k_differ_sensibly():
    acct10 = Account(account_id="10k", balance=10_000.0, leverage=100)
    acct100 = Account(account_id="100k", balance=100_000.0, leverage=100)
    cfg10 = adapt_config(acct10, PROFILE, XAU, price=2600.0)
    cfg100 = adapt_config(acct100, PROFILE, XAU, price=2600.0)

    # Same risk percent (both caps allow 1%), but the sized lot scales with
    # equity: target 100 vs 1000 USD. Lots floor independently to the
    # 0.01 grid, so the ratio is ~10x within one lot step.
    assert cfg10.risk_per_trade_pct == cfg100.risk_per_trade_pct == 1.0
    assert cfg100.lot / cfg10.lot == pytest.approx(10.0, rel=0.15)
    assert cfg100.risk_money / cfg10.risk_money == pytest.approx(10.0, rel=0.15)

    # Exposure caps differ: for 10k the margin-derived cap binds
    # (0.5*10000 / 2600 = 1.92 lots), for 100k the account limit (5) binds.
    assert cfg10.max_total_exposure_lots == pytest.approx(1.92, abs=0.01)
    assert cfg100.max_total_exposure_lots == pytest.approx(5.0)
    # Daily budget: 3% / 1% = 3 trades.
    assert cfg10.max_trades_per_day == cfg100.max_trades_per_day == 3


def test_adapt_respects_strategy_recommendation():
    conservative = StrategyProfile(
        key="x", name="x", max_dd_pct=10.0, typical_stop_distance=12.0,
        recommended_risk_pct=0.5, cost_per_round_trip=0.5)
    acct = Account(account_id="a", balance=10_000.0,
                   limits=AccountLimits(risk_per_trade_pct=2.0))
    cfg = adapt_config(acct, conservative, XAU, price=2600.0)
    assert cfg.risk_per_trade_pct == 0.5  # strategy recommendation wins


def test_adapt_margin_cap_binds_for_small_lev():
    # Leverage 5: one lot margin = 100*2600/5 = 52,000 -> cap lots =
    # 0.5*10000/52000 = 0.096 -> exposure cap below 1 lot.
    acct = Account(account_id="small", balance=10_000.0, leverage=5)
    cfg = adapt_config(acct, PROFILE, XAU, price=2600.0)
    assert cfg.max_total_exposure_lots < 1.0
    assert cfg.max_open_positions == 1  # floor of 1 keeps it tradable


# ── Suitability report ───────────────────────────────────────────────────────

def test_suitability_healthy_account_passes():
    # The account's drawdown limit must accommodate the strategy's
    # historical worst DD (35%) — that is exactly what the check verifies.
    acct = Account(account_id="10k", balance=10_000.0, leverage=100,
                   limits=AccountLimits(max_drawdown_pct=50.0))
    rep = suitability_report(acct, PROFILE, XAU, price=2600.0)
    assert rep.suitable is True
    assert all(c["passed"] for c in rep.checks)
    assert {c["name"] for c in rep.checks} == {
        "min_lot_vs_risk", "margin_headroom", "drawdown_vs_limit", "cost_impact"}


def test_suitability_flags_tiny_account_large_stop():
    # 500 USD, 1% = 5 USD intended; min lot 0.01 with a $12 stop risks
    # $12 = 2.4% of equity -> must be flagged.
    acct = Account(account_id="tiny", balance=500.0, leverage=100)
    rep = suitability_report(acct, PROFILE, XAU, price=2600.0)
    check = next(c for c in rep.checks if c["name"] == "min_lot_vs_risk")
    assert not check["passed"]
    assert rep.suitable is False


def test_suitability_flags_drawdown_over_limit():
    acct = Account(account_id="careful", balance=10_000.0,
                   limits=AccountLimits(max_drawdown_pct=20.0))
    rep = suitability_report(acct, PROFILE, XAU, price=2600.0)
    check = next(c for c in rep.checks if c["name"] == "drawdown_vs_limit")
    assert not check["passed"]          # 35% strategy DD > 20% account limit
    assert rep.suitable is False


def test_suitability_flags_cost_impact():
    # Tight $2 stop: cost 0.50/round trip on a 0.5-lot position =
    # 0.50*100*0.5 = 25 vs risk 0.5*2*100 = 100 -> 25% (passes).
    # Widen cost to 1.50 -> 75 > 33% -> flagged.
    pricey = StrategyProfile(
        key="x", name="x", max_dd_pct=10.0, typical_stop_distance=2.0,
        recommended_risk_pct=1.0, cost_per_round_trip=1.50)
    acct = Account(account_id="a", balance=10_000.0, leverage=100)
    rep = suitability_report(acct, pricey, XAU, price=2600.0)
    check = next(c for c in rep.checks if c["name"] == "cost_impact")
    assert not check["passed"]
    assert "75.00" in check["detail"]


def test_suitability_margin_headroom_flag():
    # Leverage 2: sized lot margin 0.01*100*2600/2 = 1300 > 0.8*equity? No --
    # equity 10k, 80% = 8000 -> passes. Use a 1k account: raw lot 0.0083 ->
    # below min -> sizing fails -> margin_headroom must fail.
    acct = Account(account_id="small", balance=1_000.0, leverage=2)
    rep = suitability_report(acct, PROFILE, XAU, price=2600.0)
    check = next(c for c in rep.checks if c["name"] == "margin_headroom")
    assert not check["passed"]
    assert rep.suitable is False


def test_suitability_currency_conversion_recorded():
    acct = Account(account_id="eur", balance=10_000.0, leverage=100,
                   currency="EUR",
                   limits=AccountLimits(max_drawdown_pct=50.0))
    rep = suitability_report(acct, PROFILE, XAU, price=2600.0,
                             rate=CurrencyRate(rate=0.92, timestamp_ms=1,
                                               currency="EUR"))
    assert rep.suitable is True
    detail = next(c for c in rep.checks if c["name"] == "min_lot_vs_risk")["detail"]
    assert "EUR" in detail


# ── Adapted backtest: account-specific equity curve ──────────────────────────

def test_adapted_backtest_runs_with_account_equity():
    acct = Account(account_id="10k", balance=10_000.0, leverage=100)
    cfg_a = adapt_config(acct, PROFILE, XAU, price=2600.0)
    bars = make_bars(1600)
    strat_cfg = StrategyConfig(
        sl_method="atr", sl_atr_mult=1.5, rr=1.5,
        risk_pct=cfg_a.risk_per_trade_pct,
        max_trades_per_day=cfg_a.max_trades_per_day,
        daily_loss_limit_pct=cfg_a.daily_loss_limit_pct,
    )
    equity, trades = run_backtest(bars, EMACross(strat_cfg, fast=20, slow=60),
                                  strat_cfg, start_equity=acct.balance)
    # The account-specific curve starts at the account balance.
    assert float(equity.iloc[0]) == acct.balance
    assert len(equity) == len(bars)
    assert isinstance(trades, pd.DataFrame)


# ── Continuous adaptation suggestions (suggestions ONLY) ─────────────────────

from core.adaptation import suggest_adaptation  # noqa: E402


def test_suggestion_on_deep_drawdown():
    acct = Account(account_id="a", balance=10_000.0, floating_pnl=-1_500.0,
                   limits=AccountLimits(risk_per_trade_pct=1.0))
    s = suggest_adaptation(acct, PROFILE, XAU, price=2600.0,
                           current_atr=8.0, equity_high_watermark=10_000.0)
    risk = next(x for x in s if x["what"] == "risk_per_trade_pct")
    assert risk["from"] == 1.0 and risk["to"] == 0.5
    assert "15.0% below" in risk["reason"]


def test_suggestion_on_equity_growth_capped():
    acct = Account(account_id="a", balance=10_000.0, floating_pnl=1_000.0,
                   limits=AccountLimits(risk_per_trade_pct=1.0))
    s = suggest_adaptation(acct, PROFILE, XAU, price=2600.0,
                           current_atr=8.0, equity_high_watermark=10_000.0)
    risk = next(x for x in s if x["what"] == "risk_per_trade_pct")
    assert risk["to"] == 1.25  # +25%, cautious, never doubled


def test_suggestion_on_volatility_spike():
    # Profile stop 12.0 implies ATR 8.0; current ATR 14 -> 1.75x -> suggest
    # wider stop at 1.5 * 14 = 21.
    acct = Account(account_id="a", balance=10_000.0)
    s = suggest_adaptation(acct, PROFILE, XAU, price=2600.0, current_atr=14.0)
    stop = next(x for x in s if x["what"] == "stop_distance")
    assert stop["to"] == 21.0
    assert "1.8x" in stop["reason"] or "1.75x" in stop["reason"]


def test_suggestion_on_volatility_calm():
    acct = Account(account_id="a", balance=10_000.0)
    s = suggest_adaptation(acct, PROFILE, XAU, price=2600.0, current_atr=4.0)
    stop = next(x for x in s if x["what"] == "stop_distance")
    assert stop["to"] == 6.0  # 1.5 * 4.0


def test_no_suggestion_in_normal_regime():
    acct = Account(account_id="a", balance=10_000.0)
    s = suggest_adaptation(acct, PROFILE, XAU, price=2600.0,
                           current_atr=8.0, equity_high_watermark=10_000.0)
    assert s == ()  # equity at watermark, ATR at profile level


def test_suggestions_never_mutate_account():
    acct = Account(account_id="a", balance=10_000.0, floating_pnl=-1_500.0)
    before = (acct.balance, acct.limits.risk_per_trade_pct)
    suggest_adaptation(acct, PROFILE, XAU, price=2600.0, current_atr=14.0)
    assert (acct.balance, acct.limits.risk_per_trade_pct) == before
