"""Part 7.2: position sizing must match hand calculations EXACTLY.

Every expected value below is hand-calculated for XAUUSD
(contract_size=100 oz, min_lot=0.01, lot_step=0.01, max_lot=500):

    risk_money   = lot * stop_distance * 100            (profit ccy)
    margin       = lot * 100 * price / leverage          (margin ccy)
    raw_lot      = risk_target / (stop_distance * 100)
    lot          = raw_lot rounded DOWN to 0.01 steps

Account-currency conversion: 1 profit/margin-ccy unit = ``rate`` account
units (e.g. EUR account, USD symbol -> rate = 0.92).
"""
from __future__ import annotations

import math

import pytest

from core.config import SYMBOL_SPECS
from core.sizing import (
    CurrencyRate,
    margin_per_lot,
    round_lot_down,
    size_atr,
    size_fixed_lot,
    size_fixed_risk,
)

XAU = SYMBOL_SPECS["XAUUSD"]
APPROX = {"abs": 1e-6}


# ── size_fixed_risk: happy paths ──────────────────────────────────────────────

@pytest.mark.parametrize("case", [
    # (name, equity, risk_pct, stop, price, lev, max_lot, lot, risk, margin, passes)
    pytest.param(("t01 basic 0.2 lot", 10_000, 1.0, 5.00, 2000, 100, None,
                  0.20, 100.00, 400.00, True), id="t01"),
    pytest.param(("t02 tighter stop bigger lot", 10_000, 1.0, 2.50, 2000, 100, None,
                  0.40, 100.00, 800.00, True), id="t02"),
    pytest.param(("t03 wider stop smaller lot", 10_000, 1.0, 10.00, 2000, 100, None,
                  0.10, 100.00, 200.00, True), id="t03"),
    pytest.param(("t04 double risk doubles lot", 10_000, 2.0, 5.00, 2000, 100, None,
                  0.40, 200.00, 800.00, True), id="t04"),
    pytest.param(("t05 half percent small acct", 5_000, 0.5, 5.00, 2000, 100, None,
                  0.05, 25.00, 100.00, True), id="t05"),
    pytest.param(("t06 1k account", 1_000, 1.0, 5.00, 2000, 100, None,
                  0.02, 10.00, 40.00, True), id="t06"),
    pytest.param(("t07 500 account min lot", 500, 1.0, 5.00, 2000, 100, None,
                  0.01, 5.00, 20.00, True), id="t07"),
    pytest.param(("t08 round down 3.33 stop", 10_000, 1.0, 3.33, 2000, 100, None,
                  0.30, 99.90, 600.00, True), id="t08"),
    pytest.param(("t09 round down odd equity", 7_777, 2.0, 4.00, 2000, 100, None,
                  0.38, 152.00, 760.00, True), id="t09"),
    pytest.param(("t10 round down 6.25 stop", 12_345, 1.5, 6.25, 2000, 100, None,
                  0.29, 181.25, 580.00, True), id="t10"),
    pytest.param(("t11 higher price more margin", 10_000, 1.0, 5.00, 2700, 100, None,
                  0.20, 100.00, 540.00, True), id="t11"),
    pytest.param(("t12 leverage 30 margin", 10_000, 1.0, 5.00, 2000, 30, None,
                  0.20, 100.00, 1_333.333333, True), id="t12"),
    pytest.param(("t13 leverage 10 fits", 1_000, 1.0, 5.00, 2000, 10, None,
                  0.02, 10.00, 400.00, True), id="t13"),
    pytest.param(("t14 max_lot cap 50", 1_000_000, 1.0, 1.00, 2000, 100, 50.0,
                  50.00, 5_000.00, 100_000.00, True), id="t14"),
    pytest.param(("t15 symbol max_lot 500", 10_000_000, 1.0, 0.50, 2000, 100, None,
                  500.00, 25_000.00, 1_000_000.00, True), id="t15"),
])
def test_fixed_risk_hand_calculated(case):
    name, eq, pct, stop, price, lev, cap, lot, risk, margin, ok = case
    r = size_fixed_risk(eq, pct, stop, XAU, price, lev, max_lot=cap)
    assert r.lot == pytest.approx(lot, **APPROX), name
    assert r.risk_money == pytest.approx(risk, **APPROX), name
    assert r.margin_required == pytest.approx(margin, **APPROX), name
    assert r.passes is ok, name


# ── size_fixed_risk: below-minimum and margin failures ────────────────────────

def test_t16_below_minimum_fails_closed():
    # 400 * 1% = $4 target; stop $5 -> raw 0.008 < min_lot 0.01.
    r = size_fixed_risk(400, 1.0, 5.00, XAU, 2000, 100)
    assert r.lot == 0.0 and not r.passes
    assert any("lot below minimum" in x for x in r.reasons)


def test_t17_margin_exceeds_equity_fails():
    # equity 500, 1% -> target 5 -> raw 0.01 -> lot 0.01; at leverage 2 the
    # margin is 0.01*100*2000/2 = 1000 > 500 -> rejected on margin.
    r = size_fixed_risk(500, 1.0, 5.00, XAU, 2000, 2)
    assert r.lot == pytest.approx(0.01, **APPROX)
    assert not r.passes and any("margin" in x for x in r.reasons)


# ── account currency conversion (timestamped rate input) ─────────────────────

def test_t18_eur_account_conversion():
    # 10,000 EUR, 1% = 100 EUR = 108.6956.. USD; stop 5 -> raw 0.21739 -> 0.21
    # risk = 0.21*5*100 = 105 USD = 96.60 EUR; margin 420 USD = 386.40 EUR.
    r = size_fixed_risk(10_000, 1.0, 5.00, XAU, 2000, 100,
                        rate=CurrencyRate(rate=0.92, timestamp_ms=1_700_000_000_000,
                                          currency="EUR"))
    assert r.lot == pytest.approx(0.21, **APPROX)
    assert r.risk_money == pytest.approx(96.60, **APPROX)
    assert r.margin_required == pytest.approx(386.40, **APPROX)
    assert r.passes is True


def test_t19_eur_account_tiny():
    # 1,000 EUR, 1% = 10 EUR = 10.8696 USD; stop 5 -> raw 0.0217 -> 0.02
    # risk 10 USD = 9.20 EUR; margin 40 USD = 36.80 EUR.
    r = size_fixed_risk(1_000, 1.0, 5.00, XAU, 2000, 100,
                        rate=CurrencyRate(rate=0.92, timestamp_ms=1, currency="EUR"))
    assert r.lot == pytest.approx(0.02, **APPROX)
    assert r.risk_money == pytest.approx(9.20, **APPROX)
    assert r.margin_required == pytest.approx(36.80, **APPROX)


def test_t20_gbp_account_conversion():
    # 10,000 GBP, 1% = 100 GBP = 126.5823 USD; stop 2 -> raw 0.6329 -> 0.63
    # risk 126 USD = 99.54 GBP; margin 1260 USD = 995.40 GBP.
    r = size_fixed_risk(10_000, 1.0, 2.00, XAU, 2000, 100,
                        rate=CurrencyRate(rate=0.79, timestamp_ms=1, currency="GBP"))
    assert r.lot == pytest.approx(0.63, **APPROX)
    assert r.risk_money == pytest.approx(99.54, **APPROX)
    assert r.margin_required == pytest.approx(995.40, **APPROX)


# ── fail-closed inputs ────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", [
    pytest.param(("t21 zero risk_pct", dict(equity=10_000, risk_pct=0.0,
                 stop_distance=5.0, spec=XAU, price=2000, leverage=100)), id="t21"),
    pytest.param(("t22 zero stop", dict(equity=10_000, risk_pct=1.0,
                 stop_distance=0.0, spec=XAU, price=2000, leverage=100)), id="t22"),
    pytest.param(("t23 zero equity", dict(equity=0.0, risk_pct=1.0,
                 stop_distance=5.0, spec=XAU, price=2000, leverage=100)), id="t23"),
    pytest.param(("t24 negative price", dict(equity=10_000, risk_pct=1.0,
                 stop_distance=5.0, spec=XAU, price=-2000, leverage=100)), id="t24"),
    pytest.param(("t25 zero leverage", dict(equity=10_000, risk_pct=1.0,
                 stop_distance=5.0, spec=XAU, price=2000, leverage=0)), id="t25"),
    pytest.param(("t26 nan equity", dict(equity=float("nan"), risk_pct=1.0,
                 stop_distance=5.0, spec=XAU, price=2000, leverage=100)), id="t26"),
    pytest.param(("t27 inf stop", dict(equity=10_000, risk_pct=1.0,
                 stop_distance=float("inf"), spec=XAU, price=2000, leverage=100)), id="t27"),
])
def test_fixed_risk_fail_closed(case):
    name, kwargs = case
    r = size_fixed_risk(**kwargs)
    assert r.lot == 0.0 and not r.passes and r.reasons, name


def test_t28_zero_rate_fails_closed():
    r = size_fixed_risk(10_000, 1.0, 5.0, XAU, 2000, 100,
                        rate=CurrencyRate(rate=0.0, timestamp_ms=1))
    assert not r.passes and "conversion rate must be positive" in r.reasons[0]


# ── size_atr ─────────────────────────────────────────────────────────────────

def test_t29_atr_sizing_basic():
    # stop = 1.5 * 8.0 = 12.0; target 100 -> raw 0.0833 -> 0.08
    # risk = 0.08*12*100 = 96; margin = 0.08*100*2000/100 = 160.
    r = size_atr(10_000, 1.0, atr_value=8.0, atr_mult=1.5, spec=XAU,
                 price=2000, leverage=100)
    assert r.lot == pytest.approx(0.08, **APPROX)
    assert r.risk_money == pytest.approx(96.00, **APPROX)
    assert r.margin_required == pytest.approx(160.00, **APPROX)


def test_t30_atr_sizing_mult2():
    # stop = 2.0 * 5.0 = 10.0 -> raw 0.1 -> 0.10; risk 100; margin 200.
    r = size_atr(10_000, 1.0, atr_value=5.0, atr_mult=2.0, spec=XAU,
                 price=2000, leverage=100)
    assert r.lot == pytest.approx(0.10, **APPROX)
    assert r.risk_money == pytest.approx(100.00, **APPROX)


def test_t31_atr_sizing_odd():
    # stop = 1.2 * 12.5 = 15.0; 2500*2% = 50 -> raw 0.0333 -> 0.03
    # risk 45; margin 60.
    r = size_atr(2_500, 2.0, atr_value=12.5, atr_mult=1.2, spec=XAU,
                 price=2000, leverage=100)
    assert r.lot == pytest.approx(0.03, **APPROX)
    assert r.risk_money == pytest.approx(45.00, **APPROX)
    assert r.margin_required == pytest.approx(60.00, **APPROX)


@pytest.mark.parametrize("kwargs", [
    dict(atr_value=0.0, atr_mult=1.5), dict(atr_value=-1.0, atr_mult=1.5),
    dict(atr_value=8.0, atr_mult=0.0), dict(atr_value=float("nan"), atr_mult=1.5),
])
def test_t32_atr_invalid_fails_closed(kwargs):
    r = size_atr(10_000, 1.0, spec=XAU, price=2000, leverage=100, **kwargs)
    assert r.lot == 0.0 and not r.passes and r.reasons


# ── size_fixed_lot ───────────────────────────────────────────────────────────

def test_t33_fixed_lot_basic():
    # lot 0.05, stop 5 -> risk 25; margin 0.05*100*2000/100 = 100.
    r = size_fixed_lot(0.05, XAU, stop_distance=5.0, price=2000,
                       leverage=100, equity=10_000)
    assert r.lot == pytest.approx(0.05, **APPROX)
    assert r.risk_money == pytest.approx(25.00, **APPROX)
    assert r.margin_required == pytest.approx(100.00, **APPROX)
    assert r.passes is True


def test_t34_fixed_lot_min():
    # lot 0.01, stop 10 -> risk 10; margin 20.
    r = size_fixed_lot(0.01, XAU, stop_distance=10.0, price=2000,
                       leverage=100, equity=10_000)
    assert r.risk_money == pytest.approx(10.00, **APPROX)
    assert r.margin_required == pytest.approx(20.00, **APPROX)


def test_t35_fixed_lot_one():
    # lot 1.0, stop 5 -> risk 500; margin 2000.
    r = size_fixed_lot(1.0, XAU, stop_distance=5.0, price=2000,
                       leverage=100, equity=10_000)
    assert r.risk_money == pytest.approx(500.00, **APPROX)
    assert r.margin_required == pytest.approx(2_000.00, **APPROX)


def test_t36_fixed_lot_below_min_rejected():
    r = size_fixed_lot(0.005, XAU, stop_distance=5.0, price=2000,
                       leverage=100, equity=10_000)
    assert not r.passes and "lot below symbol minimum" in r.reasons[0]


def test_t37_fixed_lot_off_grid_rejected():
    r = size_fixed_lot(0.015, XAU, stop_distance=5.0, price=2000,
                       leverage=100, equity=10_000)
    assert not r.passes and "lot-step grid" in r.reasons[0]


def test_t38_fixed_lot_above_max_rejected():
    r = size_fixed_lot(600.0, XAU, stop_distance=5.0, price=2000,
                       leverage=100, equity=10_000_000)
    assert not r.passes and "lot above symbol maximum" in r.reasons[0]


def test_t39_fixed_lot_margin_fail():
    # lot 1.0 at leverage 20 -> margin 10,000 > equity 1,500.
    r = size_fixed_lot(1.0, XAU, stop_distance=5.0, price=2000,
                       leverage=20, equity=1_500)
    assert not r.passes and any("margin" in x for x in r.reasons)


# ── helpers ──────────────────────────────────────────────────────────────────

def test_t40_margin_per_lot_values():
    assert margin_per_lot(2000, XAU, 100) == pytest.approx(2000.0, **APPROX)
    assert margin_per_lot(2700, XAU, 100) == pytest.approx(2700.0, **APPROX)
    assert margin_per_lot(2000, XAU, 500) == pytest.approx(400.0, **APPROX)


def test_t41_round_lot_down():
    assert round_lot_down(0.2, XAU) == 0.20
    assert round_lot_down(0.3003, XAU) == 0.30
    assert round_lot_down(0.008, XAU) == 0.0      # below min -> untradable
    assert round_lot_down(-1.0, XAU) == 0.0
    assert round_lot_down(float("nan"), XAU) == 0.0
    assert round_lot_down(600.0, XAU) == 500.0    # clamped to symbol max
