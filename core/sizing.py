"""Position sizing service (Master Prompt Part 7.2).

Pure, deterministic functions. No LLM involvement anywhere: every output
is a closed-form calculation from the inputs. Fail-closed: invalid or
missing inputs produce ``lot=0, passes=False`` with reason codes, never a
guess.

Conventions
-----------
- ``stop_distance`` is in PRICE units (USD on gold), strictly positive.
- Risk money = lot * stop_distance * contract_size, in the PROFIT
  currency; converted to the ACCOUNT currency with the supplied rate when
  they differ.
- Margin = lot * contract_size * price / leverage, in the MARGIN
  currency; converted to the ACCOUNT currency with the same rate when
  they differ (rate semantics: 1 unit of profit/margin currency = ``rate``
  units of account currency).
- Lots are rounded DOWN to ``lot_step`` (never up: rounding up would
  exceed the intended risk). A raw size below ``min_lot`` fails: the
  0.01-lot floor would exceed the intended risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.config import SymbolSpec


@dataclass(frozen=True)
class CurrencyRate:
    """Timestamped conversion rate into the account currency.

    ``rate`` is account-currency units per one unit of the symbol's
    profit/margin currency (e.g. account EUR, symbol USD -> rate ~ 0.92).
    """
    rate: float = 1.0
    timestamp_ms: int = 0
    currency: str = "USD"


@dataclass(frozen=True)
class SizingResult:
    lot: float
    risk_money: float          # account currency
    margin_required: float     # account currency
    passes: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    risk_target: float = 0.0   # account currency, for reference


def _fail(*reasons: str) -> SizingResult:
    return SizingResult(lot=0.0, risk_money=0.0, margin_required=0.0,
                        passes=False, reasons=reasons)


def round_lot_down(raw_lot: float, spec: SymbolSpec) -> float:
    """Floor to the lot step; 0.0 when below the tradable minimum."""
    if not math.isfinite(raw_lot) or raw_lot <= 0:
        return 0.0
    steps = math.floor(raw_lot / spec.lot_step + 1e-9)
    lot = round(steps * spec.lot_step, 10)
    if lot < spec.min_lot:
        return 0.0
    return min(lot, spec.max_lot)


def margin_per_lot(price: float, spec: SymbolSpec, leverage: float) -> float:
    """Margin for one standard lot, in the margin currency."""
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    return spec.contract_size * price / leverage


def _validate_common(equity: float, risk_pct: float, stop_distance: float,
                     price: float, leverage: float, rate: CurrencyRate | None):
    if not math.isfinite(equity) or equity <= 0:
        return "equity must be positive"
    if not math.isfinite(risk_pct) or risk_pct <= 0:
        return "risk_pct must be positive"
    if not math.isfinite(stop_distance) or stop_distance <= 0:
        return "stop_distance must be positive"
    if not math.isfinite(price) or price <= 0:
        return "price must be positive"
    if not math.isfinite(leverage) or leverage <= 0:
        return "leverage must be positive"
    if rate is not None and (not math.isfinite(rate.rate) or rate.rate <= 0):
        return "conversion rate must be positive"
    return None


def size_fixed_risk(
    equity: float,
    risk_pct: float,
    stop_distance: float,
    spec: SymbolSpec,
    price: float,
    leverage: float,
    rate: CurrencyRate | None = None,
    max_lot: float | None = None,
) -> SizingResult:
    """Risk ``risk_pct`` of equity over ``stop_distance`` of price movement.

    The classic percent-risk formula, lot rounded down to the step.
    """
    err = _validate_common(equity, risk_pct, stop_distance, price, leverage, rate)
    if err:
        return _fail(err)

    eff_rate = rate.rate if rate is not None else 1.0
    # Risk target in the profit currency.
    risk_target_account = equity * risk_pct / 100.0
    risk_target_profit_ccy = risk_target_account / eff_rate

    raw_lot = risk_target_profit_ccy / (stop_distance * spec.contract_size)
    if max_lot is not None:
        raw_lot = min(raw_lot, max_lot)
    lot = round_lot_down(raw_lot, spec)
    if lot <= 0:
        return _fail("lot below minimum: intended risk cannot be taken "
                     f"with min_lot {spec.min_lot}")

    risk_money_profit_ccy = lot * stop_distance * spec.contract_size
    risk_money = risk_money_profit_ccy * eff_rate
    margin_req = margin_per_lot(price, spec, leverage) * lot * eff_rate
    passes = margin_req <= equity
    reasons = [] if passes else ["margin required exceeds equity"]
    return SizingResult(lot=lot, risk_money=risk_money,
                        margin_required=margin_req, passes=passes,
                        reasons=tuple(reasons),
                        risk_target=risk_target_account)


def size_atr(
    equity: float,
    risk_pct: float,
    atr_value: float,
    atr_mult: float,
    spec: SymbolSpec,
    price: float,
    leverage: float,
    rate: CurrencyRate | None = None,
    max_lot: float | None = None,
) -> SizingResult:
    """Volatility-based sizing: stop distance = ``atr_mult`` x ATR."""
    if not math.isfinite(atr_value) or atr_value <= 0:
        return _fail("atr_value must be positive")
    if not math.isfinite(atr_mult) or atr_mult <= 0:
        return _fail("atr_mult must be positive")
    return size_fixed_risk(equity, risk_pct, atr_value * atr_mult, spec,
                           price, leverage, rate, max_lot)


def size_fixed_lot(
    lot: float,
    spec: SymbolSpec,
    stop_distance: float,
    price: float,
    leverage: float,
    equity: float,
    rate: CurrencyRate | None = None,
) -> SizingResult:
    """Fixed lot size: report the implied risk and margin, check limits."""
    if not math.isfinite(lot) or lot <= 0:
        return _fail("lot must be positive")
    err = _validate_common(equity, 1.0, stop_distance, price, leverage, rate)
    if err:
        return _fail(err)
    if lot < spec.min_lot - 1e-9:
        return _fail("lot below symbol minimum")
    if lot > spec.max_lot + 1e-9:
        return _fail("lot above symbol maximum")
    # Snap to the lot grid for reporting; reject off-grid sizes.
    steps = round(lot / spec.lot_step)
    if abs(steps * spec.lot_step - lot) > 1e-9:
        return _fail("lot not on the lot-step grid")

    eff_rate = rate.rate if rate is not None else 1.0
    risk_money = lot * stop_distance * spec.contract_size * eff_rate
    margin_req = margin_per_lot(price, spec, leverage) * lot * eff_rate
    passes = margin_req <= equity
    reasons = [] if passes else ["margin required exceeds equity"]
    return SizingResult(lot=lot, risk_money=risk_money,
                        margin_required=margin_req, passes=passes,
                        reasons=tuple(reasons),
                        risk_target=0.0)
