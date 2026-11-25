"""Adaptation logic and suitability report (Master Prompt Part 7.3).

Deterministic only. An AI agent may SUGGEST changes; every number here is
computed from the account model, the symbol spec and the strategy's
validated profile. Nothing is LLM-derived.

A :class:`StrategyProfile` carries only validated, measured facts about a
strategy (worst historical drawdown, typical stop distance, round-trip
cost) — produced by the Phase 1 validation stack, not invented.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.account import Account
from core.config import SymbolSpec
from core.sizing import CurrencyRate, margin_per_lot, size_fixed_risk

# Cost impact above this fraction of the intended risk is flagged.
MAX_COST_FRACTION_OF_RISK = 0.33
# Margin buffer: adapted configs may use at most this share of equity.
MARGIN_UTILIZATION_CAP = 0.5


@dataclass(frozen=True)
class StrategyProfile:
    """Validated facts about a strategy (from the Phase 1 stack)."""
    key: str
    name: str
    max_dd_pct: float                 # historical worst drawdown (positive %)
    typical_stop_distance: float      # price units
    recommended_risk_pct: float       # from validation, <= account caps
    cost_per_round_trip: float        # price units (spread + slippage + commission)


@dataclass(frozen=True)
class AdaptedConfig:
    """The strategy configured for ONE account."""
    risk_per_trade_pct: float
    max_open_positions: int
    max_total_exposure_lots: float
    max_trades_per_day: int
    daily_loss_limit_pct: float
    lot: float                        # sized lot for one typical trade
    risk_money: float
    margin_required: float


@dataclass(frozen=True)
class SuitabilityReport:
    suitable: bool
    checks: tuple[dict, ...]          # name, passed, detail


def adapt_config(
    account: Account,
    profile: StrategyProfile,
    spec: SymbolSpec,
    price: float,
    rate: CurrencyRate | None = None,
) -> AdaptedConfig:
    """Derive the account-specific configuration for a strategy.

    Deterministic rules:
    - risk per trade = min(account cap, strategy recommendation)
    - exposure cap = min(account limit, margin-derived cap): the total lots
      whose margin stays within MARGIN_UTILIZATION_CAP of equity
    - max open positions = min(account limit, exposure cap / sized lot)
    - max trades per day = floor(daily risk budget / risk per trade)
    """
    risk_pct = min(account.limits.risk_per_trade_pct,
                   profile.recommended_risk_pct)

    sizing = size_fixed_risk(
        account.equity, risk_pct, profile.typical_stop_distance,
        spec, price, account.leverage, rate,
    )
    lot = sizing.lot

    margin_one_lot = margin_per_lot(price, spec, account.leverage)
    eff_rate = rate.rate if rate is not None else 1.0
    margin_cap_lots = (account.equity * MARGIN_UTILIZATION_CAP) / (margin_one_lot * eff_rate)
    exposure_cap = min(account.limits.max_total_exposure_lots, margin_cap_lots)

    max_positions = account.limits.max_open_positions
    if lot > 0:
        max_positions = min(max_positions, int(exposure_cap / lot))
    max_positions = max(1, max_positions)

    max_trades = max(1, int(account.limits.max_daily_risk_pct / risk_pct))

    return AdaptedConfig(
        risk_per_trade_pct=risk_pct,
        max_open_positions=max_positions,
        max_total_exposure_lots=round(exposure_cap, 2),
        max_trades_per_day=max_trades,
        daily_loss_limit_pct=account.limits.max_daily_risk_pct,
        lot=lot,
        risk_money=sizing.risk_money,
        margin_required=sizing.margin_required,
    )


def suitability_report(
    account: Account,
    profile: StrategyProfile,
    spec: SymbolSpec,
    price: float,
    rate: CurrencyRate | None = None,
) -> SuitabilityReport:
    """Is this strategy viable on this account? Deterministic checks:
    1. min-lot versus intended risk (the 0.01 floor can exceed the risk
       a small account intends to take),
    2. margin headroom for the sized lot,
    3. historical worst drawdown versus the account drawdown limit,
    4. cost impact versus the intended risk.
    """
    checks = []
    eff_rate = rate.rate if rate is not None else 1.0
    risk_pct = min(account.limits.risk_per_trade_pct,
                   profile.recommended_risk_pct)
    intended_risk = account.equity * risk_pct / 100.0

    # 1) Min-lot floor versus intended risk.
    min_lot_risk = spec.min_lot * profile.typical_stop_distance * spec.contract_size * eff_rate
    ok = min_lot_risk <= intended_risk + 1e-9
    checks.append({
        "name": "min_lot_vs_risk",
        "passed": ok,
        "detail": (f"min lot {spec.min_lot} risks {min_lot_risk:.2f} {account.currency} "
                   f"({min_lot_risk / account.equity * 100:.2f}% of equity); "
                   f"intended risk {intended_risk:.2f} ({risk_pct:.2f}%)"),
    })

    # 2) Margin headroom for the sized lot.
    sizing = size_fixed_risk(account.equity, risk_pct,
                             profile.typical_stop_distance, spec, price,
                             account.leverage, rate)
    ok = sizing.passes and sizing.margin_required <= account.free_margin * 0.8 + 1e-9
    checks.append({
        "name": "margin_headroom",
        "passed": ok,
        "detail": (f"sized lot {sizing.lot} needs margin {sizing.margin_required:.2f}; "
                   f"free margin {account.free_margin:.2f} (80% usable)"),
    })

    # 3) Historical worst drawdown versus the account limit.
    ok = profile.max_dd_pct <= account.limits.max_drawdown_pct
    checks.append({
        "name": "drawdown_vs_limit",
        "passed": ok,
        "detail": (f"strategy worst historical DD {profile.max_dd_pct:.1f}% vs "
                   f"account limit {account.limits.max_drawdown_pct:.1f}%"),
    })

    # 4) Cost impact versus intended risk.
    if sizing.lot > 0:
        cost_money = (profile.cost_per_round_trip * spec.contract_size
                      * sizing.lot * eff_rate)
        cost_fraction = cost_money / sizing.risk_money if sizing.risk_money > 0 else float("inf")
        ok = cost_fraction <= MAX_COST_FRACTION_OF_RISK
        checks.append({
            "name": "cost_impact",
            "passed": ok,
            "detail": (f"round-trip cost {cost_money:.2f} = "
                       f"{cost_fraction * 100:.1f}% of the {sizing.risk_money:.2f} risk "
                       f"(cap {MAX_COST_FRACTION_OF_RISK * 100:.0f}%)"),
        })
    else:
        checks.append({"name": "cost_impact", "passed": False,
                       "detail": "no tradable lot size; cost check n/a"})

    return SuitabilityReport(suitable=all(c["passed"] for c in checks),
                             checks=tuple(checks))


def suggest_adaptation(
    account: Account,
    profile: StrategyProfile,
    spec: SymbolSpec,
    price: float,
    current_atr: float,
    equity_high_watermark: float | None = None,
    rate: CurrencyRate | None = None,
) -> tuple[dict, ...]:
    """Continuous adaptation SUGGESTIONS only (Part 7.3).

    Detects equity change (drawdown from the high watermark) and
    volatility change (current ATR versus the ATR the profile's typical
    stop was derived from, assuming the standard 1.5x multiplier).
    Returns suggestion dicts; the caller is responsible for user approval
    and audit logging. Nothing here mutates any account or config.
    """
    suggestions = []
    hwm = equity_high_watermark if equity_high_watermark is not None else account.balance
    if hwm > 0:
        dd_pct = (hwm - account.equity) / hwm * 100.0
        if dd_pct >= 10.0:
            new_risk = round(max(0.25, account.limits.risk_per_trade_pct / 2.0), 2)
            suggestions.append({
                "what": "risk_per_trade_pct",
                "from": account.limits.risk_per_trade_pct,
                "to": new_risk,
                "reason": f"equity {dd_pct:.1f}% below high watermark {hwm:.2f}",
            })
        elif dd_pct <= -5.0:  # equity ABOVE the watermark: growth
            new_risk = round(min(2.0, account.limits.risk_per_trade_pct * 1.25), 2)
            suggestions.append({
                "what": "risk_per_trade_pct",
                "from": account.limits.risk_per_trade_pct,
                "to": new_risk,
                "reason": f"equity {abs(dd_pct):.1f}% above high watermark; "
                          "scaled cautiously, capped at 2%",
            })

    implied_atr = profile.typical_stop_distance / 1.5
    if implied_atr > 0 and current_atr > 0:
        ratio = current_atr / implied_atr
        if ratio >= 1.5:
            suggestions.append({
                "what": "stop_distance",
                "from": profile.typical_stop_distance,
                "to": round(1.5 * current_atr, 2),
                "reason": f"ATR {current_atr:.2f} is {ratio:.1f}x the profile's "
                          f"implied {implied_atr:.2f}; wider stops, smaller lots",
            })
        elif ratio <= 1 / 1.5:
            suggestions.append({
                "what": "stop_distance",
                "from": profile.typical_stop_distance,
                "to": round(1.5 * current_atr, 2),
                "reason": f"ATR {current_atr:.2f} is {1/ratio:.1f}x below the "
                          f"profile's implied {implied_atr:.2f}; stops can tighten",
            })
    return tuple(suggestions)
