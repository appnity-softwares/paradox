"""Deterministic Risk Engine.

Handles ALL risk calculations and order permissions. Agents can REQUEST risk checks
but cannot bypass them. This is the single source of truth for risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.config import SYMBOL_SPECS


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str
    adjusted_volume: float
    risk_amount_usd: float
    risk_pct: float
    sl_distance: float
    tp_distance: float
    notional_usd: float


class RiskEngine:
    def __init__(self, max_risk_pct: float = 2.0, max_daily_loss_pct: float = 5.0, max_leverage: float = 10.0):
        self.max_risk_pct = max_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_leverage = max_leverage
        self.daily_pnl_pct = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 4

    def check_position_risk(
        self,
        symbol: str,
        side: str,
        volume: float,
        risk_pct: float,
        account_equity: float,
        sl_distance: float,
        tp_distance: float | None = None,
        price: float | None = None,
    ) -> RiskCheckResult:
        """Deterministic risk check for a proposed position."""
        # Clamp risk to max allowed
        effective_risk_pct = min(risk_pct, self.max_risk_pct)
        if effective_risk_pct != risk_pct:
            reason = f"Risk capped from {risk_pct}% to {self.max_risk_pct}%"
        else:
            reason = "OK"

        risk_amount = account_equity * (effective_risk_pct / 100.0)

        if sl_distance <= 0:
            # Fail closed: never guess a stop distance (Master Prompt 8.2).
            return RiskCheckResult(
                allowed=False,
                reason="rejected: missing or non-positive stop distance",
                adjusted_volume=0.0, risk_amount_usd=0.0, risk_pct=0.0,
                sl_distance=sl_distance,
                tp_distance=tp_distance or 0.0, notional_usd=0.0,
            )

        units = risk_amount / sl_distance
        ref_price = float(price) if price and price > 0 else \
            float(SYMBOL_SPECS.get(symbol, SYMBOL_SPECS["XAUUSD"]).tick_value * 2000.0)
        notional = units * ref_price

        leverage = notional / account_equity if account_equity > 0 else 0
        if leverage > self.max_leverage:
            # Reduce volume to meet leverage limit
            max_notional = account_equity * self.max_leverage
            units = max_notional / ref_price
            volume = units
            risk_amount = units * sl_distance
            effective_risk_pct = (risk_amount / account_equity * 100.0) if account_equity > 0 else 0
            reason = f"Volume reduced for leverage limit ({leverage:.1f}x > {self.max_leverage}x)"

        tp_distance = tp_distance or sl_distance * 2.0

        # Check daily loss limit
        if self.daily_pnl_pct <= -self.max_daily_loss_pct:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily loss limit breached ({self.daily_pnl_pct:.1f}% <= -{self.max_daily_loss_pct}%)",
                adjusted_volume=0.0,
                risk_amount_usd=0.0,
                risk_pct=0.0,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
                notional_usd=0.0,
            )

        # Check daily trade count
        if self.daily_trades >= self.max_daily_trades:
            return RiskCheckResult(
                allowed=False,
                reason=f"Max daily trades reached ({self.daily_trades} >= {self.max_daily_trades})",
                adjusted_volume=0.0,
                risk_amount_usd=0.0,
                risk_pct=0.0,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
                notional_usd=0.0,
            )

        return RiskCheckResult(
            allowed=True,
            reason=reason,
            adjusted_volume=round(volume, 4),
            risk_amount_usd=round(risk_amount, 2),
            risk_pct=round(effective_risk_pct, 2),
            sl_distance=round(sl_distance, 2),
            tp_distance=round(tp_distance, 2),
            notional_usd=round(notional, 2),
        )

    def record_trade_result(self, pnl_pct: float) -> None:
        """Update daily PnL tracking (called after trade closes)."""
        self.daily_pnl_pct += pnl_pct
        self.daily_trades += 1

    def reset_daily(self) -> None:
        """Reset daily counters (called at new trading day)."""
        self.daily_pnl_pct = 0.0
        self.daily_trades = 0

    def get_status(self) -> dict:
        return {
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "max_risk_pct": self.max_risk_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "blocked": self.daily_pnl_pct <= -self.max_daily_loss_pct or self.daily_trades >= self.max_daily_trades,
        }
