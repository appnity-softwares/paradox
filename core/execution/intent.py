"""Order flow types (Part 12.1).

`OrderIntent` is what a signal or a human idea produces. Only the
execution gate may mint an `ApprovedOrder`, and the broker adapter
accepts NOTHING else — skipping a step is impossible by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.execution.versions import StrategyVersion
from core.risk.engine import RiskDecision


@dataclass(frozen=True)
class OrderIntent:
    idempotency_key: str          # unique per logical order; retries reuse it
    account_id: str
    symbol: str
    side: int                     # +1 long, -1 short
    intended_risk_pct: float
    strategy_version: StrategyVersion
    sl_price: float | None
    tp_price: float | None = None
    source: str = "signal"        # "signal" | "manual"
    created_ms: int = 0
    comment: str = ""
    # "market" | "buy_stop" | "sell_stop" | "buy_limit" | "sell_limit"
    order_type: str = "market"
    limit_price: float | None = None   # trigger for pending order types

    ORDER_TYPES = ("market", "buy_stop", "sell_stop", "buy_limit",
                   "sell_limit")

    def __post_init__(self) -> None:
        if self.order_type not in self.ORDER_TYPES:
            raise ValueError(f"unknown order_type {self.order_type!r}")
        if self.order_type != "market" and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires limit_price")
        if self.order_type == "market" and self.limit_price is not None:
            raise ValueError("market order must not carry limit_price")


@dataclass(frozen=True)
class ApprovedOrder:
    """Minted ONLY by ExecutionRouter after sizing, risk, compliance and
    mode approval all passed. Carries the full decision trail."""
    intent: OrderIntent
    lot: float
    price_ref: float
    risk_decision: RiskDecision
    compliance_reasons: tuple[str, ...]
    mode: str
    approval_token: str | None
    approved_ms: int
    decision_trail: tuple[dict, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionResult:
    idempotency_key: str
    status: str                   # filled | pending_accepted | rejected | \
                                  # pending_unknown | duplicate
    reasons: tuple[str, ...] = ()
    order_ref: str | None = None
    fill_price: float | None = None
    filled_lot: float = 0.0
    steps: tuple[dict, ...] = ()
