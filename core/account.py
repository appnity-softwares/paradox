"""Account model (Master Prompt Part 7.1).

Balance, equity, free margin, used margin, margin level, leverage,
account currency, limits, open positions. Multiple accounts per user are
just multiple instances with separate settings. Pure data + derived
properties; no LLM, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class AccountLimits:
    """Per-account risk limits (adaptation and the risk engine read these)."""
    risk_per_trade_pct: float = 1.0
    max_daily_risk_pct: float = 3.0
    max_open_positions: int = 3
    max_total_exposure_lots: float = 5.0
    leverage_cap: float = 100.0
    max_drawdown_pct: float = 20.0

    def __post_init__(self):
        if not 0 < self.risk_per_trade_pct <= 100:
            raise ValueError("risk_per_trade_pct must be in (0, 100]")
        if not 0 < self.max_daily_risk_pct <= 100:
            raise ValueError("max_daily_risk_pct must be in (0, 100]")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")
        if self.max_total_exposure_lots <= 0:
            raise ValueError("max_total_exposure_lots must be positive")
        if self.leverage_cap <= 0:
            raise ValueError("leverage_cap must be positive")
        if self.max_drawdown_pct <= 0:
            raise ValueError("max_drawdown_pct must be positive")


@dataclass
class Position:
    symbol: str
    side: int                 # +1 long, -1 short
    lot: float
    entry_price: float
    margin_used: float        # account currency


@dataclass
class Account:
    account_id: str
    currency: str = "USD"
    balance: float = 0.0
    leverage: float = 100.0
    limits: AccountLimits = field(default_factory=AccountLimits)
    positions: tuple[Position, ...] = ()
    floating_pnl: float = 0.0

    def __post_init__(self):
        if self.balance < 0:
            raise ValueError("balance cannot be negative")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")

    @property
    def equity(self) -> float:
        return self.balance + self.floating_pnl

    @property
    def used_margin(self) -> float:
        return sum(p.margin_used for p in self.positions)

    @property
    def free_margin(self) -> float:
        return self.equity - self.used_margin

    @property
    def margin_level_pct(self) -> float:
        """Equity / used margin * 100; inf when no margin is used."""
        used = self.used_margin
        return float("inf") if used <= 0 else self.equity / used * 100.0

    @property
    def open_positions(self) -> int:
        return len(self.positions)

    @property
    def total_exposure_lots(self) -> float:
        return sum(p.lot for p in self.positions)

    def with_positions(self, *positions: Position) -> "Account":
        """Immutable-style update: returns a new Account with positions set."""
        return replace(self, positions=tuple(positions))
