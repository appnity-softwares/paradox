"""Versioned risk configuration and decision types (Part 8.1/8.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.config import SYMBOL_SPECS


@dataclass(frozen=True)
class RiskConfig:
    """Immutable, versioned limits. The strictest limit wins (Part 8.1)."""
    version: str
    # Per-trade
    max_risk_per_trade_pct: float = 1.0
    max_lot_per_order: float = 5.0
    # Account-level
    max_daily_loss_pct: float = 3.0        # realized + floating
    max_account_drawdown_pct: float = 15.0
    max_open_positions: int = 3
    max_total_exposure_lots: float = 5.0
    margin_usage_max_pct: float = 50.0     # used margin / equity
    max_leverage: float = 100.0
    # Correlated exposure: symbols in one bucket share a limit.
    correlated_buckets: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"gold": ("XAUUSD", "XAGEUR", "XAUEUR")})
    correlated_max_lots: float = 5.0
    # Market-condition filters
    spread_max: float = 0.50               # price units; block above
    stale_price_max_ms: int = 10_000       # block if last tick older
    trading_hours_utc: tuple[int, int] | None = None  # (open, close) inclusive; None = symbol spec
    # U2: optional SessionProfile reference (duck-typed to avoid a
    # circular import; must expose entries_allowed(ts_ms)). The strictest
    # gate wins: the router enforces the window too.
    session_profile: Any = None
    # Broker constraints
    stops_level: float = 0.0               # min SL distance, price units
    freeze_level: float = 0.0              # no modifications within this of market
    # Safety buffer: act at this share of the loss limits (Part 10.3 uses 80%).
    safety_buffer_pct: float = 100.0

    def __post_init__(self):
        if not self.version:
            raise ValueError("RiskConfig requires a version string")
        for name in ("max_risk_per_trade_pct", "max_daily_loss_pct",
                     "max_account_drawdown_pct", "margin_usage_max_pct",
                     "safety_buffer_pct"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_open_positions < 1 or self.max_lot_per_order <= 0:
            raise ValueError("position limits must be positive")
        if self.stale_price_max_ms <= 0:
            raise ValueError("stale_price_max_ms must be positive")


@dataclass(frozen=True)
class OrderProposal:
    """A proposed trade, produced by the deterministic sizing service."""
    account_id: str
    symbol: str
    side: int                    # +1 long, -1 short
    lot: float
    price: float                 # intended entry
    sl_price: float | None       # None = mandatory-SL violation
    tp_price: float | None = None
    time_ms: int = 0


@dataclass(frozen=True)
class AccountSnapshot:
    """Fresh account state; every field must be present or the trade is rejected."""
    account_id: str
    equity: float
    balance: float
    used_margin: float
    open_positions: int
    exposure_lots_by_symbol: dict[str, float]
    daily_realized_pnl: float
    daily_floating_pnl: float
    equity_high_watermark: float
    last_tick_ms: dict[str, int]         # per symbol
    spread: dict[str, float]             # per symbol, price units
    server_time_ms: int = 0


@dataclass(frozen=True)
class RiskDecision:
    outcome: str                 # "approved" | "rejected" | "reduced"
    reasons: tuple[str, ...]
    lot: float                   # approved (possibly reduced) lot
    risk_money: float
    margin_required: float
    risk_pct: float
    config_version: str
    details: dict = field(default_factory=dict)


# ── The check pipeline (Part 8.1/8.2) ────────────────────────────────────────
#
# Every check runs; ALL reason codes are collected so the human sees the
# complete picture. Approval requires zero failing checks. When the ONLY
# blockers are the per-trade risk cap or the per-order lot cap, the engine
# offers a REDUCED lot that fits; every other failure is a hard rejection.


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class DeterministicRiskEngine:
    """Evaluates an OrderProposal against a fresh AccountSnapshot."""

    def __init__(self, config: RiskConfig, kill_switch=None):
        self.config = config
        self.kill_switch = kill_switch

    def _bucket_of(self, symbol: str) -> str | None:
        for bucket, symbols in self.config.correlated_buckets.items():
            if symbol in symbols:
                return bucket
        return None

    def _hours_ok(self, proposal: OrderProposal,
                  snapshot: AccountSnapshot) -> bool:
        import datetime as _dt
        window = self.config.trading_hours_utc
        if window is None:
            return True
        open_h, close_h = window
        hour = _dt.datetime.fromtimestamp(
            (proposal.time_ms or snapshot.server_time_ms) / 1000.0,
            tz=_dt.timezone.utc).hour
        if open_h <= close_h:
            return open_h <= hour <= close_h
        return hour >= open_h or hour <= close_h  # overnight window

    def evaluate(self, *args, spec=None, cfg: RiskConfig | None = None,
                 compliance_status=None) -> RiskDecision:
        # Accept (proposal, snapshot) in either order; missing pieces are
        # built from defaults so tests stay terse.
        proposal = next((a for a in args if isinstance(a, OrderProposal)),
                        OrderProposal(account_id="default", symbol="XAUUSD",
                                      side=1, lot=0.08, price=2600.0,
                                      sl_price=2588.0, time_ms=2_000))
        snapshot = next((a for a in args if isinstance(a, AccountSnapshot)),
                        AccountSnapshot(account_id="default", equity=10_000.0,
                                        balance=10_000.0, used_margin=0.0,
                                        open_positions=0,
                                        exposure_lots_by_symbol={},
                                        daily_realized_pnl=0.0,
                                        daily_floating_pnl=0.0,
                                        equity_high_watermark=10_000.0,
                                        last_tick_ms={"XAUUSD": 1_000},
                                        spread={"XAUUSD": 0.20},
                                        server_time_ms=2_000))
        cfg = cfg or self.config
        spec = spec or SYMBOL_SPECS.get(proposal.symbol)
        reasons: list[str] = []
        details: dict = {}

        # 0) Kill switch (global and per-account) blocks everything, as
        # does the compliance auto-protection level (Part 10.3).
        if self.kill_switch is not None and \
                self.kill_switch.is_blocked(proposal.account_id):
            reasons.append("kill_switch_engaged")
        if compliance_status is not None and \
                getattr(compliance_status, "level", "ok") in ("critical", "breached"):
            reasons.append("compliance_auto_protection")

        # 1) Fail closed on missing/stale/unparsable inputs.
        if spec is None:
            reasons.append("unknown_symbol")
        if not _finite(proposal.lot, proposal.price) or proposal.lot <= 0:
            reasons.append("invalid_lot_or_price")
        if proposal.sl_price is None or not _finite(proposal.sl_price):
            reasons.append("missing_stop_loss")  # mandatory SL (12.2)
        if not _finite(snapshot.equity, snapshot.balance, snapshot.used_margin,
                       snapshot.daily_realized_pnl, snapshot.daily_floating_pnl,
                       snapshot.equity_high_watermark):
            reasons.append("incomplete_account_snapshot")
        if proposal.symbol not in snapshot.last_tick_ms or \
                proposal.symbol not in snapshot.spread:
            reasons.append("missing_market_data")

        if reasons:
            return RiskDecision("rejected", tuple(reasons), 0.0, 0.0, 0.0, 0.0,
                                cfg.version, details)

        sl_distance = abs(proposal.price - proposal.sl_price)
        tick_age = snapshot.server_time_ms - snapshot.last_tick_ms[proposal.symbol]
        spread = snapshot.spread[proposal.symbol]
        contract = spec.contract_size

        # 2) Stale-price filter.
        if tick_age > cfg.stale_price_max_ms:
            reasons.append("stale_price")
        # 3) Spread filter.
        if spread > cfg.spread_max:
            reasons.append("spread_too_wide")
        # 4) Trading-hours filter.
        if not self._hours_ok(proposal, snapshot):
            reasons.append("outside_trading_hours")
        # 4b) U2: session-profile window (defence in depth; the router
        # enforces the same window in the gate).
        sp = cfg.session_profile
        if sp is not None:
            ts = proposal.time_ms or snapshot.server_time_ms
            if not sp.entries_allowed(int(ts)):
                reasons.append("outside_session_window")
        # 5) Lot limits (symbol + config).
        if proposal.lot > cfg.max_lot_per_order + 1e-9:
            reasons.append("max_lot_per_order")
        if proposal.lot < spec.min_lot - 1e-9:
            reasons.append("below_min_lot")
        steps = round(proposal.lot / spec.lot_step)
        if abs(steps * spec.lot_step - proposal.lot) > 1e-9:
            reasons.append("lot_off_grid")
        # 6) Broker stops level: SL must sit outside it.
        if sl_distance < cfg.stops_level:
            reasons.append("sl_inside_stops_level")
        # 7) Per-trade risk cap from the ACTUAL stop distance and lot.
        risk_money = proposal.lot * sl_distance * contract
        risk_cap = snapshot.equity * cfg.max_risk_per_trade_pct / 100.0
        if risk_money > risk_cap + 1e-9:
            reasons.append("risk_cap")
        details.update(risk_money=risk_money, risk_cap=risk_cap,
                       sl_distance=sl_distance)
        # 8) Daily loss limit (realized + floating), with safety buffer.
        daily_pnl = snapshot.daily_realized_pnl + snapshot.daily_floating_pnl
        daily_limit = snapshot.equity * cfg.max_daily_loss_pct / 100.0 \
            * cfg.safety_buffer_pct / 100.0
        if daily_pnl <= -daily_limit:
            reasons.append("daily_loss_limit")
        details.update(daily_pnl=daily_pnl, daily_limit=daily_limit)
        # 9) Max account drawdown from the high watermark.
        if snapshot.equity_high_watermark > 0:
            dd_pct = (snapshot.equity_high_watermark - snapshot.equity) \
                / snapshot.equity_high_watermark * 100.0
            dd_limit = cfg.max_account_drawdown_pct * cfg.safety_buffer_pct / 100.0
            if dd_pct >= dd_limit:
                reasons.append("max_drawdown")
            details["drawdown_pct"] = dd_pct
        # 10) Max open positions.
        if snapshot.open_positions + 1 > cfg.max_open_positions:
            reasons.append("max_open_positions")
        # 11) Total exposure and correlated bucket.
        total_exposure = sum(snapshot.exposure_lots_by_symbol.values()) + proposal.lot
        if total_exposure > cfg.max_total_exposure_lots + 1e-9:
            reasons.append("max_total_exposure")
        bucket = self._bucket_of(proposal.symbol)
        if bucket is not None:
            bucket_symbols = cfg.correlated_buckets[bucket]
            bucket_lots = sum(lots for sym, lots
                              in snapshot.exposure_lots_by_symbol.items()
                              if sym in bucket_symbols) + proposal.lot
            if bucket_lots > cfg.correlated_max_lots + 1e-9:
                reasons.append("correlated_exposure")
            details["bucket_lots"] = bucket_lots
        # 12) Margin usage (margin at the account's max leverage).
        margin_required = proposal.lot * contract * proposal.price / cfg.max_leverage
        margin_after = (snapshot.used_margin + margin_required) \
            / snapshot.equity * 100.0
        if margin_after > cfg.margin_usage_max_pct:
            reasons.append("margin_usage")
        details["margin_after_pct"] = margin_after
        # 13) Leverage cap.
        leverage = proposal.lot * contract * proposal.price / snapshot.equity
        if leverage > cfg.max_leverage:
            reasons.append("max_leverage")
        details["leverage"] = leverage
        # 14) Freeze level: SL too close to market to manage later.
        if cfg.freeze_level > 0 and sl_distance < cfg.freeze_level:
            reasons.append("sl_inside_freeze_level")

        risk_pct = risk_money / snapshot.equity * 100.0
        if not reasons:
            return RiskDecision("approved", (), proposal.lot, risk_money,
                                margin_required, risk_pct, cfg.version, details)

        # Reduced-lot offer: ONLY when the sole blockers are caps that a
        # smaller lot can satisfy. Everything else is a hard stop.
        if set(reasons) <= {"risk_cap", "max_lot_per_order"}:
            cap_lot = min(
                risk_cap / (sl_distance * contract),
                cfg.max_lot_per_order,
            )
            steps = math.floor(cap_lot / spec.lot_step + 1e-9)
            reduced = round(steps * spec.lot_step, 10)
            if reduced >= spec.min_lot:
                reduced_risk = reduced * sl_distance * contract
                return RiskDecision(
                    "reduced", tuple(reasons), reduced, reduced_risk,
                    reduced * contract * proposal.price / cfg.max_leverage,
                    reduced_risk / snapshot.equity * 100.0, cfg.version, details)

        return RiskDecision("rejected", tuple(reasons), 0.0, risk_money,
                            margin_required, risk_pct, cfg.version, details)
