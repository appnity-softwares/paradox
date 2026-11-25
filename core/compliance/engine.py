"""Deterministic compliance checks (Part 10.3).

- pre_trade_check: simulates the account AFTER the proposed trade,
  including the stop-loss worst case, and rejects if any limit could be
  breached. Uses a configurable safety buffer (default 80% of limits).
- monitor: on every equity update computes remaining daily loss,
  remaining max loss, drawdown, profit-target progress, trading days,
  and an overall status (ok | warning | critical | breached).
- Auto-protection is expressed through the status; the order gate reads
  it and blocks new orders at `critical` or worse.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from core.compliance.rules import STATUS_VERIFIED, FirmRuleSet


@dataclass(frozen=True)
class AccountState:
    """Fresh compliance-relevant account state."""
    account_id: str
    balance: float
    equity: float
    # Reference values frozen at the LAST daily reset, per the firm's basis.
    balance_at_reset: float
    equity_at_reset: float
    equity_high_watermark: float
    open_positions: int
    exposure_lots: float
    # Today's realized + floating P&L in account currency.
    daily_realized_pnl: float = 0.0
    daily_floating_pnl: float = 0.0
    total_profit_so_far: float = 0.0        # closed, since challenge start
    today_profit: float = 0.0               # closed, today only
    trading_days_completed: int = 0
    open_since_ms: dict[str, int] = field(default_factory=dict)  # pos id -> open time


@dataclass(frozen=True)
class ProposedTrade:
    symbol: str
    lot: float
    price: float
    sl_price: float | None
    time_ms: int
    side: int = 1


@dataclass(frozen=True)
class ComplianceDecision:
    allowed: bool
    reasons: tuple[str, ...]
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ComplianceStatus:
    level: str                # ok | warning | critical | breached
    remaining_daily_loss: float
    remaining_max_loss: float
    current_drawdown_pct: float
    profit_target_progress_pct: float
    trading_days_completed: int
    reasons: tuple[str, ...] = ()


class ComplianceEngine:
    def __init__(self, safety_buffer_pct: float = 80.0):
        self.safety_buffer_pct = safety_buffer_pct

    # ── helpers ──────────────────────────────────────────────────────────
    def _daily_reference(self, rs: FirmRuleSet, s: AccountState) -> float:
        """The reference the firm measures daily loss against."""
        if rs.daily_loss.basis == "balance":
            return s.balance_at_reset
        if rs.daily_loss.basis == "equity":
            return s.equity_at_reset
        return max(s.balance_at_reset, s.equity_at_reset)  # balance_or_equity

    def _daily_loss_now(self, rs: FirmRuleSet, s: AccountState) -> float:
        """Positive number = loss so far today, floating included iff the
        firm says so."""
        pnl = s.daily_realized_pnl
        if rs.daily_loss.includes_floating:
            pnl += s.daily_floating_pnl
        return -pnl

    def _max_loss_reference(self, rs: FirmRuleSet, s: AccountState) -> float:
        if rs.max_loss.type == "static" or \
                rs.max_loss.basis == "starting_balance":
            return rs.starting_balance
        return s.equity_high_watermark  # trailing on equity HWM

    def _max_loss_now(self, rs: FirmRuleSet, s: AccountState) -> float:
        ref = self._max_loss_reference(rs, s)
        equity = s.equity
        if not rs.max_loss.includes_floating:
            equity = s.balance
        return max(0.0, ref - equity)

    # ── pre-trade check ──────────────────────────────────────────────────
    def pre_trade_check(self, rs: FirmRuleSet | None, state: AccountState,
                        trade: ProposedTrade,
                        calendar_events_ms: dict[int, str] | None = None,
                        spec_max_lot: float | None = None) -> ComplianceDecision:
        """calendar_events_ms: {event_start_ms: event_name}. A None calendar
        while the news restriction is enabled means BLOCK (fail closed)."""
        # 10.2 step 4: draft/missing rules grant NO permission.
        if rs is None or not rs.grants_trading_permission():
            return ComplianceDecision(
                False, ("rules_not_verified",),
                {"status": rs.status if rs else "missing"})
        problems = rs.validate()
        if problems:
            return ComplianceDecision(False, ("ruleset_invalid",),
                                      {"problems": problems})

        reasons: list[str] = []
        details: dict = {}
        buf = self.safety_buffer_pct / 100.0

        # Allowed instruments.
        if rs.allowed_symbols and trade.symbol not in rs.allowed_symbols:
            reasons.append("symbol_not_allowed")

        # Trading hours.
        if rs.trading_hours_utc is not None:
            hour = _dt.datetime.fromtimestamp(trade.time_ms / 1000.0,
                                              tz=_dt.timezone.utc).hour
            open_h, close_h = rs.trading_hours_utc
            in_hours = (open_h <= hour <= close_h) if open_h <= close_h \
                else (hour >= open_h or hour <= close_h)
            if not in_hours:
                reasons.append("outside_trading_hours")

        # Lot and position limits.
        if rs.max_lot is not None and trade.lot > rs.max_lot:
            reasons.append("max_lot")
        if spec_max_lot is not None and trade.lot > spec_max_lot:
            reasons.append("max_lot_symbol_spec")
        if rs.max_open_positions is not None and \
                state.open_positions + 1 > rs.max_open_positions:
            reasons.append("max_open_positions")
        if rs.max_exposure_lots is not None and \
                state.exposure_lots + trade.lot > rs.max_exposure_lots:
            reasons.append("max_exposure_lots")

        # News restriction: block if the trade time falls inside a window;
        # a missing calendar while enabled blocks unconditionally.
        nr = rs.news_restriction
        if nr.enabled:
            if calendar_events_ms is None:
                reasons.append("calendar_unavailable")
            else:
                before = nr.window_before_min * 60_000
                after = nr.window_after_min * 60_000
                for start_ms in calendar_events_ms:
                    if trade.time_ms is None:
                        continue
                    if start_ms - before <= trade.time_ms <= start_ms + after:
                        reasons.append("news_window")
                        details["news_event"] = calendar_events_ms[start_ms]
                        break

        # Weekend holding: no NEW positions after the Friday cutoff.
        if rs.weekend_holding == "forbidden":
            weekday = _dt.datetime.fromtimestamp(trade.time_ms / 1000.0,
                                                 tz=_dt.timezone.utc).weekday()
            hour = _dt.datetime.fromtimestamp(trade.time_ms / 1000.0,
                                              tz=_dt.timezone.utc).hour
            if weekday == 4 and hour >= rs.weekend_cutoff_utc_hour:
                reasons.append("weekend_holding_forbidden")

        # Stop-loss worst case projection (mandatory SL for the simulation).
        if trade.sl_price is None:
            reasons.append("missing_stop_loss")
        else:
            sl_distance = abs(trade.price - trade.sl_price)
            worst_case_loss = trade.lot * sl_distance * 100.0  # contract size
            # Daily loss AFTER the worst case, on the firm's basis.
            ref = self._daily_reference(rs, state)
            projected_loss = self._daily_loss_now(rs, state) + worst_case_loss
            daily_limit = ref * rs.daily_loss.limit_pct / 100.0 * buf
            if projected_loss >= daily_limit:
                reasons.append("daily_loss_worst_case")
            details["projected_daily_loss"] = projected_loss
            details["daily_limit_buffered"] = daily_limit
            # Max loss AFTER the worst case.
            max_ref = self._max_loss_reference(rs, state)
            projected_max = self._max_loss_now(rs, state) + worst_case_loss
            max_limit = max_ref * rs.max_loss.limit_pct / 100.0 * buf
            if projected_max >= max_limit:
                reasons.append("max_loss_worst_case")
            details["projected_max_loss"] = projected_max
            details["max_limit_buffered"] = max_limit

        # Consistency: block a trade whose best-day share would exceed the cap.
        if rs.consistency_rule.enabled and rs.consistency_rule.best_day_max_share:
            total = state.total_profit_so_far
            if total > 0:
                best_day = max(state.today_profit, 0.0)
                if best_day / total > rs.consistency_rule.best_day_max_share:
                    reasons.append("consistency_best_day")

        return ComplianceDecision(not reasons, tuple(reasons), details)

    # ── real-time monitoring ─────────────────────────────────────────────
    def monitor(self, rs: FirmRuleSet, state: AccountState) -> ComplianceStatus:
        buf = self.safety_buffer_pct / 100.0
        ref = self._daily_reference(rs, state)
        daily_limit = ref * rs.daily_loss.limit_pct / 100.0
        daily_used = self._daily_loss_now(rs, state)
        remaining_daily = max(0.0, daily_limit - daily_used)

        max_ref = self._max_loss_reference(rs, state)
        max_limit = max_ref * rs.max_loss.limit_pct / 100.0
        max_used = self._max_loss_now(rs, state)
        remaining_max = max(0.0, max_limit - max_used)

        hwm = state.equity_high_watermark or state.equity
        dd_pct = max(0.0, (hwm - state.equity) / hwm * 100.0) if hwm else 0.0
        target_progress = state.total_profit_so_far / rs.profit_target_amount \
            * 100.0 if rs.profit_target_amount > 0 else 0.0

        reasons: list[str] = []
        if remaining_max <= 0 or remaining_daily <= 0:
            level = "breached"
            reasons.append("loss_limit_reached")
        elif daily_used >= daily_limit * buf or max_used >= max_limit * buf:
            level = "critical"       # auto-protection: block new orders
            reasons.append("safety_buffer_hit")
        elif daily_used >= daily_limit * 0.5 or max_used >= max_limit * 0.5:
            level = "warning"
            reasons.append("half_of_limit_used")
        else:
            level = "ok"

        return ComplianceStatus(level, remaining_daily, remaining_max,
                                dd_pct, target_progress,
                                state.trading_days_completed,
                                tuple(reasons))
