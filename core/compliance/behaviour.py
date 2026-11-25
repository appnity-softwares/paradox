"""Behaviour controls (U1, Update Blueprint v2.1).

Deterministic, per-account behaviour limits that prop firms impose on
HOW an account trades (as opposed to HOW MUCH risk it may take, which
is the compliance engine's job):

- min_hold_seconds        — a position may not be closed or modified
                            before this age, EXCEPT protective actions
                            (kill switch, rule breach, stop-loss hits),
                            which are allowed and logged as exceptions.
- max_orders_per_day      — entries per UTC day.
- max_modifications_per_trade — SL/TP amendments per position.
- max_requests_per_minute — sliding window over ALL order requests.
- prohibited_styles       — martingale / grid / averaging_down detection
                            from the account's own entry history.
- hedging_policy          — forbid opposite-direction positions on the
                            same symbol.

Fail closed: a missing or stale history means the CONSERVATIVE check
applies (e.g. no history = no style violation detected, but caps still
enforced from what is known). All thresholds are configurable; the
defaults are documented in docs/behaviour_controls.md.

No LLM anywhere: every decision is arithmetic on timestamps, lots and
prices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MS_PER_MINUTE = 60_000

# Policy enum shared by ea/vps/ip/hedging/copy-trading policies.
POLICY_ALLOWED = "allowed"
POLICY_FORBIDDEN = "forbidden"
POLICY_CONDITIONAL = "conditional"
POLICY_UNKNOWN = "unknown"
POLICY_VALUES = (POLICY_ALLOWED, POLICY_FORBIDDEN,
                 POLICY_CONDITIONAL, POLICY_UNKNOWN)

PROHIBITED_STYLES = ("martingale", "grid", "averaging_down",
                     "tick_scalping", "latency_arbitrage")

# Reason codes (stable strings — they appear in audit logs and the UI).
R_MIN_HOLD = "min_hold_time"
R_MAX_ORDERS_DAY = "max_orders_per_day"
R_MAX_MODS = "max_modifications_per_trade"
R_MAX_REQUESTS = "max_requests_per_minute"
R_STYLE_PREFIX = "prohibited_style_"
R_HEDGING = "hedging_forbidden"


@dataclass
class BehaviourConfig:
    """Thresholds for the behaviour controls. `None` = control OFF.

    Defaults mirror the blueprint's safe defaults; a FirmRuleSet that
    specifies a value overrides it per account."""
    min_hold_seconds: int | None = None
    max_orders_per_day: int | None = None
    max_modifications_per_trade: int | None = None
    max_requests_per_minute: int | None = None
    # Styles actively blocked. The blueprint defaults these ON.
    prohibited_styles: tuple[str, ...] = ("martingale", "grid",
                                          "averaging_down")
    hedging_forbidden: bool = False
    # ── detection thresholds ─────────────────────────────────────────
    # martingale: a same-symbol, same-direction entry whose lot exceeds
    # the previous entry's lot by this factor after a closed LOSS.
    martingale_escalation_factor: float = 1.5
    martingale_lookback_min: int = 240
    # grid / averaging_down: this many same-direction entries at strictly
    # worsening prices within the window (grid), or while the first is
    # in loss (averaging_down).
    stack_max_entries: int = 3
    stack_window_min: int = 240

    def __post_init__(self) -> None:
        bad = [s for s in self.prohibited_styles
               if s not in PROHIBITED_STYLES]
        if bad:
            raise ValueError(f"unknown prohibited styles: {bad}")
        for name in ("min_hold_seconds", "max_orders_per_day",
                     "max_modifications_per_trade",
                     "max_requests_per_minute"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.martingale_escalation_factor <= 1.0:
            raise ValueError("martingale_escalation_factor must be > 1")
        if self.stack_max_entries < 2:
            raise ValueError("stack_max_entries must be >= 2")


@dataclass
class _Entry:
    ts_ms: int
    symbol: str
    side: int          # +1 long, -1 short
    lot: float
    price: float


@dataclass
class _OpenPos:
    position_id: str
    symbol: str
    side: int
    opened_ms: int
    entry_price: float
    lot: float = 1.0
    modifications: int = 0


@dataclass
class BehaviourDecision:
    allowed: bool
    reasons: tuple[str, ...]
    details: dict = field(default_factory=dict)


class BehaviourControls:
    """Per-account behaviour state and checks. One instance serves all
    accounts (state is keyed by account_id); construct per router."""

    def __init__(self, config: BehaviourConfig | None = None):
        self.config = config or BehaviourConfig()
        self._entries: dict[str, list[_Entry]] = {}
        self._closed: dict[str, list[dict]] = {}
        self._open: dict[str, dict[str, _OpenPos]] = {}
        self._requests: dict[str, list[int]] = {}
        self._day_orders: dict[str, dict[int, int]] = {}
        # Protective exceptions are ALLOWED but logged for the audit.
        self.exceptions: list[dict] = []

    def config_for(self, ruleset) -> BehaviourConfig:
        """Merge the base config with a FirmRuleSet's behaviour fields.
        Rulebook values (when stated) override the defaults; policy
        enums drive the style/hedging switches."""
        rs = None
        if ruleset is not None:
            rs = ruleset
        base = self.config
        if rs is None:
            return base
        styles = tuple(getattr(rs, "prohibited_styles", None)
                       or base.prohibited_styles)
        return BehaviourConfig(
            min_hold_seconds=rs.min_hold_seconds
            if rs.min_hold_seconds is not None else base.min_hold_seconds,
            max_orders_per_day=rs.max_orders_per_day
            if rs.max_orders_per_day is not None else base.max_orders_per_day,
            max_modifications_per_trade=rs.max_modifications_per_trade
            if rs.max_modifications_per_trade is not None
            else base.max_modifications_per_trade,
            max_requests_per_minute=rs.max_requests_per_minute
            if rs.max_requests_per_minute is not None
            else base.max_requests_per_minute,
            prohibited_styles=styles,
            hedging_forbidden=(getattr(rs, "hedging_policy", "unknown")
                               == "forbidden") or base.hedging_forbidden,
        )

    # ── bookkeeping (called by the router / TradeManager) ──────────────
    def record_request(self, account_id: str, ts_ms: int) -> None:
        self._requests.setdefault(account_id, []).append(int(ts_ms))

    def record_entry(self, account_id: str, position_id: str, symbol: str,
                     side: int, lot: float, price: float, ts_ms: int) -> None:
        self._entries.setdefault(account_id, []).append(
            _Entry(int(ts_ms), symbol, int(side), float(lot), float(price)))
        self._open.setdefault(account_id, {})[position_id] = _OpenPos(
            position_id, symbol, int(side), int(ts_ms), float(price),
            lot=float(lot))
        # The accepted entry counts against the daily order cap.
        day = int(ts_ms) // 86_400_000
        days = self._day_orders.setdefault(account_id, {})
        days[day] = days.get(day, 0) + 1

    def record_close(self, account_id: str, position_id: str, ts_ms: int,
                     pnl: float = 0.0) -> None:
        pos = self._open.get(account_id, {}).pop(position_id, None)
        # The close outcome feeds martingale detection: an escalation is
        # only a violation after a LOSING same-direction trade.
        if pos is not None:
            self._closed.setdefault(account_id, []).append({
                "ts_ms": int(ts_ms), "symbol": pos.symbol, "side": pos.side,
                "lot": pos.lot, "pnl": float(pnl)})

    def open_positions(self, account_id: str) -> list[_OpenPos]:
        return list(self._open.get(account_id, {}).values())

    # ── checks ─────────────────────────────────────────────────────────
    def check_request_rate(self, account_id: str, ts_ms: int) -> BehaviourDecision:
        """Sliding-window requests-per-minute over ALL order requests."""
        cap = self.config.max_requests_per_minute
        if cap is None:
            return BehaviourDecision(True, ())
        window = [t for t in self._requests.get(account_id, [])
                  if t > ts_ms - MS_PER_MINUTE]
        if len(window) >= cap:
            return BehaviourDecision(
                False, (R_MAX_REQUESTS,),
                {"requests_in_window": len(window), "cap": cap})
        return BehaviourDecision(True, ())

    def check_entry(self, account_id: str, symbol: str, side: int, lot: float,
                    price: float, ts_ms: int) -> BehaviourDecision:
        """Checks for a NEW entry (market or pending trigger). The request
        itself is counted by the caller via `record_request`."""
        reasons: list[str] = []
        details: dict = {}

        # Requests per minute.
        rate = self.check_request_rate(account_id, ts_ms)
        if not rate.allowed:
            reasons.extend(rate.reasons)
            details.update(rate.details)

        # Orders per UTC day.
        cap = self.config.max_orders_per_day
        if cap is not None:
            day = ts_ms // 86_400_000          # UTC day number
            used = self._day_orders.setdefault(account_id, {}).get(day, 0)
            if used >= cap:
                reasons.append(R_MAX_ORDERS_DAY)
                details["orders_today"] = used
                details["cap"] = cap

        # Prohibited styles (compare against this account's history).
        if self.config.prohibited_styles:
            style_reasons = self._detect_styles(
                account_id, symbol, side, lot, price, ts_ms)
            reasons.extend(style_reasons)

        return BehaviourDecision(not reasons, tuple(reasons), details)

    def _detect_styles(self, account_id: str, symbol: str, side: int,
                       lot: float, price: float, ts_ms: int) -> list[str]:
        cfg = self.config
        history = [e for e in self._entries.get(account_id, [])
                   if e.symbol == symbol and e.side == side
                   and e.ts_ms > ts_ms - cfg.stack_window_min * MS_PER_MINUTE]
        reasons: list[str] = []
        if not history:
            return reasons

        # martingale: escalation over the last CLOSED LOSING same-symbol,
        # same-direction trade within the lookback window.
        if "martingale" in cfg.prohibited_styles:
            for close in reversed(self._closed.get(account_id, [])):
                if close["symbol"] != symbol or close["side"] != side:
                    continue
                if ts_ms - close["ts_ms"] > \
                        cfg.martingale_lookback_min * MS_PER_MINUTE:
                    break
                if close["pnl"] < 0 and \
                        lot >= close["lot"] * cfg.martingale_escalation_factor:
                    reasons.append(R_STYLE_PREFIX + "martingale")
                break   # only the most recent same-direction close counts

        # grid: Nth same-direction entry at a strictly worsening price
        # within the window.
        if "grid" in cfg.prohibited_styles and len(history) + 1 >= cfg.stack_max_entries:
            worsening = all(
                (e.price > history[i + 1].price if side == 1
                 else e.price < history[i + 1].price)
                for i, e in enumerate(history[:-1]))
            last_worse = ((history[-1].price > price) if side == 1
                          else (history[-1].price < price))
            if worsening and last_worse:
                reasons.append(R_STYLE_PREFIX + "grid")

        # averaging_down: same-direction entry at a worse price than the
        # FIRST entry of the current stack while that stack is open.
        if "averaging_down" in cfg.prohibited_styles and len(history) >= 1:
            first = history[0]
            worse = (price < first.price) if side == 1 else (price > first.price)
            if worse and len(history) + 1 >= 2:
                reasons.append(R_STYLE_PREFIX + "averaging_down")
        return reasons

    def check_close(self, account_id: str, position_id: str, ts_ms: int,
                    *, protective: bool = False) -> BehaviourDecision:
        """min-hold check for closing (all or part of) a position.
        Protective actions (kill switch, rule breach) bypass the limit
        and are recorded as exceptions."""
        pos = self._open.get(account_id, {}).get(position_id)
        if pos is None:
            return BehaviourDecision(True, ())   # unknown position: not ours
        limit = self.config.min_hold_seconds
        if limit is None:
            return BehaviourDecision(True, ())
        age_s = (ts_ms - pos.opened_ms) / 1000.0
        if age_s >= limit:
            return BehaviourDecision(True, ())
        if protective:
            self.exceptions.append({
                "account": account_id, "position": position_id,
                "action": "close", "age_s": age_s, "ts_ms": ts_ms,
                "reason": "protective"})
            return BehaviourDecision(True, (), {"protective_exception": True})
        return BehaviourDecision(
            False, (R_MIN_HOLD,),
            {"age_seconds": round(age_s, 3), "min_hold_seconds": limit})

    def check_modify(self, account_id: str, position_id: str, ts_ms: int,
                     *, protective: bool = False) -> BehaviourDecision:
        """min-hold + per-trade modification cap for SL/TP amendments."""
        pos = self._open.get(account_id, {}).get(position_id)
        if pos is None:
            return BehaviourDecision(True, ())
        reasons: list[str] = []
        details: dict = {}
        limit = self.config.min_hold_seconds
        if limit is not None:
            age_s = (ts_ms - pos.opened_ms) / 1000.0
            if age_s < limit:
                if protective:
                    self.exceptions.append({
                        "account": account_id, "position": position_id,
                        "action": "modify", "age_s": age_s, "ts_ms": ts_ms,
                        "reason": "protective"})
                else:
                    reasons.append(R_MIN_HOLD)
                    details["age_seconds"] = round(age_s, 3)
        cap = self.config.max_modifications_per_trade
        if cap is not None and not protective:
            if pos.modifications >= cap:
                reasons.append(R_MAX_MODS)
                details["modifications"] = pos.modifications
                details["cap"] = cap
        if not reasons:
            pos.modifications += 1
        return BehaviourDecision(not reasons, tuple(reasons), details)

    def check_hedging(self, account_id: str, symbol: str, side: int) -> BehaviourDecision:
        """Opposite-direction position on the same symbol when hedging is
        forbidden."""
        if not self.config.hedging_forbidden:
            return BehaviourDecision(True, ())
        for pos in self._open.get(account_id, {}).values():
            if pos.symbol == symbol and pos.side == -int(side):
                return BehaviourDecision(
                    False, (R_HEDGING,),
                    {"existing_position": pos.position_id})
        return BehaviourDecision(True, ())

