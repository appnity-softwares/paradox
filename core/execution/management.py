"""Trade management (Part 12.3): trailing stop, breakeven, partial close,
time-based exit, session-end flatten. Every SL/TP modification is
re-checked: a modification may only REDUCE risk unless it is explicitly
re-approved through the same gate as a new order.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.execution.paper_broker import PaperBroker, PaperPosition


@dataclass(frozen=True)
class ManagementRules:
    """Rules come from the strategy version's config (Part 9/12.3)."""
    breakeven_at_rr: float | None = 1.0     # move SL to entry at this R multiple
    breakeven_buffer: float = 0.10          # price units beyond entry
    trail_atr_mult: float | None = None     # trail distance in price units if set
    trail_activation_rr: float = 1.0
    max_hold_ms: int | None = None          # time-based exit
    session_end_utc_hour: int | None = None # flatten at/after this hour
    weekend_flatten: bool = False


class ManagementError(Exception):
    pass


class TradeManager:
    def __init__(self, broker: PaperBroker, rules: ManagementRules,
                 audit_log=None, notifier=None,
                 behaviour=None):
        self.broker = broker
        self.rules = rules
        self.audit = audit_log
        self.notifier = notifier
        # U1: optional BehaviourControls. When absent the manager keeps
        # its pre-U1 behaviour (backward compatible); when present,
        # closes and modifications honour min-hold and modification caps
        # (protective actions are allowed and logged as exceptions).
        self.behaviour = behaviour

    def _behaviour_reject(self, action: str, account_id: str,
                          position_id: str, ts_ms: int,
                          protective: bool) -> None:
        if self.behaviour is None:
            return
        checker = self.behaviour.check_modify if action == "modify" \
            else self.behaviour.check_close
        decision = checker(account_id, position_id, ts_ms,
                           protective=protective)
        if not decision.allowed:
            raise ManagementError(
                f"{action} blocked by behaviour controls: "
                f"{decision.reasons} {decision.details}")

    # ── modifications with risk re-check ─────────────────────────────────
    def modify_sl(self, position_id: str, new_sl: float, ts_ms: int,
                  *, approval_token: str | None = None,
                  account_id: str = "default") -> None:
        pos = self.broker._positions.get(position_id)
        if pos is None:
            raise ManagementError(f"unknown position {position_id}")
        old_worst = self._worst_case(pos)
        candidate = PaperPosition(**{**pos.__dict__, "sl_price": new_sl})
        new_worst = self._worst_case(candidate)
        tightening = new_worst <= old_worst + 1e-9
        if not tightening and approval_token is None:
            raise ManagementError(
                "loosening a stop requires human re-approval (12.3)")
        # U1: min-hold + per-trade modification cap. A tightening stop is
        # risk-REDUCING, so it counts as protective and bypasses min-hold
        # (logged as an exception); the cap still applies to it.
        self._behaviour_reject("modify", account_id, position_id, ts_ms,
                               protective=tightening)
        self.broker.modify_position(position_id, sl_price=new_sl)
        if self.audit is not None:
            self.audit.append({"event": "modify_sl",
                               "position": position_id,
                               "old_sl": pos.sl_price, "new_sl": new_sl,
                               "tightening": tightening, "ts_ms": ts_ms})

    @staticmethod
    def _worst_case(pos: PaperPosition) -> float:
        """Worst-case LOSS in money. An SL beyond the entry locks profit,
        so its worst-case loss is zero — moving the stop toward or past
        entry is always risk-reducing and never needs re-approval."""
        if pos.sl_price is None:
            return float("inf")   # no SL = unbounded worst case
        loss = (pos.entry_price - pos.sl_price) * pos.side
        return max(0.0, loss) * pos.lot * 100.0

    def partial_close(self, position_id: str, fraction: float,
                      ts_ms: int, *, account_id: str = "default") -> dict:
        pos = self.broker._positions.get(position_id)
        if pos is None:
            raise ManagementError(f"unknown position {position_id}")
        # U1: a partial close is still a close — min-hold applies. It is
        # not protective (it realises profit/loss by choice).
        self._behaviour_reject("close", account_id, position_id, ts_ms,
                               protective=False)
        if not 0 < fraction < 1:
            raise ManagementError("fraction must be in (0, 1)")
        close_lot = round(pos.lot * fraction, 2)
        if close_lot <= 0:
            raise ManagementError("partial close below min lot")
        remaining = round(pos.lot - close_lot, 2)
        # Book the closed part at the current quote.
        bid, ask = self.broker.quotes.get(
            pos.symbol, (pos.entry_price, pos.entry_price))
        price = bid if pos.side == 1 else ask
        pnl = (price - pos.entry_price) * pos.side * close_lot * 100.0
        self.broker.balance += pnl
        rec = {"position_id": position_id, "closed_lot": close_lot,
               "pnl": pnl, "ts_ms": ts_ms}
        self.broker.closed.append({**rec, "reason": "partial_close",
                                   "exit_price": price})
        if remaining > 0:
            pos.lot = remaining
        else:
            self.broker._positions.pop(position_id, None)
        return rec

    # ── automatic per-tick management ────────────────────────────────────
    def on_tick(self, symbol: str, bid: float, ask: float, ts_ms: int,
                atr: float | None = None) -> None:
        self.broker.on_tick(symbol, bid, ask, ts_ms)
        for pos in list(self.broker._positions.values()):
            if pos.symbol != symbol:
                continue
            price = bid if pos.side == 1 else ask
            rr_move = (price - pos.entry_price) * pos.side
            risk_dist = abs(pos.entry_price - (pos.sl_price or price))
            rr = rr_move / risk_dist if risk_dist > 0 else 0.0

            # Breakeven.
            if (self.rules.breakeven_at_rr is not None and rr >=
                    self.rules.breakeven_at_rr):
                be = pos.entry_price + pos.side * self.rules.breakeven_buffer
                if pos.sl_price is None or \
                        (be - pos.sl_price) * pos.side > 1e-9:
                    self.modify_sl(pos.position_id, be, ts_ms)

            # Trailing stop.
            if (self.rules.trail_atr_mult is not None and atr and
                    rr >= self.rules.trail_activation_rr):
                trail = price - pos.side * self.rules.trail_atr_mult * atr
                if pos.sl_price is None or \
                        (trail - pos.sl_price) * pos.side > 1e-9:
                    self.modify_sl(pos.position_id, trail, ts_ms)

            # Time-based exit.
            if self.rules.max_hold_ms is not None and \
                    ts_ms - pos.opened_ms >= self.rules.max_hold_ms:
                self.broker.close_position(pos.position_id, ts_ms)
                if self.notifier is not None:
                    self.notifier.notify("trade_closed",
                                         f"time exit {pos.position_id}")

            # Session-end flatten.
            if self.rules.session_end_utc_hour is not None:
                import datetime as _dt
                hour = _dt.datetime.fromtimestamp(
                    ts_ms / 1000.0, tz=_dt.timezone.utc).hour
                if hour >= self.rules.session_end_utc_hour:
                    self.broker.close_position(pos.position_id, ts_ms)
                    if self.notifier is not None:
                        self.notifier.notify("trade_closed",
                                             f"session flatten {pos.position_id}")
