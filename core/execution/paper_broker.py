"""Paper broker (Part 12.1 paper row / 12.2): simulated fills at live
bid/ask from a tick feed. Same interface shape a real BrokerAdapter will
have (place/modify/cancel/close/positions/orders/history/health), so the
router cannot tell the difference — only the event source and gateway
change between paper and live (Part 9).

Fills: market orders fill at the current bid (sell) / ask (buy) plus the
cost model's slippage. SL and TP are checked on every tick, correct
sides. `place` accepts ONLY an ApprovedOrder.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from core.execution.intent import ApprovedOrder
from core.config import CostModel


@dataclass
class PaperPosition:
    position_id: str
    account_id: str
    symbol: str
    side: int
    lot: float
    entry_price: float
    sl_price: float | None
    tp_price: float | None
    opened_ms: int
    idempotency_key: str


@dataclass
class PaperOrderRecord:
    order_ref: str
    idempotency_key: str
    approved: ApprovedOrder
    fill_price: float
    fill_ms: int
    latency_ms: float = 0.0
    slippage: float = 0.0


@dataclass
class PaperPendingOrder:
    order_ref: str
    idempotency_key: str
    approved: ApprovedOrder
    order_type: str            # buy_stop | sell_stop | buy_limit | sell_limit
    trigger_price: float
    created_ms: int
    state: str = "pending"     # pending | filled | cancelled


class BrokerTimeout(Exception):
    """Simulates a send whose outcome is unknown (Part 12.2)."""


class PaperBroker:
    def __init__(self, cost_model: CostModel | None = None, **_):
        self.costs = cost_model
        self._ids = itertools.count(1)
        self._positions: dict[str, PaperPosition] = {}
        self.order_history: list[PaperOrderRecord] = []
        self.connected: bool = False
        self.last_tick_ms: dict[str, int] = {}
        self.quotes: dict[str, tuple[float, float]] = {}   # symbol -> (bid, ask)
        # Injectable faults for tests.
        self.fail_next_send_with_timeout: bool = False
        self.fail_send_if_key: str | None = None   # timeout AFTER filling (worst case)
        self.balance: float = 100_000.0
        self.closed: list[dict] = []
        self.pending: dict[str, PaperPendingOrder] = {}   # order_ref -> order

    # ── connection / info ────────────────────────────────────────────────
    def connect(self) -> None:
        self.connected = True

    def health(self) -> dict:
        return {"connected": self.connected,
                "last_tick_ms": dict(self.last_tick_ms)}

    def account_info(self) -> dict:
        return {"balance": self.balance, "equity": self.equity,
                "positions": len(self._positions)}

    # ── quotes ───────────────────────────────────────────────────────────
    def on_tick(self, symbol: str, bid: float, ask: float, ts_ms: int) -> None:
        """Feed one live tick; trigger pending orders, then manage open
        positions (SL/TP per tick)."""
        self.quotes[symbol] = (bid, ask)
        self.last_tick_ms[symbol] = ts_ms
        self._trigger_pending(symbol, bid, ask, ts_ms)
        for pos in list(self._positions.values()):
            if pos.symbol != symbol:
                continue
            if pos.side == 1:
                if pos.sl_price is not None and bid <= pos.sl_price:
                    self._close_at(pos, pos.sl_price, ts_ms, "stop_loss")
                elif pos.tp_price is not None and bid >= pos.tp_price:
                    self._close_at(pos, pos.tp_price, ts_ms, "take_profit")
            else:
                if pos.sl_price is not None and ask >= pos.sl_price:
                    self._close_at(pos, pos.sl_price, ts_ms, "stop_loss")
                elif pos.tp_price is not None and ask <= pos.tp_price:
                    self._close_at(pos, pos.tp_price, ts_ms, "take_profit")

    def _close_at(self, pos: PaperPosition, price: float, ts_ms: int,
                  reason: str) -> None:
        self._positions.pop(pos.position_id, None)
        pnl = (price - pos.entry_price) * pos.side * pos.lot * 100.0
        self.balance += pnl
        self.closed.append({"position_id": pos.position_id,
                            "account_id": pos.account_id, "pnl": pnl,
                            "exit_price": price, "reason": reason,
                            "ts_ms": ts_ms})

    # ── trading ──────────────────────────────────────────────────────────
    def order_check(self, approved: ApprovedOrder) -> dict:
        """Dry-run validation before any send (Part 12.1)."""
        if not isinstance(approved, ApprovedOrder):
            raise TypeError("broker accepts only an ApprovedOrder")
        if not self.connected:
            return {"ok": False, "detail": "not connected"}
        if approved.intent.symbol not in self.quotes:
            return {"ok": False, "detail": "no quote for symbol"}
        if approved.intent.sl_price is None:
            return {"ok": False, "detail": "missing stop loss"}
        return {"ok": True, "detail": "validated"}

    def place(self, approved: ApprovedOrder) -> PaperOrderRecord:
        """Accepts ONLY an ApprovedOrder — the type system is the gate."""
        if not isinstance(approved, ApprovedOrder):
            raise TypeError("broker accepts only an ApprovedOrder")
        if not self.connected:
            raise ConnectionError("broker not connected")
        # Duplicate protection at the broker too (client-side key).
        if any(r.idempotency_key == approved.intent.idempotency_key
               for r in self.order_history):
            raise ValueError("duplicate idempotency key")

        intent = approved.intent
        if intent.order_type != "market":
            return self._place_pending(approved)
        bid, ask = self.quotes[intent.symbol]  # order_check guarantees a quote
        fill = ask if intent.side == 1 else bid
        slippage = 0.0
        if self.costs is not None:
            slip = self.costs.slippage.market_points
            fill += intent.side * slip
            slippage = slip
        ts = intent.created_ms
        latency = 100.0

        # Simulate a timeout AFTER the fill (the worst case for
        # reconciliation): the position exists but the caller never heard
        # back.
        key = intent.idempotency_key
        if self.fail_next_send_with_timeout or self.fail_send_if_key == key:
            self.fail_next_send_with_timeout = False
            self._open_position(approved, fill, ts)
            raise BrokerTimeout(f"send timeout for {key}")

        rec = self._open_position(approved, fill, ts)
        rec.latency_ms = latency
        rec.slippage = slippage
        return {"ok": True, "order_ref": rec.order_ref,
                "fill_price": rec.fill_price, "record": rec}

    def _open_position(self, approved: ApprovedOrder, fill: float,
                       ts: int) -> PaperOrderRecord:
        intent = approved.intent
        pid = f"pos-{next(self._ids)}"
        self._positions[pid] = PaperPosition(
            pid, intent.account_id, intent.symbol, intent.side,
            approved.lot, fill, intent.sl_price, intent.tp_price, ts,
            intent.idempotency_key)
        rec = PaperOrderRecord(f"ord-{next(self._ids)}",
                               intent.idempotency_key, approved, fill, ts)
        self.order_history.append(rec)
        return rec

    def _place_pending(self, approved: ApprovedOrder) -> dict:
        """Queue a stop/limit order; it fills on a later tick."""
        intent = approved.intent
        key = intent.idempotency_key
        if any(p.idempotency_key == key for p in self.pending.values()) \
                or any(r.idempotency_key == key for r in self.order_history):
            raise ValueError("duplicate idempotency key")
        ref = f"pend-{next(self._ids)}"
        pend = PaperPendingOrder(ref, key, approved, intent.order_type,
                                 float(intent.limit_price), intent.created_ms)
        self.pending[ref] = pend
        return {"ok": True, "order_ref": ref, "fill_price": None,
                "record": pend, "pending": True}

    def _trigger_pending(self, symbol: str, bid: float, ask: float,
                         ts: int) -> None:
        """Fill pending orders whose trigger is reached. Stops fill at
        the trigger PLUS adverse slippage; limits fill at the trigger or
        better, never worse."""
        slip = self.costs.slippage.market_points if self.costs else 0.0
        for pend in list(self.pending.values()):
            if pend.state != "pending" or \
                    pend.approved.intent.symbol != symbol:
                continue
            side = pend.approved.intent.side
            t, kind = pend.trigger_price, pend.order_type
            hit = ((kind == "buy_stop" and ask >= t)
                   or (kind == "sell_stop" and bid <= t)
                   or (kind == "buy_limit" and ask <= t)
                   or (kind == "sell_limit" and bid >= t))
            if not hit:
                continue
            if kind.endswith("stop"):
                fill = t + side * slip          # adverse slippage
            else:                                # limit: trigger or better
                fill = min(t, ask) if side == 1 else max(t, bid)
            rec = self._open_position(pend.approved, fill, ts)
            rec.slippage = slip if kind.endswith("stop") else 0.0
            pend.state = "filled"
            self.pending.pop(pend.order_ref, None)

    # ── BrokerAdapter contract (Part 12.4) ────────────────────────────────
    def symbol_spec(self, symbol: str) -> dict:
        from core.config import get_symbol_spec
        s = get_symbol_spec(symbol)
        return {"symbol": symbol, "digits": s.digits, "point": s.point,
                "contract_size": s.contract_size, "min_lot": s.min_lot,
                "max_lot": s.max_lot, "lot_step": s.lot_step,
                "stops_level": 0.0}

    def quote(self, symbol: str) -> tuple[float, float]:
        bid, ask = self.quotes.get(symbol, (0.0, 0.0))
        return bid, ask

    def orders(self) -> list[dict]:
        return [{"order_ref": r.order_ref,
                 "idempotency_key": r.idempotency_key}
                for r in self.order_history]

    def positions(self, account_id: str | None = None) -> list[dict]:
        return [p.__dict__ | {"position_id": p.position_id}
                for p in self._positions.values()
                if account_id is None or p.account_id == account_id]

    def history(self) -> list[dict]:
        return [r.__dict__ | {"order_ref": r.order_ref,
                              "idempotency_key": r.idempotency_key}
                for r in self.order_history]

    def cancel_order(self, order_ref: str) -> None:
        pend = self.pending.get(order_ref)
        if pend is None:
            raise KeyError(f"no pending order {order_ref}")
        pend.state = "cancelled"
        del self.pending[order_ref]

    def pending_orders(self, account_id: str | None = None) -> list[dict]:
        return [{"order_ref": p.order_ref, "idempotency_key": p.idempotency_key,
                 "order_type": p.order_type, "trigger_price": p.trigger_price,
                 "state": p.state, "account_id":
                 p.approved.intent.account_id}
                for p in self.pending.values()
                if account_id is None
                or p.approved.intent.account_id == account_id]

    def find_by_idempotency_key(self, key: str) -> dict | None:
        for r in self.order_history:
            if r.idempotency_key == key:
                return {"ok": True, "order_ref": r.order_ref,
                        "fill_price": r.fill_price}
        for p in self.pending.values():
            if p.idempotency_key == key:
                return {"ok": True, "order_ref": p.order_ref,
                        "fill_price": None, "pending": True}
        return None

    def positions_of(self, account_id: str) -> list[PaperPosition]:
        return [p for p in self._positions.values()
                if p.account_id == account_id]

    def close_position(self, position_id: str, ts_ms: int) -> dict:
        pos = self._positions.pop(position_id)
        bid, ask = self.quotes.get(pos.symbol, (pos.entry_price,) * 2)
        price = bid if pos.side == 1 else ask
        pnl = (price - pos.entry_price) * pos.side * pos.lot * 100.0
        self.balance += pnl
        rec = {"position_id": position_id, "account_id": pos.account_id,
               "pnl": pnl, "exit_price": price,
               "reason": "manual_close", "ts_ms": ts_ms}
        self.closed.append(rec)
        return rec

    def modify_position(self, position_id: str, sl_price: float | None = None,
                        tp_price: float | None = None) -> None:
        pos = self._positions[position_id]
        if sl_price is not None:
            pos.sl_price = sl_price
        if tp_price is not None:
            pos.tp_price = tp_price

    @property
    def equity(self) -> float:
        floating = 0.0
        for p in self._positions.values():
            bid, ask = self.quotes.get(p.symbol, (p.entry_price,) * 2)
            mid = (bid + ask) / 2 if p.symbol in self.quotes else p.entry_price
            floating += (mid - p.entry_price) * p.side * p.lot * 100.0
        return self.balance + floating
