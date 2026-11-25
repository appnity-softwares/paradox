"""MT5 bridge backends (Part 12.4).

The bridge is a small authenticated service that runs on a Windows
machine or VPS BESIDE a running, logged-in MT5 terminal. The platform
talks to it over TLS with token auth; MT5 credentials live only here, in
environment variables — never in code, agent context, or logs.

`FakeMT5Backend` is a deterministic in-memory backend used for tests and
offline development; `RealMT5Backend` wraps the Windows-only
`MetaTrader5` package behind a guarded import.
"""
from __future__ import annotations

import os
import threading
import time


class MT5Unavailable(RuntimeError):
    pass


class FakeMT5Backend:
    """Deterministic MT5-shaped backend for tests (no MetaTrader5 import)."""

    name = "fake"

    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.quotes: dict[str, tuple[float, float]] = {}
        self.last_tick_ms: dict[str, int] = {}
        self.specs: dict[str, dict] = {}
        self.positions: list[dict] = []
        self.orders: list[dict] = []
        self.deals: list[dict] = []
        self.checked_keys: set[str] = set()
        self.sent_keys: set[str] = set()
        # Pending stop/limit orders queued by order_send, triggered by
        # set_quote ticks. MT5 type codes: 2 buy_limit, 3 sell_limit,
        # 4 buy_stop, 5 sell_stop.
        self.pending: list[dict] = []
        self.balance = 100_000.0
        self._ids = 0

    # ── connection ───────────────────────────────────────────────────────
    def connect(self, **kwargs) -> dict:
        self.connected = True
        return {"ok": True}

    def account_info(self) -> dict:
        return {"login": 0, "balance": self.balance, "equity": self.balance,
                "currency": "USD", "leverage": 100}

    def symbol_spec(self, symbol: str) -> dict:
        return self.specs.get(symbol, {
            "symbol": symbol, "digits": 2, "point": 0.01,
            "contract_size": 100.0, "min_lot": 0.01, "max_lot": 500.0,
            "lot_step": 0.01, "stops_level": 0.0})

    def quote(self, symbol: str) -> dict:
        bid, ask = self.quotes.get(symbol, (0.0, 0.0))
        return {"symbol": symbol, "bid": bid, "ask": ask,
                "time_ms": self.last_tick_ms.get(symbol, 0)}

    def set_quote(self, symbol: str, bid: float, ask: float, ts_ms: int) -> None:
        self.quotes[symbol] = (bid, ask)
        self.last_tick_ms[symbol] = ts_ms
        self._trigger_pending(symbol, bid, ask)

    def _trigger_pending(self, symbol: str, bid: float, ask: float) -> None:
        """Fill pending orders whose trigger is reached (stop: at trigger;
        limit: at trigger or better — the bridge reports the fill, the
        terminal's real behaviour is authoritative in production)."""
        for pend in list(self.pending):
            if pend["symbol"] != symbol or pend["state"] != "pending":
                continue
            t, kind = pend["price"], pend["type"]
            hit = ((kind == 4 and ask >= t) or (kind == 5 and bid <= t)
                   or (kind == 2 and ask <= t) or (kind == 3 and bid >= t))
            if not hit:
                continue
            side = 1 if kind in (2, 4) else -1
            fill = t                        # trigger price at the bridge
            self.positions.append({
                "ticket": pend["ticket"], "symbol": symbol, "side": side,
                "volume": pend["volume"], "price_open": fill,
                "sl": pend["sl"], "tp": pend["tp"],
                "time_ms": int(time.time() * 1000),
                "idempotency_key": pend["idempotency_key"]})
            pend["state"] = "filled"
            self.pending.remove(pend)

    def order_cancel(self, ticket: int) -> dict:
        for pend in self.pending:
            if pend["ticket"] == ticket and pend["state"] == "pending":
                pend["state"] = "cancelled"
                self.pending.remove(pend)
                return {"retcode": 10009, "comment": "cancelled"}
        return {"retcode": 10013, "comment": "no such pending order"}

    # ── trading ──────────────────────────────────────────────────────────
    def order_check(self, request: dict) -> dict:
        symbol = request.get("symbol", "")
        if symbol not in self.quotes:
            return {"ok": False, "comment": "unknown symbol"}
        lot = float(request.get("volume", 0.0))
        spec = self.symbol_spec(symbol)
        if lot < spec["min_lot"] or lot > spec["max_lot"]:
            return {"ok": False, "comment": "invalid volume"}
        if request.get("type") in (0, 1) and request.get("sl") is None:
            return {"ok": False, "comment": "missing stop loss"}
        if request.get("type") in (2, 3, 4, 5) and \
                request.get("price") is None:
            return {"ok": False, "comment": "pending order requires price"}
        self.checked_keys.add(request.get("comment", ""))
        return {"ok": True, "comment": "done"}

    def order_send(self, request: dict) -> dict:
        key = request.get("comment", "")
        # Server-side enforcement: no send without a prior check (12.4).
        if key not in self.checked_keys:
            return {"retcode": 10027, "comment": "order_check required first"}
        if key in self.sent_keys:
            return {"retcode": 10032, "comment": "duplicate key"}
        self.sent_keys.add(key)
        if request.get("type") in (2, 3, 4, 5):
            self._ids += 1
            ticket = 400_000 + self._ids
            self.pending.append({
                "ticket": ticket, "symbol": request["symbol"],
                "type": request["type"], "volume": float(request["volume"]),
                "price": float(request["price"]), "sl": request.get("sl"),
                "tp": request.get("tp"), "state": "pending",
                "idempotency_key": key})
            return {"retcode": 10009, "order": ticket, "deal": 0,
                    "price": float(request["price"]),
                    "comment": "pending placed"}
        q = self.quote(request["symbol"])
        side = 1 if request["type"] == 0 else -1
        fill = q["ask"] if side == 1 else q["bid"]
        self._ids += 1
        ticket = 500_000 + self._ids
        self.positions.append({
            "ticket": ticket, "symbol": request["symbol"], "side": side,
            "volume": float(request["volume"]), "price_open": fill,
            "sl": request.get("sl"), "tp": request.get("tp"),
            "time_ms": int(time.time() * 1000),
            "idempotency_key": key})
        return {"retcode": 10009, "order": ticket, "deal": ticket,
                "price": fill, "comment": "done"}

    def positions_get(self) -> list[dict]:
        return list(self.positions)

    def orders_get(self) -> list[dict]:
        return list(self.orders)

    def history_deals_get(self) -> list[dict]:
        return list(self.deals)

    def position_close(self, ticket: int, ts_ms: int) -> dict:
        for i, p in enumerate(self.positions):
            if p["ticket"] == ticket:
                self.positions.pop(i)
                q = self.quote(p["symbol"])
                exit_price = q["bid"] if p["side"] == 1 else q["ask"]
                pnl = (exit_price - p["price_open"]) * p["side"] \
                    * p["volume"] * p.get("contract_size", 100.0)
                self.balance += pnl
                deal = {"ticket": ticket, "pnl": pnl, "exit_price": exit_price,
                        "ts_ms": ts_ms}
                self.deals.append(deal)
                return deal
        return {"error": "unknown position"}

    def position_modify(self, ticket: int, sl: float | None = None,
                        tp: float | None = None) -> dict:
        for p in self.positions:
            if p["ticket"] == ticket:
                if sl is not None:
                    p["sl"] = sl
                if tp is not None:
                    p["tp"] = tp
                return {"ok": True}
        return {"error": "unknown position"}

    def health(self) -> dict:
        return {"backend": self.name, "connected": self.connected,
                "terminal_ok": self.connected}


class RealMT5Backend:
    """Wraps the official MetaTrader5 package (Windows-only, needs a
    running logged-in terminal). Import is guarded; credentials come from
    environment variables only."""

    name = "mt5"

    def __init__(self):
        try:
            import MetaTrader5  # noqa: F401
        except ImportError as exc:
            raise MT5Unavailable(
                "MetaTrader5 is Windows-only and requires a running, "
                "logged-in terminal; run the bridge on Windows/VPS "
                "beside the terminal") from exc
        self.mt5 = __import__("MetaTrader5")
        self.connected = False

    def connect(self, *, login: int | None = None, password: str | None = None,
                server: str | None = None, **kwargs) -> dict:
        # Credentials only from env if not explicitly provided.
        login = login or os.environ.get("MT5_LOGIN")
        password = password or os.environ.get("MT5_PASSWORD")
        server = server or os.environ.get("MT5_SERVER")
        kwargs = {k: v for k, v in
                  {"login": login, "password": password, "server": server}.items()
                  if v is not None}
        if not self.mt5.initialize(**kwargs):
            return {"ok": False, "comment": "initialize failed"}
        self.connected = True
        return {"ok": True}

    def account_info(self) -> dict:
        info = self.mt5.account_info()._asdict()
        return {k: info[k] for k in
                ("login", "balance", "equity", "currency", "leverage")
                if k in info}

    def symbol_spec(self, symbol: str) -> dict:
        s = self.mt5.symbol_info(symbol)
        if s is None:
            return {"symbol": symbol, "error": "unknown symbol"}
        d = s._asdict()
        return {"symbol": symbol, "digits": d.get("digits"),
                "point": d.get("point"),
                "contract_size": d.get("trade_contract_size"),
                "min_lot": d.get("volume_min"), "max_lot": d.get("volume_max"),
                "lot_step": d.get("volume_step"),
                "stops_level": d.get("trade_stops_level")}

    def quote(self, symbol: str) -> dict:
        t = self.mt5.symbol_info_tick(symbol)
        if t is None:
            return {"symbol": symbol, "bid": 0.0, "ask": 0.0, "time_ms": 0}
        d = t._asdict()
        return {"symbol": symbol, "bid": d.get("bid", 0.0),
                "ask": d.get("ask", 0.0),
                "time_ms": int(d.get("time", 0)) * 1000}

    def order_check(self, request: dict) -> dict:
        r = self.mt5.order_check(request)
        if r is None:
            return {"ok": False, "comment": "order_check failed"}
        d = r._asdict()
        return {"ok": d.get("retcode") == 0,
                "comment": d.get("comment", "")}

    def order_send(self, request: dict) -> dict:
        r = self.mt5.order_send(request)
        if r is None:
            return {"retcode": -1, "comment": "order_send failed"}
        d = r._asdict()
        return {"retcode": d.get("retcode"), "order": d.get("order"),
                "deal": d.get("deal"), "price": d.get("price"),
                "comment": d.get("comment", "")}

    def positions_get(self) -> list[dict]:
        pos = self.mt5.positions_get()
        return [p._asdict() for p in (pos or [])]

    def orders_get(self) -> list[dict]:
        orders = self.mt5.orders_get()
        return [o._asdict() for o in (orders or [])]

    def history_deals_get(self, **kwargs) -> list[dict]:
        deals = self.mt5.history_deals_get(**kwargs)
        return [d._asdict() for d in (deals or [])]

    def position_close(self, ticket: int, ts_ms: int) -> dict:
        # Implemented via order_send with opposing deal by the caller; the
        # bridge exposes close through order_send for real MT5.
        raise NotImplementedError("close via order_send at the bridge level")

    def position_modify(self, ticket: int, sl: float | None = None,
                        tp: float | None = None) -> dict:
        req = {"action": self.mt5.TRADE_ACTION_SLTP, "position": ticket}
        if sl is not None:
            req["sl"] = sl
        if tp is not None:
            req["tp"] = tp
        r = self.mt5.order_send(req)
        return {"ok": bool(r and r.retcode == self.mt5.TRADE_RETCODE_DONE)}

    def health(self) -> dict:
        return {"backend": self.name, "connected": self.connected,
                "terminal_ok": bool(self.mt5.terminal_info())}
