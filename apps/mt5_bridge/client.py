"""Client adapter for the MT5 bridge (Part 12.4).

Implements the full BrokerAdapter surface over HTTPS with token auth.
The token comes from the environment (`DSH_MT5_BRIDGE_TOKEN`) or is
passed explicitly by the operator — never hardcoded, never logged.
Plain HTTP is refused unless the URL is loopback (dev/tests).
"""
from __future__ import annotations

import os

import requests

from core.execution.broker_adapter import BrokerAdapter
from core.execution.intent import ApprovedOrder


class MT5BridgeError(RuntimeError):
    pass


class MT5BridgeAdapter(BrokerAdapter):
    def __init__(self, base_url: str, token: str | None = None,
                 timeout_s: float = 10.0):
        if not base_url.startswith("https://"):
            host = base_url.split("//")[-1].split("/")[0]
            if not (host.startswith("127.0.0.1") or
                    host.startswith("localhost")):
                raise MT5BridgeError(
                    "refusing plain-HTTP to a non-loopback bridge "
                    "(Part 12.4: TLS required)")
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("DSH_MT5_BRIDGE_TOKEN", "")
        if not self.token:
            raise MT5BridgeError(
                "no bridge token; set DSH_MT5_BRIDGE_TOKEN (credentials "
                "live only in the bridge or an encrypted vault)")
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.token}"

    def _call(self, method: str, path: str, *, json_body: dict | None = None,
              ok_codes: tuple[int, ...] = (200,)):
        resp = self._session.request(method, self.base_url + path,
                                     json=json_body, timeout=self.timeout_s)
        if resp.status_code not in ok_codes:
            raise MT5BridgeError(
                f"{method} {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # ── BrokerAdapter surface ────────────────────────────────────────────
    def connect(self) -> None:
        self._call("POST", "/connect")

    def health(self) -> dict:
        return self._call("GET", "/health")

    def account_info(self) -> dict:
        return self._call("GET", "/account")

    def symbol_spec(self, symbol: str) -> dict:
        return self._call("GET", f"/symbol/{symbol}")

    def quote(self, symbol: str) -> tuple[float, float]:
        q = self._call("GET", f"/quote/{symbol}")
        return q["bid"], q["ask"]

    def order_check(self, approved: ApprovedOrder) -> dict:
        if not isinstance(approved, ApprovedOrder):
            raise TypeError("adapter accepts only an ApprovedOrder")
        try:
            return self._call("POST", "/order_check",
                              json_body=self._request(approved))
        except MT5BridgeError as exc:
            return {"ok": False, "detail": str(exc)}

    def place(self, approved: ApprovedOrder) -> dict:
        if not isinstance(approved, ApprovedOrder):
            raise TypeError("adapter accepts only an ApprovedOrder")
        resp = self._session.request(
            "POST", self.base_url + "/order_send",
            json=self._request(approved), timeout=self.timeout_s)
        if resp.status_code == 200:
            body = resp.json()
            pending = body.get("deal") == 0     # queued, not filled
            return {"ok": True, "order_ref": str(body.get("order")),
                    "fill_price": None if pending else body.get("price"),
                    "pending": pending}
        # Any rejection (validation, duplicate key, broker retcode) is a
        # NOT-ok result — never silently mapped to success.
        try:
            body = resp.json()
        except ValueError:
            body = {}
        detail = body.get("detail", body) if isinstance(body, dict) else body
        if isinstance(detail, dict):
            return {"ok": False, **detail}
        return {"ok": False, "comment": str(detail)}

    def modify_position(self, position_id: str, *, sl_price: float | None = None,
                        tp_price: float | None = None) -> None:
        self._call("POST", "/position_modify",
                   json_body={"ticket": int(position_id),
                              "sl": sl_price, "tp": tp_price})

    def cancel_order(self, order_ref: str) -> dict:
        return self._call("POST", "/order_cancel",
                          json_body={"ticket": int(order_ref)})

    def close_position(self, position_id: str, ts_ms: int) -> dict:
        return self._call("POST", f"/position_close/{position_id}")

    def positions(self, account_id: str | None = None) -> list[dict]:
        return self._call("GET", "/positions")

    def orders(self) -> list[dict]:
        return self._call("GET", "/orders")

    def history(self) -> list[dict]:
        return self._call("GET", "/history")

    def find_by_idempotency_key(self, key: str) -> dict | None:
        for p in self.positions():
            if p.get("idempotency_key") == key:
                return {"order_ref": str(p["ticket"]),
                        "fill_price": p.get("price_open")}
        return None   # pending orders are found via orders() by the caller

    # MT5 order-type codes: 0 buy, 1 sell, 2 buy_limit, 3 sell_limit,
    # 4 buy_stop, 5 sell_stop.
    _TYPE_CODES = {"market": None, "buy_stop": 4, "sell_stop": 5,
                   "buy_limit": 2, "sell_limit": 3}

    @classmethod
    def _request(cls, approved: ApprovedOrder) -> dict:
        it = approved.intent
        code = cls._TYPE_CODES.get(it.order_type)
        if code is None and it.order_type != "market":
            raise MT5BridgeError(f"unsupported order type {it.order_type}")
        if code is None:                      # market: price = current ref
            return {"symbol": it.symbol, "type": 0 if it.side == 1 else 1,
                    "volume": approved.lot, "price": approved.price_ref,
                    "sl": it.sl_price, "tp": it.tp_price,
                    "comment": it.idempotency_key}
        return {"symbol": it.symbol, "type": code, "volume": approved.lot,
                "price": it.limit_price, "sl": it.sl_price,
                "tp": it.tp_price, "comment": it.idempotency_key}
