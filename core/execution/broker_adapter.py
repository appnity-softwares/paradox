"""BrokerAdapter interface (Part 12.4).

Every gateway — PaperBroker, the MT5 bridge adapter, later cTrader —
implements this exact surface, so the execution router cannot tell them
apart: only the event source and the order gateway change (Part 9).
`place` accepts ONLY an ApprovedOrder, on every adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.execution.intent import ApprovedOrder


class BrokerAdapter(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def account_info(self) -> dict: ...

    @abstractmethod
    def symbol_spec(self, symbol: str) -> dict: ...

    @abstractmethod
    def quote(self, symbol: str) -> tuple[float, float]:
        """Return (bid, ask)."""

    @abstractmethod
    def order_check(self, approved: ApprovedOrder) -> dict:
        """Broker-side dry-run validation; MUST run before order_send."""

    @abstractmethod
    def place(self, approved: ApprovedOrder) -> dict:
        """Send the order; returns {order_ref, fill_price, ...}."""

    @abstractmethod
    def modify_position(self, position_id: str, *, sl_price: float | None = None,
                        tp_price: float | None = None) -> None: ...

    @abstractmethod
    def cancel_order(self, order_ref: str) -> None: ...

    @abstractmethod
    def close_position(self, position_id: str, ts_ms: int) -> dict: ...

    @abstractmethod
    def positions(self, account_id: str | None = None) -> list[dict]: ...

    @abstractmethod
    def orders(self) -> list[dict]: ...

    @abstractmethod
    def history(self) -> list[dict]: ...

    @abstractmethod
    def find_by_idempotency_key(self, key: str) -> dict | None:
        """Used by timeout reconciliation (Part 12.2)."""


def assert_conforms(adapter: BrokerAdapter) -> None:
    """Contract check used by tests: every abstract method implemented."""
    for name in BrokerAdapter.__abstractmethods__:
        if getattr(type(adapter), name, None) is None:
            raise TypeError(f"{type(adapter).__name__} missing {name}")
