"""Shared strategy interface for the bar and tick engines (Master Prompt 6.2).

One interface, two engines:

    class Strategy:
        def on_bar(self, ctx, bar) -> list[Signal]: ...
        def on_tick(self, ctx, tick) -> list[Signal]: ...   # optional
        def on_fill(self, ctx, fill) -> None: ...

Strategies RETURN SIGNALS ONLY. They never place orders, never call a broker
and never read account secrets. The execution layer (backtest simulator or the
live router) turns signals into orders.

Look-ahead protection is structural: ``ctx.bars`` exposes CLOSED bars only.
Any attempt to look at unclosed or future data raises :class:`LookAheadError`,
which the engines treat as an invalid run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


class LookAheadError(RuntimeError):
    """Raised when a strategy tries to access data it must not see yet."""


# Order kinds understood by every engine.
KIND_MARKET = "market"
KIND_LIMIT = "limit"
KIND_STOP = "stop"

SIDE_LONG = 1
SIDE_SHORT = -1
SIDE_CLOSE = 0


@dataclass(frozen=True)
class Signal:
    """A single strategy instruction. Engines translate this into orders.

    side:      +1 open/hold long, -1 open/hold short, 0 close everything.
    kind:      "market" | "limit" | "stop" (ignored for side == 0).
    price:     trigger price for limit/stop orders (required for those kinds).
    sl / tp:   absolute stop-loss / take-profit prices for the new position.
    fraction:  optional size hint as a fraction of the engine's normal risk
               unit (0 < fraction <= 1). Engines still apply their own sizing
               and limits; a signal can never exceed configured risk.
    tag:       free-form label copied into the trade log.
    """

    side: int
    kind: str = KIND_MARKET
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    fraction: float = 1.0
    tag: str = ""

    def __post_init__(self) -> None:
        if self.side not in (SIDE_LONG, SIDE_SHORT, SIDE_CLOSE):
            raise ValueError(f"signal side must be -1, 0 or +1, got {self.side}")
        if self.kind not in (KIND_MARKET, KIND_LIMIT, KIND_STOP):
            raise ValueError(f"unknown order kind: {self.kind}")
        if self.kind in (KIND_LIMIT, KIND_STOP) and self.price is None:
            raise ValueError(f"{self.kind} signal requires a trigger price")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("signal fraction must be in (0, 1]")


@dataclass
class Fill:
    """Executed (simulated) fill reported back to the strategy."""

    order_tag: str
    side: int
    price: float
    volume: float          # lots
    time_utc_ms: int
    reason: str            # "entry" | "sl" | "tp" | "signal_exit" | "partial" | "stop_out" | "time_exit" | "end_of_data"
    pnl_realized: float    # realized P&L of this fill in account currency


class BarWindow:
    """Read-only view over CLOSED bars, ending at the bar just closed.

    The newest row is the bar that has just finished; nothing later exists.
    The engine appends bars as they close. Accessors return only closed data;
    there is no way to reach an unclosed or future bar.

    ``tail``/``frame`` build pandas objects lazily; the array accessors
    (``closes``, ``highs``, ``lows``, ``opens``) are cheap views for
    indicator math inside hot loops.
    """

    def __init__(self, max_lookback: int = 100_000):
        self._max_lookback = max_lookback
        self._times = np.empty(0, dtype=np.int64)
        self._open = np.empty(0, dtype=np.float64)
        self._high = np.empty(0, dtype=np.float64)
        self._low = np.empty(0, dtype=np.float64)
        self._close = np.empty(0, dtype=np.float64)
        self._volume = np.empty(0, dtype=np.float64)

    # -- engine-internal append (not part of the strategy API) ------------

    def _append(self, time_utc_ms: int, o: float, h: float, l: float, c: float,
                volume: float = 0.0) -> None:
        self._times = np.append(self._times, time_utc_ms)
        self._open = np.append(self._open, o)
        self._high = np.append(self._high, h)
        self._low = np.append(self._low, l)
        self._close = np.append(self._close, c)
        self._volume = np.append(self._volume, volume)

    # -- strategy-facing accessors -----------------------------------------

    def __len__(self) -> int:
        return int(self._times.size)

    def last(self) -> dict[str, Any]:
        """The bar that just closed, as a plain dict."""
        if self._times.size == 0:
            raise LookAheadError("no closed bar available yet")
        i = self._times.size - 1
        return {
            "time_utc_ms": int(self._times[i]),
            "open": float(self._open[i]),
            "high": float(self._high[i]),
            "low": float(self._low[i]),
            "close": float(self._close[i]),
            "volume": float(self._volume[i]),
        }

    def closes(self, n: int) -> np.ndarray:
        """Last n CLOSED closes, oldest first. No future rows exist."""
        self._check(n)
        return self._close[-n:]

    def highs(self, n: int) -> np.ndarray:
        self._check(n)
        return self._high[-n:]

    def lows(self, n: int) -> np.ndarray:
        self._check(n)
        return self._low[-n:]

    def opens(self, n: int) -> np.ndarray:
        self._check(n)
        return self._open[-n:]

    def tail(self, n: int) -> pd.DataFrame:
        """Last n closed bars as a DataFrame (built on demand)."""
        self._check(n)
        return self._frame_from(n)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame_from(len(self._times))

    def indicator(self, name: str, **kwargs: Any) -> Any:
        """Compute an indicator on closed bars only, via core.indicators."""
        from core import indicators as ind

        func = getattr(ind, name, None)
        if func is None:
            raise ValueError(f"unknown indicator: {name}")
        return func(self.frame, **kwargs)

    def _check(self, n: int) -> None:
        if n > self._max_lookback:
            raise LookAheadError(f"requested {n} bars exceeds lookback cap {self._max_lookback}")

    def _frame_from(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": self._open[-n:],
                "high": self._high[-n:],
                "low": self._low[-n:],
                "close": self._close[-n:],
                "volume": self._volume[-n:],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime(self._times[-n:], unit="ms", utc=True), name="time"
            ),
        )


@dataclass
class PositionView:
    """What a strategy may know about the open position."""

    side: int                 # 0 when flat
    volume: float             # open lots
    entry_price: float
    sl: Optional[float]
    tp: Optional[float]
    entry_time_utc_ms: int
    unrealized_pnl: float


@dataclass
class Context:
    """Everything a strategy may read. Nothing here can see the future."""

    symbol: str
    balance: float
    equity: float
    position: PositionView
    bars: BarWindow
    tick_time_utc_ms: int = 0
    bid: float = 0.0
    ask: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


class Strategy:
    """Base class. Subclass and override what you need.

    ``uses_ticks`` declares that ``on_tick`` is meaningful for this strategy.
    Bar-only strategies are dramatically faster on the tick engine because
    intra-bar work is vectorized; tick strategies force the per-tick path.
    """

    name: str = "unnamed"
    version: str = "0"
    uses_ticks: bool = False
    # Bars the strategy needs for warm-up; engines skip signals before this.
    warmup_bars: int = 50

    def on_bar(self, ctx: Context, bar: pd.Series) -> list[Signal]:
        return []

    def on_tick(self, ctx: Context, tick: dict[str, Any]) -> list[Signal]:
        return []

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        return []
