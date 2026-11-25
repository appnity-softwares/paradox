"""Tick-level strategies (uses_ticks=True).

These run on the tick engine's per-tick slow path: on_tick fires for
every tick, and every signal goes through the same sizing, latency,
slippage, session, news and behaviour gates as bar strategies. Slower
by design — use for precision entries, not long campaigns.

Rules documented objectively before code:
- TickReversion: mean-reversion scalp on the tick stream. Keeps a
  rolling window of recent mid prices; when the current mid drops
  `drop_ticks` threshold-units below the window mean, buy; when it
  rises above, sell. Fixed SL/TP distances in price units. One
  position at a time; cooldown after each exit.
"""
from __future__ import annotations

from collections import deque

from core.strategy_api import (Context, Signal, Strategy, SIDE_LONG,
                               SIDE_SHORT)


class TickReversion(Strategy):
    key = "tick_reversion"
    version = "1"
    uses_ticks = True
    warmup_bars = 0

    def __init__(self, window: int = 50, threshold: float = 1.0,
                 sl_distance: float = 3.0, tp_distance: float = 4.5,
                 cooldown_ticks: int = 200):
        if window < 5:
            raise ValueError("window must be >= 5")
        if threshold <= 0 or sl_distance <= 0 or tp_distance <= 0:
            raise ValueError("threshold, sl_distance, tp_distance must be > 0")
        if cooldown_ticks < 0:
            raise ValueError("cooldown_ticks must be >= 0")
        self.window = window
        self.threshold = threshold
        self.sl_distance = sl_distance
        self.tp_distance = tp_distance
        self.cooldown_ticks = cooldown_ticks
        self._mids: deque[float] = deque(maxlen=window)
        self._since_exit = cooldown_ticks  # start ready

    def on_tick(self, ctx: Context, tick: dict) -> list[Signal]:
        mid = (tick["bid"] + tick["ask"]) / 2.0
        self._since_exit += 1
        mean = (sum(self._mids) / len(self._mids)) if self._mids else mid
        self._mids.append(mid)

        if ctx.position.side != 0:
            return []
        if self._since_exit < self.cooldown_ticks or len(self._mids) < self.window:
            return []

        if mid <= mean - self.threshold:
            return [Signal(side=SIDE_LONG, kind="market", sl=mid - self.sl_distance,
                           tp=mid + self.tp_distance, tag="tick_rev_long")]
        if mid >= mean + self.threshold:
            return [Signal(side=SIDE_SHORT, kind="market", sl=mid + self.sl_distance,
                           tp=mid - self.tp_distance, tag="tick_rev_short")]
        return []

    def on_fill(self, ctx: Context, fill) -> None:
        if fill.reason == "entry":
            self._since_exit = 0
