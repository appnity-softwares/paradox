"""Bridge legacy ``generate_signals(df)`` strategies onto the shared interface.

Master Prompt 6.5: existing strategies keep working unchanged while the
platform migrates to the signal-only contract of ``core.strategy_api``.

Semantics and documented divergences vs the bar engine:

- The adapter feeds the legacy strategy the last ``max_bars`` CLOSED bars and
  reads only the newest signal row. Indicators are therefore recomputed on a
  bounded window; after the warmup the truncation error of recursive
  indicators (EMA and friends) decays exponentially and is negligible for the
  default window (1500 bars vs a 200-bar slow EMA).
- A legacy signal acts on the bar AFTER it was computed in the bar engine;
  here it acts on the first ticks after the signal bar closes. Same causality,
  different fill model.
- A direction flip takes two steps (close now, enter once flat) instead of
  the bar engine's same-bar close-and-reverse. Conservative, never
  look-ahead.
- Stop-loss / take-profit come from the legacy ``StrategyConfig``
  (``sl_method``/``sl_atr_mult``/``sl_fixed``/``rr``), anchored at the signal
  bar close instead of the unknown future fill price.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.indicators import atr as atr_indicator
from core.strategy_api import (
    Context,
    Signal,
    Strategy,
    SIDE_CLOSE,
    SIDE_LONG,
    SIDE_SHORT,
)
from strategies.base import StrategyConfig


class LegacyAdapter(Strategy):
    """Wrap a legacy strategy (``key``, ``warmup``, ``generate_signals``)."""

    version = "1"

    def __init__(self, legacy: Any, cfg: StrategyConfig, max_bars: int = 1500):
        self.legacy = legacy
        self.cfg = cfg
        self.key = getattr(legacy, "key", legacy.__class__.__name__)
        self.name = f"legacy:{self.key}"
        self.warmup_bars = int(getattr(legacy, "warmup", 60)) + 1
        if max_bars < max(self.warmup_bars + 10, 300):
            raise ValueError("max_bars too small for a stable indicator window")
        self._max_bars = int(max_bars)
        self._prev_sig = 0.0

    # -- internal window -----------------------------------------------------

    def _window(self, ctx: Context) -> pd.DataFrame:
        n = min(len(ctx.bars), self._max_bars)
        return ctx.bars.tail(n)

    def _sl_tp(self, window: pd.DataFrame, direction: int, close: float) -> tuple[float, float]:
        """Stop distance from the legacy config, anchored at the bar close."""
        if self.cfg.sl_method == "atr":
            atr_series = atr_indicator(window, length=14)
            atr_value = float(atr_series.iloc[-1])
            dist = self.cfg.sl_atr_mult * atr_value
            if not np.isfinite(dist) or dist <= 0:
                dist = float(self.cfg.sl_fixed)
        else:
            dist = float(self.cfg.sl_fixed)
        dist = max(dist, 1e-9)
        sl = close - direction * dist
        tp = close + direction * dist * float(self.cfg.rr)
        return sl, tp

    # -- shared interface ------------------------------------------------------

    def on_bar(self, ctx: Context, bar: dict) -> list[Signal]:
        window = self._window(ctx)
        if len(window) < self.warmup_bars:
            return []
        sig_series = self.legacy.generate_signals(window)
        if not len(sig_series):
            return []
        sig = float(sig_series.iloc[-1])
        if not np.isfinite(sig):
            sig = 0.0
        self._prev_sig = sig

        pos = ctx.position
        out: list[Signal] = []
        # 1) Opposite or neutral signal closes an open position.
        if pos.side == SIDE_LONG and sig <= 0:
            out.append(Signal(SIDE_CLOSE, tag=f"{self.key}:exit"))
        elif pos.side == SIDE_SHORT and sig >= 0:
            out.append(Signal(SIDE_CLOSE, tag=f"{self.key}:exit"))
        # 2) Enter only when flat (flips resolve on the next bar, see docstring).
        if pos.side == 0 and sig != 0:
            direction = 1 if sig > 0 else -1
            sl, tp = self._sl_tp(window, direction, float(bar["close"]))
            out.append(Signal(
                SIDE_LONG if direction == 1 else SIDE_SHORT,
                sl=sl, tp=tp, tag=f"{self.key}:entry",
            ))
        return out
