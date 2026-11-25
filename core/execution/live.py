"""Live signal runtime (semi-auto / paper).

Pluggable quote sources, one runner:

- ReplaySource: replays the historical tick store as if real-time
  (clock-simulated) — the development/test path, works anywhere.
- OandaStreamSource: free practice-account live stream (stub-ready;
  requires account id + token from the user).
- MT5BridgeSource: Windows-only bridge client (implemented at
  deployment time; see docs/deploy_windows_bridge.md).

Every signal goes through the SAME ExecutionRouter (compliance + risk
+ mode gate) as backtests — no path skips the gates. Paper mode fills
through PaperBroker; semi_auto mode emits alerts via the Notifier.

Bar convention matches the tick engine exactly (mid-price OHLC,
bucket-start timestamps, tick-count volume) so replay-vs-backtest
determinism can be tested.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from core.strategy_api import (Context, PositionView, Signal, SIDE_LONG,
                               SIDE_SHORT)
from core.tick_engine import BAR_MS

# ---------------------------------------------------------------------------
# Quote sources
# ---------------------------------------------------------------------------

@dataclass
class Tick:
    time_utc_ms: int
    bid: float
    ask: float


class ReplaySource:
    """Replay a tick DataFrame as a live feed.

    speed: wall-clock multiplier. 1.0 = real time, 60.0 = 1 min of
    ticks per second, None = as fast as possible (tests)."""

    def __init__(self, ticks_df: pd.DataFrame, speed: float | None = None):
        req = {"time_utc_ms", "bid", "ask"}
        if not req.issubset(ticks_df.columns):
            raise ValueError(f"ticks_df needs columns {req}")
        df = ticks_df.sort_values("time_utc_ms")
        self._t = df["time_utc_ms"].to_numpy(dtype=np.int64)
        self._bid = df["bid"].to_numpy(dtype=np.float64)
        self._ask = df["ask"].to_numpy(dtype=np.float64)
        if speed is not None and speed <= 0:
            raise ValueError("speed must be > 0 or None")
        self._speed = speed
        self._i = 0

    def __iter__(self) -> Iterator[Tick]:
        wall_start = time.monotonic()
        data_start = int(self._t[0]) if len(self._t) else 0
        for i in range(self._i, len(self._t)):
            if self._speed is not None:
                target = (self._t[i] - data_start) / 1000.0 / self._speed
                wait = target - (time.monotonic() - wall_start)
                if wait > 0:
                    time.sleep(wait)
            self._i = i + 1
            yield Tick(int(self._t[i]), float(self._bid[i]),
                       float(self._ask[i]))

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._t)


class OandaStreamSource:
    """Live XAUUSD stream from an OANDA v20 practice account.

    Constructed lazily: needs account_id + token (user-provided).
    Not auto-started anywhere; the runner only consumes __iter__."""

    def __init__(self, account_id: str, token: str,
                 instrument: str = "XAU_USD",
                 env: str = "practice"):
        if not account_id or not token:
            raise ValueError("OANDA account_id and token are required")
        self.account_id = account_id
        self.token = token
        self.instrument = instrument
        self.env = env

    def __iter__(self) -> Iterator[Tick]:
        # Implemented with the v20 streaming endpoint at integration
        # time; kept out of the deterministic core so tests never hit
        # the network.
        raise NotImplementedError(
            "OandaStreamSource: integrate v20 pricing stream at "
            "setup time (docs/deployment_topologies.md)")


# ---------------------------------------------------------------------------
# Bar aggregation (engine-identical convention)
# ---------------------------------------------------------------------------

class BarAggregator:
    """Ticks -> closed bars, mid-price OHLC, bucket-start timestamps,
    tick-count volume — identical to the tick engine's convention."""

    def __init__(self, bar_timeframe: str = "5m"):
        if bar_timeframe not in BAR_MS:
            raise ValueError(f"unknown timeframe {bar_timeframe!r}")
        self.bar_ms = BAR_MS[bar_timeframe]
        self._bucket: int | None = None
        self._mids: list[float] = []

    def feed(self, tick: Tick) -> dict | None:
        """Feed one tick. Returns the CLOSED bar dict when the tick
        starts a new bucket (engine convention: the bar is delivered
        when the next chunk's first tick arrives)."""
        bucket = tick.time_utc_ms // self.bar_ms
        mid = (tick.bid + tick.ask) / 2.0
        closed = None
        if self._bucket is not None and bucket != self._bucket:
            arr = np.asarray(self._mids, dtype=float)
            closed = {
                "time_utc_ms": self._bucket * self.bar_ms,
                "open": float(arr[0]), "high": float(arr.max()),
                "low": float(arr.min()), "close": float(arr[-1]),
                "volume": float(arr.size),
            }
            self._mids = []
        self._bucket = bucket
        self._mids.append(mid)
        return closed


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------

class LiveRunner:
    """Pull ticks from a source, aggregate bars, run the strategy on
    each closed bar, and push signals through the ExecutionRouter.

    paper mode: accepted orders fill through PaperBroker.
    semi_auto mode: accepted orders are emitted as alert dicts through
    the Notifier (no orders placed)."""

    def __init__(self, *, source, strategy, symbol: str, account_id: str,
                 mode: str, router, paper_broker=None, notifier=None,
                 bar_timeframe: str = "5m", risk_pct: float = 0.25,
                 initial_balance: float = 10_000.0,
                 session_start_min: int | None = None,
                 session_end_min: int | None = None,
                 max_bars: int = 1500):
        if mode not in ("paper", "semi_auto"):
            raise ValueError("mode must be 'paper' or 'semi_auto'")
        if mode == "paper" and paper_broker is None:
            raise ValueError("paper mode requires a PaperBroker")
        if mode == "semi_auto" and notifier is None:
            raise ValueError("semi_auto mode requires a Notifier")
        from core.execution.versions import make_version
        self.source = source
        self.strategy = strategy
        self.symbol = symbol
        self.account_id = account_id
        self.mode = mode
        self.router = router
        self.paper_broker = paper_broker
        self.notifier = notifier
        self.risk_pct = risk_pct
        self.balance = float(initial_balance)
        self.session = (session_start_min, session_end_min)
        self.aggregator = BarAggregator(bar_timeframe)
        from core.strategy_api import BarWindow
        self.bar_window = BarWindow(max_lookback=max_bars)
        self.version = make_version(type(strategy), {})
        self._counter = itertools.count(1)
        self.signals_sent: list[dict] = []
        self.ticks_seen = 0
        self.bars_closed = 0

    # -- session filter -----------------------------------------------------

    def _in_session(self, ts_ms: int) -> bool:
        if self.session == (None, None):
            return True
        start, end = self.session
        mod = (ts_ms // 60_000) % 1440
        return bool(start <= mod < end)

    # -- main loop ----------------------------------------------------------

    def run(self, max_ticks: int | None = None) -> dict:
        from core.execution.intent import OrderIntent
        pos = None   # live position mirror for PositionView
        for tick in self.source:
            self.ticks_seen += 1
            if max_ticks is not None and self.ticks_seen > max_ticks:
                break
            if not self._in_session(tick.time_utc_ms):
                continue
            closed = self.aggregator.feed(tick)
            if closed is None:
                continue
            self.bar_window._append(closed["time_utc_ms"], closed["open"],
                                    closed["high"], closed["low"],
                                    closed["close"], closed["volume"])
            self.bars_closed += 1
            # Build the strategy context from live state.
            if self.paper_broker is not None:
                mine = [p for p in self.paper_broker.positions()
                        if p.get("symbol") == self.symbol]
                if mine:
                    p = mine[-1]
                    side = int(p["side"])
                    mark = tick.bid if side == 1 else tick.ask
                    from core.config import get_symbol_spec
                    contract = get_symbol_spec(self.symbol).contract_size
                    unreal = (mark - p["entry_price"]) * side \
                        * contract * p["lot"]
                    pos = PositionView(side, p["lot"], p["entry_price"],
                                       p.get("sl_price"), p.get("tp_price"),
                                       p["opened_ms"], unreal)
                else:
                    pos = PositionView(0, 0.0, 0.0, None, None, 0, 0.0)
            else:
                pos = PositionView(0, 0.0, 0.0, None, None, 0, 0.0)
            ctx = Context(
                symbol=self.symbol, balance=self.balance,
                equity=self.balance + pos.unrealized_pnl,
                position=pos, bars=self.bar_window,
                tick_time_utc_ms=tick.time_utc_ms,
                bid=tick.bid, ask=tick.ask)
            try:
                signals = self.strategy.on_bar(ctx, closed) or []
            except Exception as exc:   # strategy bug must not kill the feed
                self.signals_sent.append(
                    {"error": repr(exc), "ts": tick.time_utc_ms})
                continue
            for sig in signals:
                if sig.side not in (SIDE_LONG, SIDE_SHORT):
                    continue
                intent = OrderIntent(
                    idempotency_key=f"live-{self.symbol}-"
                                    f"{next(self._counter)}",
                    account_id=self.account_id, symbol=self.symbol,
                    side=sig.side, intended_risk_pct=self.risk_pct,
                    strategy_version=self.version, sl_price=sig.sl,
                    tp_price=sig.tp, source="signal",
                    created_ms=tick.time_utc_ms, comment=sig.tag or "")
                result = self.router.submit(intent)
                if result.status not in ("approved", "filled",
                                         "pending_accepted"):
                    continue
                self.signals_sent.append(
                    {"ts": tick.time_utc_ms, "side": int(sig.side),
                     "sl": sig.sl, "tp": sig.tp, "tag": sig.tag,
                     "mode": self.mode, "status": result.status,
                     "lot": result.filled_lot,
                     "order_ref": result.order_ref})
                if self.mode == "paper" and result.order_ref:
                    # Paper fills are managed by the PaperBroker inside
                    # the router's send step; live SL/TP management
                    # lands with the bridge adapter (U9 prep).
                    pass
                elif self.mode == "semi_auto" and self.notifier is not None:
                    self.notifier.send(
                        f"{self.symbol} {'LONG' if sig.side == 1 else 'SHORT'} "
                        f"@{closed['close']:.2f} SL={sig.sl} TP={sig.tp} "
                        f"[{sig.tag or 'signal'}]")
        return {"ticks_seen": self.ticks_seen,
                "bars_closed": self.bars_closed,
                "signals": len(self.signals_sent)}
