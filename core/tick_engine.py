"""Production tick-by-tick backtest engine (Master Prompt 6.2–6.4).

Replays real bid/ask ticks in order from the partitioned Parquet store:

- Long entries fill at ASK, long exits at BID; short entries at BID, exits
  at ASK (correct side of the quote, always).
- SL, TP and pending order triggers are evaluated on every tick using the
  correct side. When both SL and TP fall inside one bar the TICK ORDER
  decides — the conservative bar-engine rule is only needed when ticks are
  unavailable.
- Order types: market, limit, stop. Partial closes, breakeven, trailing
  stop and time-based exit are supported.
- Costs come from ``core.config`` CostModel/SymbolSpec: recorded spread,
  commission (per lot, charged at entry and at exit, matching the bar
  engine), configurable slippage (market/stop, news-multiplied), swap with
  the triple-swap day, latency (fills use the first tick AFTER the delay),
  margin and stop-out.
- Strategies run through the shared ``core.strategy_api.Strategy`` interface
  and return signals only. Bars are built incrementally from the same ticks
  and a bar is delivered only after it closes (an unfinished final bar is
  never delivered).

Determinism: the engine uses no randomness. The same BacktestConfig plus the
same dataset hash reproduces the same trade log bit for bit.

Performance: bar-only strategies run on a vectorized fast path (per-minute
chunk scans with NumPy); strategies that declare ``uses_ticks`` fall back to
the per-tick loop. The measured runtime is reported on every run.

Known tolerances (documented, not hidden):
- A pending/stop entry and its SL can only be evaluated from the tick AFTER
  the entry tick; a same-tick stop-through is not simulated.
- Swap is charged once per UTC-midnight boundary crossed while holding
  (3x on the configured triple-swap day); weekend nights are covered by the
  triple charge, matching common broker practice.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.config import (
    DEFAULT_SYMBOL,
    get_broker_spec,
    get_cost_model,
    get_symbol_spec,
    round_lot,
)
from core.strategy_api import (
    KIND_LIMIT,
    KIND_MARKET,
    KIND_STOP,
    SIDE_CLOSE,
    SIDE_LONG,
    SIDE_SHORT,
    BarWindow,
    Context,
    Fill,
    LookAheadError,
    PositionView,
    Signal,
    Strategy,
)

ENGINE_VERSION = "tick-1.0.0"

MS_PER_MINUTE = 60_000
MS_PER_DAY = 86_400_000

BAR_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}


@dataclass
class BacktestConfig:
    """Everything needed to reproduce a run bit for bit (Master Prompt 5.1)."""

    symbol: str = DEFAULT_SYMBOL
    initial_balance: float = 100_000.0
    risk_pct: float = 1.0
    cost_key: str = "XAUUSD_ECN"
    broker_key: str = "mt5_bridge"
    dataset_hash: str = ""
    start_ms: int = 0
    end_ms: int = 0
    bar_timeframe: str = "1m"
    # Trade management (engine-level, deterministic)
    breakeven_rr: float = 0.0        # 0 = off; arm breakeven after this many R
    breakeven_offset: float = 0.0    # price units locked in at breakeven
    trail_distance: float = 0.0      # price units behind the favourable extreme; 0 = off
    time_exit_minutes: float = 0.0   # 0 = off
    require_sl: bool = True          # reject entries without a stop-loss
    default_sl_distance: float = 1.0  # sizing fallback when require_sl is False
    # Scheduled high-impact news windows (start_ms, end_ms); slippage inside
    # these windows is multiplied by CostModel.slippage.news_multiplier.
    news_windows: tuple[tuple[int, int], ...] = ()
    # ── U1 behaviour controls (backtest parity). None/empty = OFF, so
    # legacy configs behave exactly as before. The parity contract: the
    # SAME controls the live router enforces are applied here, and every
    # blocked action is counted in result["behaviour_blocks"].
    min_hold_seconds: int | None = None          # blocks signal exits only;
                                                 # SL/TP/stop-out stay protective
    max_orders_per_day: int | None = None        # entries per UTC day
    max_modifications_per_trade: int | None = None  # discrete SL ratchets
    prohibited_styles: tuple[str, ...] = ()      # "martingale" supported
    martingale_escalation_factor: float = 1.5
    martingale_lookback_min: int = 240
    # ── U2 session filter (UTC minutes-of-day; None = OFF). Backtests
    # use UTC minutes directly: DST is a live-path concern resolved by
    # SessionProfile (docs/session_profiles.md). Semantics match live:
    # entries only inside the window and before the cutoff; open
    # positions flatten when a new window starts (protective).
    # U3: when True, NEW ENTRIES inside a news window are blocked
    # (parity with the live router's block_open). Default OFF so legacy
    # configs only see the news slippage multiplier.
    block_entries_in_news_windows: bool = False
    session_start_min: int | None = None
    session_end_min: int | None = None
    session_days: tuple[int, ...] | None = None   # weekdays, Mon=0 (UTC)
    no_new_entries_before_end_min: int = 0
    flatten_at_session_end: bool = True
    engine_version: str = ENGINE_VERSION

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class _Position:
    side: int                 # +1 long, -1 short
    volume: float             # open lots (after partials)
    initial_volume: float
    entry_price: float
    entry_time_ms: int
    sl: float
    tp: float
    entry_fee: float          # commission charged at entry (per side convention)
    swap_paid: float = 0.0
    high_water: float = 0.0   # best favourable price seen (trailing anchor)
    last_swap_day: int = 0    # UTC day number of the last swap charge


@dataclass
class _PendingOrder:
    side: int
    kind: str                 # limit | stop
    price: float
    sl: float
    tp: float
    volume: float
    tag: str
    created_ms: int


@dataclass
class _MarketOrder:
    """A market/close order waiting out its latency window."""

    side: int                 # SIDE_LONG / SIDE_SHORT / SIDE_CLOSE
    sl: float
    tp: float
    volume: float
    tag: str
    target_ms: int            # first tick at/after this time fills it


class TickEngine:
    """Deterministic tick replay engine. See the module docstring."""

    def __init__(self) -> None:
        self.lookahead_violations: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, strategy: Strategy, ticks: pd.DataFrame, config: BacktestConfig) -> dict[str, Any]:
        started = time.perf_counter()
        spec = get_symbol_spec(config.symbol)
        cost = get_cost_model(config.cost_key)
        broker = get_broker_spec(config.broker_key)

        fail = self._validate_input(ticks, config)
        if fail:
            return fail

        t = ticks["time_utc_ms"].to_numpy(dtype=np.int64)
        bid = ticks["bid"].to_numpy(dtype=np.float64)
        ask = ticks["ask"].to_numpy(dtype=np.float64)

        # ── U2 session filter: load ONLY ticks inside the window ──────
        sess_active = config.session_start_min is not None and \
            config.session_end_min is not None
        if sess_active:
            s_start, s_end = int(config.session_start_min), int(config.session_end_min)
            mods = (t // 60_000) % 1440
            if s_end > s_start:
                keep = (mods >= s_start) & (mods < s_end)
            else:                                   # wraps midnight
                keep = (mods >= s_start) | (mods < s_end)
            if config.session_days is not None:
                # Epoch day 0 (1970-01-01) was a Thursday (weekday 3).
                weekdays = ((t // 86_400_000) + 3) % 7
                keep &= np.isin(weekdays, np.asarray(tuple(config.session_days)))
            t, bid, ask = t[keep], bid[keep], ask[keep]

        bar_ms = BAR_MS[config.bar_timeframe]
        chunk_starts = self._chunk_starts(t, bar_ms)

        contract = spec.contract_size
        comm_side = cost.commission_per_lot          # per side; round-trip = 2x
        lat_ms = int(cost.fill_latency_ms)
        slip_m, slip_s = cost.slippage.market_points, cost.slippage.stop_points
        news_mult = cost.slippage.news_multiplier
        news_windows = config.news_windows
        stop_out_level = broker.stop_out_level_pct / 100.0
        leverage = broker.leverage

        balance = float(config.initial_balance)
        pos: Optional[_Position] = None
        pendings: list[_PendingOrder] = []
        markets: list[_MarketOrder] = []
        trades: list[dict[str, Any]] = []
        fills: list[Fill] = []
        equity_curve: list[tuple[int, float]] = []
        lookahead: list[str] = []
        trade_seq = 0

        # ── U1 behaviour-control state ────────────────────────────────────
        # Counts of every blocked action, reported in the result dict.
        behaviour_blocks: dict[str, int] = {
            "min_hold_time": 0,
            "max_orders_per_day": 0,
            "max_modifications_per_trade": 0,
            "session_no_new_entries": 0,
            "news_window": 0,
            "prohibited_style_martingale": 0,
        }
        bh_min_hold_ms = (config.min_hold_seconds * 1000
                          if config.min_hold_seconds is not None else None)
        bh_day_orders: dict[int, int] = {}
        bh_last_close: dict[tuple[int, str], tuple[int, float, float]] = {}
        # (side, symbol) -> (ts_ms, lot, pnl) of the last closed trade
        bh_mods: dict[int, int] = {}   # id(pos) -> discrete SL ratchets used
        bh_mods_frozen: set[int] = set()

        bar_window = BarWindow()
        ctx = Context(
            symbol=config.symbol, balance=balance, equity=balance,
            position=PositionView(0, 0.0, 0.0, None, None, 0, 0.0),
            bars=bar_window,
        )

        # ---- closures over the mutable run state -------------------------

        def in_news(ts: int) -> bool:
            for lo, hi in news_windows:
                if lo <= ts <= hi:
                    return True
            return False

        def market_slip(ts: int) -> float:
            return slip_m * news_mult if in_news(ts) else slip_m

        def stop_slip(ts: int) -> float:
            return slip_s * news_mult if in_news(ts) else slip_s

        def book_trade(p: _Position, exit_price: float, exit_ms: int, reason: str,
                       volume_closed: float, fee_exit: float) -> float:
            nonlocal balance, trade_seq
            if p.side == SIDE_LONG:
                gross = (exit_price - p.entry_price) * contract * volume_closed
            else:
                gross = (p.entry_price - exit_price) * contract * volume_closed
            pnl = gross - fee_exit
            balance += pnl
            trade_seq += 1
            share = volume_closed / p.initial_volume if p.initial_volume else 0.0
            trades.append({
                "id": trade_seq,
                "symbol": config.symbol,
                "side": "LONG" if p.side == SIDE_LONG else "SHORT",
                "entry_time_utc_ms": int(p.entry_time_ms),
                "exit_time_utc_ms": int(exit_ms),
                "entry_price": round(float(p.entry_price), spec.digits + 1),
                "exit_price": round(float(exit_price), spec.digits + 1),
                "volume": round(float(volume_closed), 4),
                "pnl": round(float(pnl), 2),
                "entry_fee": round(float(p.entry_fee * share), 2),
                "exit_fee": round(float(fee_exit), 2),
                "swap": round(float(p.swap_paid * share), 2),
                "exit_reason": reason,
            })
            # U1: remember the closed trade for martingale detection.
            bh_last_close[(p.side, config.symbol)] = (
                int(exit_ms), float(volume_closed), float(pnl))
            return pnl

        def close_position(price: float, ts: int, reason: str, volume_closed: Optional[float] = None) -> None:
            """Close part or all of the open position at `price`."""
            nonlocal pos
            assert pos is not None
            vol = pos.volume if volume_closed is None else min(volume_closed, pos.volume)
            fee_exit = comm_side * vol
            book_trade(pos, price, ts, reason, vol, fee_exit)
            pos.volume -= vol
            # A remainder below the minimum lot is dust: close it out too.
            if pos.volume < spec.min_lot - 1e-12:
                pos = None

        def open_position(order_side: int, price: float, ts: int, sl: float, tp: float,
                          volume: float, tag: str) -> None:
            nonlocal pos, balance
            fee = comm_side * volume
            balance -= fee
            pos = _Position(
                side=order_side, volume=volume, initial_volume=volume,
                entry_price=price, entry_time_ms=ts, sl=sl, tp=tp,
                entry_fee=fee, high_water=price,
                last_swap_day=self._utc_day(ts),
            )
            fills.append(Fill(order_tag=tag, side=order_side, price=price,
                              volume=volume, time_utc_ms=ts, reason="entry",
                              pnl_realized=0.0))

        def submit_signals(signals: list[Signal], decision_ms: int, ref_price: float,
                           equity_now: float) -> None:
            for sig in signals:
                if sig.side == SIDE_CLOSE:
                    if pos is not None:
                        # U1 min-hold parity: a signal exit of a position
                        # younger than min_hold_seconds is blocked (the
                        # same reason code the live router uses).
                        # SL/TP/stop-out/time-exit stay protective.
                        if bh_min_hold_ms is not None and \
                                decision_ms - pos.entry_time_ms < bh_min_hold_ms:
                            behaviour_blocks["min_hold_time"] += 1
                            continue
                        vol = pos.volume * sig.fraction
                        markets.append(_MarketOrder(
                            side=SIDE_CLOSE, sl=0.0, tp=0.0, volume=vol,
                            tag=sig.tag, target_ms=decision_ms + lat_ms,
                        ))
                    continue
                if pos is not None:
                    continue  # single net position; extra entries ignored
                if config.require_sl and sig.sl is None:
                    continue  # mandatory stop-loss
                # ── U3 news-window entry block (live block_open parity) ──
                if config.block_entries_in_news_windows and \
                        config.news_windows:
                    if in_news(decision_ms):
                        behaviour_blocks["news_window"] += 1
                        continue
                # ── U2 session entry cutoff (same rule as the live gate) ──
                if sess_active:
                    mod = (decision_ms // 60_000) % 1440
                    if s_end > s_start:
                        inside = s_start <= mod < s_end
                        dist_min = s_end - mod
                    else:
                        inside = mod >= s_start or mod < s_end
                        dist_min = (s_end - mod) % 1440
                    if not inside or \
                            dist_min <= config.no_new_entries_before_end_min:
                        behaviour_blocks["session_no_new_entries"] += 1
                        continue
                # ── U1 entry gates ───────────────────────────────────────
                if config.max_orders_per_day is not None:
                    day = decision_ms // 86_400_000
                    used = bh_day_orders.get(day, 0)
                    if used >= config.max_orders_per_day:
                        behaviour_blocks["max_orders_per_day"] += 1
                        continue
                if sig.sl is not None:
                    sl_dist = abs(ref_price - sig.sl)
                    sl_price = float(sig.sl)
                else:
                    sl_dist = config.default_sl_distance
                    sl_price = 0.0
                if sl_dist <= 0:
                    continue
                risk_amount = equity_now * config.risk_pct / 100.0 * sig.fraction
                lots = round_lot(risk_amount / (sl_dist * contract), config.symbol)
                if lots <= 0:
                    continue
                # U1 martingale parity: a same-direction entry whose lot
                # escalates >= factor over the last CLOSED LOSING trade's
                # lot within the lookback window is blocked.
                if "martingale" in config.prohibited_styles:
                    last = bh_last_close.get((sig.side, config.symbol))
                    if last is not None:
                        lc_ts, lc_lot, lc_pnl = last
                        if lc_pnl < 0 and lots >= \
                                lc_lot * config.martingale_escalation_factor and \
                                decision_ms - lc_ts <= \
                                config.martingale_lookback_min * MS_PER_MINUTE:
                            behaviour_blocks["prohibited_style_martingale"] += 1
                            continue
                # Count the accepted entry against the daily cap.
                if config.max_orders_per_day is not None:
                    day = decision_ms // 86_400_000
                    bh_day_orders[day] = bh_day_orders.get(day, 0) + 1
                tp = float(sig.tp) if sig.tp is not None else 0.0
                if sig.kind == KIND_MARKET:
                    markets.append(_MarketOrder(side=sig.side, sl=sl_price, tp=tp, volume=lots,
                                                tag=sig.tag, target_ms=decision_ms + lat_ms))
                elif sig.kind in (KIND_LIMIT, KIND_STOP):
                    pendings.append(_PendingOrder(side=sig.side, kind=sig.kind,
                                                  price=float(sig.price), sl=sl_price, tp=tp,
                                                  volume=lots, tag=sig.tag,
                                                  created_ms=decision_ms))

        def deliver_fills() -> None:
            for f in fills:
                try:
                    strategy.on_fill(ctx, f)
                except LookAheadError as exc:
                    lookahead.append(f"on_fill: {exc}")
            fills.clear()

        def manage_position(scan_from: int, hi: int, ct: np.ndarray, cb: np.ndarray,
                            ca: np.ndarray) -> int:
            """Evaluate SL/TP/trailing/time-exit/stop-out on every tick in
            [scan_from, hi). Returns the index after the last processed tick
            (entry scans must start there), or `hi` if nothing triggered."""
            nonlocal pos
            start = scan_from
            while pos is not None and start < hi:
                n = hi - start
                sl_arr = ct[start:hi]
                if pos.side == SIDE_LONG:
                    mark = cb[start:hi]
                    raw_sl_hit = mark <= pos.sl if pos.sl > 0 else np.zeros(n, dtype=bool)
                    tp_hit = (mark >= pos.tp) if pos.tp > 0 else np.zeros(n, dtype=bool)
                else:
                    mark = ca[start:hi]
                    raw_sl_hit = mark >= pos.sl if pos.sl > 0 else np.zeros(n, dtype=bool)
                    tp_hit = (mark <= pos.tp) if pos.tp > 0 else np.zeros(n, dtype=bool)

                # Trailing / breakeven: ratcheted from the persisted extreme.
                eff_sl_hit = raw_sl_hit
                # U1: a discrete SL ratchet = one broker modification
                # request. Once the per-trade cap is reached the stop is
                # frozen at its last level (fail safe) and further
                # ratchets are counted as blocked.
                mods_cap = config.max_modifications_per_trade
                ratchet_allowed = True
                if mods_cap is not None and pos.entry_time_ms in bh_mods_frozen:
                    ratchet_allowed = False
                if (config.trail_distance > 0 or config.breakeven_rr > 0):
                    if not ratchet_allowed:
                        # Frozen: every chunk that would have ratcheted
                        # counts as a blocked modification request.
                        behaviour_blocks["max_modifications_per_trade"] += 1
                    else:
                        seed = pos.high_water
                        run_best = np.maximum.accumulate(np.maximum(mark, seed)) \
                            if pos.side == SIDE_LONG \
                            else np.minimum.accumulate(np.minimum(mark, seed))
                        if pos.side == SIDE_LONG:
                            cand = run_best - config.trail_distance if config.trail_distance > 0 \
                                else np.full(n, -np.inf)
                            floor = pos.sl if pos.sl > 0 else -np.inf
                            if config.breakeven_rr > 0 and pos.sl > 0:
                                r0 = abs(pos.entry_price - pos.sl)
                                be = pos.entry_price + config.breakeven_offset
                                armed = run_best >= pos.entry_price + config.breakeven_rr * r0
                                cand = np.where(armed, np.maximum(cand, be), cand)
                            trail = np.maximum(cand, floor)
                            eff_sl_hit = mark <= trail
                        else:
                            cand = run_best + config.trail_distance if config.trail_distance > 0 \
                                else np.full(n, np.inf)
                            ceil = pos.sl if pos.sl > 0 else np.inf
                            if config.breakeven_rr > 0 and pos.sl > 0:
                                r0 = abs(pos.entry_price - pos.sl)
                                be = pos.entry_price - config.breakeven_offset
                                armed = run_best <= pos.entry_price - config.breakeven_rr * r0
                                cand = np.where(armed, np.minimum(cand, be), cand)
                            trail = np.minimum(cand, ceil)
                            eff_sl_hit = mark >= trail
                        if mods_cap is not None:
                            mkey = pos.entry_time_ms
                            bh_mods[mkey] = bh_mods.get(mkey, 0) + 1
                            if bh_mods[mkey] >= mods_cap:
                                bh_mods_frozen.add(mkey)

                time_hit = (sl_arr >= pos.entry_time_ms + int(config.time_exit_minutes * MS_PER_MINUTE)) \
                    if config.time_exit_minutes > 0 else np.zeros(n, dtype=bool)

                first_sl = np.flatnonzero(eff_sl_hit)
                first_tp = np.flatnonzero(tp_hit)
                first_te = np.flatnonzero(time_hit)
                events: list[tuple[int, str]] = []
                if first_sl.size:
                    events.append((start + int(first_sl[0]), "SL"))
                if first_tp.size:
                    events.append((start + int(first_tp[0]), "TP"))
                if first_te.size:
                    events.append((start + int(first_te[0]), "time_exit"))

                # Stop-out: margin level path over the same window.
                floating = (mark - pos.entry_price) * contract * pos.volume * pos.side
                equity_path = balance + floating
                margin = pos.volume * contract * mark / leverage
                with np.errstate(divide="ignore", invalid="ignore"):
                    level = np.where(margin > 0, equity_path / margin, np.inf)
                first_so = np.flatnonzero(level <= stop_out_level)
                if first_so.size:
                    events.append((start + int(first_so[0]), "stop_out"))

                if not events:
                    # Persist the trailing extreme for the next chunk.
                    if pos.side == SIDE_LONG:
                        pos.high_water = max(pos.high_water, float(cb[start:hi].max()))
                    else:
                        pos.high_water = min(pos.high_water, float(ca[start:hi].min()))
                    return hi

                events.sort()
                k, reason = events[0]
                ts = int(ct[k])
                if reason == "stop_out":
                    price = float(cb[k] if pos.side == SIDE_LONG else ca[k])
                    close_position(price, ts, "stop_out")
                    return k + 1
                exit_price = float(cb[k] - stop_slip(ts)) if pos.side == SIDE_LONG \
                    else float(ca[k] + stop_slip(ts))
                close_position(exit_price, ts, reason)
                return k + 1
            return hi if pos is None else start

        # ---- main chunk loop ----------------------------------------------

        n_ticks = len(t)
        for chunk_no in range(len(chunk_starts)):
            lo = int(chunk_starts[chunk_no])
            hi = int(chunk_starts[chunk_no + 1]) if chunk_no + 1 < len(chunk_starts) else n_ticks
            if hi <= lo:
                continue
            ct = t[lo:hi]
            cb = bid[lo:hi]
            ca = ask[lo:hi]
            chunk_day = self._utc_day(int(ct[0]))

            # 0) U2 session flatten: a position still open from a previous
            # window closes protectively at the first tick of the new
            # window (no ticks exist in the gap, so no fills were missed).
            if sess_active and config.flatten_at_session_end and pos is not None:
                day0 = int(ct[0]) // 86_400_000
                mod0 = (int(ct[0]) // 60_000) % 1440
                cur_wid = day0 if mod0 >= s_start else day0 - 1
                eday = pos.entry_time_ms // 86_400_000
                emod = (pos.entry_time_ms // 60_000) % 1440
                if (eday if emod >= s_start else eday - 1) != cur_wid:
                    px = float(cb[0] if pos.side == SIDE_LONG else ca[0])
                    close_position(px, int(ct[0]), "session_flatten")

            scan_from = lo

            # 1) Pending order triggers (first-hit vectorized scan).
            still_pending: list[_PendingOrder] = []
            filled_a_pending = False
            for idx_po, po in enumerate(pendings):
                if po.side == SIDE_LONG:
                    trigger = ca <= po.price if po.kind == KIND_LIMIT else ca >= po.price
                else:
                    trigger = cb >= po.price if po.kind == KIND_LIMIT else cb <= po.price
                hit = np.flatnonzero(trigger)
                if hit.size == 0:
                    still_pending.append(po)
                    continue
                k = lo + int(hit[0])
                ts = int(ct[hit[0]])
                if po.kind == KIND_LIMIT:
                    fill_price = float(min(ca[hit[0]], po.price)) if po.side == SIDE_LONG \
                        else float(max(cb[hit[0]], po.price))
                else:
                    fill_price = float(ca[hit[0]] + stop_slip(ts)) if po.side == SIDE_LONG \
                        else float(cb[hit[0]] - stop_slip(ts))
                open_position(po.side, fill_price, ts, po.sl, po.tp, po.volume, po.tag)
                scan_from = max(scan_from, k + 1)
                filled_a_pending = True
                # Keep the not-yet-checked pendings alive behind the new position.
                still_pending.extend(pendings[idx_po + 1:])
                break
            # still_pending excludes the filled order and keeps the remainder.
            pendings = still_pending

            # 2) Market/close orders after their latency window.
            still_market: list[_MarketOrder] = []
            for mo in markets:
                k_rel = int(np.searchsorted(ct, mo.target_ms, side="left"))
                if k_rel >= len(ct):
                    still_market.append(mo)
                    continue
                k = lo + k_rel
                ts = int(ct[k_rel])
                if mo.side == SIDE_CLOSE:
                    if pos is not None:
                        # Signal exits pay the market side too: long closes at
                        # bid - slip, short closes at ask + slip.
                        if pos.side == SIDE_LONG:
                            price = float(cb[k_rel] - market_slip(ts))
                        else:
                            price = float(ca[k_rel] + market_slip(ts))
                        close_position(price, ts, "Signal exit", mo.volume)
                        scan_from = max(scan_from, k + 1)
                else:
                    price = float(ca[k_rel] + market_slip(ts)) if mo.side == SIDE_LONG \
                        else float(cb[k_rel] - market_slip(ts))
                    open_position(mo.side, price, ts, mo.sl, mo.tp, mo.volume, mo.tag)
                    scan_from = max(scan_from, k + 1)
            markets = still_market

            # 3) Manage the open position tick by tick (vectorized scans).
            manage_position(scan_from - lo, hi - lo, ct, cb, ca)

            # 4) Swap at UTC-midnight rollovers while holding.
            if pos is not None and chunk_day != pos.last_swap_day:
                weekday = self._utc_weekday(int(ct[0]))
                mult = 3 if weekday == cost.swap_triple_day else 1
                rate = cost.swap_long if pos.side == SIDE_LONG else cost.swap_short
                pos.swap_paid += rate * pos.volume * mult
                pos.last_swap_day = chunk_day

            # 5) Close the bar; deliver it and collect signals.
            mid = (cb + ca) / 2.0
            bar_time = self._chunk_start_ms(int(ct[0]), bar_ms)
            bar_window._append(bar_time, float(mid[0]), float(mid.max()),
                               float(mid.min()), float(mid[-1]),
                               volume=float(mid.size))

            if pos is not None:
                mark_price = float(cb[-1] if pos.side == SIDE_LONG else ca[-1])
                equity = balance + (mark_price - pos.entry_price) * contract * pos.volume * pos.side
            else:
                equity = balance
            equity_curve.append((int(ct[-1]), round(float(equity), 2)))

            if chunk_no + 1 < len(chunk_starts):
                # The bar is closed: the next chunk has already begun.
                ctx.balance = balance
                ctx.equity = equity
                ctx.position = self._position_view(pos, contract, float(cb[-1]), float(ca[-1]))
                ctx.tick_time_utc_ms = int(ct[-1])
                ctx.bid = float(cb[-1])
                ctx.ask = float(ca[-1])
                try:
                    signals = strategy.on_bar(ctx, bar_window.last())
                except LookAheadError as exc:
                    lookahead.append(f"on_bar: {exc}")
                    signals = []
                if signals:
                    decision_ms = bar_time + bar_ms
                    submit_signals(signals, decision_ms, float(mid[-1]), equity)
                deliver_fills()

        # ── End of data: liquidate at the last tick ────────────────────────
        if pos is not None:
            price = float(bid[-1] if pos.side == SIDE_LONG else ask[-1])
            close_position(price, int(t[-1]), "end_of_data")

        elapsed = time.perf_counter() - started
        return self._summarize(config=config, trades=trades, equity_curve=equity_curve,
                               elapsed_sec=elapsed, ticks_processed=n_ticks,
                               lookahead_violations=lookahead, cost=cost,
                               behaviour_blocks=behaviour_blocks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(ticks: pd.DataFrame, config: BacktestConfig) -> Optional[dict[str, Any]]:
        if ticks is None or len(ticks) == 0:
            return {"error": "no tick data supplied"}
        for column in ("time_utc_ms", "bid", "ask"):
            if column not in ticks.columns:
                return {"error": f"tick data missing column: {column}"}
        t = ticks["time_utc_ms"].to_numpy(dtype=np.int64)
        if len(t) > 1 and np.any(np.diff(t) < 0):
            return {"error": "tick timestamps must be sorted ascending"}
        if config.initial_balance <= 0:
            return {"error": "initial_balance must be positive"}
        if not 0 < config.risk_pct <= 100:
            return {"error": "risk_pct must be in (0, 100]"}
        return None

    @staticmethod
    def _chunk_start_ms(ts_ms: int, bar_ms: int) -> int:
        return int(ts_ms // bar_ms) * bar_ms

    @staticmethod
    def _utc_day(ts_ms: int) -> int:
        return int(ts_ms // MS_PER_DAY)

    @staticmethod
    def _utc_weekday(ts_ms: int) -> int:
        # 1970-01-01 was a Thursday (Monday=0 convention).
        return (int(ts_ms // MS_PER_DAY) + 3) % 7

    @staticmethod
    def _chunk_starts(t: np.ndarray, bar_ms: int) -> np.ndarray:
        if len(t) == 0:
            return np.empty(0, dtype=np.int64)
        buckets = t // bar_ms
        change = np.empty(len(t), dtype=bool)
        change[0] = True
        change[1:] = buckets[1:] != buckets[:-1]
        return np.flatnonzero(change)

    @staticmethod
    def _position_view(pos: Optional[_Position], contract: float,
                       bid: float, ask: float) -> PositionView:
        if pos is None:
            return PositionView(0, 0.0, 0.0, None, None, 0, 0.0)
        mark = bid if pos.side == SIDE_LONG else ask
        unreal = (mark - pos.entry_price) * contract * pos.volume * pos.side
        return PositionView(pos.side, pos.volume, pos.entry_price,
                            pos.sl or None, pos.tp or None, pos.entry_time_ms, unreal)

    def _summarize(self, *, config: BacktestConfig, trades: list[dict[str, Any]],
                   equity_curve: list[tuple[int, float]], elapsed_sec: float,
                   ticks_processed: int, lookahead_violations: list[str],
                   cost: Any,
                   behaviour_blocks: dict[str, int] | None = None) -> dict[str, Any]:
        pnls = [tr["pnl"] for tr in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total = float(sum(pnls))
        wr = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
        pf = (sum(wins) / abs(sum(losses))) if losses else (99.0 if wins else 0.0)
        curve = np.array([e for _, e in equity_curve], dtype=float) \
            if equity_curve else np.array([config.initial_balance])
        peak = np.maximum.accumulate(curve)
        dd = peak - curve
        max_dd_usd = float(dd.max()) if dd.size else 0.0
        fees = sum(tr["entry_fee"] + tr["exit_fee"] for tr in trades)
        swap = sum(tr["swap"] for tr in trades)
        return {
            "engine": ENGINE_VERSION,
            "engine_version": ENGINE_VERSION,
            "config_hash": config.config_hash(),
            "dataset_hash": config.dataset_hash,
            "symbol": config.symbol,
            "cost_model": cost.describe(),
            "ticks_replayed": ticks_processed,
            "bars_processed": len(equity_curve),
            "runtime_seconds": round(elapsed_sec, 3),
            "ticks_per_second": round(ticks_processed / elapsed_sec) if elapsed_sec > 0 else None,
            "total_trades": len(trades),
            "net_profit_usd": round(total, 2),
            "return_pct": round(total / config.initial_balance * 100.0, 4),
            "win_rate_pct": round(wr, 2),
            "profit_factor": round(float(pf), 3),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "max_drawdown_pct": round(max_dd_usd / config.initial_balance * 100.0, 4),
            "costs_paid_usd": {"commission": round(float(fees), 2), "swap": round(float(swap), 2)},
            "lookahead_violations": lookahead_violations,
            # U1: how many actions each behaviour control blocked (all
            # zeros when the controls are off).
            "behaviour_blocks": dict(behaviour_blocks or {}),
            "valid": len(lookahead_violations) == 0,
            "equity_curve": equity_curve,
            "trades": trades,
        }
