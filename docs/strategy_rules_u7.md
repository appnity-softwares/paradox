# U7 Campaign — Candidate Strategy Rules (written BEFORE coding)

Update Blueprint v2.1, Package U7 step 3: "Write each new strategy's
rules objectively in docs/ before coding. Prefer FEW parameters."

The 15 existing library strategies are reused as-is. This document
defines THREE new session-specific candidates. Every rule is objective
(no discretion), every parameter appears exactly once, and the default
values are the ONLY values used unless the search explicitly tunes a
listed dimension.

## Client profile (fixed for the campaign)

- Symbol: XAUUSD only.
- ONE session per deployment, at most 240 minutes. Candidates:
  London 07:00–11:00 UTC, New York 12:00–16:00 UTC, overlap
  12:00–15:00 UTC (exact hours are an optimization dimension chosen on
  TRAINING data only).
- Target max equity drawdown 2%, hard cap 3%.
- Good win rate is a goal, never a pass criterion by itself.
- Trades most days, never forced (no-trade days are allowed and
  reported).
- U1 behaviour limits enforced (min_hold_seconds, order caps), no
  martingale / grid / averaging_down.

## V1 — `london_opening_range` (opening-range breakout)

Session: London window. Rules:

1. Range = the highest high and lowest low of the FIRST
   `range_minutes` (default 15) minutes of the session, computed on
   1-minute bars built from ticks inside the session only.
2. Entry: buy-stop at range_high + `buffer` (default 0.10 USD) once,
   only if no position is open; sell-stop at range_low − buffer
   symmetrically. At most ONE entry attempt per direction per day
   (pending orders cancelled at session end — U2 flatten does this).
3. Stop: opposite side of the range ± `stop_buffer_atr × ATR(14, M1)`
   (default 0.25). Take profit at `rr × risk` (default rr = 1.5).
4. No re-entry after a stop-out on the same day.
5. Skip the day entirely if the range width > `max_range_atr` (default
   2.0 × ATR) — news-spike protection.

Parameters open to search: `range_minutes ∈ {10, 15, 30}`,
`rr ∈ {1.0, 1.5, 2.0}`. Everything else fixed.

## V2 — `trend_pullback` (pullback in trend)

Session: any of the three windows. Rules:

1. Trend filter: EMA(50, M5) vs EMA(200, M5) on bars inside the
   session; long only when fast EMA > slow EMA, short only when
   fast < slow.
2. Entry: price pulls back to the fast EMA (touch within
   `touch_bps` = 5 bps) and then prints a bar closing back in the
   trend direction. One entry per day per direction.
3. Stop: `atr_mult` (default 1.5) × ATR(14, M5) beyond the pullback
   low/high. Take profit at `rr` (default 1.5) × risk.
4. No trade if ATR(14, M5) is below its own 20th percentile over the
   trailing 5 sessions (dead market filter).

Parameters open to search: `atr_mult ∈ {1.0, 1.5, 2.0}`,
`rr ∈ {1.0, 1.5, 2.0}`.

## V3 — `ema_cross_vol_filter` (existing ema_cross + volatility gate)

Session: any window. Rules:

1. Base: the existing `ema_cross` strategy, unchanged signal logic.
2. Gate: trade only when ATR(14, M15) is above its trailing
   `atr_pct` (default 50th) percentile computed over the trailing
   10 sessions of the SAME session window (volatility filter).
3. All other parameters inherited from ema_cross defaults.

Parameters open to search: `atr_pct ∈ {40, 50, 60}`.

## Prohibited-style guard

All three candidates are single-entry, fixed-fractional, stop-based.
No martingale, no grid, no averaging down, no hedging — enforced again
at the gate by U1 behaviour controls and by the campaign's fingerprint
report (U5).

## Data split (frozen before any search)

- Dataset: 207,336,585 XAUUSD ticks, 2024-01-01 → 2026-11-22,
  dataset_hash prefix `03e63637f06597a8`.
- **Final holdout: 2026-05-22 → 2026-11-22 (last 6 months).** No
  optimization, selection, or validation step may touch it. Enforced
  in code by `core/research/holdout.py`; opening it requires an
  explicit one-time logged command.
- Training + walk-forward: 2024-01-01 → 2026-05-21.
