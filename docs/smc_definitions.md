# SMC Definitions — Formal Rules (Master Prompt 6.5)

Every Smart Money Concepts strategy in `strategies/library.py` implements
exactly the rules written here. Each definition is objective and testable on
hand-labelled bars; `tests/test_smc_definitions.py` pins the implementations
to these rules. All prices refer to 1-minute OHLC bars of the traded symbol;
`ATR` is the 14-bar ATR (Wilder, `alpha=1/14`) of the same bars.

Notation: bar `i` has `O[i], H[i], L[i], C[i]`. A "signal" is the position
state the strategy holds after bar `i` closes (+1 long, −1 short, 0 flat).
The engine acts on the signal at the first ticks after the bar closes.

---

## 1. Fair Value Gap (FVG / imbalance) — `smc_fvg`

**Detection** (a gap is registered on bar `i`, using bars `i-2, i-1, i`):

- Bullish FVG: `L[i] > H[i-2] + min_gap_atr * ATR[i]` **and** bar `i-1` is an
  up candle (`C[i-1] > O[i-1]`). Gap zone = `[H[i-2], L[i]]`, direction +1.
- Bearish FVG: `H[i] < L[i-2] - min_gap_atr * ATR[i]` **and** bar `i-1` is a
  down candle (`C[i-1] < O[i-1]`). Gap zone = `[H[i], L[i-2]]`, direction −1.

**Mitigation entry** on a later bar `j` (`j > gap bar`, age `j - bar <= max_age`):

- Bullish gap fires +1 when the bar **overlaps** the zone
  (`L[j] <= top` and `H[j] >= bottom`) — price traded back into the imbalance.
- The gap is **invalidated** (removed, no signal) when `C[j] < bottom`
  (bullish) or `C[j] > top` (bearish): the imbalance failed.
- A mitigated gap is consumed (removed from the active list).
- At most the 15 most recent gaps are tracked; `signal[j] = direction` of the
  last fired gap and the state persists until the next event.

Parameters: `min_gap_atr` (gap size in ATR, default 0.4), `max_age` (bars,
default 25). Invalid inputs (`min_gap_atr <= 0`, `max_age < 3`) raise.

---

## 2. Order Block (OB) — `smc_order_block`

**Swing reference** on bar `i`: `prev_swing_high = max(H[i-swing_len-3 : i-2])`,
`prev_swing_low = min(L[i-swing_len-3 : i-2])` — the extreme of the
`swing_len` bars ending 2 bars before the impulse.

**Block detection** (impulse on bar `i`):

- Bullish expansion: `C[i] > prev_swing_high` **and**
  `C[i] - L[i-2] > impulse_mult * ATR[i]`. The order block is the **last
  down candle** among bars `i-1, i-2, i-3` (first found scanning backwards);
  zone = `[L[ob], max(O[ob], C[ob])]`, direction +1.
- Bearish expansion: `C[i] < prev_swing_low` **and**
  `H[i-2] - C[i] > impulse_mult * ATR[i]`. The order block is the last **up
  candle** among bars `i-1..i-3`; zone = `[min(O[ob], C[ob]), H[ob]]`,
  direction −1.

**Retest entry** on bar `j` (`j > block bar`, age `<= max_bars`):

- Bullish block fires +1 when price overlaps the zone
  (`L[j] <= zone_high` and `H[j] >= zone_low`); invalidated when `C[j] < zone_low`.
- Bearish block fires −1 symmetrically; invalidated when `C[j] > zone_high`.
- At most the 10 most recent blocks are tracked; the fired direction becomes
  the persistent state.

Parameters: `swing_len` (default 5), `impulse_mult` (default 1.5),
`max_bars` (default 30). Invalid inputs raise.

---

## 3. Liquidity Sweep (turtle soup) — `smc_liquidity_sweep`

**Reference liquidity** on bar `i`: `prior_high = max(H[i-lookback : i])`,
`prior_low = min(L[i-lookback : i])` — the extremes of the previous
`lookback` bars, excluding the current bar.

**Sweep detection** on bar `i`:

- Bearish sweep (buy-side liquidity taken, then rejected): fires −1 when
  `H[i] >= prior_high + sweep_buffer` **and** `C[i] < prior_high` — price
  penetrated the old high and closed back below it.
- Bullish sweep (sell-side liquidity taken, then rejected): fires +1 when
  `L[i] <= prior_low - sweep_buffer` **and** `C[i] > prior_low`.

**Hold**: the fired state persists for `hold_bars` bars after the sweep bar,
then returns to 0 (a fresh sweep resets the hold counter). The signal is
therefore a short-lived state, not a permanent flip.

Parameters: `lookback` (default 20), `sweep_buffer` (price units, default
0.2), `hold_bars` (default 4). Invalid inputs raise.

---

## 4. Break of Structure (BOS) — `smc_bos` (new)

**Swing points** (fractal definition, confirmed with lag `swing_len`):

- A swing high at bar `s` exists when `H[s]` is the maximum of
  `H[s-swing_len : s+swing_len+1]` and is confirmed only at bar
  `s + swing_len` (the future bars of the window are all in the past by then
  — no look-ahead).
- A swing low at bar `s` exists when `L[s]` is the minimum of the same window.

**Break detection** on bar `i` (the most recent **confirmed** swing levels):

- Bullish BOS: `C[i] > last confirmed swing high level` → the close broke
  market structure upward. Fires +1 once per swing level (a level can be
  broken only once; the level is consumed).
- Bearish BOS: `C[i] < last confirmed swing low level` → fires −1, consumed.

**State**: the fired direction persists until the opposite BOS fires. A
break of the *same* side's newer level does not re-signal while the state
already matches it.

Parameters: `swing_len` (default 5), `max_levels` (default 20 tracked
levels). Invalid inputs (`swing_len < 2`) raise.

---

## Testability contract

`tests/test_smc_definitions.py` builds hand-labelled bar sequences for each
rule above (e.g. a synthetic 3-bar gap, a stop-hunt wick, a confirmed swing
break) and asserts the emitted signal series equals the labels. If an
implementation and this document ever disagree, the document is the
specification and the code is wrong.
