# U7 Campaign Summary — 2026-11-25

**Result: NO candidate passes the acceptance gates. The holdout was
never opened. This is a valid, honest campaign outcome — reported
plainly, per the blueprint.**

## Scope

- Data: 207,336,585 real XAUUSD ticks (2024-01-01 → 2026-11-22),
  dataset_hash prefix `03e63637f06597a8`. Everything below ran on the
  tick engine (recorded bid/ask, latency, slippage, commission, swap)
  with session-only loading, 5-minute bars.
- Split: training + walk-forward 2024-01-01 → 2026-05-21;
  **holdout 2026-05-22 → 2026-11-22 frozen and enforced in code**
  (`HoldoutGuard`; any overlapping search raises). The holdout remains
  closed because nothing earned the one-time opening.
- Derived risk per trade: **0.25%** (deterministic from the 2% target /
  3% hard cap drawdown limits — not hand-picked).
- U1 behaviour controls ON in every trial: min hold 5 min, max 6
  orders/day, martingale prohibited. Zero look-ahead violations in all
  runs.

## Screening (training data only)

- 540 random-search trials: 3 new session-specific candidates
  (`london_opening_range`, `trend_pullback`, `ema_cross_vol_filter` —
  rules in docs/strategy_rules_u7.md) × 3 session windows (London
  07–11, NY 12–16, overlap 12–15 UTC) × 60 trials each, every trial
  stored in reports/campaign_2026-11-25/trials.jsonl.
- Baseline: all 15 existing library strategies at DEFAULT parameters ×
  the same 3 sessions (45 runs, library_baseline.json).

### Baseline result (best of each strategy across sessions)

| Strategy | Best PF | Best-session max DD | Verdict |
|---|---|---|---|
| macd_cross | 1.01 | 25.6% | breakeven, far over DD cap |
| zscore_reversion | 0.95 | 12.0% | loses after costs |
| bollinger | 0.92 | 16.6% | loses after costs |
| ema_cross | 0.90 | 27.6% | loses |
| …the other 11 strategies | 0.52–0.89 | 17–60% | all lose |

No library strategy comes close to a 2–3% drawdown cap at any risk
level: the drawdowns above are at 0.25% risk already.

### New candidates — walk-forward (4 × ~6-month folds) + gates

Top-3 screening configs were re-run over rolling folds, with Monte
Carlo (2,000 trade-resample sims), 2x cost stress, and deflated Sharpe
(selection-corrected). All FAILED:

| Candidate | Params | WF max DD | PF @2x costs | Failed gates |
|---|---|---|---|---|
| ema_cross_vol_filter @ London | atr_pct=60 | 3.55% | 0.41 | 7 gates |
| ema_cross_vol_filter @ NY | atr_pct=50 | 4.86% | 1.64 | DD caps, streak, no-trade share |
| ema_cross_vol_filter @ NY | atr_pct=40 | 6.04% | 1.36 | DD caps, streak, no-trade share |

Deflated Sharpe probabilities ≈ 0.001–0.002: after accounting for 180
trials per strategy, the walk-forward edge is statistically
indistinguishable from noise.

## What this means (plainly)

1. **The client's stated target (high win rate, ≤2–3% drawdown, one
   ≤4h session) is not met by anything in the current candidate
   space** — neither the 15 library strategies nor the three new
   session-specific designs, on real tick data with honest costs.
2. The binding failure is DRAWDOWN, not win rate: even the
   best-behaved candidates swing 3.5–6% within single 6-month folds at
   0.25% risk. Scaling risk up to chase returns makes this worse.
3. The one variant with a positive 2x-cost expectancy (ema_cross_vol_
   filter @ NY, PF 1.36–1.64 after doubled costs) is RESEARCH_ONLY
   material at best: it still breaches the drawdown cap and fails the
   no-trade-day and streak-consequence gates. It is NOT approved for
   paper or live.
4. Holdout discipline held: because no candidate passed, the frozen
   6-month window was never opened (state/holdout_openings.jsonl is
   empty).

## Open items / next levers (research, not guarantees)

- U8 feed comparison may change cost assumptions (real firm spread can
  be better or worse than the recorded model; the 2x stress already
  bounds the upside).
- A wider search (more candidates, e.g. intraday ML filters) is
  possible but must respect the same frozen holdout; the deflated
  Sharpe penalty grows with every added trial.
- The economic-calendar vendor (U3) is still unselected, so news-window
  slippage stress is not yet applied; this can only make results worse,
  not better.
- PDF export of this report remains open from Phase 1C; CSV + JSON are
  in reports/campaign_2026-11-25/.

**Recommendation: do not deploy anything from this campaign. The
product's live-trading claim cannot be honestly made yet.**

---

# CORRECTION (2026-11-25, found during round 2)

The R-multiple computation used a `sl` key that does not exist in the
engine's trade records, so `initial_risk_usd` fell back to the FULL
entry price — R multiples came out ~200x too small. This corrupted:

- Monte Carlo DD estimates (mc95 reported ~0.04-0.05% — nonsense; the
  mc95_dd gate passed trivially),
- deflated-Sharpe inputs (skew/kurtosis of garbage R distributions),
- NOT the equity-curve metrics (PF, max DD, win rate) — the round-1
  verdict (every candidate fails the drawdown gates) is UNCHANGED.

Fix: the engine now records `initial_risk_usd` per trade (intended
risk at open, robust to breakeven/trail SL moves); run_trial uses it.
Golden files unaffected (the golden writer snapshots a fixed key
subset; config_hash unchanged). Round-2 numbers in this file use the
fixed computation.

---

# ROUND 2 (2026-11-25) — RESULT: ALL CANDIDATES FAIL, holdout stays FROZEN

Design: three NEW hypothesis classes (documented in
docs/strategy_rules_u7.md BEFORE coding) + exit management
(breakeven / trail / time-exit) searched jointly, targeting round-1's
drawdown failure. Same frozen holdout, same gates, same derived risk
(0.25%).

Trial accounting (honest):
- round 1: 540 trials + 45 library baselines
- round 2 run 1: 440 trials (metrics-correction bug found afterwards)
- round 2 run 2: 440 trials — IDENTICAL parameter draws (same seed),
  i.e. a metrics-correction replay, NOT new exploration
- cumulative UNIQUE configurations tested: 1002
  (the DSR selection count uses the conservative 1442 total)

## New candidates (screening, best of each class)

| candidate | trials | best PF | trades | max DD | win rate |
|---|---|---|---|---|---|
| asia_range_breakout | 80 | 0.78 | 21 | 1.84% | 38% |
| squeeze_breakout | 240 | 1.19 | 149 | 4.58% | 23% |
| first_move_continuation | 80 | 0.88 | 346 | 13.79% | 51% |

None of the new classes beat the round-1 ema_cross_vol_filter
screening results; none reached the walk-forward stage on merit.

## Walk-forward (top-4 by PF, all ema_cross_vol_filter with exit knobs)

Every candidate FAILED at least: oos_max_dd (4.1–6.0% vs 2.0% cap),
hard_cap_dd (vs 3.0%), mc95_dd (corrected: 11.7–15.2% vs 3.0%),
streak_consequence_dd, no_trade_day_share. Best PF after 2x costs:
1.64 (NY) — below what the DD gates require. Deflated Sharpe
probability ≈ 0.0005 (cumulative count 1442).

## Conclusion (unchanged and now better measured)

After 1002 unique configurations across 6 hypothesis classes and 2
rounds: NOTHING meets the client profile (2-3% max DD, high win rate,
intraday XAUUSD). The corrected Monte Carlo shows tail risk ~4-5x the
client's hard cap. The holdout remains untouched. Recommendation
stands: do not deploy anything from this research; the next round
should change the PROBLEM (different market/session/instrument or
explicit client-profile renegotiation), not add more trials.
