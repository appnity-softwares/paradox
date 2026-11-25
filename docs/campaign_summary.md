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
