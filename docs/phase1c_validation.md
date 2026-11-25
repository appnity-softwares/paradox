# Phase 1C — Strategies, Optimization, Validation (Progress Report)

Status: **library and validation stack complete; honest FAIL verdicts on real
data**. Remaining: Bayesian search, sensitivity heatmaps, PDF export.

## What was built

### Strategy library (Part 6.5) — 15 strategies

Existing 10 kept unchanged (EMACross, MACDCross, RSIReversal, VWAPEMA,
Supertrend, SessionBreakout, BollingerReversion, FairValueGap, SMCOrderBlock,
LiquiditySweep). Added 5, each with objective rules, validated params, warmup:

| Strategy | Key | Category |
|---|---|---|
| PSARTrend | `psar_trend` | trend (Parabolic SAR, Wilder recursion) |
| StochasticCross | `stoch_cross` | momentum (%K/%D cross in extreme zones) |
| ZScoreReversion | `zscore_reversion` | mean reversion (z-score entry/exit bands) |
| ADXTrend | `adx_trend` | trend + regime filter (ADX gate on EMA direction) |
| BreakOfStructure | `smc_bos` | SMC (fractal swings, close-through breaks) |

- `docs/smc_definitions.md` written FIRST (per 6.5): formal, testable rules
  for FVG, Order Block, Liquidity Sweep and the new BOS. The document is the
  specification; `tests/test_smc_definitions.py` pins all four SMC
  implementations to it on hand-labelled bars.
- `strategies/catalog.py`: metadata + parameter schema with ranges + code
  hash (sha256 of the class source) for every strategy.
- New indicators in `core/indicators.py`: `adx` (Wilder), `parabolic_sar`.

### Legacy adapter

`strategies/adapter.py` runs any `generate_signals(df)` strategy on the
shared signal-only interface (tick engine included), with documented
divergences: bounded indicator window (1500 bars), SL/TP from
`StrategyConfig` anchored at the signal bar close, flips resolve one bar
later (conservative, never look-ahead).

### Validation stack (Part 6.6) — `core/validate.py`

- **Monte Carlo, trade resampling**: bootstrap trade P&Ls → p5/p50/p95
  return, probability of loss, p95 max drawdown (deterministic seed).
- **Monte Carlo, randomized start dates**: same fixed params from different
  start dates; reports median/worst/positive-ratio.
- **Cost stress**: 1x / 1.5x / 2x on spread + slippage + commission.
- **Deflated Sharpe** (Bailey & López de Prado): per-trade SR deflated by the
  actual number of grid trials; single-trial case reduces to the probabilistic
  Sharpe ratio. numpy/stdlib only (no scipy dependency).
- **Robustness score** (0–100, fixed documented weights) and **verdict**
  `FAIL` / `RESEARCH_ONLY` / `PAPER_TRADE_ELIGIBLE`. Gates for paper
  eligibility: walk-forward ROBUST **and** MC p5 > 0 **and** profitable at 2x
  costs **and** deflated Sharpe > 0 **and** ≥ 50 OOS trades.
- Every grid trial is recorded (`trials_*.csv`) — anti-selection-bias.

### Reports on real data (Part 6.8: ≥ 3 strategies)

`python -m scripts.validate_strategies XAUUSD 2025 1 6` — 33.2M real ticks →
11,652 × 15-min bars (H1-2025), dataset hash recorded. Five strategies
validated (`ema_cross`, `rsi_reversal`, `supertrend`, `adx_trend`,
`bollinger`); artifacts in `reports/`:

| Strategy | OOS Ret % | Profitable folds | MC p5 | Verdict |
|---|---|---|---|---|
| ema_cross | −46.4 | 25% | −82.8% | **FAIL** |
| rsi_reversal | −49.2 | 0% | −90.7% | **FAIL** |
| supertrend | −50.6 | 0% | −88.1% | **FAIL** |
| adx_trend | −40.0 | 25% | −78.8% | **FAIL** |
| bollinger | −56.4 | 25% | −90.4% | **FAIL** |

**Honest finding:** with the default fixed-SL 3.0 / RR 2.0 configuration,
every baseline strategy loses out-of-sample on H1-2025 XAUUSD (win rates
~30% against a 2:1 reward-to-risk target). Per the Master Prompt the FAIL
verdicts are NOT softened; finding parameter sets that pass is future work
for the optimization layer, not a reason to bend the gates.

### Random search with IS/OOS split (`scripts/random_search.py`)

Wider random ranges than the fixed grids (random search over a 3x3 grid is
just a shuffle), 70/30 IS/OOS split on the real bars, deterministic seeded
sampling, every trial recorded. 40 trials x 5 strategies: the best
in-sample combos collapse out-of-sample (e.g. adx_trend IS Sharpe +0.34 ->
OOS -4.70). One bollinger cell shows +5.9% OOS but with 35 trades and a
negative IS — noise, not a candidate. **No parameter set earns
RESEARCH_ONLY or better.** Artifacts: `reports/random_search_trials_202501.csv`,
`reports/random_search_oos_202501.csv`.

### Bayesian (TPE) search (`scripts/optuna_search.py`)

Optuna TPESampler with a fixed seed (deterministic, tested), the
``fast < slow`` constraint encoded in the search space itself so recorded
parameters are exactly the parameters run, invalid combos ranked last but
kept recorded, and sub-min-trade combos scored on a sliding penalty rather
than hidden. 60 trials x 5 strategies on the same 70/30 split: TPE finds
better in-sample combos than random search (adx_trend IS Sharpe +1.25),
and **every one of them collapses out-of-sample** (OOS Sharpe -5.76 to
-7.10). The only positive OOS cells remain tiny-sample noise (bollinger,
29 trades, negative IS). Artifacts: `reports/optuna_trials_<key>_202501.csv`,
`reports/optuna_oos_202501.csv`.

**Combined honest finding (260+ trials across both searches):** the
IS-to-OOS degradation is systematic across all five baseline strategies on
H1-2025 XAUUSD 15m with the fixed SL/RR configuration. This is a property
of the strategies and the period, not of the search method — and the
validation stack detects it every time, which is exactly its job.

### Config screen: SL/TP structure beats parameter grinding

`scripts/config_screen.py`: 90 cells (5 strategies x {fixed 2.0, ATR 1.5,
ATR 2.5} x {RR 1.5/2.0/3.0} x {all, london_ny} sessions), each scored with
the honest fixed-params walk-forward OOS on real bars. Result: **every
fixed-SL cell is catastrophic (-35% to -72%); ATR-based stops flip most
cells positive** (26/90 positive OOS). Best cell
`ema_cross|atr_1.5|rr_1.5|london_ny`: OOS +61.5%, Sharpe 3.01, 100%
profitable folds, max DD -12%.

### Verification: full stack + TRUE holdout (`scripts/verify_cells.py`)

A 90-cell screen is multiple comparison, so the top three cells went
through (1) the full validation stack and (2) a holdout the screen never
saw (H2-2025, 10,516 bars):

| Cell | Verdict (H1) | WF OOS | MC p5 | 2x costs | Holdout H2-2025 |
|---|---|---|---|---|---|
| ema_cross\|atr_1.5\|rr_1.5\|london_ny | **PAPER_TRADE_ELIGIBLE** (100) | +61.5% | +10.9% | +25.1% | +1.9%, Sharpe 0.14, **MIXED** |
| ema_cross\|atr_1.5\|rr_3.0\|all | RESEARCH_ONLY (73.8) | +16.6% | -33.0% | +26.9% | +9.2%, Sharpe 0.83, MIXED |
| ema_cross\|atr_2.5\|rr_2.0\|all | RESEARCH_ONLY (73.8) | +13.7% | -13.9% | +23.7% | +0.6%, Sharpe -1.36, **FAIL** |

**Honest reading:** the first `PAPER_TRADE_ELIGIBLE` cell is real under
the H1-2025 gates, but its edge decays sharply out-of-period (MIXED on
H2-2025). Two caveats are recorded: (a) the deflated Sharpe in the stack
is adjusted by the in-stack grid trial count, not the 90-cell screen —
the holdout check is the guard against that selection, and it fired;
(b) a single 6-month holdout is itself one sample. Treat the cell as a
research candidate for further regime/structural work, not deployment.
Artifacts: `reports/config_screen_202501.csv`,
`reports/verify_cells_202501.csv`.

### Sensitivity heatmap + regime breakdown (`core/validate.py`)

- `sensitivity_heatmap`: 2-D parameter grid, every cell a full backtest;
  a smooth surface indicates robustness, an isolated spike an overfit cell.
- `regime_breakdown`: tags every trade with the regime of its entry bar
  (ATR-percentile terciles x ADX trend/range) and reports per-regime
  win rate / P&L — shows WHERE a strategy makes or loses money.

### Multi-period holdout gate (`core/validate.rolling_holdout`)

A single holdout window is one sample; `rolling_holdout` evaluates the
same fixed params on N successive unseen windows. Applied to the
PAPER_TRADE_ELIGIBLE cell over all post-selection data (2025-07 to
2026-11, 4 windows of ~6.3k bars):

- ATR cell: `[−17.7%, −14.9%, +9.1%, +25.1%]` — **2/4 positive, median −2.9%**
- Fixed-SL baseline, identical windows: `[−48.4%, −68.8%, −60.9%, −60.1%]` — 0/4

**Honest reading:** the specific PAPER_TRADE_ELIGIBLE verdict does NOT
replicate out-of-period — the gate correctly refuses deployment. But the
structural finding generalizes: ATR-based stops beat the fixed-SL
template in every single unseen window. The research direction is
confirmed; the individual cell is not.

### Performance report, Part 6.7 (`scripts/performance_report.py`)

Full-run report on real bars for any catalog key (params overridable via
`key=value` args): headline metrics CSV, full trade log CSV, monthly
returns CSV, and a PDF chart (equity curve + drawdown, matplotlib Agg,
`MPLCONFIGDIR` pinned inside the workspace). Verified on EMA 20/50 over
H1-2025: -67.2% return, Sharpe -4.25, 544 trades — consistent with the
validation verdicts.

## Verification

100 tests pass. Added in this phase: `tests/test_smc_definitions.py` (11),
`tests/test_new_strategies.py` (11), `tests/test_validate.py` (16).
`requirements.txt` (pinned, 64 packages) added for the fresh-install
acceptance criterion.

## What remains open

1. Structural edge work on the ATR-stop template: trailing stops, regime
   gating (the breakdown report exists), session-specific logic — then
   re-screen and re-verify with the full protocol (screen -> full stack ->
   multi-period holdout).
2. Regime-aware strategy selection (the breakdown report exists; using it
   as a filter is future work).
3. Phase 1 acceptance checklist is otherwise complete: 2+ years real data
   hashed, determinism tests, tick/bar consistency, cheating-strategy
   detection, runtime budget, 15 strategies with tests and reports,
   WF/MC/cost-stress for 5 strategies, no synthetic data in the validated
   path, pinned requirements, all tests green.
