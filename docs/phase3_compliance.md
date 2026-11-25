# Phase 3 — Deterministic Risk Engine and Prop Firm Compliance (Progress Report)

Status: **complete per Prompt 3 (Parts 8 and 10)**. Every check is plain
deterministic code; agents can request risk checks but cannot bypass them.

## What was built

### Risk Engine (Part 8) — `core/risk/`

- `RiskConfig` (versioned, frozen): per-trade risk cap, daily loss,
  max drawdown, open positions, total exposure, margin usage, leverage,
  correlated-exposure buckets, spread/stale-price filters, trading hours,
  broker stops/freeze levels, safety buffer.
- `DeterministicRiskEngine.evaluate(proposal, snapshot)`: runs ALL checks
  and reports every failed reason code (not fail-fast). Fail-closed on
  missing/stale/unparsable inputs: unknown symbol, missing SL (mandatory),
  NaN equity, missing market data → rejection. When the ONLY blockers are
  the risk cap or per-order lot cap, it offers a REDUCED lot that fits;
  every other failure is a hard stop.
- `KillSwitch`: global + per-account, persisted to disk (survives
  restart), corrupt state file → fail closed (blocked). Only an explicit
  human action with actor + reason disengages; every action is
  audit-logged.
- Compliance auto-protection: the engine accepts a compliance status and
  blocks new orders at `critical` or `breached` (Part 10.3 x Part 8).
- Legacy `core/risk_engine.py` fixed: removed the hardcoded 2400.0 gold
  price (now price/spec-aware) and the silent 5.0 SL fallback (now fails
  closed).

### Firm rule schema (Part 10.1) — `core/compliance/rules.py`

`FirmRuleSet` records the EXACT definition each firm uses: daily-loss
basis (balance / equity / balance_or_equity), reset time + IANA timezone,
floating inclusion, max-loss type (static / trailing) and basis, news
windows, weekend/overnight holding, min trading days with a volume
definition, consistency best-day share, allowed symbols, automation
policy. Every loss rule carries source clause citations. `validate()`
catches unknown bases and unverified sets missing citations.

### Rule lifecycle (Part 10.2) — `core/compliance/store.py`

`RuleStore`: add_draft → human verify (requires citations + clean
validation) → enforce. Versioned with field-level diffs;
`needs_reverification` flags accounts when a firm publishes a newer
version. **Draft rules grant no trading permission** — the compliance
engine returns `rules_not_verified` for them.

### Compliance Engine (Part 10.3) — `core/compliance/engine.py`

- `pre_trade_check`: simulates the account AFTER the proposed trade
  including the SL worst case, on the firm's exact basis, with a
  configurable safety buffer (default 80%). Checks: allowed symbols,
  trading hours, max lot/positions/exposure, news windows (calendar
  failure = block, not allow), weekend cutoff, daily-loss and max-loss
  worst cases, consistency best-day share.
- `monitor`: on every equity update computes remaining daily loss,
  remaining max loss, drawdown, profit-target progress, trading days,
  and status `ok | warning | critical | breached` (critical = auto-
  protection zone that blocks new orders via the risk engine).

### Challenge tracker (Part 10.3) — `core/compliance/challenge.py`

`simulate_challenge`: deterministic replay of an equity curve + trade
list against a VERIFIED rule set. Firm-day buckets follow the firm's
reset time/timezone; daily loss uses the intraday equity dip (floating
included per the firm's definition) or realized-only P&L otherwise;
trading days count only when closed volume meets the firm's definition.
Reports pass/fail, drawdown path, days used, rule hits. States clearly
that a simulation is not a guarantee.

## Real-data demo — `scripts/challenge_demo.py`

The validated ema_cross ATR cell replayed against a verified FTMO-style
rule set (100k, 8% target, 5% daily / 10% max loss, 4 min days):

| Risk/trade | Return | Max DD | Verdict |
|---|---|---|---|
| 1.0% | +72.1% | 12.4% | **FAILED** — daily loss $5,158.78 ≥ $5,000 on 2025-04-01 |
| 0.5% | +33.8% | 6.0% | **PASSED** — target hit, no breach, 126 trading days |

The honest headline: the higher-return configuration fails the challenge.
Compliance-aware sizing converts a failing config into a passing one —
the core value of the product, demonstrated on real ticks.

## Verification

252 tests pass. Added in this phase:
- `tests/test_risk_gate.py` (31): every check, fail-closed inputs,
  reduced-lot offers, exact-at-limit edges, overnight hour windows, kill
  switch persistence/audit/fail-closed, compliance gate.
- `tests/test_compliance.py` (57): schema + lifecycle (draft can never
  enable trading — tested), pre-trade scenarios (news window and calendar
  failure, weekend cutoff, worst-case projections on both bases, floating
  inclusion/exclusion, trailing HWM moves), monitor levels, and 12
  simulator scenarios (breach edges, reset-time day splitting, volume-day
  counting, weekend entries).

## What remains open

1. The audit-log sink for adaptation suggestions and kill-switch actions
   lands with the Phase 4 order flow (Part 12.1); the log formats are
   already defined.
2. Real economic-calendar feed integration (the engine blocks when the
   calendar is unavailable, so the feed is an enhancement, not a
   dependency).
3. Phase 4 next: order flow (`OrderIntent` → `ApprovedOrder`), paper
   broker, reconciliation, notifications (Parts 9, 12.1–12.3).
