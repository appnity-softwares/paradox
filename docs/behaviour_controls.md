# Behaviour Controls — Rule Schema and Enforcement (U1)

Update Blueprint v2.1, Package U1. Delivered 2026-11-24.

## 1. What the controls are

Behaviour controls govern HOW an account trades, distinct from the
compliance engine's risk-limit checks (HOW MUCH). They are deterministic
arithmetic on timestamps, lots and prices — no LLM anywhere.

| Control | FirmRuleSet field | Default (unknown rulebook) | Reason code |
| --- | --- | --- | --- |
| Minimum hold time | `min_hold_seconds` | None (off until rulebook states it) | `min_hold_time` |
| Orders per day | `max_orders_per_day` | None | `max_orders_per_day` |
| Modifications per trade | `max_modifications_per_trade` | None | `max_modifications_per_trade` |
| Requests per minute | `max_requests_per_minute` | None | `max_requests_per_minute` |
| Hedging | `hedging_policy` | `unknown` (does not enable block) | `hedging_forbidden` |
| Prohibited styles | `prohibited_styles` | martingale, grid, averaging_down ON | `prohibited_style_<style>` |
| EA / algorithmic use | `ea_policy` | `unknown` (blocks Auto) | `ea_policy_blocks_auto` |
| VPS / IP / copy trading | `vps_policy`, `ip_policy`, `copy_trading_policy` | `unknown` | recorded; enforcement lands in U5 |

## 2. Detection rules

- **Martingale** — a new entry on a symbol/direction whose lot is at
  least `martingale_escalation_factor` (default 1.5) times the lot of
  the most recent CLOSED LOSING same-symbol, same-direction trade,
  within `martingale_lookback_min` (default 240 minutes). Escalation
  after a WIN or with no prior closed trade is not flagged.
- **Grid** — the new entry would be the Nth (`stack_max_entries`,
  default 3) same-direction entry at strictly worsening prices within
  `stack_window_min` (default 240 minutes).
- **Averaging down** — a same-direction entry at a price worse than the
  first entry of the current same-direction stack (stack of >= 2).
- **Hedging** — any open opposite-direction position on the same symbol
  when `hedging_policy == "forbidden"`.

False-positive guards: equal-size re-entries after a loss are not
martingale; entries at improved prices are not grid/averaging; a win
before the escalation removes the flag; thresholds are configurable.

## 3. Where each control is enforced

- **Entries** — `ExecutionRouter._pipeline` step 3.5 (`behaviour`):
  request-rate, per-day cap, prohibited styles, hedging. Fail closed;
  new reason codes surface in the step trail and audit log.
- **Modifications** — `TradeManager.modify_sl`: a tightening stop is
  risk-reducing and counts as protective (bypasses min-hold, logged as
  an exception); the per-trade modification cap still applies.
  Loosening still requires human re-approval as before (12.3).
- **Partial closes** — `TradeManager.partial_close` enforces min-hold
  (not protective).
- **Auto mode** — `ModeGate.set_mode(..., ruleset=...)` refuses AUTO
  when the supplied ruleset's `ea_policy` does not permit it, and the
  router re-checks at order time (`ea_policy_blocks_auto`).
- **Requests per minute** — every `submit()` call is counted, including
  requests later rejected.

## 4. Backtest parity (core/tick_engine.py)

`BacktestConfig` gained optional fields (`min_hold_seconds`,
`max_orders_per_day`, `max_modifications_per_trade`,
`prohibited_styles`, `martingale_escalation_factor`,
`martingale_lookback_min`). All default OFF, so legacy configs and the
regenerated golden file are unchanged in behaviour.

- **min_hold_seconds** — blocks strategy signal exits of positions
  younger than the limit. SL hits, TP hits, stop-outs and end-of-data
  liquidation remain protective and always execute.
- **max_orders_per_day** — counts accepted entries per UTC day.
- **max_modifications_per_trade** — a discrete SL ratchet (a bar chunk
  in which the trailing/breakeven stop improves) counts as one
  modification; once the cap is reached the stop freezes and each
  further chunk counts as a blocked request.
- **prohibited styles** — martingale detection on the simulated
  sequence (lot escalation after a losing same-direction close).
  Grid/averaging cannot occur in this single-net-position engine; the
  shared detector covers them for the live path.

Every blocked action is counted in `result["behaviour_blocks"]` by
reason code. Golden file regenerated: only `config_hash` changed
(additive config fields); the trade log is bit-identical.

## 5. Known limitations (honest)

- The live/paper platform does not yet expose a full-close endpoint;
  when one is added it must call `BehaviourControls.check_close`
  (protective flag for kill-switch/rule-breach actions) before reaching
  the broker. Partial closes and SL modifications are enforced now.
- Grid/averaging-down detection is live-path only; the single-net-
  position tick engine cannot produce stacked entries.
- Inactivity enforcement (alerts on N idle days) lands with later
  packages; U1 delivers the schema and validation only.
