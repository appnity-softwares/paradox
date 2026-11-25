# Session Profiles and the Execution Gate (U2)

Update Blueprint v2.1, Package U2. Delivered 2026-11-24.

## 1. SessionProfile (core/session/profile.py)

Fields: `name`, `start`/`end` ("HH:MM", session-local), `timezone`
(IANA), `days_of_week` (Mon=0, start-day semantics), `max_duration_minutes`
(hard cap 240 — larger values raise at construction), 
`no_new_entries_after_minutes_before_end`, `flatten_by` ("HH:MM",
defaults to the window end), `max_trades_in_session` (upper cap, never
a target).

Three gate functions: `in_window(ts_ms)`, `entries_allowed(ts_ms)`
(in window AND before the no-new-entries cutoff), `must_flatten(ts_ms)`.

### DST policy (fail-safe by construction)
- Spring-forward gap (local time does not exist): resolves to the
  instant the gap began — the session can only SHRINK.
- Fall-back overlap (local time occurs twice): first occurrence wins —
  the session can never EXTEND.
- Sessions crossing local midnight wrap to the next day; a tick after
  midnight resolves to the previous day's window.
- Tests cover London (2026-03-29, 2026-10-25) and New York
  (2026-03-08) transitions.

## 2. Enforcement (gate, not strategy)

- `ExecutionRouter` step `session` (after the mode gate): new entries
  outside the window, after the cutoff, or above
  `max_trades_in_session` are rejected with reason codes
  `outside_session_window`, `session_no_new_entries`,
  `session_max_trades`. Management of open trades is never gated.
- Fail closed: binding a profile to an account refuses entries until
  the scheduler opens a window.
- Defence in depth: `RiskConfig.session_profile` makes the risk engine
  reject entries outside the window too (`outside_session_window`).
- `flatten_by` is enforced by the scheduler (below), not by the gate.

## 3. SessionScheduler (core/session/scheduler.py)

Pre-session health checks (all must pass, else SKIPPED): account
snapshot present, feed fresh per symbol, kill switch off, ruleset
verified, calendar loaded when the news restriction is enabled (U3
will populate the calendar; until then this fails closed). Every
skip writes a `session_runs` record and fires a Notifier alert.

At flatten: `router.protective_close_all(account_id, ts_ms)` closes
remaining positions as a PROTECTIVE action — bypasses entry gates and
behaviour min-hold, audited as `protective_close`, Notifier alert.

Run ledger: `SessionRunStore` (JSONL at `state/session_runs.jsonl`) —
date, deployment_id, account_id, profile, status
(skipped/completed/aborted), reason, trades, pnl, drawdown.

After the window: strategy evaluation is OFF; the deterministic
safety layer (reconciler, kill switch, feed monitor, compliance watch)
keeps running exactly as before — it never depended on the session.

## 4. Zero-LLM guarantee

The scheduler, the gate and the flatten path never import or call the
LLM layer. Test `test_zero_llm_calls_in_session_loop` patches
`core.llm.get_llm` to raise and drives a full session (open → entry →
flatten → close) — it passes only because no LLM call happens. LLM use
remains limited to explicit user-requested actions and optional
pre/post-session reports under the daily budget cap.

## 5. Backtest parity (core/tick_engine.py)

`BacktestConfig` gained `session_start_min`/`session_end_min` (UTC
minutes-of-day), `session_days` (Mon=0, UTC), 
`no_new_entries_before_end_min`, `flatten_at_session_end` (default
True). All default OFF.

- The tick loader keeps ONLY ticks inside the window (faster, less
  memory) — `ticks_replayed` reflects the filtered count.
- Entry cutoff semantics match live: a decision at or after
  `end − cutoff` is blocked and counted in
  `behaviour_blocks["session_no_new_entries"]`.
- A position still open when a new window starts flattens protectively
  at the new window's first tick (`exit_reason: session_flatten`);
  no ticks exist in the gap, so no fills are missed.

**Documented difference:** the backtest uses UTC minutes-of-day
directly; DST resolution is a live-path concern handled by
SessionProfile. Around DST shifts, live windows shift by an hour
while a backtest with fixed UTC minutes does not — run DST-spanning
evaluations with explicit UTC minutes.

## 6. API and dashboard

- `POST /api/platform/session-profiles` (trader) — create/update;
  validation errors answer 422 (e.g. `max_duration_minutes=241`).
- `GET /api/platform/session-profiles` (viewer).
- `POST /api/platform/accounts/{id}/session-profile` (trader) — bind
  (`{"profile": name}`) or unbind (`{"profile": null}`).
- `GET /api/platform/accounts/{id}/session-status` (viewer) —
  unbound/waiting/active/closed + window details + last run.
- `GET /api/platform/deployments/{id}/sessions` (viewer) — run ledger
  for a deployment (deployment id = account binding id).
- Console: a Sessions card shows the bound profile, live status and
  the last five runs.

## 7. Known limitations (honest)

- The live scheduler loop is driven by external calls
  (`run_session`/`close_session`); a cron/bridge trigger wiring it to
  wall-clock time arrives with the deployment packages (U8/U9).
- Pending orders are not cancelled at flatten; the protective close
  removes open positions, and a pending that survives to the next
  window may fill there. Cancellation-at-flatten is a candidate for
  the next revision.
- `must_flatten` returns True between sessions (any carried position
  must be flattened immediately) — deliberate fail-closed semantics.
