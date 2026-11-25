# Phase 6 — Dashboard / Platform Console (Progress Report)

Status: **complete per Prompt 6 and Part 14.1** — the console is built on
top of the Phase 3–5 engines, with honest data labels, the kill switch
always visible, and live trading controls disabled until the Phase 4
sign-off.

## What was built

### Platform state — `dashboard/platform_state.py`
Wires the new engines (DeterministicRiskEngine + KillSwitch,
ComplianceEngine + RuleStore, ExecutionRouter + PaperBroker, ModeGate,
Reconciler, FeedMonitor, AuditLog, Notifier) into one singleton,
independent of the legacy prototype state. Paper mode by default;
`record_signoff()` is the only way live controls unlock, and it is
audited.

### Platform API — `dashboard/routers/platform.py` (mounted at `/api/platform`)
- **Overview:** accounts (equity, P&L, open positions, mode), risk
  status, compliance status, alerts, kill-switch state, sign-off state.
- **Data sources with honest labels:** the parquet tick store is
  `DELAYED`, paper-broker quotes are `SYNTHETIC`, the unconnected MT5
  bridge is `STALE` — and nothing is ever labelled `LIVE` falsely
  (tested).
- **Kill switch:** GET state + POST engage/disengage with mandatory
  actor + reason (422 otherwise); every action audited; an engaged kill
  switch blocks new orders end-to-end (tested).
- **Compliance:** rule sets with the verification screen (each rule
  shown next to its cited clause); drafts show `grants_permission:
  false`; human-only `POST /compliance/verify`; monitor endpoint with
  remaining daily/max loss and target progress.
- **Trading:** order tickets flow through the FULL pipeline (mode gate →
  sizing → risk → compliance → order_check → send); duplicates
  suppressed; a 5%-risk ticket is REDUCED to the cap, not rejected
  (designed reduced-lot behaviour, tested); execution log viewer.
- **AI assistant:** chat restricted to read/propose tools — no
  execute-level tool exists in its registry (tested); proposals are
  inert data surfaced for human approval.
- **Admin:** agent run and cost logs, audit-log viewer, sign-off
  endpoints.

### Console UI — `dashboard/console.html` served at `/console`
Compact single-page terminal: overview cards, honest data-source labels,
kill switch with engage/disengage (prompting for actor + reason),
compliance verification screen, paper order ticket with the LIVE button
permanently disabled until sign-off, AI assistant, and the audit feed.

## Verification

338 tests pass (12 new in `tests/test_dashboard_platform.py`, each with
an isolated state directory)

## Post-phase additions (testing strategy + operations, Part 14.3/14.4)

- `tests/test_golden_backtest.py` — golden-file regression for the BAR
  engine on a fixed slice of REAL XAUUSD ticks (2025-01-02→08, 13
  trades locked in `tests/golden/bar_engine_golden.json`); regenerate
  only deliberately via REGEN_GOLDEN=1.
- `tests/test_properties.py` — 193 deterministic property cases:
  resampler OHLC/extremes/volume invariants, risk-money identity,
  lot-grid flooring, margin linearity, P&L symmetry and stop-loss
  equals risked money.
- `GET /api/platform/health` — composite health (broker, risk engine,
  reconciliation, data feed staleness) that reports degraded components
  instead of hiding them.
- Adaptation suggestions wired to audit + human approval
  (`/api/platform/adaptation/...`) — closing the Part 12 open item.
- `docs/operations.md` — health semantics, structured logging,
  runbooks (disconnect / mismatch / kill switch), Linux + Windows
  deployment guide.
- Suite now: 537 passed (16 dashboard-platform tests).: honest labels, sign-off lockout, kill
switch via API (engage → order blocked → disengage), draft-grants-
nothing on the verification screen, full pipeline tickets, duplicate
suppression, risk-cap reduction, execution log, assistant restrictions,
and admin cost logs.

## Security phase (Part 14.2, follow-up)

`dashboard/security.py` + `tests/test_security.py` (18 tests): session
auth (PBKDF2 hashing, bearer tokens, expiry/revocation), TOTP MFA
(stdlib RFC 6238), role hierarchy viewer < trader < admin enforced on
every endpoint (disengage/sign-off/verify are admin-only), login rate
limiting, and fail-closed lockout when `DSH_ADMIN_PASSWORD` is unset.
The console gained a login screen. Replay follow-up: 580 passed.

## What remains open

1. Next.js/TypeScript migration of the frontend (the console is
   functional vanilla JS; the API surface is the stable contract).
2. Authentication with MFA and role-based access (Part 14.2) before any
   multi-user deployment — single-user tool for now, decided explicitly.
3. Charts — DONE (follow-up): `dashboard/routers/charts.py` serves
   real-bar candlesticks (`/charts/bars`, labelled DELAYED) and a
   deterministic backtest payload (`/charts/backtest`) with equity and
   drawdown curves plus trade markers that align to bar times; the
   console renders SVG candlesticks with LONG/SHORT entry markers and
   the equity/drawdown overlay. 5 tests in `tests/test_charts.py`.
   Trade replay remains open.
4. Operational items carried from 4B/5: bridge deployment on a Windows
   VPS, paper soak period, real LLM provider backends, Telegram/email
   sinks.
