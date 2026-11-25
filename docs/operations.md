# Operations — Health, Runbooks, Deployment (Part 14.4)

## Health endpoints

`GET /api/platform/health` on the dashboard app reports a composite
status with per-component detail. It never hides a dead component:

- **broker** — `up` when connected (paper broker always reports its
  state; the MT5 bridge reports its terminal connection).
- **risk_engine** — `up`, plus the global kill-switch state.
- **reconciliation** — `paused` lists accounts with entries paused
  pending human resolution; `up` otherwise.
- **data_feed** — `stale` lists symbols whose last tick is older than
  the freshness window (`FeedMonitor.max_staleness_ms`, default 10 s),
  measured side-effect free against the tick clock.

Overall: `healthy` only when every component is `up` (or `paused`, which
is a deliberate safe state, not a fault).

## Structured logging

- The execution path logs to the JSONL **audit log**
  (`state/dashboard/audit.jsonl`): every `submit`, `execution_result`,
  kill-switch action, rule verification, sign-off, and adaptation
  decision, each with actor and reason where applicable.
- Agent activity logs to the Phase 5 **run log** (JSONL): agent, task,
  status, model, tokens, cost, duration, errors — surfaced at
  `/api/platform/admin/agent-runs`.
- Correlation id: the order `idempotency_key` threads through intent →
  ApprovedOrder → broker record → audit entries; use it to trace one
  order across all logs.

## Runbooks

### Runbook 1 — Broker disconnect
1. `FeedMonitor.check` fails → entries paused for affected accounts,
   alert `broker_disconnect` fired (visible on the console overview).
2. Operator verifies the terminal/bridge on the Windows host
   (`/health` on the bridge service).
3. After recovery, `FeedMonitor` fires `feed_recovered`; entries stay
   paused until the operator explicitly resumes.

### Runbook 2 — Reconciliation mismatch
1. `Reconciler.reconcile()` finds a fill missing at the broker or an
   unknown broker position → the account is paused automatically.
2. Operator inspects the audit log by idempotency key, resolves the
   state (filled / pending_unknown), corrects at the broker if needed.
3. `POST`-style resume with actor + reason re-enables entries. Never
   resume with unresolved positions.

### Runbook 3 — Kill switch
1. Engage via the console (always visible) or
   `POST /api/platform/kill-switch` with scope, actor, reason.
2. All new orders fail closed with reason `kill_switch_engaged`; open
   positions keep their protective SL/TP via TradeManager.
3. Disengage is human-only, requires actor + reason, and is audited.
   A corrupt kill-switch state file blocks trading (fail closed).

## Deployment guide

### Linux — API and workers
1. Python 3.14 venv; `pip install -r prototype/requirements.txt`.
2. Run the dashboard: `uvicorn dashboard.app:app --host 127.0.0.1
   --port 8123` (bind public only behind TLS + auth).
3. State lives under `prototype/state/` (audit, rule store, kill
   switch, sign-off). Back it up; the kill-switch file must be restored
   intact (corruption = blocked, by design).
4. Datasets are immutable Parquet partitions — back up separately; a
   restore is validated by `dataset_hash` matching.

### Windows — MT5 bridge
1. Install MetaTrader 5 terminal; log in to the DEMO account first.
2. Set credentials as environment variables only:
   `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`. Never in files, logs or
   prompts.
3. Run the bridge with a generated bearer token:
   `python -m apps.mt5_bridge.server --port 8000 --token <secret>` and
   TLS certfile/keyfile — the server refuses non-loopback binds without
   TLS.
4. On the Linux side, set `DSH_MT5_BRIDGE_TOKEN` and point the adapter
   at the HTTPS URL; the client refuses plain HTTP off-loopback.
5. Verify: `/health` on the bridge, then one manual-mode paper-sized
   order through the full pipeline, confirmed in the audit log.

## Security (Part 14.2)

Implemented in `dashboard/security.py`, all stdlib, fail-closed:

- **Authentication:** PBKDF2-HMAC-SHA256 password hashing (200k
  iterations, per-hash salt); bearer-token sessions (32-byte random,
  8 h expiry, revocable). The operator password comes from
  `DSH_ADMIN_PASSWORD` only and is discarded after hashing — never in
  code, logs, tests or the frontend.
- **MFA:** TOTP (RFC 6238) implemented with stdlib hmac/base32; enable
  by setting `DSH_TOTP_SECRET`. Login rejects a missing or wrong code.
- **Roles:** viewer < trader < admin. Reads need viewer; order tickets
  and kill-switch ENGAGE need trader; kill-switch DISENGAGE, rule-set
  verification, sign-off and admin logs need admin — removing a safety
  block always requires more privilege than applying one.
- **Fail closed:** without `DSH_ADMIN_PASSWORD` every mutating endpoint
  returns 403 `auth_not_configured`; only `/health` and `/auth/*` stay
  public.
- **Rate limiting:** login is limited to 5 attempts/60 s per client,
  correct credentials included — brute force fails closed.
- **Decision recorded:** single-user tool. Multi-tenancy would need
  per-user credential vaults and data isolation before any accounts
  exist.

Tests: `tests/test_security.py` (18) — hashing, TOTP vectors and
windows, login flow, expiry, revocation, unconfigured lockout, every
role boundary, rate limiting, credential hygiene.

## Notification sinks

`core/execution/notify_sinks.py` extends the in-app notifier with
outbound alerting:

- **WebhookSink** — POSTs every notification as JSON to
  `DSH_WEBHOOK_URL`, with `DSH_WEBHOOK_TOKEN` as a bearer header.
  Tested against a real local HTTP server (payload shape + auth header).
- **SmtpSink** — formats and sends email via `DSH_SMTP_HOST/PORT/USER/
  PASS/TO`; message formatting is tested, delivery requires a real
  relay and stays disabled without configuration.
- **Fail-safe:** a sink that cannot deliver logs to stderr and never
  raises — alerting must never break the trading path (tested with an
  unroutable URL).
- `attach_configured_sinks(notifier)` wires whatever the environment
  configures; nothing attaches (and nothing leaks) without it.

Telegram/Slack are thin wrappers over WebhookSink once a real token
exists — noted as deployment-time work.

## Replay soak (Part 14.4 readiness)

`scripts/soak_replay.py` fast-forwards the EXACT live pipeline — tick
feed -> PaperBroker fills + TradeManager management -> 15m resample ->
causal EMA-cross signals -> ExecutionRouter (mode gate, sizing, risk,
compliance) — over REAL historical XAUUSD ticks. Every order goes
through the same single order path the live router enforces; every
tick drives the broker and trade management exactly as the live feed
would. It compresses weeks of paper-soak wall-clock time into seconds
to shake out state bugs before real paper trading starts.

- 30 days = 5.8M real ticks in ~5 s; deterministic (two runs produce
  identical summaries — tested).
- Built-in integrity invariants: no non-positive equity, no fills
  without a fill price, no duplicate idempotency keys, every key
  resolvable, no open position without an SL. Any violation fails the
  run loudly.
- Observed on 2025-01: the risk engine correctly rejected 211 of 362
  entries with `spread_too_wide` during real spread blowouts — the
  safety systems engage on real data, which is exactly what the soak
  is for.
- Honesty: the summary is labelled `REPLAY_SOAK` — "real historical
  ticks replayed through the live pipeline — not live trading, not a
  forecast". A replay passing does NOT predict live results; the
  wall-clock paper soak remains required before any real funds.
- Tests: `tests/test_soak_replay.py` (5).
- **Sweep:** `scripts/soak_sweep.py` replays EVERY ingested month and
  aggregates. Result over all 35 months (2024-01 → 2026-11):
  211,430,119 real ticks through the live pipeline in ~166 s,
  17,565 orders, **zero integrity problems in every month**
  (`reports/soak_sweep.json`). The safety systems engaged on real
  data throughout: 13,056 `spread_too_wide`, 1,090 `risk_cap`,
  675 `max_loss_worst_case`, and 7 `daily_loss_limit` blocks — the
  last being the rule the snapshot-freeze bug had silenced. Tests:
  `tests/test_soak_sweep.py` (4).
- Bug the soak exposed and fixed: the dashboard (and the soak itself)
  never refreshed the account snapshot, so equity, spread and —
  critically — `daily_realized_pnl` stayed frozen at registration
  values and the compliance daily-loss rule could never fire.
  `PlatformState._refresh_snapshot` now rebuilds the snapshot from live
  broker state before every submit (daily window per the 17:00 UTC
  reset, per-account isolation, high-watermark tracking), and closed
  trade records carry `account_id`. Pinned by
  `tests/test_daily_pnl_wiring.py` (4): losses block trading, the
  reset window excludes pre-reset losses, other accounts' losses do
  not leak, equity/HWM track live state.
- Paper-mode feed honesty: the dashboard's synthetic quote is now
  refreshed on the real clock at every submit (labelled SYNTHETIC in
  the UI) instead of one frozen 1970 tick — the feed-staleness check
  was passing vacuously before. `live_feed_connected=False` until a
  real bridge feed is wired; the synthetic top-up stops then.

## Backups and restore (Part 14.2)

`scripts/backup_state.py` — versioned zip of the state directory
(audit log, rule store, kill switch, sign-off) with a SHA-256 manifest
and the dataset hash recorded at backup time:

    python -m scripts.backup_state backup  [--state-dir DIR] [--out FILE]
    python -m scripts.backup_state verify  BACKUP.zip
    python -m scripts.backup_state restore BACKUP.zip [--state-dir DIR]

- **Verify** checks every file against the manifest without touching
  state; tampered or missing files are reported.
- **Restore fails closed:** any manifest mismatch aborts BEFORE the
  live state is touched; a changed dataset hash also aborts — state is
  never restored onto different data.
- Tested in `tests/test_backup_restore.py` (5): round trip, tamper
  detection, missing file, fail-closed restore, dataset-hash guard.

## CI (Part 14.3)

`.github/workflows/ci.yml` installs the pinned requirements and runs
the full suite on every push/PR. Tests that need the real 207M-tick
dataset carry the `requires_dataset` marker (`tests/conftest.py`) and
are SKIPPED — visibly, never silently passed — when the store is
absent; the golden backtest and full-dataset validation stay on the
workstation. Dependency pinning was already complete (every entry in
`requirements.txt` is `==`-pinned).

## U0 hardening (Update Blueprint v2.1, 2026-11-24)

- **Legacy trade routes removed.** `POST /api/trade/order|close|
  close_all|reset_kill_switch|modify` bypassed ExecutionRouter and had
  no auth. They now answer **410 Gone**; orders go through
  `POST /api/platform/orders` (risk + compliance + mode gate +
  ApprovedOrder) and kill-switch actions through
  `POST /api/platform/kill-switch` (disengage is admin-only).
- **Route-enumeration guard** (`tests/test_route_auth.py`, 4 tests):
  every write route must carry an auth dependency or be in a reviewed
  allowlist; no unauthenticated route may reference a broker-mutating
  method (source-inspected); the removed paths must stay gone.
- **AuditLog `:memory:` fixed** — it used to materialise a real,
  ever-growing file named `:memory:` (8.9 MB deleted). Now a true
  in-memory list. `AuditLog.__init__` core/execution/router.py.
- **Hash-chained audit log.** File-mode entries carry
  `prev_hash`/`hash` (sha256 over prev_hash + canonical JSON).
  `verify_chain()` returns the first broken index or None. Legacy
  unchained segments are preserved and skipped; the chain starts at
  genesis on the first chained write. Tests:
  `tests/test_audit_chain.py` (6).
- **Hygiene:** pandas `dayofweek` → `day_of_week` (core/data.py);
  uvicorn server shutdown in tests uses `should_exit` (no un-awaited
  coroutine). Warnings: 22 → 1 (the remaining one is FastAPI's own
  httpx deprecation).
- **Python version support (PyPI-verified 2026-11-24):** dev/CI run
  3.14 (+3.13 in the CI matrix); the Windows MT5 bridge must run
  **3.13 or older** — MetaTrader5 5.0.6231 publishes wheels only for
  CPython 3.8–3.13, no 3.14 wheel yet. README + CI updated.
- Suite after U0: **624 passed** (10 new tests).

## Open questions (U0 — need human answers)

1. **Bridge VPS Python pin:** confirm the Windows VPS image can run
   Python 3.13 (3.14 has no MetaTrader5 wheel). If the image is stuck
   on an older Python, tell us which and we will pin CI to match.
2. **Legacy `index.html` Trade Desk:** its order buttons now receive
   410. Options: (a) repoint the buttons at `/api/platform/orders`
   (needs an account registered + verified ruleset), or (b) retire
   `index.html` in favour of `console.html`. Decision needed before U1.
3. **`project_code_only.zip`** (959 KB) sits in the project root,
   outside `prototype/`. Confirm it is a deliberate export and not
   needed inside the repo.

## Known limitations (honest)

- The console UI is functional vanilla JS; the Next.js/TypeScript
  migration is future work.
- Charts (candles, markers, replay) are future UI work.
- Bridge pending/limit/stop orders and the cTrader adapter are open.
