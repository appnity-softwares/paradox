# Phase 4A — Order Flow, Paper Broker, Trade Management (Progress Report)

Status: **complete per Prompt 4A (Parts 9, 12.1–12.3, 12.5, 12.6)**. The
broker gateway accepts only an `ApprovedOrder`, which only the execution
router can mint after every check passed — skipping a step is impossible
by construction.

## What was built

### Strategy versions (Part 9) — `core/execution/versions.py`
`make_version(cls, params)` hashes the strategy class source and the
sorted parameters into an immutable `StrategyVersion`. Every
`OrderIntent` carries the version that produced it; any code or param
change is a new version that must be re-validated before deployment.

### Order flow (Part 12.1) — `core/execution/{intent,router}.py`
Single path, each step recorded in an auditable trail:
`intent → mode gate → deterministic sizing → Risk Engine → Compliance
Engine → ApprovedOrder minted → broker order_check → send → fill → audit
+ notification`. `ApprovedOrder` is frozen and minted only inside the
router; the paper broker raises `TypeError` on anything else (intents,
dicts, fakes). `AuditLog` appends every decision as JSONL.

### Execution safety (Part 12.2) — router + `core/execution/monitoring.py`
- **Idempotency:** retries with the same key return the recorded outcome
  (`duplicate`) and never reach the pipeline again; the broker also
  rejects a duplicate key at its own layer.
- **Timeout reconciliation:** a send timeout is reconciled against the
  broker by idempotency key BEFORE any retry. The worst case (fill lands,
  then the connection dies) reconciles to `filled` with exactly one
  position; an unknown outcome reports `pending_unknown`, never a blind
  retry.
- **Reconciliation loop:** compares platform-tracked orders with broker
  history; any mismatch alerts and PAUSES new entries for the account
  until a human resumes (audited).
- **Feed monitoring:** a stale tick feed pauses new entries and alerts;
  open positions keep their broker-side SL/TP protection.

### Paper broker — `core/execution/paper_broker.py`
Same interface shape a real `BrokerAdapter` will have
(`order_check/place/modify/close/positions/history/health`), fills at
live bid/ask plus the cost model's slippage, per-tick SL/TP checks on the
correct sides, and injectable faults for tests (timeout-after-fill,
connection loss).

### Trade management (Part 12.3) — `core/execution/management.py`
`ManagementRules` come from the strategy version: breakeven (entry +
buffer at an R threshold, never loosening an existing tighter stop),
ATR-multiple trailing stop (ratchets only), time-based exit, session-end
flatten, partial close (books realized P&L, keeps the remainder). Every
SL modification is re-checked: tightening is always allowed; loosening
the worst-case loss requires a human approval token and is audit-logged.

### Modes (Part 12.5) — `core/execution/modes.py`
Research/Backtest place no orders. Manual/Semi-auto require a per-order
human approval token. Auto requires ALL of: verdict at
`PAPER_TRADE_ELIGIBLE` or better, ≥ 14 paper days, zero unresolved
errors, a verified rule set, and an explicit user confirmation — tested
that each missing piece refuses.

### Notifications (Part 12.6) — `core/execution/router.py::Notifier`
In-app sink plus attachable extra sinks; events: trade opened/closed,
execution failure, risk limit, compliance block, broker disconnect,
reconciliation mismatch.

## Verification

286 tests pass (32 new in this phase: 14 in `tests/test_execution_gate.py`,
18 in `tests/test_trade_management.py`). Phase 4 acceptance criteria
covered so far:
- Duplicate-order test ✓ (forced retries → one position, `duplicate` status)
- Disconnect test ✓ (timeout-after-fill reconciles to a known state)
- No code path reaches the broker without Risk + Compliance approval ✓
  (malicious test: fake objects, kill-switch-engaged router, unverified
  rules, missing SL — all blocked before `place`)
- Kill switch via API survives restart ✓ (Part 8 tests; UI lands in Phase 6)

## What remains open

1. **Prompt 4B:** real `BrokerAdapter` implementations — MT5 via the
   `apps/mt5_bridge` service (Windows/VPS beside the terminal, TLS + token
   auth, `order_check` before every send), cTrader Open API after that.
   Credentials live only in the bridge/vault.
2. **Paper-trading soak period** (e.g. 4 weeks) with logged results and
   reconciliation — operational, starts once 4B lands or against the
   paper broker with a live feed.
3. Optional: Telegram/email notification sinks (the `Notifier` sink
   interface is ready for them).

---

# Addendum — Phase 4B: Broker Integration (Part 12.4)

## What was built

### BrokerAdapter contract — `core/execution/broker_adapter.py`
The exact surface from Part 12.4 (connect, account info, symbol spec,
quote, order_check, place, modify, cancel, close, positions, orders,
history, health, find-by-idempotency-key). `assert_conforms()` is a
contract check used by tests. PaperBroker now fully conforms — same
contract tests run against every gateway (Part 9: only the event source
and the order gateway change).

### The MT5 bridge — `apps/mt5_bridge/`
A small authenticated FastAPI service that runs on a Windows machine or
VPS **beside a running, logged-in MT5 terminal**:
- **Auth:** Bearer token on every endpoint; the app refuses to start
  without a token (no unauthenticated mode exists).
- **TLS:** `run()` refuses to bind a non-loopback address without
  certfile/keyfile.
- **order_check before order_send:** enforced SERVER-side — a send
  without a prior check for the same idempotency key is refused
  (retcode 10027), and a duplicate key is refused (10032).
- **Credentials:** MT5 login/password/server come from environment
  variables inside the bridge only (`RealMT5Backend.connect`); the token
  is never logged or echoed in any response (tested).
- **Backends:** `FakeMT5Backend` (deterministic, testable) and
  `RealMT5Backend` (guarded `MetaTrader5` import — raises a clear
  `MT5Unavailable` off-Windows instead of failing silently).

### Client adapter — `apps/mt5_bridge/client.py`
`MT5BridgeAdapter` implements the full BrokerAdapter over HTTP(S) with
token auth from `DSH_MT5_BRIDGE_TOKEN`; plain HTTP to a non-loopback
bridge is refused. A rejected send (validation, duplicate key, broker
retcode) surfaces as `ok=False` — never silently mapped to success.

## Verification

302 tests pass (17 new in `tests/test_mt5_bridge.py`): auth rejections,
fail-closed startup (no token, no TLS on public bind, no token on
client), server-side order_check enforcement, duplicate-key rejection,
full adapter round trip (quote → check → send → reconcile → modify →
close), `RealMT5Backend` unavailability off-Windows, contract parity
between PaperBroker and the bridge adapter, and the Part 9 statement
"only the gateway differs" tested numerically (paper fill = live ask +
slippage).

## What remains open

1. Deploy the bridge on a Windows VPS beside a real MT5 terminal with a
   TLS certificate and a token from the vault; the code paths are
   identical to the tested fake.
2. Pending/limit/stop orders in the bridge (market orders only for now;
   `cancel_order` raises until then).
3. cTrader Open API adapter (second adapter, after MT5 is stable).
4. Paper-trading soak period with a live feed.
