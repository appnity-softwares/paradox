# Pending Stop/Limit Orders — Progress Report

Status: **complete end to end** — the last documented engine gap
("pending/limit/stop orders; market only") is closed across the intent
model, paper broker, router, MT5 bridge and adapter.

## What changed

### Intent model — `core/execution/intent.py`
`OrderIntent` gained `order_type` ("market" | "buy_stop" | "sell_stop" |
"buy_limit" | "sell_limit") and `limit_price` (the trigger). Defaults
keep every existing call site working. `__post_init__` validates: the
type must be known, pending types require a trigger, market orders must
not carry one. `ExecutionResult` gained the `pending_accepted` status.

### Paper broker — `core/execution/paper_broker.py`
- `_place_pending` queues the order (duplicate-key protection includes
  the pending queue); `pending_orders()` lists them.
- `on_tick` triggers pending orders before SL/TP management: stops fill
  at the trigger PLUS adverse slippage; limits fill at the trigger or
  better, never worse. SL/TP travel into the opened position.
- `cancel_order(order_ref)` now cancels pending orders (KeyError
  otherwise) — previously it raised NotImplementedError.
- Latent contract bug fixed: the instance dict `self.positions` shadowed
  the BrokerAdapter contract method `positions()`; renamed to
  `_positions` so `assert_conforms` now holds on instances too.

### Router — `core/execution/router.py`
A queued pending order returns `pending_accepted` (order_ref, no fill
price), audited and notified as `order_pending`. Risk and compliance
apply identically before acceptance — a 5%-risk pending ticket is
reduced to the cap exactly like a market order (tested).

### MT5 bridge — `apps/mt5_bridge/`
- `OrderRequest.price` is the pending trigger; MT5 type codes 2/3
  (limit) and 4/5 (stop) accepted; check refuses pending orders without
  a price. `FakeMT5Backend` queues, triggers on `set_quote` ticks, and
  gained `/order_cancel`. Check-before-send and duplicate-key
  enforcement unchanged.
- `MT5BridgeAdapter` maps intent order types to MT5 codes (unknown types
  raise), reports `pending: true` with `fill_price: None` for queued
  orders, implements `cancel_order`, and `find_by_idempotency_key`
  resolves the key once the pending order fills.

### Dashboard
The order ticket accepts `order_type`/`limit_price`, so pending entries
are available from the console through the full pipeline.

## Verification

15 new tests in `tests/test_pending_orders.py`: intent validation (3),
paper triggering/cancellation/duplicates (7), router end-to-end
pending_accepted + risk reduction (2), bridge round trip incl. cancel
and missing-price rejection (2), adapter round trip with check-before-
send (1). Full suite: **575 passed** (was 560; 18 trade-management tests
were updated only for the `_positions` rename, no behaviour change).

## What remains open

- RealMT5Backend pending mapping uses the documented MT5 type codes;
  full pending semantics (expiration, fills-kill vs fills-partial)
  should be verified against a real demo terminal during the Windows
  deployment — noted there.
- Remaining deployment-time items unchanged: paper soak, Next.js
  migration, real LLM providers, notification sinks, trade replay UI.
