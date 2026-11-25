# Firm Onboarding (U8 step 5 + U4 link)

## Manual step: confirm third-party bridge/EA is allowed

**Before any live or funded deployment, obtain WRITTEN confirmation
from the target firm that:**

1. A third-party bridge/EA reading quotes and placing orders through
   their MT5 server is allowed on funded accounts (not just demo).
2. The bridge's order pattern (API-driven orders, no manual
   intervention) does not violate their EA/automation rules.
3. One terminal + one bridge per account satisfies their VPS policy.
4. Their rulebook (drawdown definition, news rules, consistency rule)
   matches what `core/compliance/rules.py` encodes — this is the U4
   `firm_support_confirmations.json` input.

Record the confirmation (who, when, exact wording) in
`state/firm_support_confirmations.json` (schema in
docs/payout_protection.md). **Fail closed: until the confirmation is
recorded, the platform refuses AUTO mode for that firm's accounts.**

## cTrader (U8 step 6) — SKIPPED (for now)

No target firm currently requires cTrader. `CTraderAdapter` is NOT
implemented. If a firm is added that requires it:

- use the official Open API with OAuth (no unofficial endpoints),
- demo accounts only until the firm confirms live permission,
- implement against the existing `BrokerAdapter` protocol in
  `core/execution/broker_adapter.py` (connect/quote/order_check/place/
  modify/close/positions/history) so the compliance pipeline is
  unchanged.

Revisit this note whenever a firm is added.
