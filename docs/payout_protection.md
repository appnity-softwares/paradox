# Payout Protection (U5)

Update Blueprint v2.1, Package U5. Everything here is deterministic and
LLM-free. Purpose: protect a passing account's payout by proving the
trading was rule-compliant and by blocking the patterns prop firms ban.

## 1. Network identity registry (`core/accounts/network_identity.py`)

Per account: `vps_provider`, `region`, `public_ip`, `first_seen_ms`,
`last_seen_ms`, `bridge_instance_id` plus an IP history. The bridge
reports its public IP on `/health` → `registry.report(...)`. Stored at
`state/network_identity.json`. **No credentials are ever stored here**
(tested).

Blocking rules (fail closed):

- **Shared IP:** two accounts on one public IP while either account's
  firm `vps_policy`/`ip_policy` is not `allowed` → alert + Auto blocked
  (reason `shared_ip:<ip>`). `allowed` requires BOTH policies.
- **Unexpected IP change:** any change is recorded and flags
  `unexpected_ip_change` → alert + Auto blocked.
- Enforcement: `ModeGate.set_mode(..., network_reasons=...)` refuses
  AUTO, and the router re-checks at order time (`network_identity`
  pipeline step, reason `network_identity_conflict`) because the IP
  picture may change after set_mode.

One-bridge-per-account: `shared_bridge_conflicts()` flags two accounts
on the same `bridge_instance_id` (surfaced on
`GET /api/platform/network-status`).

## 2. Cross-account guard (`core/accounts/cross_account.py`)

Before the router submits a NEW entry it checks whether an identical
intent (same symbol, same direction, same `strategy_version_id`)
was submitted on another account within `window_ms` (default 60 s).
Blocked with reason `copy_trading_blocked` unless that account's
`copy_trading_policy` is `allowed` (`unknown` blocks — fail closed).
Closing positions is never blocked (risk-reducing). Every account on
this single-operator platform is treated as one user's accounts
(documented assumption). Blocks are audited (`copy_trading_block`) and
notified.

## 3. Payout evidence pack

`GET /api/platform/accounts/{id}/payout-evidence-pack?from&to`
(trader role) → deterministic ZIP (`core/accounts/evidence.py`):

| File | Content |
|---|---|
| `trades.csv` | every trade: strategy_version_id, signal/order/fill times, intended risk, risk decision trail, compliance reasons, ruleset version + content hash, session profile, hold duration, modification count |
| `compliance_snapshots.csv` | per-day compliance state (`state/compliance_snapshots.jsonl`) |
| `network_history.json` | IP/VPS history for the account |
| `audit_slice.jsonl` | audit entries in [from, to] |
| `chain_verification.txt` | hash-chain verdict (intact / BROKEN at index N) |
| `firm_support_confirmations.json` | from `state/firm_support_confirmations.json` (U4 populates) |
| `strategy_rationales.json` | human-written rationale per version (`state/strategy_rationales.json`) |
| `strategy_fingerprints.json` | fingerprint report (below) |
| `manifest.json` | account, range, file list, chain verdict |

Determinism: fixed ZipInfo timestamps, sorted names/rows — two builds
over the same inputs are byte-identical (tested).

## 4. Strategy fingerprint report (`core/accounts/fingerprint.py`)

Informational only, for human payout-risk review:

- `standard_public` — all indicators are standard public ones (EMA,
  RSI, ATR, Bollinger, MACD, ...) and no prohibited style.
- `contains_prohibited_style` — martingale / grid / averaging_down /
  hedging / copy_trading / tick_scalping / latency_arbitrage present.
- `custom` — anything else; `human_review_required: true`.

## Secrets policy

Credentials live only in environment variables. A test sets fake
secrets, runs the execution path, and asserts neither the audit log nor
the evidence pack contains them (`scan_for_secrets` helper is available
for ops checks).

## Tests

20 tests in `tests/test_payout_protection.py`: registry round-trip and
no-credential storage, shared-IP conflict (and the allowed-policy
negative), IP-change flag, shared bridge, shared-IP blocks Auto
(ModeGate + router order-time), correlated-intent block / allowed /
window expiry / close-side exemption, evidence pack content +
determinism + time filtering + chain verdict, real audit chain
verification, no-secrets outputs, snapshot store, fingerprints.
