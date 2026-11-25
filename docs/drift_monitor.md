# Drift Detection to Auto-Pause (U6)

Update Blueprint v2.1, Package U6. Until now the performance_analyst
agent produced `drift_flags` but nothing acted on them. U6 adds a
**deterministic** monitor (no LLM anywhere in the decision path) that
compares live per-deployment statistics with the stored out-of-sample
validation band for that strategy version, and pauses on critical
drift.

## Components (`core/monitoring/drift.py`)

| Class | Purpose |
|---|---|
| `ValidationBand` | stored OOS band per `strategy_version_id`: win rate, avg R, trade frequency, max drawdown, avg slippage, spread at entry — each with a tolerance |
| `DriftThresholds` | warn/crit deviation factors (default 1x/2x tolerance), losing-streak probability thresholds (5% warn / 1% crit), stale-feed and bridge-disconnect limits |
| `DriftMonitor` | band comparison + pause execution + report data |
| `DriftAlertStore` | persisted `drift_alerts` (JSONL) |

## Comparison logic

For each metric the deviation ratio is `|live − expected| / tolerance`
(in the harmful direction only — e.g. a HIGHER win rate than validated
is not drift):

- ratio ≤ warn_factor → inside the band, no finding
- warn_factor < ratio ≤ crit_factor → **warning**
- ratio > crit_factor → **critical**

**Losing streak vs probability:** under the VALIDATED win rate `p`,
P(≥ k consecutive losses) = (1 − p)^k. Below 5% → warning, below 1% →
critical. This catches "the strategy is losing far more consistently
than its validated win rate allows" without any curve-fitting.

## Severity and action

- `info` findings are not produced (inside-band is silent).
- `warning` → persisted alert + audit entry + notifier, **no pause**.
- `critical` → **`ModeGate.set_mode(account, 'paper')`** (deployment
  paused), audit entry (`drift_alert`), notifier alert, persisted
  alert with `action: "paused_to_paper"`.

## Operational pauses (beyond statistical drift)

`check_operational_pause(...)` pauses on: internal drawdown hard stop,
daily loss stop, reconciliation mismatch, stale feed beyond
`stale_feed_ms` (default 60 s), bridge disconnect beyond
`bridge_disconnect_ms` (default 5 min). Same audit + alert + notify
path (`operational_pause` event).

## Resuming requires an authenticated human

`POST /api/platform/accounts/{id}/resume-deployment` (trader role,
valid session token) sets the target mode (default `semi_auto`).
Without a token it returns 401 (tested). `auto` as the target still
goes through the full ModeGate prerequisites (verdict, paper days,
rules verified, confirmation) — the pause never shortcuts safety.

## Reports

`DriftMonitor.report_data(from_ms, to_ms)` returns deterministic
numbers (counts by severity, pauses, accounts). The performance_analyst
agent may write the NARRATIVE from these numbers — never the reverse.

## API

- `GET /api/platform/drift-alerts?account&min_severity` (viewer)
- `POST /api/platform/accounts/{id}/resume-deployment` (trader)

## Tests

15 tests in `tests/test_drift_monitor.py`: inside-band silence, no-band
silence, warning without pause, critical with auto-pause (mode, audit,
alert store), drawdown criticality, losing-streak probability
boundaries (6 losses warn, 5 inside band), slippage/spread drift,
operational pause triggers and below-threshold no-op, reconciliation
pause, resume 401 without token / 200 with token, alerts endpoint
401, deterministic report data, and the no-LLM guarantee (patched
`get_llm` raises).
