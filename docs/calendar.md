# Economic Calendar Integration (U3)

Update Blueprint v2.1, Package U3. The compliance engine always had a
fail-closed news check (`calendar_events_ms=None` while a news
restriction is enabled ⇒ block), but nothing ever populated it. U3 adds
the calendar plumbing so news-restricted accounts get real, current
windows and unrestricted accounts are untouched.

## Components

| File | Purpose |
|---|---|
| `core/calendar/provider.py` | `CalendarEvent`, `CalendarProvider` ABC, `FileCalendarProvider` (CSV/JSON, offline), `HttpCalendarProvider` (skeleton) |
| `core/calendar/service.py` | `CalendarService`: store (`state/calendar.json`), staleness limit (24 h default), filtered access, window helpers |
| `core/calendar/windows.py` | `build_news_windows`, `windows_from_store`, `windows_from_service` → `BacktestConfig.news_windows` |

## Failure policy (fail closed, documented)

- Account **with** a news restriction + calendar **missing / stale /
  provider error** → **no new entries**, reason `calendar_unavailable`,
  notifier alert.
- Account **without** a news restriction → calendar is irrelevant; no
  gate, no rejection.
- A successful fetch replaces the store atomically; a failed fetch
  keeps the previous data and records `last_error`. The data goes
  "stale" once `staleness_limit_s` (default 24 h) passes since the last
  successful fetch.

## Window semantics

For an event at `ts_ms` with rule-set values `window_before_min` /
`window_after_min`:

```
window = [ts_ms − before_min·60_000, ts_ms + after_min·60_000]
```

**Both ends are INCLUSIVE** — an entry exactly at `start` or exactly at
`end` is blocked; one millisecond beyond either edge is allowed. This
matches the pre-existing `ComplianceEngine.pre_trade_check` semantics
(unchanged, backward compatible) and the tick-engine `in_news()` check.

## Filtering (per rule set)

`NewsRestriction` (U3 fields, all with safe defaults):

- `currencies: tuple[str, ...] = ("USD",)` — events whose currency is
  in this set apply. XAUUSD's quote currency is USD, so this covers the
  classic gold rulebook case.
- `impacts: tuple[str, ...] = ("high",)` — impact vocabulary is
  `low | medium | high`; validation rejects unknown values.
- `position_policy: str = "keep"` — what happens to open positions
  when a window starts: `keep` (no-op), `tighten_sl` (risk-reducing SL
  move, protective, audited), `flatten` (protective close-all via
  `router.protective_close_all`, audited).

Old rule-set JSON without these keys loads with the defaults
(smoke-tested round-trip).

## Router wiring

- `router.attach_calendar(service)` — wire the service once at startup.
- Per intent, `_calendar_for()` builds the filtered
  `{event_start_ms: "title (impact)"}` map around the intent time and
  passes it to the existing compliance check. Missing/stale ⇒ `None`
  ⇒ engine's fail-closed `calendar_unavailable`.
- `router.enforce_news_position_policy(account_id, ts_ms)` — call this
  from the live loop when a window starts (deterministic, no LLM).

## Scheduler

The U2 pre-check `calendar_not_loaded` is now service-aware: with a
news restriction enabled, a **missing** calendar skips the session with
`calendar_not_loaded`, a **stale** one with `calendar_stale`. Both are
fail-closed skips with audit + notification.

## Backtest parity

`BacktestConfig.news_windows` existed (news slippage multiplier). U3
adds `block_entries_in_news_windows: bool = False` (default OFF for
backward compatibility): when True, entries whose decision instant
falls inside a window are blocked and counted in
`behaviour_blocks["news_window"]` — the same gate the live router
applies. Build the tuple from the same stored calendar the live path
uses:

```python
from core.calendar import windows_from_store
cfg.news_windows = windows_from_store("state/calendar.json",
                                      currencies=("USD",), impacts=("high",),
                                      before_min=2, after_min=2)
```

**Documented difference:** the live path filters by currency/impact at
gate time from the freshest calendar; a backtest freezes the windows
from the store at build time. Rebuild windows per backtest period.

## Providers

### FileCalendarProvider (works today, offline)

CSV: `ts_ms,currency,impact,title` — JSON: list of objects, or
`{"events": [...]}`; ISO `time_utc` accepted instead of `ts_ms`.
Manual export from any source is fine for paper trading.

### HttpCalendarProvider (SKELETON — deliberately vendor-neutral)

Configurable URL, API key read from an environment variable **at
request time** (never stored, logged, or serialized), timeout, retry
with backoff, disk cache with TTL. `fetch()` fails fast with
`NotImplementedError` until a vendor subclass implements
`_parse_response` — no vendor is silently assumed.

## Questions to answer before activating the HTTP provider

1. **Vendor choice** — ForexFactory weekly CSV export (free, manual),
   Finnhub / FMP / Trading Economics (API, paid tiers), or a broker
   feed? Cost, licensing and redistribution terms?
2. **Endpoint + schema** — exact URL, auth header format, response
   fields, pagination.
3. **Impact vocabulary** — how do vendor labels map to
   low/medium/high? (e.g. "red" → high?)
4. **Currency coverage** — is the vendor's currency list complete for
   USD (and any other symbols we trade later)?
5. **Update frequency** — how often does the vendor revise event times?
   (Drives the staleness limit; 24 h is the default.)
6. **Historical archive** — do they sell historical events? Needed to
   backtest news windows over 2024-2026 honestly; otherwise news
   windows are live-only and backtests must document that gap.

## Tests

21 tests in `tests/test_calendar.py`: providers (CSV/JSON, skeleton
fails fast, no-key storage), service (fresh/stale/missing, provider
error keeps old data), filtering, router (window block, missing/stale
calendar fail-closed, unrestricted account unaffected, inclusive edges
at both boundaries ± 1 ms), position policy (keep / tighten_sl /
flatten), scheduler skip reasons, and backtest parity (blocked entry
counted, control run trades).
