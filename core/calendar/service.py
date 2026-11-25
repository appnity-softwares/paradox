"""CalendarService: store, staleness and filtered windows (U3).

FAILURE POLICY (fail closed, documented):
- An account WITH a news restriction that faces a missing, stale or
  erroring calendar gets NO new entries and an alert.
- Accounts WITHOUT a news restriction are not affected.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from core.calendar.provider import CalendarEvent, CalendarProvider

DEFAULT_STALENESS_LIMIT_S = 24 * 3600    # refresh required within 24 h


class CalendarService:
    """Loads events through a provider, stores them with fetched_at and
    source, and serves filtered windows with a staleness limit."""

    def __init__(self, provider: CalendarProvider,
                 store_path: str | Path = "state/calendar.json",
                 staleness_limit_s: float = DEFAULT_STALENESS_LIMIT_S):
        self.provider = provider
        self.store_path = Path(store_path)
        self.staleness_limit_s = staleness_limit_s
        self.last_error: str | None = None
        self._load_store()

    # ── store ────────────────────────────────────────────────────────────

    def _load_store(self) -> None:
        self.events: list[CalendarEvent] = []
        self.fetched_at_ms: int | None = None
        self.source: str | None = None
        if not self.store_path.exists():
            return
        try:
            blob = json.loads(self.store_path.read_text())
            self.fetched_at_ms = blob.get("fetched_at_ms")
            self.source = blob.get("source")
            self.events = [CalendarEvent(**e) for e in blob.get("events", [])]
        except (OSError, ValueError, KeyError, TypeError):
            self.events = []
            self.fetched_at_ms = None

    def _save_store(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps({
            "fetched_at_ms": self.fetched_at_ms,
            "source": self.source,
            "events": [{"ts_ms": e.ts_ms, "currency": e.currency,
                        "impact": e.impact, "title": e.title,
                        "source": e.source} for e in self.events],
        }, indent=1))

    # ── refresh and status ───────────────────────────────────────────────

    def refresh(self, from_ms: int, to_ms: int, now_ms: int | None = None) -> str:
        """Fetch through the provider; on success replace the store. On
        failure keep the previous data and record the error (status
        becomes 'stale' once the limit passes). Returns the status."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        try:
            events = self.provider.fetch(from_ms, to_ms)
        except Exception as exc:                      # provider error
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.status(now)
        self.events = sorted(events, key=lambda e: e.ts_ms)
        self.fetched_at_ms = now
        self.source = self.provider.name
        self.last_error = None
        self._save_store()
        return "ok"

    def status(self, now_ms: int | None = None) -> str:
        """'ok' | 'stale' | 'missing'."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        if self.fetched_at_ms is None:
            return "missing"
        if now - self.fetched_at_ms > self.staleness_limit_s * 1000:
            return "stale"
        return "ok"

    # ── filtered access ──────────────────────────────────────────────────

    def events_filtered(self, currencies: tuple[str, ...] | None,
                        impacts: tuple[str, ...] | None,
                        from_ms: int, to_ms: int) -> list[CalendarEvent]:
        """Events in [from_ms, to_ms) filtered by currency and impact.
        None = no filter on that axis."""
        out = []
        for e in self.events:
            if not (from_ms <= e.ts_ms < to_ms):
                continue
            if currencies is not None and e.currency not in currencies:
                continue
            if impacts is not None and e.impact not in impacts:
                continue
            out.append(e)
        return out

    def calendar_map(self, currencies: tuple[str, ...] | None,
                     impacts: tuple[str, ...] | None,
                     around_ms: int, span_days: int = 2,
                     ) -> dict[int, str] | None:
        """The dict form the ComplianceEngine expects:
        {event_start_ms: "title (impact)"}. Returns None when the
        calendar is missing or stale (fail closed upstream)."""
        st = self.status()
        if st != "ok":
            return None
        lo = around_ms - span_days // 2 * 86_400_000
        hi = around_ms + span_days * 86_400_000
        return {e.ts_ms: f"{e.title} ({e.impact})"
                for e in self.events_filtered(currencies, impacts, lo, hi)}

    def windows(self, currencies: tuple[str, ...] | None,
                impacts: tuple[str, ...] | None,
                before_min: int, after_min: int,
                from_ms: int, to_ms: int) -> list[tuple[int, int]]:
        """Expanded windows [start - before, start + after] (both ends
        INCLUSIVE — matches the ComplianceEngine semantics)."""
        out = []
        for e in self.events_filtered(currencies, impacts, from_ms, to_ms):
            out.append((e.ts_ms - before_min * 60_000,
                        e.ts_ms + after_min * 60_000))
        return sorted(out)

    def next_window_start(self, currencies: tuple[str, ...] | None,
                          impacts: tuple[str, ...] | None,
                          before_min: int, ts_ms: int,
                          horizon_days: int = 3) -> int | None:
        """First window START at or after ts_ms (for position policy)."""
        lo, hi = ts_ms, ts_ms + horizon_days * 86_400_000
        for e in self.events_filtered(currencies, impacts, lo, hi):
            if e.ts_ms - before_min * 60_000 >= ts_ms:
                return e.ts_ms
        return None
