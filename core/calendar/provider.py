"""Calendar providers (Update Blueprint v2.1, U3).

A CalendarProvider yields economic-calendar events for a time range.
Two implementations:
- FileCalendarProvider: manual CSV/JSON import, works offline.
- HttpCalendarProvider: SKELETON with configurable URL, API key from
  an environment variable (never hard coded), timeout, retry, disk
  cache. The vendor-specific response mapping is a marked TODO — no
  vendor is assumed.
"""
from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

IMPACTS = ("low", "medium", "high")


@dataclass(frozen=True)
class CalendarEvent:
    ts_ms: int              # event instant, UTC
    currency: str           # e.g. "USD"
    impact: str             # low | medium | high
    title: str
    source: str

    def __post_init__(self):
        if self.impact not in IMPACTS:
            raise ValueError(f"impact must be one of {IMPACTS}, "
                             f"got {self.impact!r}")


class CalendarProvider(ABC):
    """fetch(from_ms, to_ms) -> events with start in [from_ms, to_ms)."""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, from_ms: int, to_ms: int) -> list[CalendarEvent]:
        ...


class FileCalendarProvider(CalendarProvider):
    """Manual import: CSV or JSON file, offline-safe.

    CSV columns: ts_ms,currency,impact,title   (ts_ms = UTC epoch ms)
    JSON: [{"ts_ms": ..., "currency": ..., "impact": ..., "title": ...}]
    A `time_utc` ISO column is accepted instead of ts_ms in JSON.
    """

    name = "file"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, from_ms: int, to_ms: int) -> list[CalendarEvent]:
        text = self.path.read_text(encoding="utf-8")
        raw: list[dict]
        if self.path.suffix.lower() == ".json":
            raw = json.loads(text)
            if isinstance(raw, dict):        # {"events": [...]}
                raw = raw.get("events", [])
        else:
            raw = list(csv.DictReader(text.splitlines()))
        events: list[CalendarEvent] = []
        for row in raw:
            ts = int(row["ts_ms"]) if row.get("ts_ms") else _iso_ms(row["time_utc"])
            events.append(CalendarEvent(
                ts_ms=ts, currency=str(row["currency"]).upper(),
                impact=str(row["impact"]).lower(),
                title=str(row.get("title", "")), source=self.name))
        return [e for e in events if from_ms <= e.ts_ms < to_ms]


def _iso_ms(value: str) -> int:
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class HttpCalendarProvider(CalendarProvider):
    """HTTP skeleton. NOT vendor-specific by design.

    TODO(vendor): to activate against a real vendor you must answer:
      1. Which vendor? (e.g. ForexFactory CSV export, Finnhub, FMP, ...)
      2. Exact endpoint URL and response schema.
      3. Impact vocabulary mapping (vendor labels -> low/medium/high).
      4. Currency list coverage and update frequency.
    Implement `_parse_response()` below; everything else is ready.
    """

    name = "http"
    # Set True ONLY in a vendor-specific subclass that implements
    # _parse_response. Until then fetch() fails fast, offline.
    vendor_ready = False

    def __init__(self, url: str, api_key_env: str = "",
                 timeout_s: float = 10.0, retries: int = 2,
                 backoff_s: float = 1.0, cache_path: str | Path | None = None,
                 cache_ttl_s: float = 3600.0):
        self.url = url
        self.api_key_env = api_key_env      # NAME of the env var only
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_ttl_s = cache_ttl_s

    def _headers(self) -> dict[str, str]:
        # The key is read from the environment AT REQUEST TIME and is
        # never stored on the instance, logged, or serialized.
        headers = {"Accept": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def fetch(self, from_ms: int, to_ms: int) -> list[CalendarEvent]:
        if not self.vendor_ready:
            raise NotImplementedError(
                "HttpCalendarProvider is a skeleton: vendor mapping is a "
                "TODO — answer docs/calendar.md questions first")
        cached = self._read_cache(from_ms, to_ms)
        if cached is not None:
            return cached
        body = self._http_get(self._params(from_ms, to_ms))
        events = self._parse_response(body, from_ms, to_ms)
        self._write_cache(events)
        return events

    def _params(self, from_ms: int, to_ms: int) -> str:
        # TODO(vendor): vendor-specific query parameters.
        return f"{self.url}?from={from_ms}&to={to_ms}"

    def _http_get(self, url: str) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, headers=self._headers())
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise ConnectionError(f"calendar fetch failed: {last_exc}")

    def _parse_response(self, body: bytes,
                        from_ms: int, to_ms: int) -> list[CalendarEvent]:
        # TODO(vendor): map the vendor JSON to CalendarEvent list.
        # Deliberately unimplemented so no vendor is silently assumed.
        raise NotImplementedError(
            "HttpCalendarProvider._parse_response: vendor mapping is a "
            "TODO — answer docs/calendar.md questions first")

    def _read_cache(self, from_ms: int, to_ms: int) -> list[CalendarEvent] | None:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        try:
            blob = json.loads(self.cache_path.read_text())
            if time.time() - blob["fetched_at_s"] > self.cache_ttl_s:
                return None
            events = [CalendarEvent(**e, source=self.name)
                      for e in blob["events"]]
            return [e for e in events if from_ms <= e.ts_ms < to_ms]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, events: list[CalendarEvent]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({
            "fetched_at_s": time.time(),
            "events": [{"ts_ms": e.ts_ms, "currency": e.currency,
                        "impact": e.impact, "title": e.title}
                       for e in events]}))
