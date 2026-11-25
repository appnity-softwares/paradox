"""News-window helpers: stored calendar -> BacktestConfig.news_windows."""
from __future__ import annotations

import json
from pathlib import Path

from core.calendar.service import CalendarService


def build_news_windows(events: list[dict], currencies: tuple[str, ...] | None,
                       impacts: tuple[str, ...] | None,
                       before_min: int, after_min: int,
                       from_ms: int = 0, to_ms: int = 1 << 62,
                       ) -> tuple[tuple[int, int], ...]:
    """Deterministic windows from raw event dicts (as stored in the
    calendar store). Edges are INCLUSIVE on both ends, matching the
    ComplianceEngine's news check."""
    svc_events = []
    for e in events:
        svc_events.append((int(e["ts_ms"]), str(e["currency"]).upper(),
                           str(e["impact"]).lower()))
    picked = []
    for ts, cur, imp in svc_events:
        if not (from_ms <= ts < to_ms):
            continue
        if currencies is not None and cur not in currencies:
            continue
        if impacts is not None and imp not in impacts:
            continue
        picked.append(ts)
    picked.sort()
    return tuple((ts - before_min * 60_000, ts + after_min * 60_000)
                 for ts in picked)


def windows_from_store(store_path: str | Path,
                       currencies: tuple[str, ...] | None,
                       impacts: tuple[str, ...] | None,
                       before_min: int, after_min: int,
                       from_ms: int = 0, to_ms: int = 1 << 62,
                       ) -> tuple[tuple[int, int], ...]:
    """Read a calendar store written by CalendarService and build the
    news_windows tuple for BacktestConfig — so a backtest honours the
    same restrictions as the live path."""
    blob = json.loads(Path(store_path).read_text())
    return build_news_windows(blob.get("events", []), currencies, impacts,
                              before_min, after_min, from_ms, to_ms)


def windows_from_service(service: CalendarService,
                         currencies: tuple[str, ...] | None,
                         impacts: tuple[str, ...] | None,
                         before_min: int, after_min: int,
                         from_ms: int = 0, to_ms: int = 1 << 62,
                         ) -> tuple[tuple[int, int], ...]:
    events = service.events_filtered(currencies, impacts, from_ms, to_ms)
    return tuple((e.ts_ms - before_min * 60_000, e.ts_ms + after_min * 60_000)
                 for e in events)
