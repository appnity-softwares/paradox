"""Session profiles (Update Blueprint v2.1, U2).

A SessionProfile bounds WHEN the platform may open new trades and when
it must flatten. All conversions are DST-safe via zoneinfo: for any
calendar date the window is resolved to exact UTC instants.

DST policy (documented, fail-safe):
- Spring-forward gap (a local wall time that does not exist): the wall
  time maps to the instant just AFTER the gap starts (PEP 495 fold=0
  conversion), so the session is effectively shorter. Never longer.
- Fall-back overlap (a local wall time that occurs twice): the FIRST
  occurrence is used, so the session is never extended.

These rules can only shrink or keep a session, never extend it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

MS_PER_DAY = 86_400_000
MAX_DURATION_CAP_MIN = 240          # hard cap from the blueprint


def _parse_hhmm(value: str, field_name: str) -> time:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a 'HH:MM' string")
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"{field_name} must be 'HH:MM', got {value!r}")
    try:
        hh, mm = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"{field_name} must be 'HH:MM', got {value!r}")
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"{field_name} out of range: {value!r}")
    return time(hh, mm)


@dataclass(frozen=True)
class SessionProfile:
    """One bounded trading session (e.g. London+NY overlap)."""

    name: str
    start: str                          # "HH:MM" session-local
    end: str                            # "HH:MM" session-local
    timezone_name: str                  # IANA, e.g. "Europe/London"
    days_of_week: tuple[int, ...] = (0, 1, 2, 3, 4)   # Mon=0, start-day
    max_duration_minutes: int = MAX_DURATION_CAP_MIN
    no_new_entries_after_minutes_before_end: int = 0
    flatten_by: str | None = None       # "HH:MM" local; None = end
    max_trades_in_session: int | None = None   # upper cap, never a target

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string")
        try:
            ZoneInfo(self.timezone_name)
        except Exception as exc:
            raise ValueError(f"unknown IANA timezone "
                             f"{self.timezone_name!r}") from exc
        if not self.days_of_week:
            raise ValueError("days_of_week must not be empty")
        for d in self.days_of_week:
            if not 0 <= d <= 6:
                raise ValueError(f"day out of range 0..6: {d}")
        if not (1 <= self.max_duration_minutes <= MAX_DURATION_CAP_MIN):
            raise ValueError(
                f"max_duration_minutes must be 1..{MAX_DURATION_CAP_MIN} "
                f"(got {self.max_duration_minutes}); longer sessions are "
                "rejected by policy")
        if self.no_new_entries_after_minutes_before_end < 0:
            raise ValueError("no_new_entries_after_minutes_before_end >= 0")
        if self.max_trades_in_session is not None and \
                self.max_trades_in_session < 1:
            raise ValueError("max_trades_in_session must be >= 1 or None")
        self.__dict__["_start_t"] = _parse_hhmm(self.start, "start")
        self.__dict__["_end_t"] = _parse_hhmm(self.end, "end")
        # Window length (mod midnight) must fit inside the cap.
        start_min = self._start_t.hour * 60 + self._start_t.minute
        end_min = self._end_t.hour * 60 + self._end_t.minute
        length = (end_min - start_min) % 1440
        if length == 0:
            raise ValueError("session start and end must differ")
        if length > self.max_duration_minutes:
            raise ValueError(
                f"session length {length} min exceeds "
                f"max_duration_minutes={self.max_duration_minutes}")
        self.__dict__["_flatten_t"] = (
            _parse_hhmm(self.flatten_by, "flatten_by")
            if self.flatten_by is not None else self._end_t)

    # ── DST-safe UTC conversion ──────────────────────────────────────────

    def _local_to_utc_ms(self, day: date, t: time) -> int:
        """Convert a local wall time on `day` to a UTC millisecond instant.

        Nonexistent local times (spring-forward gap) resolve to the
        instant the gap began (the session shrinks). Ambiguous local
        times (fall-back) resolve to the first occurrence.
        """
        dt = datetime.combine(day, t, tzinfo=ZoneInfo(self.timezone_name))
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)

    def window_utc(self, day: date) -> tuple[int, int]:
        """(start_ms, end_ms) in UTC for the session starting on `day`
        (start-day semantics; a session that wraps local midnight ends
        on the next calendar day)."""
        start_ms = self._local_to_utc_ms(day, self._start_t)
        end_day = day if self._end_t > self._start_t else day + timedelta(days=1)
        end_ms = self._local_to_utc_ms(end_day, self._end_t)
        return start_ms, end_ms

    def _flatten_instant(self, day: date) -> int:
        flat_day = day if self._flatten_t > self._start_t \
            else day + timedelta(days=1)
        return self._local_to_utc_ms(flat_day, self._flatten_t)

    def window_for_ts(self, ts_ms: int) -> tuple[int, int] | None:
        """The window containing ts_ms (checks the ts day and the
        previous day, so wrapped sessions resolve correctly)."""
        day = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).date()
        for cand in (day - timedelta(days=1), day):
            if cand.weekday() not in self.days_of_week:
                continue
            start_ms, end_ms = self.window_utc(cand)
            if start_ms <= ts_ms < end_ms:
                return start_ms, end_ms
        return None

    # ── the three blueprint functions ────────────────────────────────────

    def in_window(self, ts_ms: int) -> bool:
        return self.window_for_ts(ts_ms) is not None

    def entries_allowed(self, ts_ms: int) -> bool:
        win = self.window_for_ts(ts_ms)
        if win is None:
            return False
        _, end_ms = win
        cutoff = end_ms - \
            self.no_new_entries_after_minutes_before_end * 60_000
        return ts_ms < cutoff

    def must_flatten(self, ts_ms: int) -> bool:
        """True at or after the flatten instant of the most recent
        session day whose start is at or before ts_ms. Between sessions
        this is True: any position carried out of its window must be
        flattened immediately."""
        day = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).date()
        for cand in (day, day - timedelta(days=1)):
            if cand.weekday() not in self.days_of_week:
                continue
            start_ms, _ = self.window_utc(cand)
            if start_ms <= ts_ms:
                return ts_ms >= self._flatten_instant(cand)
        return False

    # ── serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name, "start": self.start, "end": self.end,
            "timezone": self.timezone_name,
            "days_of_week": list(self.days_of_week),
            "max_duration_minutes": self.max_duration_minutes,
            "no_new_entries_after_minutes_before_end":
                self.no_new_entries_after_minutes_before_end,
            "flatten_by": self.flatten_by,
            "max_trades_in_session": self.max_trades_in_session,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionProfile":
        return cls(
            name=data["name"], start=data["start"], end=data["end"],
            timezone_name=data.get("timezone", "UTC"),
            days_of_week=tuple(data.get("days_of_week", (0, 1, 2, 3, 4))),
            max_duration_minutes=data.get("max_duration_minutes",
                                          MAX_DURATION_CAP_MIN),
            no_new_entries_after_minutes_before_end=
                data.get("no_new_entries_after_minutes_before_end", 0),
            flatten_by=data.get("flatten_by"),
            max_trades_in_session=data.get("max_trades_in_session"),
        )
