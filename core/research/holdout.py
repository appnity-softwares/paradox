"""Holdout guard (Update Blueprint v2.1, U7 step 1).

The LAST 6 months of the dataset are a final holdout that no
optimization or selection step may touch. Enforcement is IN CODE:
- every search/validation range is checked against the holdout and a
  HoldoutViolation error is raised on overlap;
- the holdout is opened only by an explicit ONE-TIME command that is
  logged to state/holdout_openings.jsonl; a second attempt refuses.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class HoldoutViolation(RuntimeError):
    pass


class HoldoutGuard:
    def __init__(self, holdout_start_ms: int, dataset_end_ms: int,
                 log_path: str | Path = "state/holdout_openings.jsonl"):
        if holdout_start_ms >= dataset_end_ms:
            raise ValueError("holdout start must be before dataset end")
        self.holdout_start_ms = holdout_start_ms
        self.dataset_end_ms = dataset_end_ms
        self.log_path = Path(log_path)
        self._opened = self._already_opened()

    def _already_opened(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    # ── enforcement ──────────────────────────────────────────────────────

    def check_range(self, start_ms: int, end_ms: int,
                    purpose: str = "unspecified") -> None:
        """Raises HoldoutViolation when [start, end) overlaps the
        holdout. Used by EVERY search and validation step."""
        if end_ms <= start_ms:
            raise ValueError("end must be after start")
        if end_ms > self.holdout_start_ms:
            raise HoldoutViolation(
                f"range [{start_ms}, {end_ms}) overlaps the frozen "
                f"holdout (starts {self.holdout_start_ms}); purpose="
                f"{purpose!r}. Holdout is opened only by the explicit "
                f"one-time holdout command.")

    def check_not_opened(self, purpose: str = "unspecified") -> None:
        if self._opened:
            raise HoldoutViolation(
                f"holdout already opened ({len(self._opened)} time(s)); "
                f"purpose={purpose!r} refused — one-time only")

    # ── the explicit one-time opening ────────────────────────────────────

    def open_holdout(self, by: str, reason: str,
                     ts_ms: int | None = None) -> dict:
        """Explicit, logged, ONE-TIME. Any later open attempt refuses."""
        if self._opened:
            raise HoldoutViolation(
                "holdout already opened — one-time only; "
                f"{len(self._opened)} opening(s) on record")
        now = ts_ms if ts_ms is not None else int(time.time() * 1000)
        rec = {"ts_ms": now, "by": by, "reason": reason,
               "holdout_start_ms": self.holdout_start_ms,
               "dataset_end_ms": self.dataset_end_ms}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._opened.append(rec)
        return rec

    @property
    def opened(self) -> bool:
        return bool(self._opened)

    def openings(self) -> list[dict]:
        return list(self._opened)
