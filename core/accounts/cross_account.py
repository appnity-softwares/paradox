"""Cross-account correlation guard (Update Blueprint v2.1, U5).

Before the router submits an intent, check whether an identical or
highly correlated intent (same symbol, same direction, within a
configurable time window, same strategy_version) was submitted on
another account. Default: BLOCK on prop accounts unless that firm's
copy_trading_policy is 'allowed' (unknown blocks too — fail closed).

Assumption (documented): this platform is single-operator, so every
registered account belongs to the same user.
"""
from __future__ import annotations

import time

DEFAULT_WINDOW_MS = 60_000


class CrossAccountGuard:
    def __init__(self, window_ms: int = DEFAULT_WINDOW_MS):
        self.window_ms = window_ms
        self._events: list[dict] = []   # submitted intents, newest last

    def record(self, account_id: str, symbol: str, side: int,
               strategy_version_id: str, ts_ms: int | None = None) -> None:
        ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
        self._events.append({"account_id": account_id, "symbol": symbol,
                             "side": int(side),
                             "strategy_version_id": strategy_version_id,
                             "ts_ms": int(ts)})

    def correlated(self, account_id: str, symbol: str, side: int,
                   strategy_version_id: str, ts_ms: int) -> list[dict]:
        """Events on OTHER accounts matching symbol+direction+version
        within the window, newest first."""
        lo, hi = ts_ms - self.window_ms, ts_ms + self.window_ms
        out = []
        for ev in reversed(self._events):
            if ev["account_id"] == account_id:
                continue
            if not (lo <= ev["ts_ms"] <= hi):
                continue
            if (ev["symbol"] == symbol and ev["side"] == int(side)
                    and ev["strategy_version_id"] == strategy_version_id):
                out.append(ev)
        return out

    def check(self, account_id: str, symbol: str, side: int,
              strategy_version_id: str, ts_ms: int,
              copy_trading_policy: str) -> dict:
        """Decision for a NEW entry (side != close). Returns
        {'blocked': bool, 'matches': [...]}."""
        matches = self.correlated(account_id, symbol, side,
                                  strategy_version_id, ts_ms)
        blocked = bool(matches) and copy_trading_policy != "allowed"
        return {"blocked": blocked, "matches": matches}
