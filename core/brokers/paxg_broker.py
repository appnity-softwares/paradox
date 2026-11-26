"""Binance PAXG spot-proxy provider (24/7 tokenised gold).

PAXGUSDT tracks one troy ounce of allocated gold, so it is a usable *proxy* for
XAUUSD when the broker feed is unavailable — but it is NOT the broker's quote.
Every result is therefore labelled ``proxy=True`` and the provider name says
"spot proxy", so the UI can never present it as broker data.

Bars come from the public klines REST endpoint; ticks from the public
bookTicker endpoint (real bid/ask, no fabrication).
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from core.broker import BrokerStatus, DataProvider

REST = "https://api.binance.com/api/v3"
SYMBOL = "PAXGUSDT"
_INTERVALS = {
    "1m": "1m", "2m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
}


def _ssl_context() -> ssl.SSLContext:
    """Use the OS trust store when available (corporate/root CAs), never skip verify."""
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


def _get(path: str, timeout: float = 6.0):
    request = urllib.request.Request(
        f"{REST}/{path}", headers={"User-Agent": "quant-platform/0.4"}
    )
    with urllib.request.urlopen(request, timeout=timeout, context=_SSL) as response:
        return json.loads(response.read().decode())


class PaxgBroker(DataProvider):
    """24/7 gold spot proxy. Read-only: it can never execute orders."""

    def name(self) -> str:
        return "Binance PAXG (spot proxy)"

    def key(self) -> str:
        return "paxg"

    def is_available(self) -> BrokerStatus:
        try:
            _get(f"ticker/price?symbol={SYMBOL}", timeout=4.0)
            available, reason = True, "Binance PAXGUSDT reachable (24/7 tokenised gold)"
        except Exception as exc:
            available, reason = False, f"Binance unreachable: {exc}"
        return BrokerStatus(
            name=self.name(),
            available=available,
            reason=reason,
            real_data=True,
            spot=True,
            execution=False,
            delayed=False,
            proxy=True,
        )

    def fetch_ohlcv(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        # Honesty guard: this provider is a spot GOLD proxy (PAXGUSDT).
        # Its prices must never be served for another instrument.
        if symbol.upper() not in ("XAUUSD", "PAXGUSDT"):
            raise ValueError(f"PAXG proxy only covers XAUUSD, not {symbol}")
        interval = _INTERVALS.get(timeframe.lower())
        if interval is None:
            raise ValueError(f"PAXG does not support timeframe {timeframe!r}")
        limit = max(1, min(int(bars), 1000))
        rows = _get(f"klines?symbol={SYMBOL}&interval={interval}&limit={limit}")
        if not rows:
            raise RuntimeError("Binance returned no PAXG klines")
        records = [{
            "time": datetime.fromtimestamp(row[0] / 1000.0, timezone.utc),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        } for row in rows]
        df = pd.DataFrame(records).set_index("time").sort_index()
        # The last kline is still forming; drop it so no incomplete bar is
        # presented as a closed candle.
        return df.iloc[:-1] if len(df) > 1 else df

    def fetch_tick(self, symbol: str):
        if symbol.upper() not in ("XAUUSD", "PAXGUSDT"):
            return None
        try:
            book = _get(f"ticker/bookTicker?symbol={SYMBOL}", timeout=4.0)
        except Exception:
            return None
        bid, ask = float(book["bidPrice"]), float(book["askPrice"])
        if bid <= 0 or ask <= 0:
            return None
        return {
            "time": datetime.now(timezone.utc),
            "bid": bid,
            "ask": ask,
            "last": (bid + ask) / 2,
            "volume": float(book.get("bidQty", 0.0)) + float(book.get("askQty", 0.0)),
        }

    def fetch_ticks(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        if symbol.upper() not in ("XAUUSD", "PAXGUSDT"):
            raise ValueError(f"PAXG proxy only covers XAUUSD, not {symbol}")
        raise RuntimeError(
            "Binance PAXG provides a live quote stream, not historical ticks. "
            "Historical tick validation requires the MT5 export feed."
        )

    def account_info(self):
        raise RuntimeError("PAXG is a market-data proxy and holds no trading account.")
