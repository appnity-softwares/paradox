"""Broker and market-data provider abstractions.

The platform never talks to a single data source directly. It talks to a
`DataProvider`, and the `DataManager` picks the best one available:

  1. MetaTrader5  (Windows only, real spot XAUUSD ticks + execution)
  2. Yahoo Finance (macOS/Linux, real gold FUTURES, 15m/1h)
  3. Synthetic     (offline fallback, clearly labeled)

When MT5 is present (Windows machine with terminal installed) the platform
uses it automatically - the user never has to touch MT5 by hand.
"""
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class BrokerStatus:
    name: str
    available: bool
    reason: str = ""
    real_data: bool = False
    spot: bool = False
    execution: bool = False
    delayed: bool = False
    # True when the feed tracks the instrument but is not the broker's own
    # quote (e.g. tokenised gold). Must be surfaced in the UI.
    proxy: bool = False


@dataclass
class MarketDataResult:
    """Data plus provenance. A DataFrame alone cannot say if it is stale/synthetic."""

    data: Any
    provider: str
    provider_key: str
    symbol: str
    timeframe: str
    real_data: bool
    spot: bool
    delayed: bool
    synthetic: bool = False
    stale: bool = False
    proxy: bool = False
    fetched_at: datetime | None = None
    data_as_of: datetime | None = None
    sanitized_bars: int = 0

    def __post_init__(self) -> None:
        if self.fetched_at is None:
            self.fetched_at = datetime.now(timezone.utc)
        if self.data_as_of is None and getattr(self.data, "index", None) is not None and len(self.data):
            ts = self.data.index[-1]
            self.data_as_of = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

    def metadata(self) -> dict:
        return {
            "provider": self.provider,
            "provider_key": self.provider_key,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "real_data": self.real_data,
            "spot": self.spot,
            "delayed": self.delayed,
            "synthetic": self.synthetic,
            "stale": self.stale,
            "proxy": self.proxy,
            "sanitized_bars": self.sanitized_bars,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "data_as_of": self.data_as_of.isoformat() if self.data_as_of else None,
        }


class DataProvider(ABC):
    """Common interface for every data source."""

    @abstractmethod
    def name(self) -> str: ...

    def key(self) -> str:
        return self.__class__.__name__.replace("Broker", "").lower()

    @abstractmethod
    def is_available(self) -> BrokerStatus: ...

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, bars: int) -> "pd.DataFrame": ...

    def fetch_tick(self, symbol: str):
        """Optional real-time tick. Returns None if unsupported."""
        return None

    def place_order(self, *a, **kw):
        raise NotImplementedError("execution not supported by this provider")

    def positions(self):
        return []

    def account_info(self):
        return None


def is_windows() -> bool:
    return platform.system() == "Windows"
