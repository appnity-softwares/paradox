"""Yahoo Finance data provider - real gold FUTURES (GC=F).

Works on macOS/Linux/Windows. Used as the primary source when MT5 is
unavailable. Provides real market data (not synthetic) but:
  - GC=F gold FUTURES, not spot XAUUSD
  - 15m limited to last 60 days, 1h to 730 days
  - no execution capability
"""
import pandas as pd

from core.broker import DataProvider, BrokerStatus


YAHOO_PERIODS = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d",
    "30m": "60d", "1h": "730d", "4h": "730d", "1d": "max", "1wk": "max", "1mo": "max",
}


class YahooBroker(DataProvider):
    def __init__(self):
        self._yf = None
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            self._yf = None

    def name(self) -> str:
        return "YahooFinance (GC=F futures)"

    def is_available(self) -> BrokerStatus:
        if self._yf is None:
            return BrokerStatus(
                name=self.name(),
                available=False,
                reason="yfinance not installed (pip install yfinance)",
                real_data=True,
                spot=False,
                execution=False,
                delayed=True,
            )
        return BrokerStatus(
            name=self.name(),
            available=True,
            reason="real gold futures (GC=F), no execution",
            real_data=True,
            spot=False,
            execution=False,
            delayed=True,
        )

    def fetch_ohlcv(self, symbol="XAUUSD", timeframe="15m", bars=5000) -> pd.DataFrame:
        # Yahoo uses GC=F for gold futures regardless of symbol passed
        period = YAHOO_PERIODS.get(timeframe, "60d")
        raw = self._yf.download("GC=F", interval=timeframe, period=period, progress=False, auto_adjust=False)
        if raw is None or len(raw) == 0:
            raise RuntimeError("Yahoo returned no data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        df = pd.DataFrame({
            "open": raw["Open"].astype(float),
            "high": raw["High"].astype(float),
            "low": raw["Low"].astype(float),
            "close": raw["Close"].astype(float),
            "volume": raw["Volume"].astype(float),
        })
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index().dropna()
        return df.tail(bars)

    def fetch_tick(self, symbol="XAUUSD"):
        return None  # Yahoo has no real-time tick in free tier
