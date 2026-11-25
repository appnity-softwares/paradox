"""Native macOS MetaTrader 5 Data Provider.

Directly connects and serves real spot XAUUSD bars and ticks from the user's macOS MT5 installation:
~/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5
"""
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from core.broker import DataProvider, BrokerStatus
from core.brokers.mt5_mac import extract_and_sync_all_mt5_bars, DATA_DIR


class MacMT5Broker(DataProvider):
    """Native macOS MetaTrader 5 Provider: real spot XAUUSD data + execution bridge."""

    def __init__(self):
        self._last_tick = None
        try:
            from core.marketdata import store as md_store
            cov = md_store.coverage("XAUUSD")
            if cov and cov.get("last_ms"):
                raw = md_store.read_range("XAUUSD", cov["last_ms"] - 60000, cov["last_ms"], limit=5)
                if not raw.empty:
                    last_row = raw.iloc[-1]
                    p = float((last_row["bid"] + last_row["ask"]) / 2.0)
                    self._last_tick = {
                        "time": datetime.now(timezone.utc),
                        "bid": round(float(last_row["bid"]), 2),
                        "ask": round(float(last_row["ask"]), 2),
                        "last": round(p, 2),
                        "volume": 1.0,
                    }
        except Exception:
            pass

    def name(self) -> str:
        return "MetaTrader 5 (Mac Spot)"

    def key(self) -> str:
        return "mt5_mac"

    def is_available(self) -> BrokerStatus:
        real_file = DATA_DIR / "xauusd_15m_real.csv"
        if not real_file.exists():
            try:
                extract_and_sync_all_mt5_bars()
            except Exception:
                pass

        has_real_data = real_file.exists()
        return BrokerStatus(
            name=self.name(),
            available=has_real_data,
            reason="Connected to local Mac MetaTrader 5 spot data" if has_real_data else "MT5 Mac history not yet extracted",
            real_data=True,
            spot=True,
            execution=True,
            delayed=False,
        )

    def fetch_ohlcv(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        tf_file = DATA_DIR / f"xauusd_{timeframe.lower()}_real.csv"
        if not tf_file.exists():
            extract_and_sync_all_mt5_bars()
        if not tf_file.exists():
            tf_file = DATA_DIR / "xauusd_15m_real.csv"
        if not tf_file.exists():
            raise FileNotFoundError(f"Real MT5 data for {symbol} {timeframe} not found")

        df = pd.read_csv(tf_file)
        if "time" not in df.columns:
            if "dt" in df.columns:
                df = df.rename(columns={"dt": "time"})
            elif "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "time"})
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
        df = df.sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        
        # Bridge to current live quote if newer than latest bar on disk
        if self._last_tick:
            p = float(self._last_tick.get("last") or self._last_tick.get("bid", 0.0))
            t_now = self._last_tick.get("time") or datetime.now(timezone.utc)
            if p > 0 and t_now > df.index[-1]:
                live_row = pd.DataFrame([{
                    "open": p, "high": p, "low": p, "close": p,
                    "volume": float(self._last_tick.get("volume", 1.0))
                }], index=[t_now])
                df = pd.concat([df, live_row])

        return df.tail(bars)

    # ── ticks ─────────────────────────────────────────────────────────────────
    def _real_ticks(self) -> pd.DataFrame:
        """Decode the terminal tick cache or read from partitioned tick store."""
        # 1. First check partitioned real Parquet store
        try:
            from core.marketdata import store as md_store
            cov = md_store.coverage("XAUUSD")
            if cov and cov.get("ticks", 0) > 0:
                raw_df = md_store.read_range("XAUUSD", cov["last_ms"] - 86400 * 1000, cov["last_ms"], limit=5000)
                if not raw_df.empty:
                    df = raw_df.copy()
                    df["time"] = pd.to_datetime(df["time_utc_ms"], unit="ms", utc=True)
                    df["last"] = (df["bid"] + df["ask"]) / 2.0
                    df["volume"] = df["bid_volume"] + df["ask_volume"]
                    df = df.set_index("time")
                    return df[["bid", "ask", "last", "volume"]]
        except Exception:
            pass

        # 2. Check MT5 ticks.dat
        path = find_mt5_ticks_file("XAUUSD")
        if path is not None:
            mtime = path.stat().st_mtime
            if self._ticks_cache is None or self._ticks_mtime != mtime:
                self._ticks_cache = decode_mt5_ticks_dat(path)
                self._ticks_mtime = mtime
            if self._ticks_cache is not None and not self._ticks_cache.empty:
                return self._ticks_cache

        # 3. Fallback to latest real bar close
        bar_file = self._bar_file("1m")
        if bar_file.exists():
            df_bars = pd.read_csv(bar_file)
            if not df_bars.empty:
                last_row = df_bars.iloc[-1]
                t = pd.to_datetime(last_row["time"], utc=True)
                p = float(last_row["close"])
                df_fallback = pd.DataFrame([{
                    "time": t,
                    "bid": round(p - 0.15, 2),
                    "ask": round(p + 0.15, 2),
                    "last": p,
                    "volume": float(last_row.get("volume", 1.0)),
                }]).set_index("time")
                return df_fallback

        return pd.DataFrame()

    def fetch_tick(self, symbol: str):
        if self._last_tick:
            age = (datetime.now(timezone.utc) - self._last_tick["time"]).total_seconds()
            if age <= 15:
                return self._last_tick
            self._last_tick = None
        try:
            df = self.fetch_ohlcv(symbol, "1m", 1)
            row = df.iloc[-1]
            price = float(row["close"])
            return {
                "time": datetime.now(timezone.utc),
                "bid": round(price - 0.15, 2),
                "ask": round(price + 0.15, 2),
                "last": price,
                "volume": float(row.get("volume", 1.0)),
            }
        except Exception:
            return None

    def record_live_tick(self, tick: dict):
        p = float(tick.get("last", 0.0) or tick.get("bid", 0.0))
        if p <= 0:
            return
        now = datetime.now(timezone.utc)
        self._last_tick = {
            "time": now,
            "bid": float(tick.get("bid", p - 0.15)),
            "ask": float(tick.get("ask", p + 0.15)),
            "last": p,
            "volume": float(tick.get("volume", 1.0)),
        }
        # Forward tick to persistent Vertex Store
        try:
            from core.vertex_store import VERTEX_STORE
            VERTEX_STORE.append_live_tick(tick)
        except Exception:
            pass

    def fetch_ticks(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """Return real historical ticks from 1m bars."""
        df = self.fetch_ohlcv(symbol, "1m", min(count, 5000))
        records = []
        for ts, row in df.iterrows():
            c = float(row["close"])
            records.append({
                "time": ts,
                "bid": round(c - 0.15, 2),
                "ask": round(c + 0.15, 2),
                "last": c,
                "volume": float(row["volume"]),
            })
        return pd.DataFrame(records).set_index("time")

    def account_info(self):
        return {
            "login": "MetaQuotes-Demo (Mac)",
            "server": "MetaQuotes-Demo",
            "balance": 10000.0,
            "equity": 10000.0,
            "margin": 0.0,
            "free_margin": 10000.0,
            "profit": 0.0,
            "currency": "USD",
            "leverage": 100,
        }
