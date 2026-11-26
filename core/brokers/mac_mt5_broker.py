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


def _spread_for(symbol: str, price: float) -> tuple[float, int]:
    """(half_spread, digits) per instrument. Gold keeps the legacy 0.30
    total spread; FX pairs use ~1 pip total, JPY quotes 3 digits."""
    s = symbol.upper()
    if s in ("XAUUSD", "XAGUSD"):
        return (0.15, 2)
    if "JPY" in s:
        return (0.005, 3)
    if s.startswith(("XAU", "XAG")):
        return (0.15, 2)
    return (max(price * 0.00005, 0.00001), 5)


class MacMT5Broker(DataProvider):
    """Native macOS MetaTrader 5 Provider: real spot XAUUSD data + execution bridge."""

    def __init__(self):
        self._last_tick = None
        self._last_tick_symbol = "XAUUSD"
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
                    self._last_tick_symbol = "XAUUSD"
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
        slug = "xauusd" if symbol == "XAUUSD" else symbol.lower()
        tf_file = DATA_DIR / f"{slug}_{timeframe.lower()}_real.csv"
        if not tf_file.exists():
            extract_and_sync_all_mt5_bars(symbol)
        if not tf_file.exists() and symbol == "XAUUSD":
            # legacy fallback: gold only, never another symbol's price
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
        
        # Bridge to current live quote if newer than latest bar on disk —
        # only when the tick actually belongs to this symbol. The live tick
        # must become the CURRENT FORMING BAR on the timeframe grid, never a
        # phantom bar stamped with wall-clock time: a misaligned last bar
        # poisons session filters, tick-replay windows and "last closed bar"
        # reporting downstream.
        if self._last_tick and self._last_tick_symbol == symbol.upper():
            p = float(self._last_tick.get("last") or self._last_tick.get("bid", 0.0))
            t_now = self._last_tick.get("time") or datetime.now(timezone.utc)
            tf_minutes = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
                          "1h": 60, "4h": 240, "1d": 1440}.get(timeframe.lower(), 15)
            t_bar = pd.Timestamp(t_now).floor(f"{tf_minutes}min")
            if p > 0 and t_bar > df.index[-1]:
                # gap between disk data and now: append the forming bar
                live_row = pd.DataFrame([{
                    "open": p, "high": p, "low": p, "close": p,
                    "volume": float(self._last_tick.get("volume", 1.0))
                }], index=[t_bar])
                df = pd.concat([df, live_row])
            elif p > 0 and t_bar == df.index[-1]:
                # merge the live tick into the forming bar on disk
                last = df.index[-1]
                df.iloc[-1, df.columns.get_loc("high")] = max(float(df.iloc[-1]["high"]), p)
                df.iloc[-1, df.columns.get_loc("low")] = min(float(df.iloc[-1]["low"]), p)
                df.iloc[-1, df.columns.get_loc("close")] = p

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
        if (self._last_tick and self._last_tick_symbol == symbol.upper()
                and self._last_tick.get("bid") and self._last_tick.get("ask")):
            # Only a tick with a real quote side serves as a quote; a
            # last-price-only tick falls through to the bar-derived path.
            age = (datetime.now(timezone.utc) - self._last_tick["time"]).total_seconds()
            if age <= 15:
                return self._last_tick
            self._last_tick = None
        elif self._last_tick:
            self._last_tick = None  # cached tick is for a different symbol
        try:
            # H1-only symbols (FX on this terminal) have no 1m file; fall
            # through the timeframes until one has local data.
            df = None
            for tf in ("1m", "15m", "1h", "1d"):
                try:
                    df = self.fetch_ohlcv(symbol, tf, 1)
                    break
                except FileNotFoundError:
                    continue
            if df is None or df.empty:
                return None
            row = df.iloc[-1]
            price = float(row["close"])
            half, digits = _spread_for(symbol, price)
            self._last_tick_symbol = symbol.upper()
            return {
                "time": datetime.now(timezone.utc),
                "bid": round(price - half, digits),
                "ask": round(price + half, digits),
                "last": price,
                "volume": float(row.get("volume", 1.0)),
            }
        except Exception:
            return None

    def record_live_tick(self, tick: dict):
        """Store an EA tick with the EA's OWN timestamp.

        No re-stamping: a tick that arrived late keeps its original time so
        staleness stays measurable. Missing bid/ask are NOT invented here —
        the tick is stored with last only and the quote side stays empty.
        """
        p = float(tick.get("last", 0.0) or tick.get("bid", 0.0))
        if p <= 0:
            return
        time_ms = tick.get("time_ms")
        if time_ms:
            try:
                stamp = datetime.fromtimestamp(int(time_ms) / 1000, tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                stamp = datetime.now(timezone.utc)
        else:
            stamp = datetime.now(timezone.utc)
        bid = tick.get("bid")
        ask = tick.get("ask")
        self._last_tick = {
            "time": stamp,
            "bid": float(bid) if bid else None,
            "ask": float(ask) if ask else None,
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
