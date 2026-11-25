"""MetaTrader5 connector - runs only on Windows with a locally installed terminal.

The platform uses this automatically when available. On macOS/Linux it reports
"MT5 not available" and the DataManager falls back to Yahoo Finance (real
gold futures) without the user touching anything.

MT5 setup (one-time, on the Windows machine):
  1. Install MetaTrader5 terminal from your broker (Exness/IC Markets/etc.)
  2. Log in to your account in the terminal
  3. pip install MetaTrader5
  4. That's it - the platform auto-connects on startup
"""
import os
import platform
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from core.broker import DataProvider, BrokerStatus

load_dotenv()

# MT5 is Windows-only. Import lazily so the module loads on macOS/Linux.
_mt5 = None
_MT5_AVAILABLE = False
try:
    import MetaTrader5 as _mt5
    _MT5_AVAILABLE = True
except ImportError:
    _mt5 = None


# timeframe string -> mt5 constant
TF_MAP = {
    "1m": "TIMEFRAME_M1",
    "2m": "TIMEFRAME_M2",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
    "1w": "TIMEFRAME_W1",
    "1mo": "TIMEFRAME_MN1",
}


class MT5Broker(DataProvider):
    """Full MT5 connector: real spot XAUUSD data + execution."""

    def __init__(self, login=None, password=None, server=None, path=None):
        self.login = login or os.getenv("MT5_LOGIN")
        self.password = password or os.getenv("MT5_PASSWORD")
        self.server = server or os.getenv("MT5_SERVER")
        self.path = path or os.getenv("MT5_PATH")
        self._connected = False
        self._last_error = None

    def name(self) -> str:
        return "MetaTrader5"

    def is_available(self) -> BrokerStatus:
        # Allow override: if credentials are set and we're not on Windows,
        # still try to connect (e.g., via Wine, VM, or remote setup)
        force_try = bool(self.login and self.password and self.server)
        if platform.system() != "Windows" and not force_try:
            return BrokerStatus(
                name="MetaTrader5",
                available=False,
                reason="MT5 runs only on Windows. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env to try anyway.",
                real_data=True,
                spot=True,
                execution=True,
            )
        if not _MT5_AVAILABLE:
            return BrokerStatus(
                name="MetaTrader5",
                available=False,
                reason="MetaTrader5 Python package not installed - run: pip install MetaTrader5",
                real_data=True,
                spot=True,
                execution=True,
            )
        try:
            ok = _mt5.initialize(
                path=self.path,
                login=int(self.login) if self.login else None,
                password=self.password,
                server=self.server,
            )
            if not ok:
                err = _mt5.last_error()
                return BrokerStatus(
                    name="MetaTrader5",
                    available=False,
                    reason=f"MT5 initialize failed: {err}. Is the terminal running and logged in?",
                    real_data=True,
                    spot=True,
                    execution=True,
                )
            info = _mt5.terminal_info()
            if info is None:
                return BrokerStatus(
                    name="MetaTrader5",
                    available=False,
                    reason="MT5 terminal not responding",
                    real_data=True,
                    spot=True,
                    execution=True,
                )
            account = _mt5.account_info()
            execution = bool(
                getattr(info, "trade_allowed", False)
                and not getattr(info, "tradeapi_disabled", False)
                and account is not None
                and getattr(account, "trade_allowed", False)
            )
            self._connected = True
            return BrokerStatus(
                name="MetaTrader5",
                available=True,
                reason=f"connected (build {info.build}, company: {info.company})",
                real_data=True,
                spot=True,
                execution=execution,
                delayed=False,
            )
        except Exception as e:
            return BrokerStatus(
                name="MetaTrader5",
                available=False,
                reason=f"MT5 error: {e}",
                real_data=True,
                spot=True,
                execution=True,
            )

    def _ensure_connected(self):
        if not self._connected:
            st = self.is_available()
            if not st.available:
                raise RuntimeError(st.reason)

    def _resolve_symbol(self, requested: str) -> str:
        """Resolve broker suffixes such as XAUUSDm, XAUUSD.a or GOLD."""
        self._ensure_connected()
        if _mt5.symbol_info(requested) is not None:
            actual = requested
        else:
            requested_upper = requested.upper()
            aliases = [requested_upper]
            if requested_upper == "XAUUSD":
                aliases.extend(["GOLD", "XAU"])
            symbols = _mt5.symbols_get() or ()
            candidates = []
            for symbol in symbols:
                name_upper = symbol.name.upper()
                score = next(
                    (len(alias) for alias in aliases if name_upper == alias or name_upper.startswith(alias)),
                    0,
                )
                if score:
                    candidates.append((score, -len(symbol.name), symbol.name))
            if not candidates:
                raise RuntimeError(f"MT5 symbol not found: {requested}")
            actual = max(candidates)[2]
        if not _mt5.symbol_select(actual, True):
            raise RuntimeError(f"MT5 could not select {actual}: {_mt5.last_error()}")
        return actual

    def fetch_ticks(self, symbol: str = "XAUUSD", count: int = 1000) -> pd.DataFrame:
        """Fetch recent ticks from MT5. Returns tick-level data."""
        self._ensure_connected()
        actual = self._resolve_symbol(symbol)
        ticks = _mt5.copy_ticks_from(
            actual,
            datetime.now(timezone.utc) - timedelta(hours=1),
            count,
            _mt5.COPY_TICKS_ALL,
        )
        if ticks is None or len(ticks) == 0:
            ticks = _mt5.copy_ticks_range(
                actual,
                datetime.now(timezone.utc) - timedelta(days=1),
                datetime.now(timezone.utc),
                _mt5.COPY_TICKS_ALL,
            )
            if ticks is not None and len(ticks) > count:
                ticks = ticks[-count:]
        if ticks is None or len(ticks) == 0:
            raise RuntimeError(f"MT5 returned no ticks for {actual}: {_mt5.last_error()}")
        df = pd.DataFrame(ticks)
        if "time_msc" in df.columns:
            df["time"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        else:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        # build OHLCV-like from ticks (1 tick = 1 row with bid/ask)
        result = pd.DataFrame({
            "bid": df["bid"],
            "ask": df["ask"],
            "last": df["last"],
            "volume": df["volume"],
        })
        return result

    def fetch_ohlcv(self, symbol: str = "XAUUSD", timeframe: str = "15m", bars: int = 5000) -> pd.DataFrame:
        self._ensure_connected()
        actual = self._resolve_symbol(symbol)
        tf_const = getattr(_mt5, TF_MAP.get(timeframe, "TIMEFRAME_M15"))
        rates = _mt5.copy_rates_from_pos(actual, tf_const, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 returned no data for {actual} {timeframe}: {_mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        return df.astype(float)

    def fetch_tick(self, symbol: str = "XAUUSD"):
        self._ensure_connected()
        actual = self._resolve_symbol(symbol)
        tick = _mt5.symbol_info_tick(actual)
        if tick is None:
            return None
        time_msc = getattr(tick, "time_msc", 0)
        return {
            "time": pd.Timestamp(time_msc, unit="ms", tz="UTC") if time_msc else pd.Timestamp(tick.time, unit="s", tz="UTC"),
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
        }

    def symbol_info(self, symbol: str = "XAUUSD"):
        self._ensure_connected()
        actual = self._resolve_symbol(symbol)
        info = _mt5.symbol_info(actual)
        if info is None:
            return None
        return {
            "name": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }

    def account_info(self):
        self._ensure_connected()
        a = _mt5.account_info()
        if a is None:
            return None
        return {
            "login": a.login,
            "balance": a.balance,
            "equity": a.equity,
            "margin": a.margin,
            "free_margin": a.margin_free,
            "profit": a.profit,
            "currency": a.currency,
            "leverage": a.leverage,
            "server": a.server,
        }

    def positions(self, symbol: str = None):
        self._ensure_connected()
        if symbol:
            pos = _mt5.positions_get(symbol=symbol)
        else:
            pos = _mt5.positions_get()
        if pos is None:
            return []
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "time": pd.Timestamp(p.time, unit="s", tz="UTC"),
            }
            for p in pos
        ]

    def place_market_order(self, symbol: str, side: str, volume: float,
                           sl: float = 0.0, tp: float = 0.0, comment: str = "AI-Quant"):
        """Place a market order. side = 'BUY' or 'SELL'."""
        self._ensure_connected()
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        actual = self._resolve_symbol(symbol)
        info = _mt5.symbol_info(actual)
        if info is None:
            raise RuntimeError(f"symbol {actual} not found")

        step = float(info.volume_step or 0.01)
        volume = round(round(float(volume) / step) * step, 8)
        volume = min(max(volume, float(info.volume_min)), float(info.volume_max))

        tick = _mt5.symbol_info_tick(actual)
        if tick is None:
            raise RuntimeError(f"no tick for {actual}")

        order_type = _mt5.ORDER_TYPE_BUY if side == "BUY" else _mt5.ORDER_TYPE_SELL
        price = tick.ask if side == "BUY" else tick.bid
        filling = info.filling_mode
        valid_filling = {
            _mt5.ORDER_FILLING_FOK,
            _mt5.ORDER_FILLING_IOC,
            _mt5.ORDER_FILLING_RETURN,
        }
        if filling not in valid_filling:
            filling = _mt5.ORDER_FILLING_IOC

        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": actual,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 20261117,
            "comment": comment,
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        checked = _mt5.order_check(request)
        if checked is None:
            raise RuntimeError(f"order_check returned None: {_mt5.last_error()}")
        if checked.retcode not in (0, _mt5.TRADE_RETCODE_DONE):
            raise RuntimeError(f"order_check failed: retcode={checked.retcode} comment={checked.comment}")
        result = _mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send returned None: {_mt5.last_error()}")
        if result.retcode != _mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"order failed: retcode={result.retcode} comment={result.comment}")
        return {
            "ticket": result.order,
            "volume": result.volume,
            "price": result.price,
            "retcode": result.retcode,
        }

    def close_position(self, ticket: int):
        self._ensure_connected()
        pos = _mt5.positions_get(ticket=ticket)
        if not pos:
            raise RuntimeError(f"position {ticket} not found")
        p = pos[0]
        side = "SELL" if p.type == 0 else "BUY"
        info = _mt5.symbol_info(p.symbol)
        tick = _mt5.symbol_info_tick(p.symbol)
        if info is None or tick is None:
            raise RuntimeError(f"cannot read symbol/tick for {p.symbol}")
        price = tick.bid if side == "SELL" else tick.ask
        filling = info.filling_mode
        if filling not in {
            _mt5.ORDER_FILLING_FOK,
            _mt5.ORDER_FILLING_IOC,
            _mt5.ORDER_FILLING_RETURN,
        }:
            filling = _mt5.ORDER_FILLING_IOC
        request = {
            "action": _mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": _mt5.ORDER_TYPE_SELL if side == "SELL" else _mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 20261117,
            "comment": "AI-Quant close",
            "type_time": _mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        checked = _mt5.order_check(request)
        if checked is None or checked.retcode not in (0, _mt5.TRADE_RETCODE_DONE):
            raise RuntimeError(f"close order_check failed: {checked}")
        result = _mt5.order_send(request)
        if result is None or result.retcode != _mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"close failed: {result}")
        return {"ticket": result.order, "retcode": result.retcode}

    def update_position_sl_tp(self, ticket: int, sl: float, tp: float) -> dict:
        """Update stop loss / take profit on an open position (SLTP action)."""
        self._ensure_connected()
        pos = _mt5.positions_get(ticket=ticket)
        if not pos:
            raise RuntimeError(f"position {ticket} not found")
        p = pos[0]
        info = _mt5.symbol_info(p.symbol)
        tick = _mt5.symbol_info_tick(p.symbol)
        if info is None or tick is None:
            raise RuntimeError(f"cannot read symbol/tick for {p.symbol}")
        request = {
            "action": _mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        checked = _mt5.order_check(request)
        if checked is None or checked.retcode not in (0, _mt5.TRADE_RETCODE_DONE):
            raise RuntimeError(f"SL/TP order_check failed: {checked}")
        result = _mt5.order_send(request)
        if result is None or result.retcode != _mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"SL/TP update failed: {result}")
        return {"ticket": ticket, "retcode": result.retcode, "sl": sl, "tp": tp}

    def shutdown(self):
        if _mt5 is not None and self._connected:
            _mt5.shutdown()
            self._connected = False


# quick test when run directly
if __name__ == "__main__":
    b = MT5Broker()
    st = b.is_available()
    print("MT5 status:", st)
    if st.available:
        print("account:", b.account_info())
        print("symbol:", b.symbol_info("XAUUSD"))
        print("tick:", b.fetch_tick("XAUUSD"))
        df = b.fetch_ohlcv("XAUUSD", "15m", 10)
        print(df.tail(3))
        b.shutdown()
