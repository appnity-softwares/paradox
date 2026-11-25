"""Complete quantitative strategy library.
Includes classical rule-based strategies and Smart Money Concepts (SMC):
- EMACross (key: "ema_cross")
- MACDCross (key: "macd_cross")
- RSIReversal (key: "rsi_reversal")
- VWAPEMA (key: "vwap_ema")
- Supertrend (key: "supertrend")
- SessionBreakout (key: "session_breakout")
- BollingerReversion (key: "bollinger")
- FairValueGap (key: "smc_fvg")
- SMCOrderBlock (key: "smc_order_block")
- LiquiditySweep (key: "smc_liquidity_sweep")

Every strategy exposes:
  key     -> stable id used by CLI and dashboard
  warmup  -> bars needed before signals are valid (used by walk-forward buffering)
"""
import numpy as np
import pandas as pd

from strategies.base import StrategyConfig
from core.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    macd,
    parabolic_sar,
    rsi,
    sma,
    stochastic,
    supertrend,
    vwap_session,
)


class EMACross:
    """EMA fast/slow crossover trend-following."""
    key = "ema_cross"

    def __init__(self, cfg: StrategyConfig, fast: int = 20, slow: int = 50):
        self.cfg = cfg
        self.fast, self.slow = int(fast), int(slow)
        if self.fast < 2 or self.slow <= self.fast:
            raise ValueError("EMA requires 2 <= fast < slow")
        self.warmup = self.slow + 10
        cfg.name = f"EMA {self.fast}/{self.slow}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = np.sign(ema(df["close"], self.fast) - ema(df["close"], self.slow))
        sig.iloc[: self.warmup] = 0.0
        return sig


class MACDCross:
    """MACD line vs signal line."""
    key = "macd_cross"

    def __init__(self, cfg: StrategyConfig, fast: int = 12, slow: int = 26, signal: int = 9):
        self.cfg = cfg
        self.fast, self.slow, self.signal = int(fast), int(slow), int(signal)
        if self.fast < 2 or self.slow <= self.fast or self.signal < 2:
            raise ValueError("MACD requires 2 <= fast < slow and signal >= 2")
        self.warmup = self.slow + self.signal + 10
        cfg.name = f"MACD {self.fast}/{self.slow}/{self.signal}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        line, sl, _ = macd(df["close"], self.fast, self.slow, self.signal)
        sig = np.sign(line - sl)
        sig.iloc[: self.warmup] = 0.0
        return sig


class RSIReversal:
    """Mean reversion: buy when RSI exits oversold, short when it exits overbought.
    Flat again when the opposite extreme is reached."""
    key = "rsi_reversal"

    def __init__(self, cfg: StrategyConfig, length: int = 14, oversold: float = 30, overbought: float = 70):
        self.cfg = cfg
        self.length = int(length)
        self.os, self.ob = float(oversold), float(overbought)
        if self.length < 2 or not 0 < self.os < self.ob < 100:
            raise ValueError("RSI requires length >= 2 and 0 < oversold < overbought < 100")
        self.warmup = self.length * 3
        cfg.name = f"RSI {self.length} Rev"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.length).to_numpy(dtype=float)
        sig = np.zeros(len(r))
        state = 0.0
        for i in range(1, len(r)):
            prev, cur = r[i - 1], r[i]
            if np.isnan(prev) or np.isnan(cur):
                state = 0.0
            else:
                if state == 0.0:
                    if prev < self.os and cur >= self.os:
                        state = 1.0
                    elif prev > self.ob and cur <= self.ob:
                        state = -1.0
                elif state == 1.0 and cur >= self.ob:
                    state = 0.0
                elif state == -1.0 and cur <= self.os:
                    state = 0.0
            sig[i] = state
        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class VWAPEMA:
    """Notebook strategy (3-4 July page): Single EMA + VWAP filter.

    Trigger: price X EMA with VWAP filter.
    Long  = price above VWAP AND above EMA (optional: EMA rising).
    Short = price below VWAP AND below EMA (optional: EMA falling).
    """
    key = "vwap_ema"

    def __init__(self, cfg: StrategyConfig, ema_length: int = 20, slope: int = 1):
        self.cfg = cfg
        self.ema_length = int(ema_length)
        self.slope = bool(int(slope))
        if self.ema_length < 2:
            raise ValueError("EMA length must be at least 2")
        self.warmup = self.ema_length + 10
        cfg.name = f"VWAP+EMA {self.ema_length}" + ("+slope" if self.slope else "")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vwap = vwap_session(df)
        ev = ema(df["close"], self.ema_length)
        close = df["close"]

        long_ok = (close > vwap) & (close > ev)
        short_ok = (close < vwap) & (close < ev)
        if self.slope:
            e_diff = ev.diff()
            long_ok = long_ok & (e_diff > 0)
            short_ok = short_ok & (e_diff < 0)

        sig = pd.Series(0.0, index=df.index)
        sig[long_ok.fillna(False)] = 1.0
        sig[short_ok.fillna(False)] = -1.0
        sig.iloc[: self.warmup] = 0.0
        return sig


class Supertrend:
    """ATR trailing-stop trend following (notebook advanced list)."""
    key = "supertrend"

    def __init__(self, cfg: StrategyConfig, length: int = 10, mult: float = 3.0):
        self.cfg = cfg
        self.length = int(length)
        self.mult = float(mult)
        if self.length < 2 or self.mult <= 0:
            raise ValueError("Supertrend requires length >= 2 and multiplier > 0")
        self.warmup = self.length * 3
        cfg.name = f"Supertrend {self.length}x{self.mult:g}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return supertrend(df, self.length, self.mult)


class SessionBreakout:
    """Range breakout: Asia session high/low break, traded in London/NY.

    Notebook basis: "Only trades = Tokyo or London or NY" + Breakout strategy.
    """
    key = "session_breakout"

    def __init__(self, cfg: StrategyConfig, buffer: float = 0.0, range_end: int = 7):
        self.cfg = cfg
        self.buffer = float(buffer)
        self.range_end = int(range_end)   # UTC hour when Asia range completes
        if self.buffer < 0 or not 1 <= self.range_end <= 23:
            raise ValueError("Session breakout requires buffer >= 0 and range_end in 1..23")
        self.trade_end = 21               # stop opening new logic after NY close (UTC)
        self.warmup = 5
        cfg.name = f"Brk {self.range_end}h/{self.buffer:g}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        index = df.index
        hours = index.hour.to_numpy(dtype=int)
        day = index.normalize()
        close = df["close"].to_numpy(dtype=float)

        ahigh = (
            df["high"].where(hours < self.range_end).groupby(day).cummax().groupby(day).ffill()
        ).to_numpy(dtype=float)
        alow = (
            df["low"].where(hours < self.range_end).groupby(day).cummin().groupby(day).ffill()
        ).to_numpy(dtype=float)

        n = len(df)
        sig = np.zeros(n)
        state = 0.0
        prev_day = None
        for i in range(n):
            if day[i] != prev_day:
                state = 0.0
                prev_day = day[i]
            if hours[i] < self.range_end or hours[i] >= self.trade_end or np.isnan(ahigh[i]) or np.isnan(alow[i]):
                state = 0.0
            else:
                if state <= 0 and close[i] > ahigh[i] + self.buffer:
                    state = 1.0
                elif state >= 0 and close[i] < alow[i] - self.buffer:
                    state = -1.0
            sig[i] = state
        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class BollingerReversion:
    """Mean reversion from band touches, exit back at mid band."""
    key = "bollinger"

    def __init__(self, cfg: StrategyConfig, length: int = 20, num_std: float = 2.0):
        self.cfg = cfg
        self.length = int(length)
        self.num_std = float(num_std)
        if self.length < 2 or self.num_std <= 0:
            raise ValueError("Bollinger requires length >= 2 and num_std > 0")
        self.warmup = self.length + 10
        cfg.name = f"BB {self.length}/{self.num_std:g}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lower, mid, upper = bollinger(df["close"], self.length, self.num_std)
        lower = lower.to_numpy(dtype=float)
        mid = mid.to_numpy(dtype=float)
        upper = upper.to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        sig = np.zeros(len(close))
        state = 0.0
        for i in range(len(close)):
            if np.isnan(lower[i]):
                state = 0.0
            else:
                if state == 0.0:
                    if close[i] < lower[i]:
                        state = 1.0
                    elif close[i] > upper[i]:
                        state = -1.0
                elif state == 1.0 and close[i] >= mid[i]:
                    state = 0.0
                elif state == -1.0 and close[i] <= mid[i]:
                    state = 0.0
            sig[i] = state
        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class FairValueGap:
    """Smart Money Concepts (SMC) - Fair Value Gap (FVG) / Imbalance Strategy.
    
    Identifies 3-bar imbalance:
    - Bullish FVG: Low[i] > High[i-2] with min gap threshold
    - Bearish FVG: High[i] < Low[i-2] with min gap threshold
    Enters when subsequent price retests/mitigates the open gap in trend direction.
    """
    key = "smc_fvg"

    def __init__(self, cfg: StrategyConfig, min_gap_atr: float = 0.4, max_age: int = 25):
        self.cfg = cfg
        self.min_gap_atr = float(min_gap_atr)
        self.max_age = int(max_age)
        if self.min_gap_atr <= 0 or self.max_age < 3:
            raise ValueError("FVG requires min_gap_atr > 0 and max_age >= 3")
        self.warmup = 30
        cfg.name = f"SMC FVG {self.min_gap_atr:g}xATR"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        sig = np.zeros(n)
        if n < 4:
            return pd.Series(sig, index=df.index)

        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        o = df["open"].to_numpy(dtype=float)
        atr_vals = atr(df, 14).to_numpy(dtype=float)

        active_gaps = []
        state = 0.0

        for i in range(2, n):
            curr_atr = atr_vals[i] if not np.isnan(atr_vals[i]) and atr_vals[i] > 0 else 1.0
            min_gap_size = self.min_gap_atr * curr_atr

            # Bullish FVG: low[i] > high[i-2]
            if l[i] > h[i - 2] + min_gap_size and c[i - 1] > o[i - 1]:
                active_gaps.append({
                    "type": 1,
                    "bottom": h[i - 2],
                    "top": l[i],
                    "bar": i,
                })
            # Bearish FVG: high[i] < low[i-2]
            elif h[i] < l[i - 2] - min_gap_size and c[i - 1] < o[i - 1]:
                active_gaps.append({
                    "type": -1,
                    "bottom": h[i],
                    "top": l[i - 2],
                    "bar": i,
                })

            surviving_gaps = []
            signal_triggered = 0.0

            for g in active_gaps:
                if i - g["bar"] > self.max_age:
                    continue

                if g["type"] == 1:
                    if c[i] < g["bottom"]:
                        continue  # invalidated
                    if l[i] <= g["top"] and h[i] >= g["bottom"] and i > g["bar"]:
                        signal_triggered = 1.0
                        continue  # mitigated
                else:
                    if c[i] > g["top"]:
                        continue  # invalidated
                    if h[i] >= g["bottom"] and l[i] <= g["top"] and i > g["bar"]:
                        signal_triggered = -1.0
                        continue  # mitigated

                surviving_gaps.append(g)

            active_gaps = surviving_gaps[-15:]

            if signal_triggered != 0.0:
                state = signal_triggered
            sig[i] = state

        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class SMCOrderBlock:
    """Smart Money Concepts (SMC) - Institutional Order Block (OB) Strategy.
    
    Identifies high-probability institutional footprints:
    - Bullish OB: The last down candle before an impulsive bullish expansion that breaks structure (BOS).
    - Bearish OB: The last up candle before an impulsive bearish expansion that breaks structure (BOS).
    Enters on pullback retest/mitigation of the Order Block zone.
    """
    key = "smc_order_block"

    def __init__(self, cfg: StrategyConfig, swing_len: int = 5, impulse_mult: float = 1.5, max_bars: int = 30):
        self.cfg = cfg
        self.swing_len = int(swing_len)
        self.impulse_mult = float(impulse_mult)
        self.max_bars = int(max_bars)
        if self.swing_len < 2 or self.impulse_mult <= 0 or self.max_bars < 5:
            raise ValueError("SMCOrderBlock invalid parameters")
        self.warmup = self.swing_len * 3 + 10
        cfg.name = f"SMC OB {self.swing_len}s/{self.impulse_mult:g}x"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        sig = np.zeros(n)
        if n < self.warmup:
            return pd.Series(sig, index=df.index)

        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        o = df["open"].to_numpy(dtype=float)
        atr_vals = atr(df, 14).to_numpy(dtype=float)

        active_obs = []
        state = 0.0

        for i in range(self.swing_len * 2, n):
            curr_atr = atr_vals[i] if not np.isnan(atr_vals[i]) and atr_vals[i] > 0 else 1.0

            prev_swing_high = np.max(h[i - self.swing_len - 3: i - 2])
            prev_swing_low = np.min(l[i - self.swing_len - 3: i - 2])

            # Bullish expansion: strong upward move breaking swing high
            if c[i] > prev_swing_high and (c[i] - l[i - 2]) > (self.impulse_mult * curr_atr):
                for back in range(1, 4):
                    ob_idx = i - back
                    if c[ob_idx] < o[ob_idx]:
                        active_obs.append({
                            "type": 1,
                            "zone_low": l[ob_idx],
                            "zone_high": max(o[ob_idx], c[ob_idx]),
                            "bar": i,
                        })
                        break

            # Bearish expansion: strong downward move breaking swing low
            elif c[i] < prev_swing_low and (h[i - 2] - c[i]) > (self.impulse_mult * curr_atr):
                for back in range(1, 4):
                    ob_idx = i - back
                    if c[ob_idx] > o[ob_idx]:
                        active_obs.append({
                            "type": -1,
                            "zone_low": min(o[ob_idx], c[ob_idx]),
                            "zone_high": h[ob_idx],
                            "bar": i,
                        })
                        break

            surviving_obs = []
            signal_triggered = 0.0

            for ob in active_obs:
                if i - ob["bar"] > self.max_bars:
                    continue

                if ob["type"] == 1:
                    if c[i] < ob["zone_low"]:
                        continue
                    if l[i] <= ob["zone_high"] and h[i] >= ob["zone_low"] and i > ob["bar"]:
                        signal_triggered = 1.0
                        continue
                else:
                    if c[i] > ob["zone_high"]:
                        continue
                    if h[i] >= ob["zone_low"] and l[i] <= ob["zone_high"] and i > ob["bar"]:
                        signal_triggered = -1.0
                        continue

                surviving_obs.append(ob)

            active_obs = surviving_obs[-10:]

            if signal_triggered != 0.0:
                state = signal_triggered
            sig[i] = state

        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class LiquiditySweep:
    """Smart Money Concepts (SMC) - Liquidity Sweep / Turtle Soup Strategy.
    
    Detects false breakouts (stop hunts) above key highs or below key lows:
    - Bearish Sweep: price penetrates previous swing high (taking buy-side liquidity),
      then immediately rejects and closes back below the high with upper wick.
    - Bullish Sweep: price penetrates previous swing low (taking sell-side liquidity),
      then immediately rejects and closes back above the low with lower wick.
    """
    key = "smc_liquidity_sweep"

    def __init__(self, cfg: StrategyConfig, lookback: int = 20, sweep_buffer: float = 0.2, hold_bars: int = 4):
        self.cfg = cfg
        self.lookback = int(lookback)
        self.sweep_buffer = float(sweep_buffer)
        self.hold_bars = int(hold_bars)
        if self.lookback < 5 or self.sweep_buffer < 0 or self.hold_bars < 1:
            raise ValueError("LiquiditySweep invalid parameters")
        self.warmup = self.lookback + 5
        cfg.name = f"SMC Sweep {self.lookback}b/{self.sweep_buffer:g}pt"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        sig = np.zeros(n)
        if n < self.warmup:
            return pd.Series(sig, index=df.index)

        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)

        state = 0.0
        bars_held = 0

        for i in range(self.lookback, n):
            prior_high = np.max(h[i - self.lookback: i])
            prior_low = np.min(l[i - self.lookback: i])

            curr_h = h[i]
            curr_l = l[i]
            curr_c = c[i]

            if curr_h >= prior_high + self.sweep_buffer and curr_c < prior_high:
                state = -1.0
                bars_held = 0
            elif curr_l <= prior_low - self.sweep_buffer and curr_c > prior_low:
                state = 1.0
                bars_held = 0
            elif state != 0.0:
                bars_held += 1
                if bars_held >= self.hold_bars:
                    state = 0.0

            sig[i] = state

        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


# ---------------------------------------------------------------------------
# Phase 1C additions (Master Prompt 6.5): trend / momentum / mean-reversion /
# SMC break of structure. Same contract as the strategies above: key, warmup,
# validated params, generate_signals -> {+1, -1, 0} Series.
# ---------------------------------------------------------------------------


class PSARTrend:
    """Parabolic SAR trend following: long while close > SAR, short below.

    Params: af_start/af_step/af_max are the standard Wilder acceleration
    factors. Signal is the SAR side of the close, 0 during warmup.
    """
    key = "psar_trend"
    PARAM_SCHEMA = {"af_start": (0.01, 0.05), "af_step": (0.01, 0.05), "af_max": (0.1, 0.3)}

    def __init__(self, cfg: StrategyConfig, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
        self.cfg = cfg
        self.af_start, self.af_step, self.af_max = float(af_start), float(af_step), float(af_max)
        if not 0 < self.af_start <= self.af_step <= self.af_max:
            raise ValueError("PSAR requires 0 < af_start <= af_step <= af_max")
        self.warmup = 10
        cfg.name = f"PSAR {self.af_step:g}/{self.af_max:g}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sar = parabolic_sar(df, self.af_start, self.af_step, self.af_max)
        close = df["close"].to_numpy(dtype=float)
        sig = np.where(close > sar.to_numpy(dtype=float), 1.0, -1.0)
        sig[np.isnan(sar.to_numpy(dtype=float))] = 0.0
        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class StochasticCross:
    """Stochastic %K/%D cross filtered to the extreme zones.

    Long when %K crosses above %D while both are below `oversold`;
    short when %K crosses below %D while both are above `overbought`.
    The state holds until the opposite signal fires (persistent, like the
    other trend strategies in this library).
    """
    key = "stoch_cross"
    PARAM_SCHEMA = {"length": (7, 21), "oversold": (15, 30), "overbought": (70, 85)}

    def __init__(self, cfg: StrategyConfig, length: int = 14, oversold: float = 25.0, overbought: float = 75.0):
        self.cfg = cfg
        self.length = int(length)
        self.oversold = float(oversold)
        self.overbought = float(overbought)
        if self.length < 3 or not 0 <= self.oversold < self.overbought <= 100:
            raise ValueError("Stochastic requires length >= 3 and oversold < overbought")
        self.warmup = self.length + 10
        cfg.name = f"Stoch {self.length}/{self.oversold:g}-{self.overbought:g}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        k, d = stochastic(df, self.length, k_smooth=3)
        k = k.to_numpy(dtype=float)
        d = d.to_numpy(dtype=float)
        n = len(df)
        sig = np.zeros(n)
        state = 0.0
        for i in range(1, n):
            cross_up = k[i - 1] <= d[i - 1] and k[i] > d[i]
            cross_dn = k[i - 1] >= d[i - 1] and k[i] < d[i]
            if cross_up and max(k[i], d[i]) < self.oversold:
                state = 1.0
            elif cross_dn and min(k[i], d[i]) > self.overbought:
                state = -1.0
            sig[i] = state
        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class ZScoreReversion:
    """Mean reversion on the z-score of close vs its SMA.

    Long when z <= -entry_z (price stretched below the mean), short when
    z >= +entry_z, flat again once |z| <= exit_z.
    """
    key = "zscore_reversion"
    PARAM_SCHEMA = {"length": (20, 60), "entry_z": (1.5, 3.0), "exit_z": (0.0, 1.0)}

    def __init__(self, cfg: StrategyConfig, length: int = 30, entry_z: float = 2.0, exit_z: float = 0.5):
        self.cfg = cfg
        self.length = int(length)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        if self.length < 5 or self.entry_z <= self.exit_z or self.exit_z < 0:
            raise ValueError("ZScore requires length >= 5 and exit_z < entry_z")
        self.warmup = self.length + 5
        cfg.name = f"ZScore {self.length}/{self.entry_z:g}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mean = sma(close, self.length)
        std = close.rolling(self.length).std()
        z = (close - mean) / (std + 1e-12)
        z = z.to_numpy(dtype=float)
        n = len(df)
        sig = np.zeros(n)
        state = 0.0
        for i in range(n):
            if not np.isfinite(z[i]):
                sig[i] = state
                continue
            if state == 0.0:
                if z[i] <= -self.entry_z:
                    state = 1.0
                elif z[i] >= self.entry_z:
                    state = -1.0
            elif state == 1.0 and z[i] >= -self.exit_z:
                state = 0.0
            elif state == -1.0 and z[i] <= self.exit_z:
                state = 0.0
            sig[i] = state
        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)


class ADXTrend:
    """EMA direction gated by ADX trend strength (Part 6.5 regime filter).

    Long when EMA(fast) > EMA(slow) AND ADX >= threshold; short on the
    mirror; flat when the trend is too weak.
    """
    key = "adx_trend"
    PARAM_SCHEMA = {"fast": (8, 30), "slow": (40, 120), "adx_length": (10, 20), "adx_threshold": (15, 30)}

    def __init__(self, cfg: StrategyConfig, fast: int = 20, slow: int = 60,
                 adx_length: int = 14, adx_threshold: float = 20.0):
        self.cfg = cfg
        self.fast, self.slow = int(fast), int(slow)
        self.adx_length = int(adx_length)
        self.adx_threshold = float(adx_threshold)
        if self.fast >= self.slow or self.adx_length < 5 or not 0 < self.adx_threshold < 60:
            raise ValueError("ADXTrend requires fast < slow, adx_length >= 5, 0 < threshold < 60")
        self.warmup = max(self.slow, self.adx_length * 3) + 5
        cfg.name = f"ADX {self.fast}/{self.slow}@{self.adx_threshold:g}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        direction = np.sign(ema(df["close"], self.fast) - ema(df["close"], self.slow))
        strength = adx(df, self.adx_length)
        sig = direction.where(strength >= self.adx_threshold, 0.0)
        sig.iloc[: self.warmup] = 0.0
        return sig.fillna(0.0)


class BreakOfStructure:
    """SMC Break of Structure — formal rules in docs/smc_definitions.md §4.

    Fractal swings (confirmed after `swing_len` bars) define structure;
    a CLOSE through a confirmed swing level breaks it. Each level breaks
    once; the fired direction persists until the opposite break.
    """
    key = "smc_bos"
    PARAM_SCHEMA = {"swing_len": (3, 10), "max_levels": (10, 40)}

    def __init__(self, cfg: StrategyConfig, swing_len: int = 5, max_levels: int = 20):
        self.cfg = cfg
        self.swing_len = int(swing_len)
        self.max_levels = int(max_levels)
        if self.swing_len < 2 or self.max_levels < 2:
            raise ValueError("BOS requires swing_len >= 2 and max_levels >= 2")
        self.warmup = self.swing_len * 2 + 5
        cfg.name = f"SMC BOS {self.swing_len}s"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        n = len(df)
        sig = np.zeros(n)
        state = 0.0
        swing_highs: list[float] = []   # confirmed swing high levels
        swing_lows: list[float] = []
        w = self.swing_len
        for i in range(n):
            # Confirm the swing that completed at bar i - w (window fully past).
            s = i - w
            if s >= w:
                window_h = h[s - w: s + w + 1]
                window_l = l[s - w: s + w + 1]
                if h[s] >= window_h.max() and h[s] > np.max(np.delete(window_h, w)):
                    swing_highs.append(float(h[s]))
                    swing_highs = swing_highs[-self.max_levels:]
                if l[s] <= window_l.min() and l[s] < np.min(np.delete(window_l, w)):
                    swing_lows.append(float(l[s]))
                    swing_lows = swing_lows[-self.max_levels:]
            # Break checks against the most recent confirmed levels.
            if swing_highs and c[i] > swing_highs[-1]:
                if state <= 0:
                    state = 1.0
                swing_highs.pop()   # level consumed
            elif swing_lows and c[i] < swing_lows[-1]:
                if state >= 0:
                    state = -1.0
                swing_lows.pop()
            sig[i] = state
        sig[: self.warmup] = 0.0
        return pd.Series(sig, index=df.index)
