"""Technical indicators - pure pandas, no external TA library."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out[loss == 0] = 100.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, length: int = 20, num_std: float = 2.0):
    mid = sma(close, length)
    sd = close.rolling(length).std()
    return mid - num_std * sd, mid, mid + num_std * sd


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def stochastic(df: pd.DataFrame, length: int = 14, k_smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K / %D oscillator. Returns (%K, %D)."""
    low = df["low"].rolling(length).min()
    high = df["high"].rolling(length).max()
    k_line = 100.0 * (df["close"] - low) / (high - low + 1e-12)
    d_line = k_line.ewm(span=max(1, k_smooth), adjust=False).mean()
    return k_line, d_line


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP (resets every UTC day). Typical price = HLC3 (notebook default)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    if isinstance(df.index, pd.DatetimeIndex):
        key = df.index.normalize()
    else:
        key = pd.Series(np.arange(len(df)) // 96, index=df.index)
    pv = (tp * df["volume"]).groupby(key).cumsum()
    vv = df["volume"].groupby(key).cumsum().replace(0, np.nan)
    return pv / vv


def supertrend(df: pd.DataFrame, length: int = 10, mult: float = 3.0) -> pd.Series:
    """Classic ATR trailing-stop trend indicator. Returns +1 (up) / -1 (down) / 0 (warmup)."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    a = atr(df, length).to_numpy(dtype=float)
    n = len(df)
    fub = np.full(n, np.nan)
    flb = np.full(n, np.nan)
    trend = np.zeros(n)
    prev_t = 0.0
    for i in range(n):
        if np.isnan(a[i]):
            continue
        hub = (h[i] + l[i]) / 2 + mult * a[i]
        lub = (h[i] + l[i]) / 2 - mult * a[i]
        if np.isnan(fub[i - 1]) if i > 0 else True:
            fub[i] = hub
            flb[i] = lub
            trend[i] = 0.0
            prev_t = 0.0
            continue
        fub[i] = hub if (hub < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lub if (lub > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if prev_t == 0.0:
            prev_t = 1.0 if c[i] > fub[i] else (-1.0 if c[i] < flb[i] else 0.0)
        elif prev_t == 1.0 and c[i] < flb[i]:
            prev_t = -1.0
        elif prev_t == -1.0 and c[i] > fub[i]:
            prev_t = 1.0
        trend[i] = prev_t
    return pd.Series(trend, index=df.index)


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average Directional Index (Wilder smoothing). Trend strength, 0..100."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    if n < 2:
        return pd.Series(np.full(n, np.nan), index=df.index)
    up_move = np.diff(h)   # H[i] - H[i-1]
    down_move = -np.diff(l)  # L[i-1] - L[i]
    tr = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(l[1:] - c[:-1]),
    ])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    alpha = 1.0 / length
    atr_w = np.empty(n - 1)
    pdm = np.empty(n - 1)
    mdm = np.empty(n - 1)
    atr_w[0], pdm[0], mdm[0] = tr[0], plus_dm[0], minus_dm[0]
    for i in range(1, n - 1):
        atr_w[i] = atr_w[i - 1] + alpha * (tr[i] - atr_w[i - 1])
        pdm[i] = pdm[i - 1] + alpha * (plus_dm[i] - pdm[i - 1])
        mdm[i] = mdm[i - 1] + alpha * (minus_dm[i] - mdm[i - 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * pdm / atr_w
        minus_di = 100.0 * mdm / atr_w
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    adx_w = np.full(n, np.nan)
    adx_w[1] = dx[0]
    for i in range(2, n):
        adx_w[i] = adx_w[i - 1] + alpha * (dx[i - 1] - adx_w[i - 1])
    return pd.Series(adx_w, index=df.index)


def parabolic_sar(
    df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
) -> pd.Series:
    """Parabolic SAR (Wilder). Trend reads as sign(close - SAR).

    The SAR trails the trend, accelerating by ``af_step`` per new extreme
    toward ``af_max``, and flips when price crosses it. The SAR never
    penetrates the previous two bars' range against the current trend.
    """
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    n = len(df)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)
    trend = 1.0 if h[1] >= h[0] else -1.0
    ep = h[1] if trend == 1.0 else l[1]
    af = af_start
    sar[1] = l[0] if trend == 1.0 else h[0]
    for i in range(2, n):
        nxt = sar[i - 1] + af * (ep - sar[i - 1])
        if trend == 1.0:
            nxt = min(nxt, l[i - 1], l[i - 2])
        else:
            nxt = max(nxt, h[i - 1], h[i - 2])
        flipped = (trend == 1.0 and l[i] < nxt) or (trend == -1.0 and h[i] > nxt)
        if flipped:
            trend = -trend
            sar[i] = ep
            ep = l[i] if trend == -1.0 else h[i]
            af = af_start
        else:
            sar[i] = nxt
            if trend == 1.0 and h[i] > ep:
                ep = h[i]
                af = min(af + af_step, af_max)
            elif trend == -1.0 and l[i] < ep:
                ep = l[i]
                af = min(af + af_step, af_max)
    return pd.Series(sar, index=df.index)
