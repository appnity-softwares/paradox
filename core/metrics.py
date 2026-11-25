"""Performance analytics - the notebook dashboard table:
Win Rate | DD | Total Trades | Sharpe | PF | RR | Max Trades/Day | Con.Win | Con.Loss | ...
"""
import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, cfg, tf_label: str) -> dict:
    total = len(trades)
    if total == 0:
        return {
            "Strategy": cfg.name, "Win Rate %": 0.0, "Max DD %": 0.0,
            "Total Trades": 0, "Sharpe": 0.0, "Profit Factor": 0.0,
            "RR (realized)": 0.0, "Max Trades/Day": 0, "Con. Wins": 0,
            "Con. Losses": 0, "Net P&L $": 0.0, "Return %": 0.0,
            "Risk/Trade %": cfg.risk_pct, "TF": tf_label,
            "Session": cfg.session, "Order": "Market",
        }

    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = len(wins) / total * 100.0
    gross_win = float(wins.sum())
    gross_loss = abs(losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else None
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    rr_realized = avg_win / avg_loss if avg_loss > 0 else None

    # Drawdown from mark-to-market equity curve
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100.0
    max_dd = dd.min()

    # Daily Sharpe avoids overstating results by annualizing every intraday bar
    # as if the instrument traded 24/7 without serial correlation.
    if isinstance(equity.index, pd.DatetimeIndex):
        daily_equity = equity.resample("1D").last().dropna()
        rets = daily_equity.pct_change().dropna()
    else:
        rets = equity.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0

    # Max trades per day (actual)
    if "entry_time" in trades.columns:
        tpd = trades.groupby(trades["entry_time"].dt.date).size()
        max_trades_day = int(tpd.max())
    else:
        max_trades_day = 0

    # Consecutive wins / losses
    max_cw = max_cl = cw = cl = 0
    for p in pnl:
        if p > 0:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)

    net = float(equity.iloc[-1] - equity.iloc[0])
    expectancy = float(pnl.mean()) if total else 0.0

    def rounded(value, digits=2):
        return round(float(value), digits) if value is not None and np.isfinite(value) else None

    return {
        "Strategy": cfg.name,
        "Win Rate %": round(win_rate, 1),
        "Max DD %": round(max_dd, 1),
        "Total Trades": total,
        "Sharpe": rounded(sharpe),
        "Profit Factor": rounded(profit_factor),
        "RR (realized)": rounded(rr_realized),
        "Max Trades/Day": max_trades_day,
        "Con. Wins": max_cw,
        "Con. Losses": max_cl,
        "Net P&L $": round(net, 2),
        "Return %": round(net / equity.iloc[0] * 100, 1),
        "Ending Equity $": round(float(equity.iloc[-1]), 2),
        "Expectancy $": round(expectancy, 2),
        "Risk/Trade %": cfg.risk_pct,
        "TF": tf_label,
        "Session": cfg.session,
        "Order": "Market",
    }
