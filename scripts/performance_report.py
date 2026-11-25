"""Part 6.7 performance report: CSV + PDF export for one strategy run.

Deterministic, real data only. For a given strategy (default parameters or
explicit ``key=value`` overrides) this runs the bar engine on real resampled
bars and exports:

- ``reports/perf_<key>_<stamp>_summary.csv``   headline metrics
- ``reports/perf_<key>_<stamp>_trades.csv``    full trade log
- ``reports/perf_<key>_<stamp>_monthly.csv``   monthly returns table
- ``reports/perf_<key>_<stamp>.pdf``           equity curve + drawdown chart

Usage:
    python -m scripts.performance_report [symbol] [year] [month] [months] \
        [strategy_key] [key=value ...]
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# matplotlib config dir must be writable; keep it inside the workspace.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplcache")
)

from core.backtest import run_backtest  # noqa: E402
from core.marketdata import store  # noqa: E402
from core.marketdata.resampler import resample_ticks_to_bars  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from strategies import library as strategy_library  # noqa: E402
from strategies.base import StrategyConfig  # noqa: E402
from strategies.catalog import catalog  # noqa: E402


def _coerce(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main() -> int:
    args = sys.argv[1:]
    symbol = args[0] if args else "XAUUSD"
    year = int(args[1]) if len(args) > 1 else 2025
    month = int(args[2]) if len(args) > 2 else 1
    months = int(args[3]) if len(args) > 3 else 6
    key = args[4] if len(args) > 4 else "ema_cross"
    overrides = {}
    for token in args[5:]:
        if "=" in token:
            k, v = token.split("=", 1)
            overrides[k] = _coerce(v)

    cat = catalog()
    if key not in cat:
        print(f"unknown strategy key {key!r}; available: {sorted(cat)}")
        return 1

    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_year = year + (month + months - 1) // 12
    end_month = (month + months - 1) % 12 + 1
    end = int(datetime(end_year, end_month, 1, tzinfo=timezone.utc).timestamp() * 1000)

    ticks = store.read_range(symbol, start, end)
    if ticks.empty:
        print(f"no ticks stored for {symbol} [{start},{end})")
        return 1
    bars = resample_ticks_to_bars(ticks, "15m", price_col="mid")

    cfg = StrategyConfig(sl_method="fixed", sl_fixed=2.0, rr=2.0, risk_pct=1.0)
    cls = getattr(strategy_library, cat[key]["class"])
    strat = cls(cfg, **overrides)
    equity, trades = run_backtest(bars, strat, cfg)
    metrics = compute_metrics(equity, trades, cfg, "15m")

    stamp = f"{year}{month:02d}"
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    base = f"perf_{key}_{stamp}"

    # Summary: headline metrics + provenance.
    summary = dict(metrics)
    summary["Strategy Key"] = key
    summary["Params"] = str(overrides or "defaults")
    summary["Bars"] = len(bars)
    summary["Dataset Hash Prefix"] = store.dataset_hash(symbol)[:16]
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
        out_dir / f"{base}_summary.csv"
    )
    trades.to_csv(out_dir / f"{base}_trades.csv", index=False)

    # Monthly returns from the equity curve.
    eq = pd.Series(equity, index=bars.index) if not isinstance(equity, pd.Series) else equity
    monthly = eq.resample("ME").last().pct_change().dropna() * 100.0
    monthly.rename("return_pct").to_csv(out_dir / f"{base}_monthly.csv")

    # PDF: equity curve + drawdown, Agg backend (headless).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dd = eq / eq.cummax() - 1.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   height_ratios=[2, 1])
    ax1.plot(eq.index, eq.values, lw=0.8, color="#1f77b4")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"{cfg.name} — {symbol} 15m {stamp}  |  "
                  f"Ret {metrics['Return %']:.1f}%  Sharpe {metrics['Sharpe']:.2f}  "
                  f"Trades {metrics['Total Trades']}  PF {metrics['Profit Factor']:.2f}")
    ax1.grid(alpha=0.3)
    ax2.fill_between(dd.index, dd.values * 100.0, 0.0, color="#d62728", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    pdf_path = out_dir / f"{base}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"strategy  : {cfg.name}  params={overrides or 'defaults'}")
    print(f"bars      : {len(bars)}  trades: {metrics['Total Trades']}")
    for k in ("Return %", "Sharpe", "Profit Factor", "Max DD %", "Win Rate %"):
        print(f"{k:<14}: {metrics.get(k)}")
    print(f"reports   : {base}_summary.csv, {base}_trades.csv, "
          f"{base}_monthly.csv, {pdf_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
