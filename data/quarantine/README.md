# Quarantined data — mislabeled provider mix

Moved here on 2026-11-23 during the Phase 0 audit.

`MacMT5Broker._stitch_latest_live_bars()` silently concatenated three different
feeds into `data/xauusd_*_real.csv` and `data/cache/mt5_mac/`:

1. MT5 Mac terminal cache (real XAUUSD spot, old)
2. Yahoo Finance GC=F (CME gold **futures**, different price level)
3. Binance PAXGUSDT (tokenised gold **spot proxy**, 24/7)

Everything was then served as `provider="MetaTrader 5 (Mac Spot)"`,
`real_data=True`, `spot=True`, `delayed=False`. That violates the Master Prompt
rule "never present delayed, synthetic or proxy data as live or real broker
data", and it produced the impossible wicks on the chart (futures ~4080 next to
spot ~3914).

Nothing here is deleted. The files are kept for reference only and must not be
used for backtesting or validation. Real data must come from a single,
correctly labelled provider (see Phase 1A: MT5 export on Windows).
