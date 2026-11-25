import unittest

import pandas as pd

from core.backtest import run_backtest
from strategies.base import StrategyConfig


class AlwaysLong:
    def generate_signals(self, df):
        return pd.Series(1.0, index=df.index)


class BacktestEventOrderTests(unittest.TestCase):
    def test_new_entry_is_exposed_to_same_bar_stop(self):
        index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100, 100, 100],
                "high": [101, 101, 101],
                "low": [99, 90, 99],
                "close": [100, 100, 100],
                "volume": [1, 1, 1],
            },
            index=index,
        )
        cfg = StrategyConfig(
            sl_method="fixed",
            sl_fixed=5,
            rr=2,
            max_trades_per_day=1,
        )
        equity, trades = run_backtest(
            df,
            AlwaysLong(),
            cfg,
            spread=0,
            slippage=0,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["reason"], "SL")
        self.assertEqual(trades.iloc[0]["exit"], 95.0)
        self.assertEqual(equity.iloc[-1], 9900.0)

    def test_open_position_is_closed_at_end_of_data(self):
        index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100, 100, 101],
                "high": [101, 102, 103],
                "low": [99, 99, 100],
                "close": [100, 101, 102],
                "volume": [1, 1, 1],
            },
            index=index,
        )
        cfg = StrategyConfig(sl_method="fixed", sl_fixed=20, rr=5)
        _, trades = run_backtest(df, AlwaysLong(), cfg, spread=0, slippage=0)
        self.assertEqual(trades.iloc[-1]["reason"], "End of data")


if __name__ == "__main__":
    unittest.main()
