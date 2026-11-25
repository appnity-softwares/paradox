import unittest

import pandas as pd

from core.optimize import fixed_walk_forward
from strategies.base import StrategyConfig


class AlwaysLongStrategy:
    warmup = 20

    def __init__(self, cfg, **params):
        self.cfg = cfg
        self.warmup = 20
        cfg.name = "Always Long"

    def generate_signals(self, df):
        return pd.Series(1.0, index=df.index)


class WalkForwardBoundaryTests(unittest.TestCase):
    def test_warmup_bars_cannot_open_oos_trades(self):
        index = pd.date_range("2025-01-01", periods=1800, freq="15min", tz="UTC")
        close = pd.Series(range(1800), index=index, dtype=float) * 0.01 + 2000
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 100,
            },
            index=index,
        )
        cfg = StrategyConfig(sl_method="fixed", sl_fixed=50, rr=10, max_trades_per_day=1)
        _, trades, _ = fixed_walk_forward(
            df,
            AlwaysLongStrategy,
            {},
            cfg,
            "15m",
            n_folds=2,
        )
        segment = len(df) // 3
        for fold, start_position in ((1, segment), (2, segment * 2)):
            fold_trades = trades[trades["fold"] == fold]
            if len(fold_trades):
                self.assertGreaterEqual(fold_trades["entry_time"].min(), index[start_position])


if __name__ == "__main__":
    unittest.main()
