import unittest

from core.paper_trading import PaperTradingEngine
from strategies.base import StrategyConfig
from strategies.library import EMACross


def make_bar(t: str, o: float, h: float, l: float, c: float, v: float = 100.0) -> dict:
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def rising_ema_strategy():
    cfg = StrategyConfig(
        risk_pct=1.0,
        sl_method="fixed",
        sl_fixed=5.0,
        rr=2.0,
        session="all",
        max_trades_per_day=4,
        daily_loss_limit_pct=3.0,
    )
    return EMACross(cfg, fast=2, slow=3)


class PaperTradingEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PaperTradingEngine()

    def test_signal_zero_before_warmup(self):
        strat = rising_ema_strategy()
        self.engine.start(strat, strat.cfg, "XAUUSD")
        self.assertEqual(self.engine._compute_signal(), 0.0)

    def _feed_rising_bars(self, n: int = 30):
        strat = rising_ema_strategy()
        self.engine.start(strat, strat.cfg, "XAUUSD")
        base = 100.0
        for i in range(n):
            t = f"2026-11-20 10:{i:02d}:00+00:00"
            self.engine.process_bar(make_bar(t, base, base + 1, base - 1, base + 0.5 + i * 0.5))

    def test_opens_position_after_warmup_on_uptrend(self):
        self._feed_rising_bars(30)
        self.assertEqual(self.engine._last_signal, 1.0)
        self.assertTrue(self.engine.get_status()["running"])

    def test_sl_close_reduces_position(self):
        self._feed_rising_bars(30)
        # Position may or may not be open at this point; use a fresh engine flow:
        self.engine2 = PaperTradingEngine()
        strat = rising_ema_strategy()
        self.engine2.start(strat, strat.cfg, "XAUUSD")
        base = 100.0
        opened = False
        for i in range(30):
            t = f"2026-11-20 10:{i:02d}:00+00:00"
            self.engine2.process_bar(make_bar(t, base, base + 1, base - 1, base + 0.5 + i * 0.5))
            if self.engine2.get_status()["open_position"]:
                opened = True
                break
        self.assertTrue(opened, "expected the paper engine to open a position on uptrend")

        trade = self.engine2.process_bar(make_bar("2026-11-20 11:00:00+00:00", 90.0, 95.0, 88.0, 92.0))
        self.assertIsNotNone(trade)
        self.assertEqual(trade.reason, "SL")
        self.assertFalse(self.engine2.get_status()["open_position"])
        self.assertEqual(len(self.engine2.get_trades()), 1)

    def test_kill_switch_blocks_new_entries(self):
        self._feed_rising_bars(30)
        # Use engine that definitely has an open position: fresh feed then kill
        self.engine3 = PaperTradingEngine()
        strat = rising_ema_strategy()
        self.engine3.start(strat, strat.cfg, "XAUUSD")
        base = 100.0
        opened = False
        for i in range(30):
            t = f"2026-11-20 10:{i:02d}:00+00:00"
            self.engine3.process_bar(make_bar(t, base, base + 1, base - 1, base + 0.5 + i * 0.5))
            if self.engine3.get_status()["open_position"]:
                opened = True
                break
        self.assertTrue(opened)

        result = self.engine3.trigger_kill_switch()
        self.assertTrue(result["killed"])
        self.assertEqual(len(self.engine3.get_positions()), 0)

        trade = self.engine3.process_bar(make_bar("2026-11-20 11:00:00+00:00", 90.0, 95.0, 88.0, 92.0))
        self.assertIsNone(trade)


if __name__ == "__main__":
    unittest.main()
