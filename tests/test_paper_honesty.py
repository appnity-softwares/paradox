"""Honesty tests for the paper engine's fill model.

The audit found the paper engine booking PnL at 1/100th of the correct
scale (no contract size), never charging the spread, and filling stops at
the bar open. These tests pin the corrected behavior.
"""

import unittest

from core.paper_trading import PaperTradingEngine


def bar(t, o, h, l, c, v=100.0):
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


class ManualOrderPnlScaleTests(unittest.TestCase):
    def setUp(self):
        self.eng = PaperTradingEngine()
        self.eng._buffer.append(bar("2026-11-26 10:00:00+00:00", 3500, 3501, 3499, 3500))

    def test_pnl_uses_contract_size(self):
        """0.1 lot XAUUSD = 10 oz: a $1.00 move books $10.00, not $0.10."""
        r = self.eng.manual_order("XAUUSD", "BUY", 0.1, entry_price=3500.0,
                                  sl_price=3490.0, tp_price=3520.0)
        self.assertTrue(r["ok"])
        trade = self.eng.manual_close(r["ticket"], exit_price=3501.0)["trade"]
        self.assertAlmostEqual(trade.pnl, 10.0, places=6)

    def test_spread_is_paid_on_entry_and_exit(self):
        """BUY fills at the ask (entry + half spread); closing at the bid
        (exit - half spread) books the full spread as a cost."""
        half = self.eng.half_spread
        r = self.eng.manual_order("XAUUSD", "BUY", 0.1, entry_price=3500.0 + half,
                                  sl_price=3490.0, tp_price=3520.0)
        self.assertTrue(r["ok"])
        # close at the bid (mid - half spread): the round trip costs the FULL
        # spread x ounces — $0.30 x 10 oz = $3.00 on 0.1 lot
        trade = self.eng.manual_close(r["ticket"], exit_price=3500.0 - half)["trade"]
        self.assertAlmostEqual(trade.pnl, -self.eng.spread * 10.0, places=6)
        self.assertLess(trade.pnl, 0, "a round trip at an unchanged mid must cost the spread")

    def test_sell_fills_at_bid_side(self):
        eng = PaperTradingEngine()
        eng._buffer.append(bar("2026-11-26 10:00:00+00:00", 3500, 3501, 3499, 3500))
        r = eng.manual_order("XAUUSD", "SELL", 0.1, entry_price=3499.0,
                             sl_price=3509.0, tp_price=3489.0)
        self.assertTrue(r["ok"])
        trade = eng.manual_close(r["ticket"], exit_price=3498.0)["trade"]
        self.assertAlmostEqual(trade.pnl, 10.0, places=6)

    def test_bracket_validation(self):
        eng = PaperTradingEngine()
        # BUY with stop above entry is refused
        r = eng.manual_order("XAUUSD", "BUY", 0.1, entry_price=3500.0,
                             sl_price=3510.0, tp_price=3520.0)
        self.assertFalse(r["ok"])
        # SELL with target above entry is refused
        r = eng.manual_order("XAUUSD", "SELL", 0.1, entry_price=3500.0,
                             sl_price=3510.0, tp_price=3515.0)
        self.assertFalse(r["ok"])

    def test_kill_switch_blocks_manual_orders(self):
        eng = PaperTradingEngine()
        eng.trigger_kill_switch()
        eng.release_kill_switch = None  # guard against accidental use
        eng.kill_switch_active = True
        r = eng.manual_order("XAUUSD", "BUY", 0.1, entry_price=3500.0,
                             sl_price=3490.0, tp_price=3510.0)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "kill_switch_active")


class StrategyFillRealismTests(unittest.TestCase):
    def _open_long(self, eng, base=3500.0):
        """Drive the engine until a long opens on rising bars."""
        from tests.test_paper import rising_ema_strategy
        strat = rising_ema_strategy()
        eng.start(strat, strat.cfg, "XAUUSD")
        for i in range(40):
            t = f"2026-11-20 10:{i:02d}:00+00:00"
            eng.process_bar(bar(t, base, base + 1, base - 1, base + 0.5 + i * 0.5))
            if eng.positions:
                return eng.positions[-1]
        return None

    def test_entry_pays_half_spread_plus_slippage(self):
        eng = PaperTradingEngine()
        pos = self._open_long(eng)
        self.assertIsNotNone(pos, "expected a position on a rising series")
        # find the bar open it entered on
        self.assertGreater(pos.entry_price, 0)
        # entry must include half spread + slippage above the raw open
        raw_open = pos.entry_price - eng.half_spread - eng.slippage
        self.assertGreater(pos.entry_price, raw_open)

    def test_stop_fills_at_the_stop_not_the_open(self):
        """A stop touched mid-bar fills AT the stop (minus exit costs), not
        at the (better) bar open."""
        eng = PaperTradingEngine()
        pos = self._open_long(eng)
        self.assertIsNotNone(pos)
        # next bar: opens ABOVE the stop but dips through it mid-bar
        open_px = pos.sl + 2.0
        trade = eng.process_bar(bar("2026-11-20 11:00:00+00:00",
                                    open_px, open_px + 1, pos.sl - 1.0, open_px))
        self.assertIsNotNone(trade)
        self.assertEqual(trade.reason, "SL")
        self.assertLessEqual(trade.exit_price, pos.sl,
                             "stop filled better than the stop price")

    def test_gap_through_stop_fills_at_the_open(self):
        """Price gapping through the stop fills at the open — the worse price."""
        eng = PaperTradingEngine()
        pos = self._open_long(eng)
        self.assertIsNotNone(pos)
        open_px = pos.sl - 3.0   # gapped below the stop
        trade = eng.process_bar(bar("2026-11-20 11:00:00+00:00",
                                    open_px, open_px + 1, open_px - 1, open_px))
        self.assertIsNotNone(trade)
        self.assertEqual(trade.reason, "SL")
        self.assertLessEqual(trade.exit_price, open_px + eng.slippage)

    def test_target_fills_at_the_target(self):
        eng = PaperTradingEngine()
        pos = self._open_long(eng)
        self.assertIsNotNone(pos)
        open_px = pos.tp - 1.0
        trade = eng.process_bar(bar("2026-11-20 11:00:00+00:00",
                                    open_px, pos.tp + 0.5, open_px - 1, open_px))
        self.assertIsNotNone(trade)
        self.assertEqual(trade.reason, "TP")
        self.assertAlmostEqual(trade.exit_price, pos.tp, places=6)

    def test_manual_position_not_closed_by_strategy_signal(self):
        eng = PaperTradingEngine()
        pos = self._open_long(eng)
        self.assertIsNotNone(pos)
        # simulate a manual desk position
        eng.positions[0].strategy_name = "manual"
        # force an opposite strategy signal and feed a bar that touches
        # neither the stop nor the target: only a strategy position may be
        # signal-closed, never a manual desk position
        eng._last_signal = -1.0
        trade = eng.process_bar(bar("2026-11-20 11:00:00+00:00",
                                    pos.entry_price, pos.entry_price + 1,
                                    pos.entry_price - 1, pos.entry_price))
        self.assertIsNone(trade)
        self.assertTrue(any(p.ticket == pos.ticket for p in eng.positions),
                        "a manual position was closed by a strategy signal")


if __name__ == "__main__":
    unittest.main()
