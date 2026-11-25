import os
import sys
import unittest

PROTOTYPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROTOTYPE_ROOT not in sys.path:
    sys.path.insert(0, PROTOTYPE_ROOT)

from core.config import get_symbol_spec
from core.execution_engine import ExecutionEngine


class ExecutionEngineQuoteGateTests(unittest.TestCase):
    def test_simulated_fill_requires_a_real_quote(self):
        engine = ExecutionEngine()
        result = engine.place_market_order("XAUUSD", "BUY", 0.1)
        self.assertFalse(result.success)
        self.assertIn("No live quote", result.reason)

    def test_long_fills_at_ask_and_short_at_bid(self):
        engine = ExecutionEngine()
        engine.update_market_price("XAUUSD", 2400.0, 2400.3)

        buy = engine.place_market_order("XAUUSD", "BUY", 0.1)
        self.assertTrue(buy.success)
        self.assertEqual(buy.filled_price, 2400.3)

        sell = engine.place_market_order("XAUUSD", "SELL", 0.1)
        self.assertTrue(sell.success)
        self.assertEqual(sell.filled_price, 2400.0)

    def test_contract_size_comes_from_the_symbol_spec(self):
        engine = ExecutionEngine()
        engine.update_market_price("XAUUSD", 2400.0, 2400.3)
        result = engine.place_market_order("XAUUSD", "BUY", 0.5)

        position = engine.positions[result.order_id]
        self.assertEqual(position.contract_size, get_symbol_spec("XAUUSD").contract_size)

    def test_unknown_symbol_is_rejected(self):
        engine = ExecutionEngine()
        engine.update_market_price("ZZZUSD", 1.0, 1.1)
        result = engine.place_market_order("ZZZUSD", "BUY", 0.1)
        self.assertFalse(result.success)
        self.assertIn("not found", result.reason)


if __name__ == "__main__":
    unittest.main()
