import unittest

from ai.builder import build, parse_command, validate


class BuilderValidationTests(unittest.TestCase):
    def test_fixed_stop_reaches_strategy_config(self):
        spec = validate(parse_command("fixed 12 dollar stop EMA 20/50 crossover"))
        _, cfg = build(spec)
        self.assertEqual(cfg.sl_method, "fixed")
        self.assertEqual(cfg.sl_fixed, 12.0)

    def test_invalid_macd_fast_slow_is_corrected(self):
        spec = validate(parse_command("MACD 30 10 9"))
        self.assertEqual(spec["params"]["fast"], 10)
        self.assertEqual(spec["params"]["slow"], 30)
        self.assertTrue(any("swap" in warning for warning in spec["warnings"]))


if __name__ == "__main__":
    unittest.main()
