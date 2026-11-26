"""Tests for the prop-firm rules registry and breach evaluator."""

from __future__ import annotations

import unittest

from core.prop_firms import PROP_FIRMS, evaluate, get_firm, list_firms


def state(**kw):
    base = {
        "initial_balance": 100_000.0,
        "day_start_balance": 100_000.0,
        "day_start_equity": 100_000.0,
        "balance": 100_000.0,
        "equity": 100_000.0,
        "peak_equity": 100_000.0,
        "trading_days": 0,
        "best_day_pnl": 0.0,
        "total_pnl": 0.0,
    }
    base.update(kw)
    return base


class RegistryTests(unittest.TestCase):
    def test_all_requested_firms_present(self):
        expected = {"ftmo", "fundednext", "fundingpips", "the5ers", "goatfunded",
                    "blueguardian", "fxify", "alphacapital", "orion"}
        self.assertEqual(expected, set(PROP_FIRMS.keys()))

    def test_no_firm_is_marked_verified(self):
        """Nothing may claim verified status until someone actually checks
        the firm's own rules page — that check has not happened."""
        for firm in PROP_FIRMS.values():
            self.assertFalse(firm.verified, f"{firm.slug} claims verified")
            self.assertIn(firm.confidence, ("high", "medium", "low"))

    def test_every_firm_has_a_verify_url_except_placeholder(self):
        for firm in PROP_FIRMS.values():
            if firm.slug != "orion":   # explicitly a placeholder config
                self.assertTrue(firm.verify_url.startswith("https://"))

    def test_list_firms_shape(self):
        firms = list_firms()
        self.assertEqual(len(firms), 9)
        by_slug = {f["slug"]: f for f in firms}
        self.assertTrue(by_slug["ftmo"]["ea_allowed"])
        self.assertEqual(by_slug["ftmo"]["phases"][0]["profit_target_pct"], 10.0)


class FTMOEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.rules = get_firm("ftmo")

    def test_fresh_account_is_ok(self):
        r = evaluate(self.rules, 0, state())
        self.assertEqual(r["status"], "OK")
        self.assertFalse(r["breaches"])
        self.assertEqual(r["daily_dd_limit_pct"], 5.0)
        self.assertEqual(r["total_dd_limit_pct"], 10.0)

    def test_daily_breach_at_5pct_below_day_start(self):
        # FTMO daily DD is 5% of initial balance; equity fell $5,001 intraday
        r = evaluate(self.rules, 0, state(equity=94_999.0, balance=94_999.0))
        self.assertEqual(r["status"], "BREACHED")
        self.assertIn("max_daily_loss", [b["rule"] for b in r["breaches"]])

    def test_daily_breach_counts_floating_loss(self):
        # balance intact but open positions drag equity below the floor
        r = evaluate(self.rules, 0, state(balance=100_000.0, equity=94_900.0))
        self.assertEqual(r["status"], "BREACHED")

    def test_total_breach_at_10pct_below_initial(self):
        r = evaluate(self.rules, 0, state(equity=89_999.0, balance=89_999.0,
                                          day_start_balance=92_000.0,
                                          day_start_equity=92_000.0))
        self.assertIn("max_total_loss", [b["rule"] for b in r["breaches"]])

    def test_warning_inside_20pct_headroom(self):
        # $800 above the daily floor = 0.8% of initial < 20% of the 5% limit
        r = evaluate(self.rules, 0, state(equity=95_800.0))
        self.assertEqual(r["status"], "WARNING")

    def test_target_progress(self):
        r = evaluate(self.rules, 0, state(total_pnl=5_000.0, balance=105_000.0,
                                          equity=105_000.0))
        self.assertAlmostEqual(r["target_progress_pct"], 50.0, places=6)
        self.assertFalse(r["target_hit"])

    def test_phase2_target_is_5pct(self):
        r = evaluate(self.rules, 1, state(total_pnl=5_000.0, balance=105_000.0,
                                          equity=105_000.0))
        self.assertTrue(r["target_hit"])
        self.assertEqual(r["phase"], "Verification")


class TrailingDDTests(unittest.TestCase):
    def test_trailing_floor_follows_peak(self):
        """A trailing-DD firm: after equity peaks at 105k, the floor rises."""
        firm = get_firm("fundingpips")
        # FundingPips is static; synthesize a trailing variant of it
        from dataclasses import replace
        trailing = replace(firm, total_dd_basis="trailing_peak")
        # peak 105k -> 10% trailing floor = 94.5k; equity 95k is above it
        r = evaluate(trailing, 0, state(equity=95_000.0, peak_equity=105_000.0))
        self.assertNotIn("max_total_loss", [b["rule"] for b in r["breaches"]])
        # equity 94k is below the 94.5k trailing floor -> breach
        r = evaluate(trailing, 0, state(equity=94_000.0, peak_equity=105_000.0))
        self.assertIn("max_total_loss", [b["rule"] for b in r["breaches"]])


class ConsistencyRuleTests(unittest.TestCase):
    def test_consistency_breach_when_best_day_dominates(self):
        from dataclasses import replace
        firm = replace(get_firm("fxify"), consistency_max_best_day_pct=40.0)
        r = evaluate(firm, 0, state(total_pnl=10_000.0, best_day_pnl=6_000.0))
        self.assertIn("consistency", [b["rule"] for b in r["breaches"]])
        r = evaluate(firm, 0, state(total_pnl=10_000.0, best_day_pnl=3_000.0))
        self.assertTrue(r["consistency_ok"])


class MinDaysTests(unittest.TestCase):
    def test_target_without_min_days_warns(self):
        rules = get_firm("ftmo")   # 4 min days
        r = evaluate(rules, 0, state(total_pnl=10_500.0, trading_days=2))
        self.assertTrue(r["target_hit"])
        self.assertFalse(r["days_ok"])
        self.assertIn("min_trading_days", [w["rule"] for w in r["warnings"]])


if __name__ == "__main__":
    unittest.main()
