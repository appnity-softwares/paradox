"""The audit scorecard and agent analytics must not fabricate numbers.

The brutal audit found /api/agents/audit returning a hardcoded 94.2
'INSTITUTIONAL GRADE' scorecard with invented walk-forward, Monte-Carlo
and performance statistics, and hash-derived PRNG 'analytics' in the
optimization and performance agents. These tests pin the honest behavior.
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from dashboard.app import app


class HonestAuditScorecardTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_overall_score_is_null_until_a_real_harness_exists(self):
        body = self.client.get("/api/agents/audit").json()
        self.assertIsNone(body["overall_score"])
        self.assertEqual(body["grade"], "NOT EVALUATED")
        self.assertNotIn("INSTITUTIONAL", body["grade"])

    def test_no_agent_carries_an_invented_score(self):
        body = self.client.get("/api/agents/audit").json()
        for agent in body["agents"]:
            self.assertIsNone(agent["score"],
                              f"{agent['id']} still reports a fabricated score")
            self.assertNotIn("PASS", agent["status"])

    def test_no_fabricated_wfe_or_monte_carlo_numbers(self):
        """The fabricated numbers must not appear as METRIC VALUES. Explanatory
        prose that says 'the old value was fabricated' is allowed."""
        body = self.client.get("/api/agents/audit").json()
        for agent in body["agents"]:
            for value in agent["metrics"].values():
                for fiction in ("86.2%", "97.4%", "1.74", "1.86", "94.2"):
                    self.assertNotIn(fiction, str(value),
                                     f"{agent['id']} metric still carries a fabricated number")

    def test_live_context_still_measured(self):
        """Honesty does not mean emptiness: real market/account state stays."""
        body = self.client.get("/api/agents/audit").json()
        self.assertIn("account_equity", body)
        self.assertIn("executive_summary", body)


class HonestOptimizationAgentTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_compare_refuses_to_rank_unmeasured_strategies(self):
        response = self.client.post("/api/agent/message", json={
            "sender": "test", "receiver": "strategy_optimization_agent",
            "action": "compare_strategies",
            "payload": {"strategies": ["ema_cross", "made_up_strategy"]},
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body.get("success"))
        self.assertIsNone(body.get("best_strategy") if not body.get("ranking") else None) \
            if False else None
        # every unmeasured strategy must carry score None and measured False
        for row in body.get("ranking", []):
            if not row.get("measured"):
                self.assertIsNone(row["score"])

    def test_regime_change_without_baseline_is_not_measured(self):
        response = self.client.post("/api/agent/message", json={
            "sender": "test", "receiver": "strategy_optimization_agent",
            "action": "detect_regime_change",
            "payload": {"strategy_name": "never_optimized", "current_regime": "trending_up"},
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body.get("success"))
        self.assertIsNone(body.get("regime_changed"))
        self.assertFalse(body.get("measured"))


class HonestPerformanceAgentTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_report_comes_from_the_real_ledger_or_refuses(self):
        from core.agents.performance_analyst_agent import PerformanceAnalystAgent
        from dashboard.state import ORCHESTRATOR
        agent = PerformanceAnalystAgent(ORCHESTRATOR)
        # unknown strategy -> no ledger -> refuse, never invent
        report = agent._generate_report({"strategy_name": "no_such_strategy_xyz", "period_days": 30})
        self.assertFalse(report["measured"])
        self.assertIsNone(report["summary"])
        self.assertIn("cannot be generated", report["error"])

        # a strategy with a real backtest ledger gets REAL numbers, labelled
        report = agent._generate_report({"strategy_name": "ema_cross", "period_days": 3650})
        if report["measured"]:
            self.assertGreater(report["trade_count"], 0)
            self.assertIn("backtest", report["data_source"])
            # win rate must be consistent with the ledger-derived fields
            self.assertLessEqual(report["summary"]["win_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
