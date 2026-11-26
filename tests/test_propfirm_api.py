"""API tests for the prop-firm selection and live guard endpoints."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from dashboard.app import app
from dashboard.state import PAPER_ENGINE, PROP_FIRM_SELECTION


class PropFirmApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # always leave the selection cleared after each test
        self.addCleanup(PROP_FIRM_SELECTION.update, {"slug": None, "phase_index": 0})

    def test_all_requested_firms_listed(self):
        body = self.client.get("/api/propfirms").json()
        slugs = {f["slug"] for f in body["firms"]}
        self.assertLessEqual({"ftmo", "fundednext", "fundingpips", "the5ers",
                              "goatfunded", "blueguardian", "fxify",
                              "alphacapital", "orion"}, slugs)
        self.assertIn("disclaimer", body)
        self.assertIn("unverified", body["disclaimer"].lower())

    def test_select_and_status_roundtrip(self):
        r = self.client.post("/api/propfirm/select",
                             json={"slug": "ftmo", "phase_index": 0})
        self.assertEqual(r.json()["selected"], "ftmo")
        st = self.client.get("/api/propfirm/status").json()
        self.assertEqual(st["selected"], "ftmo")
        self.assertFalse(st["verified"])          # honesty: unverified data
        self.assertIn("evaluation", st)
        self.assertEqual(st["evaluation"]["daily_dd_limit_pct"], 5.0)

    def test_select_unknown_firm_404(self):
        r = self.client.post("/api/propfirm/select",
                             json={"slug": "no_such_firm"})
        self.assertEqual(r.status_code, 404)

    def test_select_bad_phase_422(self):
        r = self.client.post("/api/propfirm/select",
                             json={"slug": "ftmo", "phase_index": 9})
        self.assertEqual(r.status_code, 422)

    def test_clear_selection(self):
        self.client.post("/api/propfirm/select", json={"slug": "ftmo"})
        r = self.client.post("/api/propfirm/select", json={"slug": None})
        self.assertEqual(r.json()["selected"], None)
        st = self.client.get("/api/propfirm/status").json()
        self.assertIsNone(st["selected"])

    def test_daily_breach_fires_paper_kill_switch(self):
        """A hard daily-DD breach must engage the paper kill switch."""
        PAPER_ENGINE.kill_switch_active = False
        PAPER_ENGINE.equity = 94_000.0          # $6k below day start: > 5% FTMO daily
        PAPER_ENGINE.day_start_equity = 100_000.0
        PAPER_ENGINE.peak_equity = 100_000.0
        self.client.post("/api/propfirm/select", json={"slug": "ftmo"})
        st = self.client.get("/api/propfirm/status").json()
        self.assertEqual(st["evaluation"]["status"], "BREACHED")
        self.assertTrue(st["kill_switch_active"])
        self.assertTrue(PAPER_ENGINE.kill_switch_active)
        # cleanup
        PAPER_ENGINE.kill_switch_active = False
        PAPER_ENGINE.equity = PAPER_ENGINE.starting_equity
        PAPER_ENGINE.day_start_equity = PAPER_ENGINE.starting_equity
        PAPER_ENGINE.peak_equity = PAPER_ENGINE.starting_equity

    def test_status_without_selection(self):
        st = self.client.get("/api/propfirm/status").json()
        self.assertIsNone(st["selected"])
        self.assertIn("note", st)

    def test_simulator_profiles_carry_honesty_flags(self):
        body = self.client.get("/api/prop_firm/profiles").json()
        self.assertIn("disclaimer", body)
        ids = {p["id"] for p in body["profiles"]}
        for slug in ("ftmo", "fundednext", "fundingpips", "the5ers",
                     "goatfunded", "blueguardian", "fxify", "alphacapital", "orion"):
            self.assertIn(f"{slug}_p1", ids)
        for p in body["profiles"]:
            self.assertIn("verified", p)
            self.assertFalse(p["verified"])


if __name__ == "__main__":
    unittest.main()


class NewEngineApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_tick_replay_endpoint(self):
        r = self.client.post("/api/tick_replay", json={
            "strategy": "ema_cross", "bars": 300,
            "sl_method": "fixed", "sl_fixed": 5.0})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["result"]["engine"], "tick_replay")
        self.assertIn("ticks_scanned", d["result"]["coverage"])
        self.assertIn("Win Rate %", d["metrics"])

    def test_tick_replay_unknown_strategy(self):
        r = self.client.post("/api/tick_replay", json={"strategy": "nope"})
        self.assertEqual(r.status_code, 400)

    def test_timemachine_endpoint(self):
        r = self.client.post("/api/timemachine", json={
            "entry": 4400.0, "stop": 4395.0, "target": 4410.0,
            "side": "buy", "replays": 500})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["plan"]["r_at_target"], 2.0)
        self.assertIn("verdict", d["simulation"])
        self.assertIn("triggered", d["history"])

    def test_timemachine_invalid_plan_422(self):
        r = self.client.post("/api/timemachine", json={
            "entry": 4400.0, "stop": 4401.0, "target": 4410.0})
        self.assertEqual(r.status_code, 422)
