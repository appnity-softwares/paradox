import unittest

from fastapi.testclient import TestClient

from dashboard.app import app


class PaperApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_paper_status_returns_clean_state(self):
        response = self.client.get("/api/paper/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("running", body)
        self.assertIn("open_position", body)

    def test_paper_stop_without_task(self):
        response = self.client.post("/api/paper/stop", json={})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["stopped"])

    def test_paper_replay_rejects_unknown_strategy(self):
        response = self.client.post(
            "/api/paper/replay",
            json={"strategy": "not_a_real_strategy", "timeframe": "15m"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
