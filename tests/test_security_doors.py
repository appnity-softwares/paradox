"""Behavioral security tests for the doors the U0 route audit only
string-matched. A guard that can be defeated by one function call is
theater — these tests exercise the actual HTTP paths."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from dashboard.app import app
from dashboard.state import EXECUTION_ENGINE, PAPER_ENGINE


class AgentBusOrderDoorTests(unittest.TestCase):
    """POST /api/agent/message must NOT be able to place, modify or close
    orders — the payload is refused before any engine is reached."""

    def setUp(self):
        self.client = TestClient(app)

    def test_place_market_order_via_agent_bus_is_refused(self):
        before = len(EXECUTION_ENGINE.positions)
        response = self.client.post("/api/agent/message", json={
            "sender": "attacker",
            "receiver": "execution_engine",
            "action": "place_market_order",
            "payload": {"symbol": "XAUUSD", "side": "BUY", "volume": 1.0},
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body.get("success"))
        self.assertEqual(body.get("refused_action"), "place_market_order")
        self.assertEqual(len(EXECUTION_ENGINE.positions), before,
                         "an order was created through the agent bus")

    def test_close_position_via_agent_bus_is_refused(self):
        response = self.client.post("/api/agent/message", json={
            "sender": "attacker",
            "receiver": "execution_engine",
            "action": "close_position",
            "payload": {"position_id": 1},
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json().get("success"))

    def test_kill_switch_via_agent_bus_is_refused(self):
        response = self.client.post("/api/agent/message", json={
            "sender": "attacker",
            "receiver": "execution_engine",
            "action": "emergency_kill_switch",
            "payload": {},
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json().get("success"))

    def test_execution_agent_endpoint_refuses_orders(self):
        response = self.client.post("/api/agents/execution", json={
            "sender": "attacker",
            "receiver": "execution_agent",
            "action": "place_order",
            "payload": {"symbol": "XAUUSD", "side": "BUY", "volume": 1.0},
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body.get("success"))
        self.assertTrue(body.get("refused"))


class MT5BridgeLoopbackTests(unittest.TestCase):
    """The MT5 bridge routes are loopback-only and never invent a price."""

    def setUp(self):
        self.client = TestClient(app)

    def test_tick_without_bid_ask_is_rejected_not_fabricated(self):
        response = self.client.post("/api/mt5/tick", json={
            "symbol": "XAUUSD", "last": 3945.0,
        })
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        text = detail if isinstance(detail, str) else str(detail)
        self.assertIn("bid", text.lower())

    def test_tick_rejects_the_hardcoded_fallback_price(self):
        """A tick carrying neither bid/ask nor a plausible price must never
        produce the old hardcoded 3945.60/3945.90 fill pair."""
        response = self.client.post("/api/mt5/tick", json={"symbol": "XAUUSD"})
        self.assertEqual(response.status_code, 422)
        # and nothing was fed to the execution engine's price state
        self.assertNotEqual(EXECUTION_ENGINE.last_bid, 3945.60)

    def test_config_endpoint_stays_available_locally(self):
        # loopback (TestClient) may still configure the bridge; the endpoint
        # must exist and answer JSON, not 404/403-for-everyone.
        response = self.client.post("/api/mt5/config", json={})
        self.assertIn(response.status_code, (200, 422))


class PaperDeskTests(unittest.TestCase):
    """The terminal's manual desk is paper-only and spread-honest."""

    def setUp(self):
        self.client = TestClient(app)

    def test_paper_order_requires_a_real_quote(self):
        # With no provider active the desk must refuse, not invent a price.
        PAPER_ENGINE.positions.clear()
        response = self.client.post("/api/paper/order", json={
            "symbol": "NOSUCH", "side": "BUY", "volume": 0.01,
            "sl": 1.0, "tp": 2.0,
        })
        self.assertIn(response.status_code, (503, 422))

    def test_paper_positions_endpoint_shape(self):
        response = self.client.get("/api/paper/positions")
        self.assertEqual(response.status_code, 200)
        self.assertIn("positions", response.json())


if __name__ == "__main__":
    unittest.main()
