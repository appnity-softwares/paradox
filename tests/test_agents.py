import unittest

from core.execution_engine import ExecutionEngine
from core.orchestrator import AgentOrchestrator
from core.risk_engine import RiskEngine
from core.agents import (
    ComplianceAgent,
    ExecutionAgent,
    PerformanceAnalystAgent,
    PortfolioRiskAgent,
    ResearchAgent,
    StrategyAdaptationAgent,
)
from dashboard.app import app
from dashboard.state import ORCHESTRATOR
from fastapi.testclient import TestClient


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.orch = AgentOrchestrator()
        self.research = ResearchAgent(self.orch)
        self.compliance = ComplianceAgent(self.orch)
        self.execution = ExecutionAgent(self.orch)
        self.performance = PerformanceAnalystAgent(self.orch)
        self.portfolio = PortfolioRiskAgent(self.orch)
        self.adapt = StrategyAdaptationAgent(self.orch)
        for a in (self.research, self.compliance, self.execution, self.performance, self.portfolio, self.adapt):
            self.orch.register(a)

    def test_agent_status_lists_all_agents(self):
        status = self.orch.get_agent_status()
        self.assertEqual(len(status), 6)
        self.assertIn("research_agent", status)

    def test_research_agent_returns_regime_response(self):
        import asyncio

        from core.orchestrator import AgentMessage, MessageType

        msg = AgentMessage(
            sender="user",
            receiver="research_agent",
            msg_type=MessageType.REQUEST,
            payload={"action": "market_research", "payload": {"symbol": "XAUUSD", "timeframe": "15m", "lookback_days": 30}},
        )
        response = asyncio.run(self.orch.route(msg))
        self.assertIsNotNone(response)
        self.assertTrue(response.payload.get("success"))
        self.assertIn(response.payload.get("regime"), {"trending_up", "trending_down", "ranging", "high_volatility", "low_volatility"})

    def test_unknown_action_returns_clean_error(self):
        import asyncio

        from core.orchestrator import AgentMessage, MessageType

        msg = AgentMessage(
            sender="user",
            receiver="compliance_agent",
            msg_type=MessageType.REQUEST,
            payload={"action": "not_a_real_action"},
        )
        response = asyncio.run(self.orch.route(msg))
        self.assertIsNotNone(response)
        self.assertFalse(response.payload.get("success"))
        self.assertIn("Unknown action", response.payload.get("error", ""))


class RiskEngineTests(unittest.TestCase):
    def test_blocks_when_daily_loss_breached(self):
        risk = RiskEngine(max_risk_pct=2.0, max_daily_loss_pct=5.0, max_leverage=10.0)
        result = risk.check_position_risk(
            symbol="XAUUSD",
            side="BUY",
            volume=0.01,
            risk_pct=1.0,
            account_equity=10000.0,
            sl_distance=5.0,
        )
        self.assertTrue(result.allowed)

        risk.record_trade_result(pnl_pct=-8.0)
        blocked = risk.check_position_risk(
            symbol="XAUUSD",
            side="BUY",
            volume=0.01,
            risk_pct=1.0,
            account_equity=10000.0,
            sl_distance=5.0,
        )
        self.assertFalse(blocked.allowed)
        self.assertIn("Daily loss limit", blocked.reason)


class ExecutionAgentFlowTests(unittest.TestCase):
    def setUp(self):
        self.orch = AgentOrchestrator()
        self.execution = ExecutionAgent(self.orch)
        self.engine = ExecutionEngine()
        self.orch.register_handler("execution_engine", self._engine_handler)
        self.orch.register(self.execution)

    def _engine_handler(self, message):
        payload = message.payload
        action = payload.get("action", "")

        async def respond(payload_out):
            from core.orchestrator import AgentMessage, MessageType

            return AgentMessage(
                sender="execution_engine",
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                payload=payload_out,
                correlation_id=message.id,
            )

        if action == "place_market_order":
            result = self.engine.place_market_order(
                symbol=payload.get("symbol", "XAUUSD"),
                side=payload.get("side", "BUY"),
                volume=float(payload.get("volume", 0.01)),
                sl=float(payload.get("sl", 0.0)),
                tp=float(payload.get("tp", 0.0)),
                strategy_name=payload.get("strategy_name", "unknown"),
                comment=payload.get("comment", "AI-Quant"),
            )
            return respond({"success": result.success, "order_id": result.order_id, "filled_price": result.filled_price, "volume": result.volume, "reason": result.reason})

        if action == "emergency_kill_switch":
            result = self.engine.emergency_kill_switch()
            return respond(result)

        from core.orchestrator import AgentMessage, MessageType

        return AgentMessage(
            sender="execution_engine",
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown execution action: {action}"},
            correlation_id=message.id,
        )

    def test_execution_agent_refuses_kill_switch(self):
        """U0-hardening: the agent bus is not an order path. Mutating actions
        must be refused with a pointer to the ExecutionRouter, never executed."""
        import asyncio

        from core.orchestrator import AgentMessage, MessageType

        msg = AgentMessage(
            sender="user",
            receiver="execution_agent",
            msg_type=MessageType.COMMAND,
            payload={"action": "emergency_kill_switch"},
        )
        response = asyncio.run(self.orch.route(msg))
        self.assertIsNotNone(response)
        self.assertFalse(response.payload.get("success"))
        self.assertTrue(response.payload.get("refused"))
        self.assertIn("/api/platform/orders", response.payload.get("reason", ""))


class AgentApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_research_agent_endpoint_returns_regime(self):
        response = self.client.post(
            "/api/agents/research",
            json={"sender": "user", "receiver": "research_agent", "action": "market_research", "payload": {"symbol": "XAUUSD", "timeframe": "15m", "lookback_days": 30}},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body.get("success"))
        self.assertIn(body.get("regime"), {"trending_up", "trending_down", "ranging", "high_volatility", "low_volatility"})

    def test_compliance_agent_endpoint_checks_trade(self):
        response = self.client.post(
            "/api/agents/compliance",
            json={"sender": "user", "receiver": "compliance_agent", "action": "check_trade_permission", "payload": {"symbol": "XAUUSD", "side": "BUY", "volume": 0.01, "risk_pct": 1.0, "account_equity": 10000}},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body.get("success"))
        self.assertTrue(body.get("allowed"))


class ExecutionEngineBrokerTests(unittest.TestCase):
    def test_update_sl_tp_routes_to_broker(self):
        engine = ExecutionEngine()

        class FakeBroker:
            def __init__(self):
                self.calls = []

            def is_available(self):
                return True

            def place_market_order(self, **kw):
                return {"ticket": 777, "retcode": 0, "price": 2400.0, "volume": 0.1}

            def update_position_sl_tp(self, ticket, sl, tp):
                self.calls.append((ticket, sl, tp))
                return {"ticket": ticket, "retcode": 0, "sl": sl, "tp": tp}

            def close_position(self, ticket):
                return {"ticket": ticket, "retcode": 0}

        fb = FakeBroker()
        engine.set_broker(fb)

        placed = engine.place_market_order("XAUUSD", "BUY", 0.1, sl=2.0, tp=4.0, strategy_name="test")
        self.assertTrue(placed.success)
        pos_id = list(engine.positions.keys())[0]

        updated = engine.update_sl_tp(pos_id, sl=3.0, tp=6.0)
        self.assertTrue(updated.get("success"))
        self.assertEqual(fb.calls, [(pos_id, 3.0, 6.0)])

    def test_kill_switch_via_broker_flow(self):
        engine = ExecutionEngine(broker_connector=None)

        class FakeBroker:
            def __init__(self):
                self.closed = []

            def is_available(self):
                return True

            def place_market_order(self, **kw):
                return {"ticket": 778, "retcode": 0, "price": 2400.0, "volume": 0.1}

            def close_position(self, ticket):
                self.closed.append(ticket)
                return {"ticket": ticket, "retcode": 0}

        fb = FakeBroker()
        engine.set_broker(fb)
        placed = engine.place_market_order("XAUUSD", "BUY", 0.1, sl=2.0, tp=4.0, strategy_name="test")
        self.assertTrue(placed.success)
        pos_id = list(engine.positions.keys())[0]

        result = engine.close_position(pos_id, reason="manual")
        self.assertTrue(result.get("success"))
        self.assertEqual(fb.closed, [pos_id])


if __name__ == "__main__":
    unittest.main()
