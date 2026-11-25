"""Console AI assistant (Part 14.1): chat that can call read and propose
tools ONLY. Built on the Phase 5 orchestrator; no execute-level tool
exists, and every proposal is inert data for human approval.
"""
from __future__ import annotations

import sys, os
PROTOTYPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROTOTYPE_ROOT)

from pydantic import BaseModel, Field

from core.ai import (
    FakeModel,
    ModelRoute,
    ModelRouter,
    Orchestrator,
    ToolRegistry,
    build_agent_specs,
    register_tools,
)
from dashboard.platform_state import PLATFORM


class ChatReply(BaseModel):
    reply: str = Field(min_length=1)
    proposal: dict | None = None


class ConsoleAssistant:
    def __init__(self):
        self.registry = ToolRegistry()
        register_tools(self.registry, store=None, broker=PLATFORM.broker,
                       accounts=PLATFORM.accounts, reports_dir=None)
        # Read-only canned backend: the console assistant answers with
        # structured summaries; it works with no provider configured.
        router = ModelRouter()
        router.add_route("console_chat", ModelRoute(FakeModel({
            "console_chat": {"reply": "Ready. Ask about positions, "
                                      "compliance status or propose a "
                                      "ticket.",
                             "actions": []}})))
        self.orchestrator = Orchestrator(router, self.registry)
        spec = build_agent_specs()["execution_assistant"]
        from core.ai.orchestrator import AgentSpec
        self.orchestrator.agents["console_assistant"] = AgentSpec(
            name="console_assistant", description="Console chat agent",
            permissions=("read", "propose"),
            output_schema=ChatReply, task="console_chat")

    def respond(self, message: str) -> dict:
        run = self.orchestrator.run("console_assistant", {"message": message})
        if run.status != "ok":
            return {"reply": "assistant failed safely; nothing was executed",
                    "status": run.status, "errors": run.errors,
                    "actions": []}
        out = run.output or {}
        # A proposal is INERT: surfaced for human approval, never submitted.
        proposal = None
        if out.get("proposal"):
            proposal = {**out["proposal"],
                        "note": "proposal only — submit via the order "
                                "ticket with your approval"}
        return {"reply": out.get("reply", ""),
                "status": run.status, "proposal": proposal,
                "actions": [], "model": run.model,
                "cost_usd": run.cost_usd}

    def run_log(self, limit: int = 100) -> list[dict]:
        return [{"agent": r.agent, "task": r.task, "status": r.status,
                 "model": r.model, "tokens": r.tokens,
                 "cost_usd": r.cost_usd, "duration_ms": r.duration_ms,
                 "errors": r.errors}
                for r in self.orchestrator.runs[-limit:]]


ASSISTANT = ConsoleAssistant()
