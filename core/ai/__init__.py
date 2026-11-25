"""AI layer (Part 11): agents, orchestrator, strategy builder."""
from core.ai.agents import build_agent_specs, register_tools
from core.ai.models import (
    BudgetExceeded,
    FakeModel,
    ModelBackend,
    ModelError,
    ModelRoute,
    ModelRouter,
    RateLimited,
    sanitize_external_text,
)
from core.ai.orchestrator import AgentRun, AgentSpec, Orchestrator
from core.ai.sandbox import UnsafeCodeError, run_generated, static_check
from core.ai.spec_builder import (
    GeneratedStrategy,
    StrategySpec,
    nl_to_spec,
    spec_to_signals,
    version_spec,
)
from core.ai.tools import ToolPermissionError, ToolRegistry

__all__ = [
    "AgentRun", "AgentSpec", "BudgetExceeded", "FakeModel", "GeneratedStrategy",
    "ModelBackend", "ModelError", "ModelRoute", "ModelRouter", "Orchestrator",
    "RateLimited", "StrategySpec", "ToolPermissionError", "ToolRegistry",
    "UnsafeCodeError", "build_agent_specs", "nl_to_spec", "register_tools",
    "run_generated", "sanitize_external_text", "spec_to_signals",
    "static_check", "version_spec",
]
