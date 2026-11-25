"""Orchestrator (Part 11.2): typed messages, validated structured
outputs (one retry, then fail safe), and a stored record for every agent
run: input, tools called, output, model, tokens, cost, duration, errors.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from core.ai.models import ModelError, ModelRouter, sanitize_external_text
from core.ai.tools import ToolRegistry


@dataclass
class AgentSpec:
    name: str
    description: str
    permissions: tuple[str, ...]          # max permission levels
    output_schema: type[BaseModel]
    task: str                             # model-route key


@dataclass
class AgentRun:
    agent: str
    task: str
    input: dict
    tools_called: list[dict] = field(default_factory=list)
    output: dict | None = None
    model: str = ""
    tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    status: str = "ok"                    # ok | failed | failed_safe


class Orchestrator:
    def __init__(self, router: ModelRouter, registry: ToolRegistry,
                 run_log_path: str | Path | None = None):
        self.router = router
        self.registry = registry
        self.agents: dict[str, AgentSpec] = {}
        self.runs: list[AgentRun] = []
        self._run_log = Path(run_log_path) if run_log_path else None

    def register_agent(self, spec: AgentSpec) -> None:
        self.agents[spec.name] = spec

    def run(self, agent_name: str, payload: dict, *,
            user: str = "default", external_data: dict[str, str] | None = None
            ) -> AgentRun:
        """Run an agent. `external_data` values (rulebook text, news,
        uploads) are sanitized as data-only before reaching the model."""
        spec = self.agents.get(agent_name)
        if spec is None:
            raise ModelError(f"unknown agent {agent_name!r}")
        run = AgentRun(agent=agent_name, task=spec.task, input=payload)
        started = time.monotonic()

        # External text is data, never instructions (Part 11.2).
        context: dict = {"request": payload}
        for source, text in (external_data or {}).items():
            context[f"data_{source}"] = sanitize_external_text(text, source)

        # Structured output: validate; ONE retry; then fail safe.
        output = None
        for attempt in (1, 2):
            try:
                reply = self.router.complete(spec.task,
                                             spec.output_schema.model_json_schema(),
                                             context, user=user)
                run.model, run.tokens = reply.model, \
                    reply.tokens_in + reply.tokens_out
                run.cost_usd += reply.cost_usd
                output = spec.output_schema.model_validate(reply.output)
                break
            except ValidationError as exc:
                run.errors.append(f"schema_validation_attempt_{attempt}: "
                                  f"{exc.errors()[0]['msg']}")
            except ModelError as exc:
                run.errors.append(f"model_error_attempt_{attempt}: {exc}")
                break        # provider failure: no point retrying the schema
        run.duration_ms = (time.monotonic() - started) * 1000
        if output is None:
            run.status = "failed_safe" if run.errors else "failed"
            self._store(run)
            return run
        run.output = output.model_dump()
        self._store(run)
        return run

    def call_tool(self, run: AgentRun, name: str, **kwargs):
        """Agents call tools through the orchestrator so every call is
        recorded and permission-checked."""
        spec = self.agents[run.agent]
        result = self.registry.call(run.agent, spec.permissions, name,
                                    **kwargs)
        run.tools_called.append({"tool": name,
                                 "args_keys": sorted(kwargs)})
        return result

    def _store(self, run: AgentRun) -> None:
        self.runs.append(run)
        if self._run_log is not None:
            self._run_log.parent.mkdir(parents=True, exist_ok=True)
            with self._run_log.open("a") as f:
                f.write(json.dumps({
                    "agent": run.agent, "task": run.task,
                    "input": run.input, "tools_called": run.tools_called,
                    "output": run.output, "model": run.model,
                    "tokens": run.tokens, "cost_usd": run.cost_usd,
                    "duration_ms": run.duration_ms, "errors": run.errors,
                    "status": run.status}) + "\n")
