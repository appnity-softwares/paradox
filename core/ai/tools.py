"""Tool registry with permission levels (Part 11.2).

Permissions: read < propose < execute. NO agent ever gets `execute` on
broker orders — the registry refuses to register an execute-level tool
at all, so the violation is impossible by construction. Execution
happens only through the Execution Engine after Risk and Compliance
approval (Parts 8/10/12.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

READ = "read"
PROPOSE = "propose"
EXECUTE = "execute"
_LEVELS = {READ: 0, PROPOSE: 1, EXECUTE: 2}


class ToolPermissionError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    permission: str
    description: str
    fn: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, permission: str, description: str,
                 fn: Callable[..., Any]) -> Tool:
        if permission not in _LEVELS:
            raise ToolPermissionError(f"unknown permission {permission!r}")
        if permission == EXECUTE:
            # Part 11.2: no agent has execute on broker orders.
            raise ToolPermissionError(
                "execute-level tools cannot be registered; execution "
                "happens only through the Execution Engine after Risk and "
                "Compliance approval")
        self._tools[name] = Tool(name, permission, description, fn)
        return self._tools[name]

    def allowed(self, agent_permissions: tuple[str, ...]) -> tuple[str, ...]:
        """Tool names the agent may call given its max permission."""
        max_level = max(_LEVELS[p] for p in agent_permissions)
        return tuple(name for name, tool in self._tools.items()
                     if _LEVELS[tool.permission] <= max_level)

    def call(self, agent_name: str, agent_permissions: tuple[str, ...],
             name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ToolPermissionError(f"unknown tool {name!r}")
        tool = self._tools[name]
        if _LEVELS[tool.permission] > max(_LEVELS[p]
                                          for p in agent_permissions):
            raise ToolPermissionError(
                f"agent {agent_name!r} may not call {name!r} "
                f"({tool.permission} > its permission)")
        return tool.fn(**kwargs)
