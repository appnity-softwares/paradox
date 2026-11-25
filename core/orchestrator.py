"""Shared Multi-Agent Orchestrator and Base Agent.

Design principles:
- Agents NEVER touch risk calculations or order permissions directly.
- Deterministic engines (risk, execution, compliance) own the final say.
- Agents communicate ONLY through the shared Orchestrator.
- Every agent action is logged and auditable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("ai_quant.agents")


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ALERT = "alert"
    COMMAND = "command"


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = "orchestrator"
    msg_type: MessageType = MessageType.REQUEST
    payload: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str | None = None
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "msg_type": self.msg_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
        }


@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: dict
    output_schema: dict


class BaseAgent(ABC):
    """Base class for all AI agents in the platform."""

    def __init__(self, agent_id: str, name: str, orchestrator: "AgentOrchestrator"):
        self.agent_id = agent_id
        self.name = name
        self.orchestrator = orchestrator
        self.status = AgentStatus.IDLE
        self.capabilities: list[AgentCapability] = []
        self.message_history: list[AgentMessage] = []
        self.metrics = {
            "messages_handled": 0,
            "errors": 0,
            "avg_response_ms": 0.0,
        }

    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process an incoming message and optionally return a response."""
        ...

    @abstractmethod
    def get_capabilities(self) -> list[AgentCapability]:
        """Return list of capabilities this agent exposes."""
        ...

    async def send(self, receiver: str, payload: dict, msg_type: MessageType = MessageType.REQUEST, priority: int = 0) -> AgentMessage | None:
        """Send a message through the orchestrator."""
        msg = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
        )
        return await self.orchestrator.route(msg)

    async def reply_to(self, original: AgentMessage, payload: dict, msg_type: MessageType = MessageType.RESPONSE) -> AgentMessage | None:
        """Reply to a specific message."""
        msg = AgentMessage(
            sender=self.agent_id,
            receiver=original.sender,
            msg_type=msg_type,
            payload=payload,
            correlation_id=original.id,
            priority=original.priority,
        )
        return await self.orchestrator.route(msg)

    async def emit_event(self, event_name: str, data: dict, priority: int = 0) -> None:
        """Broadcast an event to all agents."""
        msg = AgentMessage(
            sender=self.agent_id,
            receiver="orchestrator",
            msg_type=MessageType.EVENT,
            payload={"event": event_name, **data},
            priority=priority,
        )
        await self.orchestrator.route(msg)

    def _record_handled(self, start_ms: float) -> None:
        self.metrics["messages_handled"] += 1
        elapsed = (time.perf_counter() - start_ms) * 1000
        self.metrics["avg_response_ms"] = (
            (self.metrics["avg_response_ms"] * (self.metrics["messages_handled"] - 1) + elapsed)
            / self.metrics["messages_handled"]
        )

    def _record_error(self) -> None:
        self.metrics["errors"] += 1


class AgentOrchestrator:
    """Central communication hub for all agents.

    Agents never talk to each other directly. All messages flow through here.
    """

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}
        self.message_bus: list[AgentMessage] = []
        self.message_handlers: dict[str, Callable[[AgentMessage], Any]] = {}
        self.message_history: list[AgentMessage] = []
        self.max_history = 10_000
        self._lock = asyncio.Lock()

    def register(self, agent: BaseAgent) -> None:
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent registered: {agent.agent_id} ({agent.name})")

    def unregister(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)
        logger.info(f"Agent unregistered: {agent_id}")

    async def route(self, message: AgentMessage) -> AgentMessage | None:
        """Route a message to its target agent or handler."""
        async with self._lock:
            self.message_bus.append(message)
            self.message_history.append(message)
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history:]

        # Route to specific agent
        if message.receiver in self.agents:
            target = self.agents[message.receiver]
            target.status = AgentStatus.BUSY
            start_ms = time.perf_counter()
            try:
                response = await target.handle_message(message)
                target._record_handled(start_ms)
                target.status = AgentStatus.IDLE
                if response:
                    await self._deliver(response)
                return response
            except Exception as exc:
                target._record_error()
                target.status = AgentStatus.ERROR
                logger.exception(f"Agent {target.agent_id} failed: {exc}")
                # Return error response
                return AgentMessage(
                    sender=message.receiver,
                    receiver=message.sender,
                    msg_type=MessageType.RESPONSE,
                    payload={"error": str(exc), "success": False},
                    correlation_id=message.id,
                )

        # Route to registered handler (for deterministic engines)
        if message.receiver in self.message_handlers:
            handler = self.message_handlers[message.receiver]
            return await handler(message)

        # No handler found - a RESPONSE addressed to a non-agent (e.g. replying
        # to a user) is simply returned as the answer to the awaiting caller.
        if message.msg_type == MessageType.RESPONSE:
            return message

        logger.warning(f"No handler for receiver: {message.receiver}")
        return AgentMessage(
            sender="orchestrator",
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"No handler for {message.receiver}", "success": False},
            correlation_id=message.id,
        )

    async def _deliver(self, message: AgentMessage) -> None:
        async with self._lock:
            self.message_history.append(message)

    def register_handler(self, name: str, handler: Callable[[AgentMessage], Any]) -> None:
        """Register a deterministic engine handler (risk, execution, compliance engine)."""
        self.message_handlers[name] = handler
        logger.info(f"Handler registered: {name}")

    def get_agent_status(self) -> dict[str, dict]:
        return {
            aid: {
                "name": a.name,
                "status": a.status.value,
                "metrics": a.metrics,
                "capabilities": [c.name for c in a.capabilities],
            }
            for aid, a in self.agents.items()
        }
