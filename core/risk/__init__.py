"""Deterministic Risk Engine (Master Prompt Part 8).

Sits between every proposed trade and the execution layer. Plain code,
no LLM dependency, fail-closed: a missing, stale or unparsable input
REJECTS the trade.

Outputs a structured :class:`RiskDecision` (approved | rejected | reduced)
with reason codes, the computed numbers and the risk configuration
version. Limits change only through a versioned :class:`RiskConfig`.
"""
from core.risk.engine import (
    AccountSnapshot,
    DeterministicRiskEngine,
    OrderProposal,
    RiskConfig,
    RiskDecision,
)
from core.risk.kill_switch import KillSwitch

__all__ = [
    "AccountSnapshot",
    "DeterministicRiskEngine",
    "OrderProposal",
    "RiskConfig",
    "RiskDecision",
    "KillSwitch",
]
