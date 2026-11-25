"""Trading modes and the mode gate (Part 12.5).

Research and Backtest never place orders. Manual and Semi-auto require a
human approval token per order. Moving an account to Auto requires ALL
of: a strategy verdict at PAPER_TRADE_ELIGIBLE or better, a minimum
paper-trading period with no unresolved errors, a verified rule set (for
prop accounts), and an explicit user confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

RESEARCH = "research"
BACKTEST = "backtest"
PAPER = "paper"
MANUAL = "manual"
SEMI_AUTO = "semi_auto"
AUTO = "auto"

ORDER_MODES = {PAPER, MANUAL, SEMI_AUTO, AUTO}
HUMAN_APPROVAL_MODES = {MANUAL, SEMI_AUTO}

# Auto-mode prerequisites.
MIN_PAPER_DAYS = 14
ELIGIBLE_VERDICTS = {"PAPER_TRADE_ELIGIBLE"}


class ModeError(Exception):
    pass


@dataclass
class AccountMode:
    mode: str = RESEARCH
    # Evidence recorded when Auto was granted.
    verdict: str | None = None
    paper_days: int = 0
    rules_verified: bool = False
    confirmed_by: str | None = None
    unresolved_errors: int = 0


class ModeGate:
    def __init__(self):
        self._modes: dict[str, AccountMode] = {}

    def mode_of(self, account_id: str) -> str:
        return self._modes.get(account_id, AccountMode()).mode

    def set_mode(self, account_id: str, mode: str, *,
                 verdict: str | None = None, paper_days: int = 0,
                 rules_verified: bool = False,
                 confirmed_by: str | None = None,
                 unresolved_errors: int = 0,
                 ruleset=None,
                 network_reasons: tuple[str, ...] = ()) -> AccountMode:
        """`ruleset` (U1): when a FirmRuleSet is supplied and mode is
        AUTO, an ea_policy of `unknown` or `forbidden` refuses Auto
        (fail closed). Manual/Semi-auto stay available."""
        if mode not in {RESEARCH, BACKTEST, PAPER, MANUAL, SEMI_AUTO, AUTO}:
            raise ModeError(f"unknown mode {mode}")
        if mode == AUTO:
            missing = []
            if verdict not in ELIGIBLE_VERDICTS:
                missing.append(f"verdict {verdict!r} not in {ELIGIBLE_VERDICTS}")
            if paper_days < MIN_PAPER_DAYS:
                missing.append(f"only {paper_days} paper days < {MIN_PAPER_DAYS}")
            if unresolved_errors > 0:
                missing.append(f"{unresolved_errors} unresolved errors")
            if not rules_verified:
                missing.append("rule set not verified")
            if not confirmed_by:
                missing.append("no explicit user confirmation")
            if ruleset is not None and not ruleset.allows_auto():
                missing.append(
                    f"ea_policy {ruleset.ea_policy!r} does not permit "
                    f"auto mode")
            # U5: network identity conflicts (shared IP with a firm
            # whose vps/ip policy is not 'allowed', unexpected IP
            # change) block Auto.
            if network_reasons:
                missing.append("network identity conflict: "
                               + "; ".join(network_reasons))
            if missing:
                raise ModeError("auto mode refused: " + "; ".join(missing))
        am = AccountMode(mode, verdict, paper_days, rules_verified,
                         confirmed_by, unresolved_errors)
        self._modes[account_id] = am
        return am

    def check_order_permission(self, account_id: str, *,
                               approval_token: str | None = None) -> None:
        """Raises ModeError when the mode does not allow this order."""
        am = self._modes.get(account_id, AccountMode())
        if am.mode not in ORDER_MODES:
            raise ModeError(
                f"mode {am.mode!r} places no orders (account {account_id})")
        if am.mode in HUMAN_APPROVAL_MODES and not approval_token:
            raise ModeError(
                f"mode {am.mode} requires a human approval token per order")
