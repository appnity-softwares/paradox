"""Firm rule schema (Part 10.1) and the draft/verify/enforce lifecycle
(Part 10.2).

The schema records the EXACT definition each firm uses (daily-loss basis,
reset time and timezone, floating inclusion, trailing basis) because the
wrong definition breaks accounts. An LLM may PROPOSE a rule set with
clause citations; only a human verification flips status to `verified`,
and only `verified` rule sets grant trading permission.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

from core.compliance.behaviour import (
    POLICY_ALLOWED,
    POLICY_CONDITIONAL,
    POLICY_UNKNOWN,
    POLICY_VALUES,
    PROHIBITED_STYLES,
)

STATUS_DRAFT = "draft"
STATUS_VERIFIED = "verified"
STATUS_RETIRED = "retired"


@dataclass
class DailyLossRule:
    limit_pct: float
    # "balance" | "equity" | "balance_or_equity" — measured against the
    # balance/equity AT THE DAILY RESET, exactly as the firm defines it.
    basis: str
    reset_time: str                 # "17:00"
    reset_tz: str                   # IANA name, e.g. "America/New_York"
    includes_floating: bool


@dataclass
class MaxLossRule:
    limit_pct: float
    type: str                       # "static" | "trailing"
    # Static: measured from starting_balance. Trailing: from the equity
    # high watermark (or balance HWM when basis says so).
    basis: str = "starting_balance"  # "starting_balance" | "equity_hwm" | "balance_hwm"
    includes_floating: bool = True


@dataclass
class NewsRestriction:
    enabled: bool
    window_before_min: int = 2
    window_after_min: int = 2
    applies_to: str = "open_and_close"   # "open" | "close" | "open_and_close"
    # U1: separate flags for WHAT is blocked inside the window. Defaults
    # are the SAFE values (block everything) — a rulebook must state an
    # exception for it to be relaxed.
    block_open: bool = True              # new market entries
    block_close: bool = True             # manual/signal closes
    block_pending: bool = True           # new pending orders
    block_sltp: bool = True              # SL/TP hits inside the window


@dataclass
class ConsistencyRule:
    enabled: bool
    # No single day may exceed this share of total profit (e.g. 0.30 = 30%).
    best_day_max_share: float = 0.30


@dataclass
class InactivityRule:
    """U1: account-level inactivity limit (e.g. 'max N days without a
    trade or the challenge expires')."""
    enabled: bool = False
    max_inactive_days: int | None = None


@dataclass
class AutomationPolicy:
    ea_allowed: bool = True
    copy_trading_allowed: bool = False
    hft_allowed: bool = False
    notes: str = ""


@dataclass
class FirmRuleSet:
    firm: str
    program: str
    version: str
    status: str = STATUS_DRAFT
    verified_by: str | None = None
    source_url: str = ""
    # Per-field source clause citations from the rulebook, e.g.
    # {"daily_loss": ["3.2"], "max_loss": ["5.1"]}.
    source_clause_refs: dict[str, list[str]] = field(default_factory=dict)
    rulebook_hash: str = ""              # hash of the ingested rulebook text

    starting_balance: float = 100_000.0
    profit_target_pct: float = 8.0
    daily_loss: DailyLossRule = field(
        default_factory=lambda: DailyLossRule(5.0, "balance", "17:00",
                                              "America/New_York", True))
    max_loss: MaxLossRule = field(
        default_factory=lambda: MaxLossRule(10.0, "static",
                                            "starting_balance", True))
    min_trading_days: int = 4
    # A trading day counts when closed volume reaches this many lots.
    min_day_volume_lots: float = 0.01
    max_lot: float | None = None
    max_open_positions: int | None = None
    max_exposure_lots: float | None = None
    news_restriction: NewsRestriction = field(
        default_factory=lambda: NewsRestriction(enabled=False))
    weekend_holding: str = "forbidden"   # "forbidden" | "allowed"
    overnight_holding: str = "allowed"   # "allowed" | "forbidden"
    # Friday cutoff (UTC hour) after which new positions may not be opened
    # when weekend holding is forbidden.
    weekend_cutoff_utc_hour: int = 20
    consistency_rule: ConsistencyRule = field(
        default_factory=lambda: ConsistencyRule(enabled=False))
    allowed_symbols: list[str] = field(default_factory=list)   # empty = all
    trading_hours_utc: tuple[int, int] | None = None           # None = 24/5
    automation_policy: AutomationPolicy = field(default_factory=AutomationPolicy)

    # ── U1: behaviour-control schema ─────────────────────────────────────
    # All new fields default to the SAFE value when unknown: None limits
    # are "not stated in the rulebook" and stay unenforced ONLY until the
    # rulebook is verified; policy enums default to `unknown`, which
    # blocks Auto mode (fail closed).
    min_hold_seconds: int | None = None
    max_orders_per_day: int | None = None
    max_modifications_per_trade: int | None = None
    max_requests_per_minute: int | None = None
    ea_policy: str = POLICY_UNKNOWN      # allowed|forbidden|conditional|unknown
    vps_policy: str = POLICY_UNKNOWN
    ip_policy: str = POLICY_UNKNOWN
    hedging_policy: str = POLICY_UNKNOWN
    copy_trading_policy: str = POLICY_UNKNOWN
    policy_notes: dict[str, str] = field(default_factory=dict)
    inactivity_rule: InactivityRule = field(default_factory=InactivityRule)
    platform: str = ""                   # e.g. "MT5" — optional, from rulebook
    server: str = ""                     # broker server name, optional
    # Styles the firm prohibits. Blueprint default ON for the three the
    # blueprint names; a rulebook may extend the list.
    prohibited_styles: list[str] = field(
        default_factory=lambda: ["martingale", "grid", "averaging_down"])

    # ── derived amounts ──────────────────────────────────────────────────
    @property
    def profit_target_amount(self) -> float:
        return self.starting_balance * self.profit_target_pct / 100.0

    @property
    def daily_loss_amount(self) -> float:
        return self.starting_balance * self.daily_loss.limit_pct / 100.0

    @property
    def max_loss_amount(self) -> float:
        return self.starting_balance * self.max_loss.limit_pct / 100.0

    # ── lifecycle ────────────────────────────────────────────────────────
    def grants_trading_permission(self) -> bool:
        """Only a human-verified rule set grants permission (Part 10.2)."""
        return self.status == STATUS_VERIFIED

    def allows_auto(self) -> bool:
        """U1 RULE: ea_policy `unknown` or `forbidden` means Auto mode is
        NOT permitted. `conditional` permits Auto (conditions live in
        policy_notes and are the operator's responsibility to honour);
        `allowed` permits Auto."""
        return self.ea_policy in (POLICY_ALLOWED, POLICY_CONDITIONAL)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FirmRuleSet":
        d = dict(d)
        d["daily_loss"] = DailyLossRule(**d["daily_loss"])
        d["max_loss"] = MaxLossRule(**d["max_loss"])
        d["news_restriction"] = NewsRestriction(**d["news_restriction"])
        d["consistency_rule"] = ConsistencyRule(**d["consistency_rule"])
        d["automation_policy"] = AutomationPolicy(**d["automation_policy"])
        # U1 fields are optional in old JSON; missing keys take the
        # dataclass defaults (safe values). New-style JSON may carry the
        # nested inactivity rule as a dict.
        if isinstance(d.get("inactivity_rule"), dict):
            d["inactivity_rule"] = InactivityRule(**d["inactivity_rule"])
        if d.get("trading_hours_utc") is not None:
            d["trading_hours_utc"] = tuple(d["trading_hours_utc"])
        if d.get("prohibited_styles") is not None:
            d["prohibited_styles"] = list(d["prohibited_styles"])
        return cls(**d)

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def validate(self) -> list[str]:
        """Schema-level validation; returns a list of problems (empty = ok)."""
        problems = []
        if self.status == STATUS_VERIFIED and not self.verified_by:
            problems.append("verified ruleset requires verified_by")
        if self.daily_loss.basis not in ("balance", "equity", "balance_or_equity"):
            problems.append(f"unknown daily loss basis: {self.daily_loss.basis}")
        if self.max_loss.type not in ("static", "trailing"):
            problems.append(f"unknown max loss type: {self.max_loss.type}")
        if self.max_loss.basis not in ("starting_balance", "equity_hwm", "balance_hwm"):
            problems.append(f"unknown max loss basis: {self.max_loss.basis}")
        if self.daily_loss.limit_pct <= 0 or self.max_loss.limit_pct <= 0:
            problems.append("loss limits must be positive")
        if self.weekend_holding not in ("forbidden", "allowed"):
            problems.append(f"unknown weekend_holding: {self.weekend_holding}")
        if self.overnight_holding not in ("forbidden", "allowed"):
            problems.append(f"unknown overnight_holding: {self.overnight_holding}")
        if self.news_restriction.enabled and self.news_restriction.applies_to \
                not in ("open", "close", "open_and_close"):
            problems.append("unknown news applies_to")
        if self.consistency_rule.enabled and not \
                0 < self.consistency_rule.best_day_max_share <= 1.0:
            problems.append("consistency share must be in (0, 1]")
        # ── U1 schema validation ─────────────────────────────────────────
        for name in ("min_hold_seconds", "max_orders_per_day",
                     "max_modifications_per_trade", "max_requests_per_minute"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or v < 0):
                problems.append(f"{name} must be a non-negative int or null")
        for name in ("ea_policy", "vps_policy", "ip_policy",
                     "hedging_policy", "copy_trading_policy"):
            if getattr(self, name) not in POLICY_VALUES:
                problems.append(f"unknown {name}: {getattr(self, name)}")
        if self.inactivity_rule.enabled and \
                (self.inactivity_rule.max_inactive_days is None
                 or self.inactivity_rule.max_inactive_days < 1):
            problems.append("inactivity_rule needs max_inactive_days >= 1")
        unknown_styles = [s for s in self.prohibited_styles
                          if s not in PROHIBITED_STYLES]
        if unknown_styles:
            problems.append(f"unknown prohibited styles: {unknown_styles}")
        # Every loss-related field should carry a clause citation once verified.
        if self.status == STATUS_VERIFIED:
            for key in ("daily_loss", "max_loss"):
                if not self.source_clause_refs.get(key):
                    problems.append(f"missing clause citation for {key}")
        return problems


def diff_rulesets(old: FirmRuleSet, new: FirmRuleSet) -> dict[str, tuple]:
    """Field-level diff between two versions (Part 10.2 step 5)."""
    a, b = old.to_dict(), new.to_dict()
    changes = {}
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            changes[key] = (a.get(key), b.get(key))
    return changes
