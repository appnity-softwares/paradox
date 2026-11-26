"""Prop-firm evaluation rules: registry + breach evaluator.

HONESTY CONTRACT (this codebase's core rule):
- The rules ENGINE here is exact arithmetic against the configured limits.
- The rules DATA is pre-filled from best-known public information and each
  firm carries a `verified` flag, a `confidence` level and a `verify_url`
  pointing at the firm's own rules page. Rules change often; nothing here
  should be trusted for a live purchase without checking `verify_url`.
- No endpoint or UI may present an unverified rule as fact.

DD semantics supported:
- static_initial : total DD is a fixed line at initial_balance * (1 - limit)
- trailing_peak  : total DD trails the highest reached equity/balance
- daily balance/equity basis with a configurable reset time
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class Phase:
    """One evaluation phase (e.g. FTMO Challenge -> Verification)."""
    name: str
    profit_target_pct: float
    max_daily_loss_pct: float
    max_total_loss_pct: float
    min_trading_days: int = 0
    max_trading_days: Optional[int] = None  # None = unlimited


@dataclass(frozen=True)
class PropFirmRules:
    slug: str
    name: str
    program: str                      # flagship program this config describes
    phases: list                      # list[Phase]
    total_dd_basis: str               # "static_initial" | "trailing_peak"
    daily_dd_basis: str               # "balance" | "equity" (day-start basis)
    daily_dd_reset: str               # "midnight_cect" | "midnight_utc" | "5pm_et" | "rolling_24h"
    consistency_max_best_day_pct: Optional[float]  # None = no consistency rule
    news_restriction: str             # "none" | "funded_only" | "all"
    weekend_holding: bool
    ea_allowed: bool
    payout_split_pct: int             # profit share on funded account
    payout_min_days: int              # min days before first payout
    confidence: str                   # "high" | "medium" | "low"
    verified: bool                    # True ONLY after checking verify_url
    verified_asof: Optional[str]      # date the rules were checked
    verify_url: str
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phases"] = [asdict(p) for p in self.phases]
        return d


# ---------------------------------------------------------------------------
# Registry. confidence reflects how sure we are of EACH number as a whole;
# `verified` stays False until someone opens verify_url and confirms.
# ---------------------------------------------------------------------------

PROP_FIRMS: dict[str, PropFirmRules] = {}


def _register(rules: PropFirmRules) -> None:
    PROP_FIRMS[rules.slug] = rules


_register(PropFirmRules(
    slug="ftmo",
    name="FTMO",
    program="Challenge + Verification (Normal)",
    phases=[
        Phase("Challenge", 10.0, 5.0, 10.0, min_trading_days=4),
        Phase("Verification", 5.0, 5.0, 10.0, min_trading_days=4),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_cect",
    consistency_max_best_day_pct=None,
    news_restriction="funded_only",
    weekend_holding=True,   # funded account; NOT in challenge/verification
    ea_allowed=True,
    payout_split_pct=80,
    payout_min_days=14,
    confidence="high",
    verified=False,
    verified_asof=None,
    verify_url="https://ftmo.com/en/trading-objectives/",
    notes=[
        "Daily loss counts balance+floating at midnight CE(S)T against a 5% of initial-balance cap.",
        "Weekend holding: allowed on the funded account, NOT in Challenge/Verification.",
        "News restriction applies on the funded account (Swing accounts differ).",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="fundednext",
    name="FundedNext",
    program="Stellar 2-Step",
    phases=[
        Phase("Stellar Phase 1", 8.0, 5.0, 10.0, min_trading_days=5),
        Phase("Stellar Phase 2", 5.0, 5.0, 10.0, min_trading_days=5),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="funded_only",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=90,
    payout_min_days=14,
    confidence="medium",
    verified=False,
    verified_asof=None,
    verify_url="https://fundednext.com/trading-rules/",
    notes=[
        "15% profit share from Phase 1 is a FundedNext differentiator; funded split advertised up to 90%+.",
        "News restriction applies on funded accounts for Stellar.",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="fundingpips",
    name="FundingPips",
    program="2-Step Evaluation",
    phases=[
        Phase("Evaluation Phase 1", 8.0, 5.0, 10.0, min_trading_days=0),
        Phase("Evaluation Phase 2", 5.0, 5.0, 10.0, min_trading_days=0),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="equity",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="none",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=80,
    payout_min_days=14,
    confidence="medium",
    verified=False,
    verified_asof=None,
    verify_url="https://fundingpips.com/evaluation-process",
    notes=[
        "Advertises no minimum trading days and no consistency rule.",
        "Daily DD basis (balance vs equity) has changed before — verify.",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="the5ers",
    name="The5ers",
    program="High Stakes",
    phases=[
        Phase("High Stakes", 6.0, 5.0, 10.0, min_trading_days=0),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="none",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=80,
    payout_min_days=0,
    confidence="low",
    verified=False,
    verified_asof=None,
    verify_url="https://the5ers.com/evaluation-traders/",
    notes=[
        "The5ers runs several programs (High Stakes, Hyper Growth, Bootcamp) with DIFFERENT DD styles; this config covers High Stakes only.",
        "Numbers here are the least certain in this registry — verify every field on verify_url.",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="goatfunded",
    name="Goat Funded",
    program="2-Step (Goat)",
    phases=[
        Phase("Phase 1", 8.0, 5.0, 10.0, min_trading_days=0),
        Phase("Phase 2", 5.0, 5.0, 10.0, min_trading_days=0),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="none",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=80,
    payout_min_days=14,
    confidence="low",
    verified=False,
    verified_asof=None,
    verify_url="https://goatfunded.com/",
    notes=[
        "Goat Funded runs 1-step, 2-step and instant-funding programs with different consistency rules; this config covers the standard 2-step.",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="blueguardian",
    name="Blue Guardian",
    program="Guardian (2-Step)",
    phases=[
        Phase("Phase 1", 8.0, 4.0, 8.0, min_trading_days=0),
        Phase("Phase 2", 4.0, 4.0, 8.0, min_trading_days=0),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="none",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=85,
    payout_min_days=14,
    confidence="low",
    verified=False,
    verified_asof=None,
    verify_url="https://blueguardian.com/",
    notes=[
        "Blue Guardian has changed its program structure more than once (Guardian/Pilot variants).",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="fxify",
    name="FXIFY",
    program="2-Phase",
    phases=[
        Phase("Phase 1", 8.0, 5.0, 10.0, min_trading_days=0),
        Phase("Phase 2", 5.0, 5.0, 10.0, min_trading_days=0),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="none",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=85,
    payout_min_days=14,
    confidence="medium",
    verified=False,
    verified_asof=None,
    verify_url="https://fxify.com/",
    notes=[
        "FXIFY also sells 1-phase and 3-phase variants with different targets.",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="alphacapital",
    name="Alpha Capital Group",
    program="2-Phase Evaluation",
    phases=[
        Phase("Phase 1", 8.0, 5.0, 10.0, min_trading_days=3),
        Phase("Phase 2", 5.0, 5.0, 10.0, min_trading_days=3),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="none",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=80,
    payout_min_days=14,
    confidence="low",
    verified=False,
    verified_asof=None,
    verify_url="https://alphacapitalgroup.uk/",
    notes=[
        "Alpha Capital has scaled-plan and altitude variants; this config covers the standard 2-phase evaluation.",
        "UNVERIFIED snapshot — confirm on verify_url before purchasing.",
    ],
))

_register(PropFirmRules(
    slug="orion",
    name="Orion",
    program="2-Step (placeholder config)",
    phases=[
        Phase("Phase 1", 8.0, 5.0, 10.0, min_trading_days=0),
        Phase("Phase 2", 5.0, 5.0, 10.0, min_trading_days=0),
    ],
    total_dd_basis="static_initial",
    daily_dd_basis="balance",
    daily_dd_reset="midnight_utc",
    consistency_max_best_day_pct=None,
    news_restriction="none",
    weekend_holding=True,
    ea_allowed=True,
    payout_split_pct=80,
    payout_min_days=14,
    confidence="low",
    verified=False,
    verified_asof=None,
    verify_url="",   # fill in the official rules URL before trusting anything here
    notes=[
        "PLACEHOLDER CONFIG: this firm's real program structure could not be verified at all.",
        "Every number below is a generic industry-default guess and MUST be replaced from the firm's official rules page.",
    ],
))


# ---------------------------------------------------------------------------
# Evaluator — exact arithmetic, no opinions.
# ---------------------------------------------------------------------------

def evaluate(rules: PropFirmRules, phase_index: int, state: dict) -> dict:
    """Evaluate an account state against one phase of a firm's rules.

    state keys:
      initial_balance   float  — account starting balance
      day_start_balance float  — balance at the daily reset
      day_start_equity  float  — equity at the daily reset (for equity-basis DD)
      balance           float  — current balance
      equity            float  — current equity (balance + floating)
      peak_equity       float  — highest equity ever reached (for trailing DD)
      trading_days      int    — distinct trading days so far
      best_day_pnl      float  — largest single-day profit so far
      total_pnl         float  — closed PnL so far (target progress basis)

    Returns dict with usage percentages, breach list, warning list and
    target progress. Breach means the firm would fail the account.
    """
    phase = rules.phases[phase_index]
    initial = float(state.get("initial_balance", 0.0))
    equity = float(state.get("equity", initial))
    balance = float(state.get("balance", initial))
    day_start_balance = float(state.get("day_start_balance", initial))
    day_start_equity = float(state.get("day_start_equity", initial))
    peak = float(state.get("peak_equity", initial))
    total_pnl = float(state.get("total_pnl", 0.0))
    best_day = float(state.get("best_day_pnl", 0.0))
    trading_days = int(state.get("trading_days", 0))

    breaches, warnings = [], []

    # --- total (max) drawdown -------------------------------------------------
    if rules.total_dd_basis == "trailing_peak":
        dd_floor = peak * (1.0 - phase.max_total_loss_pct / 100.0)
        dd_desc = f"trailing {phase.max_total_loss_pct:.0f}% from peak ${peak:,.2f}"
    else:
        dd_floor = initial * (1.0 - phase.max_total_loss_pct / 100.0)
        dd_desc = f"static {phase.max_total_loss_pct:.0f}% below initial ${initial:,.2f}"
    # distance below the floor, expressed as % of initial balance
    total_used_pct = max(0.0, (dd_floor - equity) / initial * 100.0) if initial > 0 else 0.0
    if equity <= dd_floor:
        breaches.append({
            "rule": "max_total_loss",
            "detail": f"equity ${equity:,.2f} breached the {dd_desc} floor ${dd_floor:,.2f}",
        })
    elif initial > 0:
        headroom_pct = (equity - dd_floor) / initial * 100.0
        if headroom_pct < phase.max_total_loss_pct * 0.2:
            warnings.append({
                "rule": "max_total_loss",
                "detail": f"only ${equity - dd_floor:,.2f} ({headroom_pct:.2f}% of initial) above the max-loss floor",
            })

    # --- daily drawdown -------------------------------------------------------
    day_base = day_start_equity if rules.daily_dd_basis == "equity" else day_start_balance
    day_floor = day_base * (1.0 - phase.max_daily_loss_pct / 100.0)
    daily_used_pct = max(0.0, (day_floor - equity) / initial * 100.0) if initial > 0 else 0.0
    if equity <= day_floor:
        breaches.append({
            "rule": "max_daily_loss",
            "detail": (f"equity ${equity:,.2f} breached the daily floor ${day_floor:,.2f} "
                       f"({phase.max_daily_loss_pct:.0f}% below day-start {rules.daily_dd_basis} ${day_base:,.2f}, "
                       f"reset {rules.daily_dd_reset})"),
        })
    elif day_base > 0:
        day_headroom = (equity - day_floor) / day_base * 100.0
        if day_headroom < phase.max_daily_loss_pct * 0.2:
            warnings.append({
                "rule": "max_daily_loss",
                "detail": f"only {day_headroom:.2f}% of day-start {rules.daily_dd_basis} left before the daily limit",
            })

    # --- profit target --------------------------------------------------------
    target_amount = initial * phase.profit_target_pct / 100.0
    target_progress_pct = (total_pnl / target_amount * 100.0) if target_amount > 0 else 0.0
    target_hit = total_pnl >= target_amount

    # --- min trading days -----------------------------------------------------
    days_ok = trading_days >= phase.min_trading_days
    if not days_ok and target_hit:
        warnings.append({
            "rule": "min_trading_days",
            "detail": f"target hit but only {trading_days}/{phase.min_trading_days} trading days recorded",
        })

    # --- consistency rule -----------------------------------------------------
    consistency_ok = True
    if rules.consistency_max_best_day_pct and total_pnl > 0 and best_day > 0:
        best_share = best_day / total_pnl * 100.0
        consistency_ok = best_share <= rules.consistency_max_best_day_pct
        if not consistency_ok:
            breaches.append({
                "rule": "consistency",
                "detail": (f"best day ${best_day:,.2f} is {best_share:.1f}% of total profit "
                           f"(limit {rules.consistency_max_best_day_pct:.0f}%)"),
            })

    return {
        "firm": rules.slug,
        "firm_name": rules.name,
        "phase": phase.name,
        "daily_dd_used_pct": round(daily_used_pct, 3),
        "daily_dd_limit_pct": phase.max_daily_loss_pct,
        "daily_floor": round(day_floor, 2),
        "total_dd_used_pct": round(total_used_pct, 3),
        "total_dd_limit_pct": phase.max_total_loss_pct,
        "total_floor": round(dd_floor, 2),
        "total_dd_desc": dd_desc,
        "target_amount": round(target_amount, 2),
        "target_progress_pct": round(target_progress_pct, 2),
        "target_hit": target_hit,
        "min_trading_days": phase.min_trading_days,
        "trading_days": trading_days,
        "days_ok": days_ok,
        "consistency_ok": consistency_ok,
        "breaches": breaches,
        "warnings": warnings,
        "status": "BREACHED" if breaches else ("WARNING" if warnings else "OK"),
        "verified": rules.verified,
        "confidence": rules.confidence,
    }


def get_firm(slug: str) -> Optional[PropFirmRules]:
    return PROP_FIRMS.get(slug)


def list_firms() -> list[dict]:
    return [f.to_dict() for f in PROP_FIRMS.values()]
