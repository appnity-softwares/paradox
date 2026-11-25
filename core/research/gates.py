"""Acceptance gates and derived risk (Update Blueprint v2.1, U7).

Gates are configurable and reported PASS/FAIL per gate. Win rate is
REPORTED but is never a pass criterion by itself. Risk per trade is
DERIVED by deterministic arithmetic from the drawdown caps — never
hand-picked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ClientProfile:
    """Configurable client profile (defaults = the stated targets)."""
    symbol: str = "XAUUSD"
    target_max_dd_pct: float = 2.0      # out-of-sample max drawdown
    hard_cap_dd_pct: float = 3.0        # never above this
    mc95_max_dd_pct: float = 3.0        # Monte Carlo 95th pct drawdown
    min_trades: int = 30                # statistical meaning
    min_trades_per_day_avg: float = 0.2
    max_no_trade_day_share: float = 0.75
    max_losing_streak_dd_pct: float = 3.0   # streak consequence cap
    session_max_minutes: int = 240


def derive_risk_pct(profile: ClientProfile, avg_loss_r: float = 1.0,
                    consecutive_losses_assumed: int = 4,
                    safety_factor: float = 0.5) -> float:
    """Deterministic risk-per-trade derivation.

    A losing streak of `consecutive_losses_assumed` trades at risk r
    each costs roughly r * avg_loss_r * sum of the compounded losses.
    We require that cost to stay within `safety_factor` of the target
    drawdown, and the hard cap to hold even at 1.5x the streak.
    Returns the risk percentage per trade (of equity)."""
    if consecutive_losses_assumed < 1:
        raise ValueError("consecutive_losses_assumed must be >= 1")
    if safety_factor <= 0 or safety_factor >= 1:
        raise ValueError("safety_factor must be in (0, 1)")
    # Simple (non-compounded, conservative-linear) streak cost:
    # k losses of r each ≈ k * r * avg_loss_r (in equity fraction).
    # r = safety_factor * target_dd / (k * avg_loss_r)
    k = consecutive_losses_assumed
    r_target = safety_factor * (profile.target_max_dd_pct / 100.0) / (k * avg_loss_r)
    # Hard cap: 1.5x the streak must stay under the hard cap.
    r_hard = (profile.hard_cap_dd_pct / 100.0) / (1.5 * k * avg_loss_r)
    return round(min(r_target, r_hard) * 100.0, 3)


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float | str
    threshold: float | str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "value": self.value, "threshold": self.threshold}


def evaluate_gates(metrics: dict, profile: ClientProfile | None = None,
                   ) -> list[GateResult]:
    """metrics keys (all deterministic, from the tick engine +
    Monte Carlo + cost stress):
      oos_max_dd_pct, mc95_dd_pct, pf_after_2x_costs,
      expectancy_r_after_2x_costs, n_trades, max_losing_streak,
      streak_dd_pct, trades_per_day_avg, no_trade_day_share,
      win_rate (reported only).
    """
    p = profile or ClientProfile()
    gates = [
        GateResult("oos_max_dd", metrics["oos_max_dd_pct"] <= p.target_max_dd_pct,
                   metrics["oos_max_dd_pct"], f"<= {p.target_max_dd_pct}"),
        GateResult("hard_cap_dd", metrics["oos_max_dd_pct"] <= p.hard_cap_dd_pct,
                   metrics["oos_max_dd_pct"], f"<= {p.hard_cap_dd_pct}"),
        GateResult("mc95_dd", metrics["mc95_dd_pct"] <= p.mc95_max_dd_pct,
                   metrics["mc95_dd_pct"], f"<= {p.mc95_max_dd_pct}"),
        GateResult("pf_after_2x_costs",
                   metrics["pf_after_2x_costs"] > 1.0,
                   metrics["pf_after_2x_costs"], "> 1.0"),
        GateResult("expectancy_after_2x_costs",
                   metrics["expectancy_r_after_2x_costs"] > 0.0,
                   metrics["expectancy_r_after_2x_costs"], "> 0"),
        GateResult("min_trades", metrics["n_trades"] >= p.min_trades,
                   metrics["n_trades"], f">= {p.min_trades}"),
        GateResult("streak_consequence_dd",
                   metrics["streak_dd_pct"] <= p.max_losing_streak_dd_pct,
                   metrics["streak_dd_pct"],
                   f"<= {p.max_losing_streak_dd_pct}"),
        GateResult("trades_per_day_avg",
                   metrics["trades_per_day_avg"] >= p.min_trades_per_day_avg,
                   metrics["trades_per_day_avg"],
                   f">= {p.min_trades_per_day_avg}"),
        GateResult("no_trade_day_share",
                   metrics["no_trade_day_share"] <= p.max_no_trade_day_share,
                   metrics["no_trade_day_share"],
                   f"<= {p.max_no_trade_day_share}"),
    ]
    return gates


def all_gates_passed(gates: list[GateResult]) -> bool:
    return all(g.passed for g in gates)


def streak_dd_pct(equity_curve: list[tuple[int, float]]) -> float:
    """Drawdown of the equity curve, in percent (peak-to-trough)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    worst = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak * 100.0)
    return round(worst, 3)


def longest_losing_streak(pnls: list[float]) -> int:
    streak = best = 0
    for pnl in pnls:
        if pnl < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def trade_calendar_stats(trade_times_ms: list[int],
                         session_days: int) -> tuple[float, float]:
    """(trades_per_day_avg, no_trade_day_share) over the session days
    the deployment was active."""
    if session_days <= 0:
        return 0.0, 1.0
    days_with_trades = len({t // 86_400_000 for t in trade_times_ms})
    per_day = len(trade_times_ms) / session_days
    no_trade_share = 1.0 - days_with_trades / session_days
    return round(per_day, 4), round(no_trade_share, 4)
