"""Smart Money Concepts & Quantitative Prop-Firm Challenge Simulator.

Implements:
1. Standard Institutional Prop Firm Profiles (FTMO, Funding Pips, MFF, Alpha Capital).
2. Monte Carlo Bootstrap Simulation (1,000 iterations).
3. Deterministic Daily Drawdown & Total Drawdown rule validation.
4. Pass / Ruin / Inconclusive probability distributions & sample equity paths.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class PropFirmProfile:
    id: str
    name: str
    initial_balance: float
    profit_target_pct: float
    max_daily_loss_pct: float
    max_total_loss_pct: float
    min_trading_days: int = 4
    max_trading_days: Optional[int] = None
    leverage: str = "1:100"
    news_trading_allowed: bool = True
    weekend_holding_allowed: bool = True

    @property
    def target_amount(self) -> float:
        return self.initial_balance * (self.profit_target_pct / 100.0)

    @property
    def daily_loss_limit_amount(self) -> float:
        return self.initial_balance * (self.max_daily_loss_pct / 100.0)

    @property
    def max_total_loss_amount(self) -> float:
        return self.initial_balance * (self.max_total_loss_pct / 100.0)


# Standard Institutional Prop Firm Challenge Profiles
PROP_FIRM_PROFILES: Dict[str, PropFirmProfile] = {
    "ftmo_100k": PropFirmProfile(
        id="ftmo_100k",
        name="FTMO 100K Challenge (Normal)",
        initial_balance=100_000.0,
        profit_target_pct=10.0,       # $10,000 target
        max_daily_loss_pct=5.0,        # $5,000 max daily loss
        max_total_loss_pct=10.0,       # $10,000 max total loss
        min_trading_days=4,
        max_trading_days=None,         # Unlimited time
        leverage="1:100",
    ),
    "funding_pips_100k": PropFirmProfile(
        id="funding_pips_100k",
        name="Funding Pips 100K (2-Step Evaluation)",
        initial_balance=100_000.0,
        profit_target_pct=8.0,        # $8,000 target
        max_daily_loss_pct=5.0,        # $5,000 max daily loss
        max_total_loss_pct=10.0,       # $10,000 max total loss
        min_trading_days=0,            # No minimum trading days
        max_trading_days=None,
        leverage="1:100",
    ),
    "mff_100k": PropFirmProfile(
        id="mff_100k",
        name="MyFundedFX / MFF 100K Challenge",
        initial_balance=100_000.0,
        profit_target_pct=8.0,        # $8,000 target
        max_daily_loss_pct=5.0,        # $5,000 max daily loss
        max_total_loss_pct=12.0,       # $12,000 max total loss (generous)
        min_trading_days=5,
        max_trading_days=None,
        leverage="1:100",
    ),
    "alpha_capital_100k": PropFirmProfile(
        id="alpha_capital_100k",
        name="Alpha Capital 100K Evaluation",
        initial_balance=100_000.0,
        profit_target_pct=10.0,
        max_daily_loss_pct=4.0,        # 4% tighter daily drawdown
        max_total_loss_pct=8.0,        # 8% tighter total drawdown
        min_trading_days=3,
        max_trading_days=None,
        leverage="1:100",
    ),
}


class PropFirmChallengeSimulator:
    """Simulates prop firm challenge pass rates using Monte Carlo resampling."""

    def __init__(self, profile: PropFirmProfile):
        self.profile = profile

    def simulate_from_trades(
        self,
        trades: pd.DataFrame,
        num_simulations: int = 1000,
        max_trades_per_attempt: int = 150,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Runs Monte Carlo simulation on the trade list.
        
        trades DataFrame requires either 'pnl' or ('pnl_pct'/'return').
        """
        if trades is None or len(trades) < 3:
            return {
                "error": "Not enough trade samples to run Monte Carlo (minimum 3 trades required)",
                "passed_count": 0,
                "failed_count": 0,
                "pass_rate_pct": 0.0,
            }

        # Extract PnL values
        if "pnl" in trades.columns:
            trade_pnls = trades["pnl"].to_numpy(dtype=float)
        elif "net_pnl" in trades.columns:
            trade_pnls = trades["net_pnl"].to_numpy(dtype=float)
        elif "profit" in trades.columns:
            trade_pnls = trades["profit"].to_numpy(dtype=float)
        else:
            return {"error": "Trades DataFrame missing 'pnl' column"}

        # Filter out NaN/Inf
        trade_pnls = trade_pnls[np.isfinite(trade_pnls)]
        if len(trade_pnls) < 3:
            return {"error": "Insufficient valid trades after filtering NaNs"}

        # Determine trade timestamps/dates if available to model daily clustering
        has_dates = "entry_time" in trades.columns and hasattr(trades["entry_time"], "dt")

        rng = np.random.default_rng(random_seed)
        n_samples = len(trade_pnls)

        passed_count = 0
        failed_daily_count = 0
        failed_max_dd_count = 0
        inconclusive_count = 0

        trades_to_pass_list: List[int] = []
        max_daily_dd_list: List[float] = []
        max_total_dd_list: List[float] = []

        sample_paths: List[List[float]] = []

        target_usd = self.profile.target_amount
        max_daily_usd = self.profile.daily_loss_limit_amount
        max_total_usd = self.profile.max_total_loss_amount
        init_bal = self.profile.initial_balance

        for sim_idx in range(num_simulations):
            # Resample trades with replacement
            sim_trades = rng.choice(trade_pnls, size=max_trades_per_attempt, replace=True)

            balance = init_bal
            peak_balance = init_bal
            passed = False
            failed_daily = False
            failed_total = False
            trades_count = 0

            # Assume on average 2.5 trades per day if dates aren't distinct
            trades_per_day = 3
            current_day_loss = 0.0

            trajectory = [balance]
            worst_daily_dd = 0.0
            worst_total_dd = 0.0

            for t_idx, pnl in enumerate(sim_trades):
                trades_count += 1
                balance += pnl
                trajectory.append(balance)

                # Day start tracking
                if t_idx % trades_per_day == 0:
                    current_day_loss = 0.0

                if pnl < 0:
                    current_day_loss += abs(pnl)
                else:
                    current_day_loss = max(0.0, current_day_loss - pnl * 0.5)

                if balance > peak_balance:
                    peak_balance = balance

                total_dd = peak_balance - balance
                if total_dd > worst_total_dd:
                    worst_total_dd = total_dd

                if current_day_loss > worst_daily_dd:
                    worst_daily_dd = current_day_loss

                # Rule checks
                if current_day_loss >= max_daily_usd:
                    failed_daily = True
                    break

                if total_dd >= max_total_usd:
                    failed_total = True
                    break

                if (balance - init_bal) >= target_usd:
                    passed = True
                    break

            worst_daily_dd_pct = (worst_daily_dd / init_bal) * 100.0
            worst_total_dd_pct = (worst_total_dd / init_bal) * 100.0
            max_daily_dd_list.append(worst_daily_dd_pct)
            max_total_dd_list.append(worst_total_dd_pct)

            if passed:
                passed_count += 1
                trades_to_pass_list.append(trades_count)
            elif failed_daily:
                failed_daily_count += 1
            elif failed_total:
                failed_max_dd_count += 1
            else:
                inconclusive_count += 1

            # Save 25 sample paths for charting
            if len(sample_paths) < 25:
                # Downsample trajectory to 20 points
                step = max(1, len(trajectory) // 20)
                sub_traj = trajectory[::step][:20]
                if len(sub_traj) < 20:
                    sub_traj.append(trajectory[-1])
                sample_paths.append([round(p, 2) for p in sub_traj])

        pass_rate = round((passed_count / num_simulations) * 100.0, 1)
        daily_breach_rate = round((failed_daily_count / num_simulations) * 100.0, 1)
        max_dd_breach_rate = round((failed_max_dd_count / num_simulations) * 100.0, 1)
        inconclusive_rate = round((inconclusive_count / num_simulations) * 100.0, 1)

        p95_daily_dd = round(float(np.percentile(max_daily_dd_list, 95)), 2)
        p95_max_dd = round(float(np.percentile(max_total_dd_list, 95)), 2)
        avg_trades_pass = round(float(np.mean(trades_to_pass_list)), 1) if trades_to_pass_list else 0.0

        # Mathematical Expectancy
        win_trades = trade_pnls[trade_pnls > 0]
        loss_trades = trade_pnls[trade_pnls < 0]
        win_rate = (len(win_trades) / len(trade_pnls)) * 100.0 if len(trade_pnls) > 0 else 0.0
        avg_win = float(np.mean(win_trades)) if len(win_trades) > 0 else 0.0
        avg_loss = float(abs(np.mean(loss_trades))) if len(loss_trades) > 0 else 0.0
        expectancy = (win_rate / 100.0 * avg_win) - ((100.0 - win_rate) / 100.0 * avg_loss)

        # Compliance Grade
        if pass_rate >= 80.0 and p95_daily_dd < self.profile.max_daily_loss_pct * 0.8:
            grade = "A+ (INSTITUTIONAL GRADE)"
            status = "PASSED"
        elif pass_rate >= 65.0:
            grade = "B+ (PASSABLE)"
            status = "CONDITIONALLY APPROVED"
        elif pass_rate >= 50.0:
            grade = "C (HIGH RISK)"
            status = "BORDERLINE"
        else:
            grade = "D (UNACCEPTABLE RISK)"
            status = "HIGH RUIN RISK"

        return {
            "profile": asdict(self.profile),
            "simulations": num_simulations,
            "pass_rate_pct": pass_rate,
            "daily_breach_rate_pct": daily_breach_rate,
            "max_dd_breach_rate_pct": max_dd_breach_rate,
            "inconclusive_rate_pct": inconclusive_rate,
            "p95_daily_drawdown_pct": p95_daily_dd,
            "p95_max_total_drawdown_pct": p95_max_dd,
            "avg_trades_to_pass": avg_trades_pass,
            "win_rate_pct": round(win_rate, 1),
            "expectancy_usd": round(expectancy, 2),
            "grade": grade,
            "status": status,
            "sample_paths": sample_paths,
        }
