"""Deterministic Optimization Engine.

Wraps the existing grid_search and walk_forward logic from core/optimize.py.
Agents request optimization but this engine performs the actual computation.
"""
from __future__ import annotations

from typing import Any

from core.optimize import grid_search, walk_forward, PARAM_GRID


class OptimizationEngine:
    def __init__(self):
        self.param_grids = dict(PARAM_GRID)

    def run_optimization(
        self,
        df,
        strategy_cls,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        method: str = "walk_forward",
        param_ranges: dict | None = None,
        min_trades: int = 15,
        n_folds: int = 4,
        cfg=None,
    ) -> dict:
        """Run optimization and return results."""
        from copy import deepcopy

        grid = param_ranges if param_ranges else self.param_grids.get(strategy_name, {})

        if not grid:
            return {
                "success": False,
                "reason": f"No parameter grid for strategy {strategy_name}",
            }

        try:
            if method == "walk_forward":
                folds_df, pooled, summary = walk_forward(
                    df,
                    strategy_cls,
                    grid,
                    cfg or self._make_cfg(strategy_name),
                    timeframe,
                    n_folds=n_folds,
                    min_trades=min_trades,
                )
                return {
                    "success": True,
                    "method": method,
                    "best_params": summary.get("best_params", {}),
                    "oos_performance": {
                        "ret_pct": summary.get("OOS Ret %", 0),
                        "sharpe": summary.get("OOS Sharpe (med)", 0),
                        "win_rate": summary.get("OOS Win %", 0),
                        "max_dd": summary.get("OOS Max DD %", 0),
                        "total_trades": summary.get("OOS Trades", 0),
                    },
                    "verdict": summary.get("Verdict", "FAIL"),
                    "params_tested": summary.get("Param Sets Used", 0),
                    "profitable_folds_pct": summary.get("Profitable Folds %", 0),
                }
            else:
                # Grid search on full data (in-sample)
                res, best = grid_search(
                    df,
                    strategy_cls,
                    grid,
                    cfg or self._make_cfg(strategy_name),
                    timeframe,
                    min_trades=min_trades,
                )
                return {
                    "success": True,
                    "method": method,
                    "best_params": {k: best[k] for k in grid},
                    "oos_performance": {
                        "ret_pct": best.get("Return %", 0),
                        "sharpe": best.get("Sharpe", 0),
                        "win_rate": best.get("Win Rate %", 0),
                        "max_dd": best.get("Max DD %", 0),
                        "total_trades": best.get("Total Trades", 0),
                    },
                    "verdict": "IS_ONLY",  # In-sample only, not validated
                    "params_tested": len(res),
                }
        except Exception as exc:
            return {
                "success": False,
                "reason": str(exc),
            }

    def _make_cfg(self, strategy_name: str):
        from strategies.base import StrategyConfig
        return StrategyConfig(
            risk_pct=1.0,
            rr=2.0,
            sl_method="atr",
            sl_atr_mult=1.5,
            session="all",
            max_trades_per_day=4,
            daily_loss_limit_pct=3.0,
        )

    def get_available_grids(self) -> dict:
        return dict(self.param_grids)
