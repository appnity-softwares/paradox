"""Strategy Optimization Agent - Parameter optimization and regime detection.

Responsibilities:
- Coordinate optimization runs (grid, Bayesian, genetic) via deterministic engine
- Detect strategy regime changes and suggest re-optimization
- Compare strategy performance across timeframes
- NEVER executes trades directly
"""
from __future__ import annotations

from typing import Any

from core.orchestrator import AgentCapability, AgentMessage, BaseAgent, MessageType


class StrategyOptimizationAgent(BaseAgent):
    def __init__(self, orchestrator):
        super().__init__("strategy_optimization_agent", "Strategy Optimization Agent", orchestrator)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="optimize_strategy",
                description="Request parameter optimization for a strategy via deterministic engine",
                input_schema={"strategy_name": "str", "symbol": "str", "timeframe": "str", "method": "str", "param_ranges": "dict"},
                output_schema={"best_params": "dict", "oos_performance": "dict", "verdict": "str"},
            ),
            AgentCapability(
                name="detect_regime_change",
                description="Detect if current market regime has changed from optimization baseline",
                input_schema={"strategy_name": "str", "symbol": "str", "current_regime": "str"},
                output_schema={"regime_changed": "bool", "recommend_reoptimize": "bool", "confidence": "float"},
            ),
            AgentCapability(
                name="compare_strategies",
                description="Compare multiple strategies on same data",
                input_schema={"strategies": "list[str]", "symbol": "str", "timeframe": "str"},
                output_schema={"ranking": "list[dict]", "best_strategy": "str"},
            ),
        ]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        payload = message.payload
        action = payload.get("action", "")

        if action == "optimize_strategy":
            return await self._handle_optimize(message, payload)

        if action == "detect_regime_change":
            return await self._handle_regime_change(message, payload)

        if action == "compare_strategies":
            return await self._handle_compare(message, payload)

        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown action: {action}", "success": False},
            correlation_id=message.id,
        )

    async def _handle_optimize(self, message: AgentMessage, payload: dict) -> AgentMessage | None:
        strategy_name = payload.get("strategy_name", "ema_cross")
        symbol = payload.get("symbol", "XAUUSD")
        timeframe = payload.get("timeframe", "15m")
        method = payload.get("method", "walk_forward")
        param_ranges = payload.get("param_ranges", {})

        # Send to deterministic optimization engine (registered as "optimization_engine")
        opt_msg = await self.orchestrator.route(
            AgentMessage(
                sender=self.agent_id,
                receiver="optimization_engine",
                msg_type=MessageType.REQUEST,
                payload={
                    "action": "run_optimization",
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "method": method,
                    "param_ranges": param_ranges,
                },
                priority=1,
            )
        )

        if opt_msg and opt_msg.payload.get("success"):
            return await self.reply_to(
                message,
                {
                    "success": True,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "method": method,
                    "best_params": opt_msg.payload.get("best_params", {}),
                    "oos_performance": opt_msg.payload.get("oos_performance", {}),
                    "verdict": opt_msg.payload.get("verdict", "UNKNOWN"),
                    "params_tested": opt_msg.payload.get("params_tested", 0),
                },
            )

        return await self.reply_to(
            message,
            {
                "success": False,
                "reason": opt_msg.payload.get("error", "Optimization engine unavailable") if opt_msg else "No engine",
            },
        )

    async def _handle_regime_change(self, message: AgentMessage, payload: dict) -> AgentMessage | None:
        """Regime-change verdicts require a measured baseline.

        The old implementation derived 'regime_changed' and a 'confidence'
        from a sha256 hash of the input string — deterministic noise with a
        confident tone. Without a real optimization baseline for this
        strategy the honest answer is: not measured.
        """
        strategy_name = payload.get("strategy_name", "ema_cross")
        symbol = payload.get("symbol", "XAUUSD")
        current_regime = payload.get("current_regime", "unknown")

        baseline = self._load_optimization_baseline(strategy_name, symbol)
        if baseline is None:
            return await self.reply_to(
                message,
                {
                    "success": False,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "current_regime": current_regime,
                    "regime_changed": None,
                    "recommend_reoptimize": None,
                    "confidence": 0.0,
                    "measured": False,
                    "reason": ("no optimization baseline on record for "
                               f"{strategy_name} on {symbol}; run a real "
                               "optimization first — regime-change detection "
                               "is not available"),
                },
            )

        baseline_regime = baseline.get("regime")
        regime_changed = baseline_regime is not None and current_regime != baseline_regime
        return await self.reply_to(
            message,
            {
                "success": True,
                "strategy_name": strategy_name,
                "symbol": symbol,
                "current_regime": current_regime,
                "baseline_regime": baseline_regime,
                "regime_changed": regime_changed,
                "recommend_reoptimize": regime_changed,
                "measured": True,
                "reason": (f"baseline regime was {baseline_regime!r}" if regime_changed
                           else f"current regime matches baseline {baseline_regime!r}"),
            },
        )

    def _load_optimization_baseline(self, strategy_name: str, symbol: str) -> dict | None:
        """Read the recorded optimization result for a strategy, if any."""
        from pathlib import Path
        import csv

        path = Path(__file__).resolve().parents[2] / "results" / "optimization_summary.csv"
        if not path.exists():
            return None
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("strategy", "").lower() == strategy_name.lower()
                            and row.get("symbol", symbol).upper() == symbol.upper()):
                        return dict(row)
        except Exception:
            return None
        return None

    async def _handle_compare(self, message: AgentMessage, payload: dict) -> AgentMessage | None:
        """Rank strategies ONLY on recorded optimization results.

        The old implementation generated win rates, drawdowns and Sharpes
        from a sha256 hash of the strategy name — pure fiction presented as
        a ranking. Now: strategies with a recorded, in-sample AND
        out-of-sample verified result are ranked by that; the rest are
        reported as unmeasured, never ranked.
        """
        strategies = payload.get("strategies", ["ema_cross", "macd_cross", "rsi_reversal"])
        symbol = payload.get("symbol", "XAUUSD")
        timeframe = payload.get("timeframe", "15m")

        ranked, unmeasured = [], []
        for s in strategies:
            row = self._load_optimization_baseline(s, symbol)
            if row and row.get("oos_pass") not in (None, "", "False", "false", "FAIL"):
                ranked.append({
                    "strategy": s,
                    "win_rate": row.get("oos_win_rate"),
                    "max_dd": row.get("oos_max_dd"),
                    "sharpe": row.get("oos_sharpe"),
                    "total_return": row.get("oos_total_return"),
                    "score": row.get("oos_score"),
                    "measured": True,
                })
            else:
                unmeasured.append({
                    "strategy": s,
                    "win_rate": None, "max_dd": None, "sharpe": None,
                    "total_return": None, "score": None,
                    "measured": False,
                    "note": "no verified out-of-sample result on record",
                })

        ranked.sort(key=lambda x: (x["score"] is not None, x["score"] or 0), reverse=True)

        return await self.reply_to(
            message,
            {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "ranking": ranked + unmeasured,
                "best_strategy": ranked[0]["strategy"] if ranked else None,
                "unmeasured": [u["strategy"] for u in unmeasured],
                "comparison_count": len(strategies),
                "measured": bool(ranked),
                "note": (None if ranked else
                         "no strategy has a verified out-of-sample result; "
                         "nothing is ranked — run real optimizations first"),
            },
        )
