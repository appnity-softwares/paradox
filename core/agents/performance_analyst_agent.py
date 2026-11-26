"""Performance Analyst Agent - Trade analytics, reporting, improvement suggestions.

Responsibilities:
- Analyze trade history and performance metrics
- Identify weaknesses in strategy execution
- Suggest concrete improvements
- Generate performance reports
- NEVER executes trades directly
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.orchestrator import AgentCapability, AgentMessage, BaseAgent, MessageType


class PerformanceAnalystAgent(BaseAgent):
    def __init__(self, orchestrator):
        super().__init__("performance_analyst_agent", "Performance Analyst Agent", orchestrator)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                name="analyze_performance",
                description="Analyze strategy/trade performance and identify weaknesses",
                input_schema={"trades": "list[dict]", "equity_curve": "list[float]", "strategy_name": "str"},
                output_schema={"metrics": "dict", "weaknesses": "list[str]", "suggestions": "list[str]"},
            ),
            AgentCapability(
                name="generate_report",
                description="Generate performance report for a strategy or account",
                input_schema={"strategy_name": "str", "period_days": "int", "include_charts": "bool"},
                output_schema={"report_id": "str", "summary": "dict", "recommendations": "list[str]"},
            ),
            AgentCapability(
                name="compare_with_benchmark",
                description="Compare strategy performance against benchmark (buy & hold, index)",
                input_schema={"strategy_returns": "list[float]", "benchmark_returns": "list[float]"},
                output_schema={"alpha": "float", "beta": "float", "information_ratio": "float"},
            ),
        ]

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        payload = message.payload
        action = payload.get("action", "")

        if action == "analyze_performance":
            result = self._analyze_performance(payload)
            return await self.reply_to(message, {"success": True, **result})

        if action == "generate_report":
            result = self._generate_report(payload)
            return await self.reply_to(message, {"success": True, **result})

        if action == "compare_with_benchmark":
            result = self._compare_with_benchmark(payload)
            return await self.reply_to(message, {"success": True, **result})

        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            payload={"error": f"Unknown action: {action}", "success": False},
            correlation_id=message.id,
        )

    def _analyze_performance(self, payload: dict) -> dict:
        trades = payload.get("trades", [])
        equity_curve = payload.get("equity_curve", [])
        strategy_name = payload.get("strategy_name", "unknown")

        total_trades = len(trades)
        if total_trades == 0:
            return {
                "strategy_name": strategy_name,
                "error": "No trades to analyze",
                "metrics": {},
                "weaknesses": ["No trade data available"],
                "suggestions": ["Run backtest or collect live trade data first"],
            }

        pnls = [float(t.get("pnl", 0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / total_trades * 100
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        net_pnl = sum(pnls)

        # Consecutive losses
        max_consec_losses = 0
        current = 0
        for p in pnls:
            if p <= 0:
                current += 1
                max_consec_losses = max(max_consec_losses, current)
            else:
                current = 0

        # Drawdown from equity curve
        max_dd_pct = 0.0
        if equity_curve and len(equity_curve) > 1:
            peak = equity_curve[0]
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                max_dd_pct = max(max_dd_pct, dd)

        # Weaknesses identification
        weaknesses = []
        if win_rate < 40:
            weaknesses.append(f"Low win rate ({win_rate:.1f}%) - strategy may need entry filter refinement")
        if profit_factor < 1.0:
            weaknesses.append(f"Profit factor below 1.0 ({profit_factor:.2f}) - strategy loses money overall")
        if max_consec_losses >= 8:
            weaknesses.append(f"Long losing streak ({max_consec_losses} consecutive) - consider max consecutive loss filter")
        if max_dd_pct > 20:
            weaknesses.append(f"Deep drawdown ({max_dd_pct:.1f}%) - position sizing too aggressive")
        if avg_loss and avg_win and abs(avg_loss) > avg_win * 1.5:
            weaknesses.append(f"Average loss ({abs(avg_loss):.2f}) much larger than average win ({avg_win:.2f}) - RR imbalance")

        # Suggestions
        suggestions = []
        if win_rate < 40:
            suggestions.append("Add session filter (trade only London/NY) to improve win rate")
        if profit_factor < 1.0:
            suggestions.append("Tighten stop loss or increase take profit target to improve RR")
        if max_consec_losses >= 8:
            suggestions.append("Implement circuit breaker: stop trading after N consecutive losses")
        if max_dd_pct > 20:
            suggestions.append("Reduce position size to 0.5% per trade to lower drawdown")
        if avg_loss and avg_win and abs(avg_loss) > avg_win * 1.5:
            suggestions.append("Adjust SL/TP ratio to achieve at least 1:2 risk-reward")

        if not weaknesses:
            suggestions.append("Strategy metrics look healthy - consider increasing position size gradually")
            suggestions.append("Add more instruments to diversify")

        metrics = {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "max_drawdown_pct": round(max_dd_pct, 1),
            "max_consecutive_losses": max_consec_losses,
        }

        return {
            "strategy_name": strategy_name,
            "metrics": metrics,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
        }

    def _generate_report(self, payload: dict) -> dict:
        """Performance reports come from the REAL trade ledger, or not at all.

        The old implementation fabricated every number in the summary from a
        sha256 hash of the strategy name and attached three canned
        'recommendations' that had nothing to do with any data. A report
        without measured trades is refused.
        """
        strategy_name = payload.get("strategy_name", "unknown")
        period_days = int(payload.get("period_days", 30))
        include_charts = payload.get("include_charts", False)

        trades, source = self._measured_trades(strategy_name, period_days)
        if not trades:
            return {
                "report_id": None,
                "strategy_name": strategy_name,
                "period_days": period_days,
                "measured": False,
                "summary": None,
                "recommendations": [],
                "include_charts": include_charts,
                "error": ("no measured trades on record for this strategy and "
                          "period; a performance report cannot be generated "
                          "without real trade data"),
            }

        import statistics

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        total_return = sum(pnls)
        peak, dd = 0.0, 0.0
        equity = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            dd = min(dd, equity - peak)
        gross_win = sum(wins)
        gross_loss = -sum(p for p in pnls if p < 0)
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        sharpe = (statistics.mean(pnls) / statistics.pstdev(pnls) * (len(pnls) ** 0.5)
                  if len(pnls) > 1 and statistics.pstdev(pnls) > 0 else 0.0)

        recs = []
        if total_return < 0:
            recs.append("strategy is net negative over the measured period — review before scaling")
        if dd < -0.15 * max(1.0, abs(total_return)):
            recs.append("drawdown is large relative to net return — check stop placement")
        if not recs:
            recs.append("no specific recommendation derivable from this small sample")

        return {
            "report_id": f"RPT-{strategy_name.upper()}-{len(trades)}trades",
            "strategy_name": strategy_name,
            "period_days": period_days,
            "generated_at": __import__("datetime").datetime.now(timezone.utc).isoformat(),
            "measured": True,
            "data_source": source,
            "trade_count": len(trades),
            "summary": {
                "total_return": round(total_return, 2),
                "win_rate": round(100.0 * len(wins) / len(pnls), 1),
                "max_dd": round(dd, 2),
                "sharpe": round(sharpe, 2),
                "total_trades": len(pnls),
                "profit_factor": round(pf, 2) if pf != float("inf") else None,
            },
            "recommendations": recs,
            "include_charts": include_charts,
        }

    def _measured_trades(self, strategy_name: str, period_days: int) -> tuple[list[dict], str]:
        """Read the real trade ledgers (results/trades_{strategy}.csv from
        backtests), filtered by period. Returns (trades, source_label).
        Empty when there is nothing measured."""
        from pathlib import Path
        import csv
        from datetime import datetime, timedelta, timezone

        results_dir = Path(__file__).resolve().parents[2] / "results"
        if strategy_name in ("unknown", "all"):
            paths = sorted(results_dir.glob("trades_*.csv"))
            label = "backtest ledger (all strategies)"
        else:
            path = results_dir / f"trades_{strategy_name.lower()}.csv"
            paths = [path] if path.exists() else []
            label = "backtest ledger (in-sample)"
        if not paths:
            return [], label
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
        trades = []
        for path in paths:
            try:
                with open(path, newline="") as fh:
                    for row in csv.DictReader(fh):
                        try:
                            when = datetime.fromisoformat(row["exit_time"])
                            if when.tzinfo is None:
                                when = when.replace(tzinfo=timezone.utc)
                        except (KeyError, ValueError):
                            when = None
                        if when and when < cutoff:
                            continue
                        try:
                            trades.append({"pnl": float(row["pnl"]), "exit_time": row["exit_time"]})
                        except (KeyError, ValueError):
                            continue
            except Exception:
                continue
        return trades, label

    def _compare_with_benchmark(self, payload: dict) -> dict:
        strategy_returns = payload.get("strategy_returns", [])
        benchmark_returns = payload.get("benchmark_returns", [])

        import statistics

        if not strategy_returns or not benchmark_returns:
            return {
                "success": False,
                "error": "Both strategy_returns and benchmark_returns required",
            }

        n = min(len(strategy_returns), len(benchmark_returns))
        strat = strategy_returns[:n]
        bench = benchmark_returns[:n]

        mean_strat = statistics.mean(strat)
        mean_bench = statistics.mean(bench)

        # Simple alpha (excess return)
        alpha = mean_strat - mean_bench

        # Simple beta
        cov = sum((s - mean_strat) * (b - mean_bench) for s, b in zip(strat, bench)) / n
        var_bench = sum((b - mean_bench) ** 2 for b in bench) / n
        beta = cov / var_bench if var_bench > 0 else 0.0

        # Information ratio
        active_returns = [s - b for s, b in zip(strat, bench)]
        ir = (statistics.mean(active_returns) / statistics.stdev(active_returns)) if statistics.stdev(active_returns) > 0 else 0.0

        return {
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "information_ratio": round(ir, 4),
            "strategy_avg_return": round(mean_strat, 4),
            "benchmark_avg_return": round(mean_bench, 4),
            "n_periods": n,
        }
