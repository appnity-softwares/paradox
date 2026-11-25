"""The five agents (Part 11.1). Each has a clear job, a toolset within
its permission level, and a validated structured output. Agents PROPOSE;
they never execute — every recommendation still flows through the
deterministic engines before anything happens.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from core.ai.orchestrator import AgentSpec


# ── Structured outputs ───────────────────────────────────────────────────────

class ResearchProposal(BaseModel):
    ideas: list[str] = Field(min_length=1)
    rationale: str
    suggested_experiments: list[str] = []
    weaknesses: list[str] = []


class AdaptationRecommendation(BaseModel):
    recommended_risk_pct: float = Field(gt=0, le=5)
    recommended_exposure_cap_lots: float = Field(gt=0)
    suitability_summary: str
    warnings: list[str] = []


class DraftRuleSet(BaseModel):
    firm: str
    program: str
    rules_text: str
    clause_citations: list[str] = Field(min_length=1)


class TradeTicketProposal(BaseModel):
    symbol: str
    side: Literal["long", "short"]
    intended_risk_pct: float = Field(gt=0, le=5)
    sl_price: float = Field(gt=0)
    tp_price: float | None = None
    rationale: str


class PerformanceReport(BaseModel):
    period: str
    summary: str
    drift_flags: list[str] = []
    recommendations: list[str] = []


# ── Agent definitions ────────────────────────────────────────────────────────

def build_agent_specs() -> dict[str, AgentSpec]:
    specs = [
        AgentSpec(
            name="research_strategy",
            description="Analyse market data, propose strategy ideas and "
                        "specs, suggest experiments, explain weaknesses",
            permissions=("read",),
            output_schema=ResearchProposal,
            task="research_proposal"),
        AgentSpec(
            name="account_adaptation",
            description="Recommend settings for an account and explain the "
                        "suitability report",
            permissions=("read",),
            output_schema=AdaptationRecommendation,
            task="adaptation_recommendation"),
        AgentSpec(
            name="compliance",
            description="Extract draft rules from rulebooks with citations, "
                        "explain and compare rules, summarize status",
            permissions=("read", "propose"),   # draft write only
            output_schema=DraftRuleSet,
            task="draft_rules"),
        AgentSpec(
            name="execution_assistant",
            description="Propose trade tickets for human approval, "
                        "summarize execution status",
            permissions=("read", "propose"),   # intent only, never submit
            output_schema=TradeTicketProposal,
            task="trade_ticket"),
        AgentSpec(
            name="performance_analyst",
            description="Compare live versus backtest, detect drift, write "
                        "daily and weekly reports",
            permissions=("read", "propose"),   # report writer
            output_schema=PerformanceReport,
            task="performance_report"),
    ]
    return {s.name: s for s in specs}


def register_tools(registry: Any, *, store=None, broker=None,
                   accounts: dict | None = None, reports_dir=None) -> None:
    """Wire read/propose tools to the real services. No execute-level
    tool exists anywhere (enforced by the registry)."""
    # Research and Strategy Agent tools.
    registry.register(
        "query_bars", "read",
        "Read resampled OHLCV bars for a symbol and time range (read-only)",
        fn=_make_query_bars())
    registry.register(
        "run_backtest", "read",
        "Run the deterministic backtest engine on a strategy config "
        "(compute only, no orders)",
        fn=_make_run_backtest())

    # Account Adaptation Agent tools.
    registry.register(
        "read_account", "read",
        "Read account balance, equity, limits (read-only)",
        fn=lambda account_id: _account_snapshot(accounts, account_id))
    registry.register(
        "calc_suitability", "read",
        "Run the deterministic suitability calculator",
        fn=lambda account_id, profile: _suitability(accounts, account_id,
                                                    profile))

    # Compliance Agent tools.
    if store is not None:
        registry.register(
            "add_draft_ruleset", "propose",
            "Store a DRAFT rule set (drafts grant no trading permission; "
            "a human must verify)",
            fn=lambda ruleset_dict: _add_draft(store, ruleset_dict))
    registry.register(
        "read_compliance_status", "read",
        "Read the current compliance monitor status",
        fn=lambda ruleset, state: _compliance_status(ruleset, state))

    # Execution Assistant tools.
    registry.register(
        "create_order_intent", "propose",
        "Create an OrderIntent object for human approval. NOT submitted; "
        "submission happens only through the Execution Router",
        fn=_make_intent_creator())
    if broker is not None:
        registry.register(
            "read_positions", "read",
            "Read open positions (read-only)",
            fn=lambda account_id: [
                p.__dict__ for p in broker.positions_of(account_id)])

    # Performance Analyst tools.
    if reports_dir is not None:
        registry.register(
            "write_report", "propose",
            "Write a markdown report into the reports directory",
            fn=lambda filename, text: _write_report(reports_dir, filename,
                                                    text))


# ── tool implementations (thin wrappers over existing services) ─────────────

def _make_query_bars():
    def query_bars(symbol: str, start_ms: int, end_ms: int,
                   timeframe: str = "15m") -> dict:
        from core.marketdata.resampler import TIMEFRAME_MAP, \
            resample_ticks_to_bars
        from core.marketdata.store import read_range
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        ticks = read_range(symbol, start_ms, end_ms)
        bars = resample_ticks_to_bars(ticks, timeframe)
        return {"symbol": symbol, "timeframe": timeframe, "n_bars": len(bars),
                "start": str(bars.index[0]) if len(bars) else None,
                "end": str(bars.index[-1]) if len(bars) else None}
    return query_bars


def _make_run_backtest():
    def run_backtest_tool(strategy_name: str, params: dict,
                          bars: "pd.DataFrame") -> dict:
        """Deterministic backtest on provided bars (compute only)."""
        import numpy as np
        from core.backtest import run_backtest
        from strategies.base import StrategyConfig
        from strategies import library
        cls = getattr(library, {"ema_cross": "EMACross"}.get(
            strategy_name, strategy_name), None)
        if cls is None or not hasattr(cls, "generate_signals"):
            raise ValueError(f"unknown strategy {strategy_name!r}")
        cfg = StrategyConfig(**{k: v for k, v in params.items()
                                if k in ("sl_method", "sl_atr_mult", "rr",
                                         "session", "max_trades_per_day",
                                         "daily_loss_limit_pct", "risk_pct")})
        kwargs = {k: v for k, v in params.items()
                  if k not in cfg.model_fields}
        equity, trades = run_backtest(bars, cls(cfg, **kwargs), cfg)
        rets = equity.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) \
            if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0
        max_dd = float((equity / equity.cummax() - 1.0).min() * 100.0)
        return {"n_trades": int(len(trades)),
                "total_return_pct": float(equity.iloc[-1] / equity.iloc[0]
                                          - 1.0) * 100.0,
                "max_drawdown_pct": max_dd, "sharpe": sharpe}
    return run_backtest_tool


def _add_draft(store, ruleset_dict: dict):
    from core.compliance import FirmRuleSet
    rs = FirmRuleSet.from_dict(ruleset_dict)
    return store.add_draft(rs)


def _account_snapshot(accounts, account_id):
    acct = accounts.get(account_id)
    if acct is None:
        raise ValueError(f"unknown account {account_id!r}")
    return {"account_id": acct.account_id, "balance": acct.balance,
            "equity": acct.equity, "leverage": acct.leverage}


def _suitability(accounts, account_id, profile):
    from core.adaptation import suitability_report
    from core.config import SYMBOL_SPECS
    acct = accounts.get(account_id)
    if acct is None:
        raise ValueError(f"unknown account {account_id!r}")
    return suitability_report(acct, profile, SYMBOL_SPECS["XAUUSD"],
                              price=2600.0)


def _compliance_status(ruleset, state):
    from core.compliance import ComplianceEngine
    return ComplianceEngine().monitor(ruleset, state).__dict__


def _make_intent_creator():
    from core.execution import OrderIntent, make_version

    def create_order_intent(idempotency_key: str, account_id: str,
                            symbol: str, side: str, intended_risk_pct: float,
                            sl_price: float, tp_price: float | None = None,
                            strategy_version=None) -> OrderIntent:
        side_int = 1 if side == "long" else -1
        if strategy_version is None:
            strategy_version = make_version(type("ManualIdea", (), {}), {})
        return OrderIntent(
            idempotency_key=idempotency_key, account_id=account_id,
            symbol=symbol, side=side_int,
            intended_risk_pct=intended_risk_pct,
            strategy_version=strategy_version, sl_price=sl_price,
            tp_price=tp_price, source="manual")
    return create_order_intent


def _write_report(reports_dir, filename: str, text: str) -> str:
    from pathlib import Path
    safe = Path(filename).name          # no path traversal
    if not safe.endswith(".md"):
        raise ValueError("reports must be markdown")
    path = Path(reports_dir) / safe
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)
