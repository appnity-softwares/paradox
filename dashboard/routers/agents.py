"""AI layer: strategy builder, copilot chat and the agent council."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from dashboard.state import (
    AIChatReq,
    AgentMessage,
    AgentMessageReq,
    BuildReq,
    COMPLIANCE_AGENT,
    DATA_MANAGER,
    EXECUTION_ENGINE,
    MARKET_INTELLIGENCE_AGENT,
    MARKET_RAG_ENGINE,
    MessageType,
    ORCHESTRATOR,
    PAPER_ENGINE,
    PORTFOLIO_RISK_AGENT,
    RESEARCH_AGENT,
    SingleAgentExecReq,
    datetime,
    equity_to_curve,
    fetch_market_data,
    infer_tf_label,
    os,
    run_pipeline,
    timezone,
    trades_to_records,
)

router = APIRouter(tags=["agents"])


@router.post("/api/build")
def build(req: BuildReq):
    snapshot = fetch_market_data(req.timeframe, 20_000, allow_synthetic=False)
    df = snapshot.data
    tf_label = infer_tf_label(df.index)
    spec, metrics, wf, explanation, trades, wf_folds, equity = run_pipeline(
        req.text,
        df,
        tf_label,
        folds=req.folds,
    )
    return {
        "spec": spec,
        "metrics": metrics,
        "wf": wf or {},
        "wf_folds": wf_folds.to_dict("records") if wf_folds is not None else [],
        "explanation": explanation,
        "equity": equity_to_curve(equity),
        "trades": trades_to_records(trades),
        "market": snapshot.metadata(),
    }


@router.post("/api/agent/message")
async def agent_message(req: AgentMessageReq):
    """Send a message from one agent to another (or to an engine) through the orchestrator."""
    payload = dict(req.payload or {})
    payload.setdefault("action", req.action)
    msg = AgentMessage(
        sender=req.sender,
        receiver=req.receiver,
        msg_type=MessageType.REQUEST,
        payload=payload,
        priority=req.priority,
    )
    response = await ORCHESTRATOR.route(msg)
    if response is None:
        return {"success": False, "error": "No response from receiver"}
    return response.payload


@router.get("/api/agents/status")
def agent_status():
    return ORCHESTRATOR.get_agent_status()


@router.get("/api/agents/activity")
def agent_activity(limit: int = Query(50, ge=1, le=500)):
    """Recent agent message-bus activity (auditable trail)."""
    history = list(ORCHESTRATOR.message_history[-limit:])
    return {
        "count": len(history),
        "messages": [m.to_dict() for m in history],
    }


@router.get("/api/agents/{agent_id}/capabilities")
def agent_capabilities(agent_id: str):
    agent = ORCHESTRATOR.agents.get(agent_id)
    if agent is None:
        raise HTTPException(404, f"Agent {agent_id} not found")
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "status": agent.status.value,
        "capabilities": [
            {
                "name": c.name,
                "description": c.description,
                "input_schema": c.input_schema,
                "output_schema": c.output_schema,
            }
            for c in agent.capabilities
        ],
    }


@router.post("/api/agents/research")
async def agent_research(req: AgentMessageReq):
    """Convenience endpoint: send message to Research Agent."""
    req.sender = req.sender or "user"
    req.receiver = "research_agent"
    return await agent_message(req)


@router.post("/api/agents/adapt")
async def agent_adapt(req: AgentMessageReq):
    """Convenience endpoint: send message to Strategy Adaptation Agent."""
    req.sender = req.sender or "user"
    req.receiver = "strategy_adaptation_agent"
    return await agent_message(req)


@router.post("/api/agents/compliance")
async def agent_compliance(req: AgentMessageReq):
    """Convenience endpoint: send message to Compliance Agent."""
    req.sender = req.sender or "user"
    req.receiver = "compliance_agent"
    return await agent_message(req)


@router.post("/api/agents/execution")
async def agent_execution(req: AgentMessageReq):
    """Convenience endpoint: send message to Execution Agent."""
    req.sender = req.sender or "user"
    req.receiver = "execution_agent"
    return await agent_message(req)


@router.post("/api/agents/market_intelligence")
async def agent_market_intelligence(req: AgentMessageReq):
    """Convenience endpoint: send message to Market Intelligence Agent."""
    req.sender = req.sender or "user"
    req.receiver = "market_intelligence_agent"
    return await agent_message(req)


@router.post("/api/agents/optimization")
async def agent_optimization(req: AgentMessageReq):
    """Convenience endpoint: send message to Strategy Optimization Agent."""
    req.sender = req.sender or "user"
    req.receiver = "strategy_optimization_agent"
    return await agent_message(req)


@router.post("/api/agents/portfolio_risk")
async def agent_portfolio_risk(req: AgentMessageReq):
    """Convenience endpoint: send message to Portfolio Risk Agent."""
    req.sender = req.sender or "user"
    req.receiver = "portfolio_risk_agent"
    return await agent_message(req)


@router.post("/api/agents/performance_analyst")
async def agent_performance_analyst(req: AgentMessageReq):
    """Convenience endpoint: send message to Performance Analyst Agent."""
    req.sender = req.sender or "user"
    req.receiver = "performance_analyst_agent"
    return await agent_message(req)


@router.post("/api/ai/chat")
async def ai_chat(req: AIChatReq):
    """Interactive Quantitative Trading Copilot inspired by roboquant.dev."""
    user_msg = req.message.strip()
    ctx = req.context or {}
    strat = ctx.get("strategy", "ema_cross")
    tf = ctx.get("timeframe", "15m")
    last_p = float(ctx.get("lastPrice", 3557.10) or 3557.10)
    metrics = ctx.get("metrics", {})

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your_google_ai_studio_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            system_prompt = (
                f"You are the Institutional AI Quant Copilot for XAUUSD (Gold) on AI Quant Terminal. "
                f"Current Gold Price: {last_p}. Active Strategy: {strat} on {tf}. "
                f"Current Strategy Metrics: {metrics}. "
                f"Provide concise, mathematically rigorous, professional quant guidance (Hinglish/English friendly). "
                f"Cover risk management, parameter tuning, market microstructure, and trading session timing."
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [{"text": f"{system_prompt}\n\nUser Question: {user_msg}"}]}],
            )
            return {"reply": response.text, "model": "gemini-2.0-flash", "source": "llm"}
        except Exception as exc:
            pass

    msg_lower = user_msg.lower()
    rag_info = MARKET_RAG_ENGINE.get_market_pattern_rag_context(last_p)

    if any(k in msg_lower for k in ["rag", "vertex", "vector", "pattern", "similar", "match", "retrieval"]):
        matches = rag_info.get("matches") or []
        if matches:
            top_match = matches[0]
            reply = (
                f"### 🧠 Market RAG Pattern Retrieval\n\n"
                f"- **Top Historical Match:** `{top_match.get('timestamp', 'n/a')}` with **{top_match.get('similarity_pct', 0)}% similarity**.\n"
                f"- **Historical Outcome:** {rag_info.get('win_rate_pct', 'n/a')}% bullish expansion, avg forward drift `{rag_info.get('avg_forward_return_bps', 'n/a')} bps`.\n"
                f"- Source: the live pattern store built from recorded market data."
            )
        else:
            reply = (
                "### 🧠 Market RAG Pattern Retrieval\n\n"
                "- **No similar patterns on record.** The pattern store has no "
                "measured history for the current market state (no tick "
                "recorder has run yet), so there is nothing to retrieve — "
                "any similarity number here would be invented."
            )
    elif any(k in msg_lower for k in ["analyze", "audit", "performance", "result", "sharpe", "win rate"]):
        paper = PAPER_ENGINE.get_status()
        realized = paper.get("realized_pnl")
        trades = paper.get("trades", 0)
        if trades:
            reply = (
                f"### 📊 Measured Paper Performance ({strat} context, {tf})\n\n"
                f"- **Realized PnL (paper):** ${realized:,.2f} over **{trades} closed trades**.\n"
                f"- **Open positions:** {paper.get('positions', 0)}. Kill switch: "
                f"{'ENGAGED' if paper.get('kill_switch_active') else 'armed'}.\n\n"
                "**Note:** these are PAPER fills with modeled spread/slippage — "
                "not live results. Win rate / PF / Sharpe per strategy require "
                "a trade ledger, which does not exist yet."
            )
        else:
            reply = (
                "### 📊 Performance: NOT MEASURED\n\n"
                "- There is **no trade ledger yet** — no closed paper or live "
                "trades are on record, so win rate, profit factor, Sharpe and "
                "drawdown cannot be reported without inventing them.\n"
                "- The numbers this chat used to show here (64.2% WR, PF 2.18, "
                "+18.4%) were hardcoded fiction and have been removed.\n\n"
                "**To get real numbers:** run the paper engine or a backtest "
                "with real spread/swap costs, then ask again."
            )
    elif any(k in msg_lower for k in ["regime", "market", "volatility", "gold", "xau", "trend"]):
        reply = (
            f"### 🌐 Gold (XAUUSD) Market Microstructure Analysis\n\n"
            f"- **Current Spot Anchor:** `${last_p:.2f}`\n"
            f"- **Market Regime:** computed live from data (see the regime card) — not a fixed label.\n"
            f"- **Session Note:** XAUUSD movement typically peaks around the **13:30 - 16:30 UTC** window (US data & Comex open). "
            f"Specific support/resistance levels are NOT provided here — the levels previously quoted in this reply were invented, not computed."
        )
    elif any(k in msg_lower for k in ["optimize", "parameters", "sl", "tp", "risk", "stop loss"]):
        reply = (
            f"### ⚡ Parameter Optimization Matrix for Gold ({tf})\n\n"
            f"1. **Dynamic Volatility Stops (ATR):** Gold noise requires `1.4 - 1.8x ATR(14)`. Fixed pip stops often suffer premature stop-outs during London sweeps.\n"
            f"2. **Risk-to-Reward Ratio (R:R):** Target `2.2 : 1`. Even with a 45% win rate, a 2.2 R:R generates positive mathematical expectancy.\n"
            f"3. **Max Leverage Cap:** Limit effective leverage to **1:10** to preserve prop-firm compliance (never exceed 1% account risk per trade)."
        )
    elif any(k in msg_lower for k in ["mt5", "mac", "download", "real data", "data"]):
        reply = (
            f"### 🔌 Real MT5 Mac Integration Guide\n\n"
            f"- **MT5 Path (typical):** `~/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/Program Files/MetaTrader 5` — verify in the MT5 tab, which scans the real path.\n"
            f"- **Tick files on disk:** check the MT5 tab for the actual scan result — this chat does not detect them itself.\n"
            f"- **Export script:** run the `Export_XAUUSD` script inside MT5 (`MQL5/Scripts`), then click **'Sync from MT5'** in the MT5 tab.\n"
            f"  1. In MT5, double-click `Export_XAUUSD` under Scripts.\n"
            f"  2. Click **'Sync from MT5'** in the MT5 tab on the dashboard to immediately inject it into the backtester!"
        )
    else:
        reply = (
            f"### 🤖 AI Quant Copilot Ready (roboquant mode)\n\n"
            f"I am monitoring XAUUSD at `${last_p:.2f}` with strategy `{strat}` ({tf}).\n\n"
            f"Ask me anything about:\n"
            f"- **Strategy Diagnostics:** *'Analyze my strategy performance'* or *'Why did my trade lose?'*\n"
            f"- **Market Intelligence:** *'What is the current gold volatility regime?'*\n"
            f"- **Optimization:** *'Suggest best ATR multiplier and risk parameters'*.\n"
            f"- **MT5 Integration:** *'How to sync real MT5 data from Mac'*."
        )

    return {"reply": reply, "model": "quant-heuristic-engine", "source": "rule_based"}


@router.get("/api/agents/audit")
def agents_audit_endpoint():
    """Agent audit scorecard — HONEST version.

    The previous version of this endpoint returned a hardcoded 94.2
    "INSTITUTIONAL GRADE" scorecard: fixed scores per agent, invented
    walk-forward efficiency (86.2%), Monte Carlo pass rates (97.4%), Sharpe
    (1.86) and profit factor (1.74) that no code ever computed. None of it
    was measured. This version reports ONLY what is actually measured:

    - live market context (regime/session/price/spread) when data exists
    - live account and position state from the paper engine
    - portfolio risk from the risk agent on the real positions
    - every capability score is NOT EVALUATED until a real audit harness
      measures it — there is no such harness yet, so nothing is scored
    """
    from datetime import datetime, timezone

    regime_data = RESEARCH_AGENT._detect_regime("XAUUSD", "15m", 30)
    session_data = MARKET_INTELLIGENCE_AGENT._analyze_session("XAUUSD")
    mkt_ctx = MARKET_INTELLIGENCE_AGENT._get_market_context("XAUUSD")
    acct = PAPER_ENGINE.get_status()
    positions = PAPER_ENGINE.get_positions()
    risk_assessment = PORTFOLIO_RISK_AGENT._assess_portfolio_risk({
        "positions": positions,
        "account_equity": acct["equity"],
    })

    cur_price = mkt_ctx.get("price")
    spread = mkt_ctx.get("spread_points")
    atr_val = regime_data.get("metrics", {}).get("atr")
    rsi_val = regime_data.get("metrics", {}).get("rsi")
    regime = regime_data.get("regime")
    regime_measured = regime_data.get("measured", regime not in (None, "unavailable"))

    def _na(v, fmt=None):
        if v is None:
            return "NOT MEASURED"
        return fmt(v) if fmt else v

    scorecard = {
        "overall_score": None,
        "grade": "NOT EVALUATED",
        "status": "AWAITING_REAL_AUDIT",
        "note": ("No quantitative audit harness exists in this codebase. The "
                 "scores previously shown here (94.2 'INSTITUTIONAL GRADE', "
                 "per-agent 89-98) were hardcoded and never measured. This "
                 "scorecard reports only live, measured state until a real "
                 "audit harness is built."),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account_equity": acct["equity"],
        "max_prop_firm_drawdown_limit_pct": 5.0,
        "current_drawdown_pct": 0.0 if not positions else None,
        "agents": [
            {
                "id": "compliance_agent",
                "name": "Prop-Firm & Regulatory Compliance Agent",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Enforces prop-firm drawdown rules (daily loss limit, leverage caps, mandatory SL).",
                "metrics": {
                    "Daily Loss Limit": "3.0% (config value)",
                    "Kill Switch": "ENGAGED" if acct["kill_switch_active"] else "armed",
                    "Open Paper Positions": str(len(positions)),
                    "Measured Compliance Run": "NOT MEASURED",
                },
                "audit_verdict": "Rules exist in code; no measured compliance record exists yet.",
            },
            {
                "id": "portfolio_risk_agent",
                "name": "Portfolio Risk & Capital Preservation Agent",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Portfolio risk on open positions.",
                "metrics": {
                    "Current Risk % of Equity": f"{risk_assessment.get('risk_pct_of_equity', 0.0)}%",
                    "Total Notional USD": f"${risk_assessment.get('total_notional_usd', 0.0):,.2f}",
                    "Leverage Exposure": f"{risk_assessment.get('leverage', 0.0)}x",
                    "VaR / Ruin Probability": "NOT MEASURED",
                },
                "audit_verdict": "Position-based risk computed live; VaR/ruin models are not implemented.",
            },
            {
                "id": "execution_agent",
                "name": "Execution & Slippage",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Order execution path and cost model.",
                "metrics": {
                    "Effective Spread": _na(spread, lambda v: f"${v:.2f}/oz"),
                    "Execution Mode": "PAPER ONLY (no live broker attached)",
                    "Contract Standard": "100 oz per Lot",
                    "Measured Slippage": "NOT MEASURED",
                },
                "audit_verdict": "No live execution has occurred; slippage and latency are unmeasured.",
            },
            {
                "id": "market_intelligence_agent",
                "name": "Market Intelligence & Session Agent",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Session classification and tradeable-window logic.",
                "metrics": {
                    "Active Session": _na(session_data.get("session")),
                    "Session Volatility": _na(session_data.get("volatility_expectation")),
                    "Liquidity Status": _na(mkt_ctx.get("liquidity")),
                    "Predictive Accuracy": "NOT MEASURED",
                },
                "audit_verdict": "Session logic runs live; its predictive value has never been measured.",
            },
            {
                "id": "research_agent",
                "name": "Alpha Research & Signal Edge Agent",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Regime detection and strategy candidates.",
                "metrics": {
                    "Detected Regime": _na(regime, lambda v: str(v).replace("_", " ").title()),
                    "Regime Measured From Data": "YES" if regime_measured else "NO",
                    "Current Price": _na(cur_price, lambda v: f"${v:.2f}"),
                    "Signal Edge": "NOT MEASURED",
                },
                "audit_verdict": "Regime detection runs on real data; no signal edge has ever been demonstrated.",
            },
            {
                "id": "strategy_optimization_agent",
                "name": "Walk-Forward & Anti-Overfitting Agent",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Walk-forward optimization and parameter stability.",
                "metrics": {
                    "Walk-Forward Efficiency": "NOT MEASURED",
                    "Monte Carlo Pass": "NOT MEASURED",
                    "Out-of-Sample Retention": "NOT MEASURED",
                    "Verified OOS Results On Record": "0 strategies",
                },
                "audit_verdict": ("No walk-forward validation has been run on any strategy "
                                  "in this codebase (results/optimization_summary.csv holds one "
                                  "in-sample FAIL). Previously displayed WFE/Monte-Carlo numbers were fabricated."),
            },
            {
                "id": "strategy_adaptation_agent",
                "name": "Dynamic Strategy Adaptation Agent",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Regime-based parameter adaptation.",
                "metrics": {
                    "Regime Strategy Fit": _na(regime, lambda v: str(v).upper() if regime_measured else "NOT MEASURED"),
                    "Adaptation Events": "NOT MEASURED",
                    "Benefit Over Static Params": "NOT MEASURED",
                },
                "audit_verdict": "No adaptation loop has been validated against a static baseline.",
            },
            {
                "id": "performance_analyst_agent",
                "name": "Quantitative Performance Analyst",
                "score": None,
                "status": "NOT EVALUATED",
                "role": "Realized trade performance analytics.",
                "metrics": {
                    "Realized Trades Analyzed": "0 (no trade ledger on record)",
                    "Profit Factor": "NOT MEASURED",
                    "Annualized Sharpe": "NOT MEASURED",
                    "Expectancy": "NOT MEASURED",
                },
                "audit_verdict": ("No live or paper trade ledger exists yet. The previously displayed "
                                  "PF 1.74 / Sharpe 1.86 / 12,600-bar claim was fabricated."),
            },
        ],
        "executive_summary": (
            "NOT EVALUATED. This scorecard previously asserted a 94.2/100 "
            "'INSTITUTIONAL GRADE' verdict with hardcoded per-agent scores and "
            "invented walk-forward, Monte-Carlo and performance statistics. No "
            "audit harness measures any of that. Live measured context only: "
            f"price {_na(cur_price, lambda v: f'${v:.2f}')}, session "
            f"{_na(session_data.get('session'))}, paper equity "
            f"${acct['equity']:,.2f}, open positions {len(positions)}."
        ),
    }
    return scorecard
