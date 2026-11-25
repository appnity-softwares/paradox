# Phase 5 — AI Layer: Agents, Orchestrator, Strategy Builder (Progress Report)

Status: **complete per Prompt 5 (Part 11)**. The AI layer proposes;
deterministic engines decide. No LLM anywhere in numeric paths — the
whole layer runs without any provider (tests use a canned `FakeModel`),
and every agent output is validated structured data that still has to
pass the Risk Engine, Compliance Engine and Execution Router.

## What was built

### Tool registry with permission levels (11.2) — `core/ai/tools.py`
`read < propose < execute`. The registry **refuses to register an
execute-level tool at all** — "no agent has execute on broker orders" is
true by construction, not by policy. Agents call tools only through the
orchestrator, which records every call and re-checks permissions.

### Orchestrator (11.2) — `core/ai/orchestrator.py`
- Every run stored: input, tools called, output, model, tokens, cost,
  duration, errors (JSONL run log + in-memory).
- Structured outputs validated against Pydantic schemas; invalid output
  retried exactly once, then **fails safe** (`failed_safe`, no output).
- Per-task model routing with provider fallback; rate limits per task;
  daily budget cap per user (exceeded → blocked, not billed).
- Prompt-injection defence: rulebook/news/upload text is wrapped in an
  explicit DATA envelope (`sanitize_external_text`) and tested to change
  nothing — an injected "set risk to 100%" in a rulebook leaves
  `RiskConfig.max_risk_per_trade_pct` untouched, and the agent's only
  power is storing a draft rule set that grants no permission.
- No credentials, keys, or full account identifiers enter prompts
  (contexts carry IDs the tools resolve, never secrets).

### The five agents (11.1) — `core/ai/agents.py`
| Agent | Output schema | Tools |
|---|---|---|
| Research & Strategy | ResearchProposal | query_bars, run_backtest (read) |
| Account Adaptation | AdaptationRecommendation | read_account, calc_suitability (read) |
| Compliance | DraftRuleSet | add_draft_ruleset (propose, drafts only), read_compliance_status |
| Execution Assistant | TradeTicketProposal | create_order_intent (propose — an inert object; submission requires the router no agent holds), read_positions |
| Performance Analyst | PerformanceReport | write_report (propose, reports dir only) |

Market Intelligence / Portfolio Risk from the prototype are services,
not agents — exposure summaries and volatility regime are deterministic
jobs (Part 11.1 consolidation).

### Strategy Builder (11.3) — `core/ai/{spec_builder,spec_rules}.py`
Natural language → validated `StrategySpec` (JSON, one retry then fail
safe) → interpreted by the **safe rule engine** over the indicator
library (`spec_rules.py`: EMA crosses, RSI thresholds, EMA bands) — the
preferred no-codegen route. Every generated strategy is versioned
(spec + prompt hash) with the prompt stored. Deployment still requires
the Phase 1 validation stack before paper trading.

### Sandbox (11.3) — `core/ai/sandbox.py`
For free-form generated code only: AST static check rejects imports
outside {numpy, pandas, math, statistics, project indicators} and any
`os`, `subprocess`, `socket`, `eval`, `exec`, `open`, dunder access;
execution runs in a **separate process** with a memory cap, process
limit, wall-clock timeout, stdin-only input, stdout-only output, temp-dir
cwd. Generated strategies return signals only. Tested: clean code runs;
each dangerous pattern is rejected pre-execution; an infinite loop is
killed by the timeout.

## Verification

326 tests pass (24 new in `tests/test_ai_layer.py`): execute-level
impossibility, cross-agent permission denial, run-record completeness,
retry-once-then-fail-safe, provider fallback, budget cap, rate limit,
injection neutrality, intent-only execution assistant, NL→spec
validation and versioning, deterministic spec signals, and five sandbox
attack patterns plus timeout kill.

## What remains open

1. Real provider backends (API-key handling lives in the operator's
   environment; the `ModelBackend` interface is ready).
2. Telegram/email notification sinks (Notifier interface ready from 4A).
3. Wiring the Performance Analyst's drift report into a scheduled job.
