# Deployment Topologies

Five realistic ways to run the platform for prop-firm trading,
ordered by simplicity. The ModeGate already supports `semi_auto`
(see core/execution mode gating), which unlocks Option 1 with zero
extra infrastructure.

## Option 1 — Semi-auto (no bridge, no Windows, no permissions)

Platform runs anywhere (Ubuntu VPS or a laptop). It generates
signals only after every compliance/risk gate passes, and delivers
them as alerts (Telegram/email). The operator executes manually in
the firm's MT5 mobile/desktop app.

- Pros: zero Windows dependency; manual trading needs no EA/bridge
  permission from any firm; the full compliance layer still gates
  every signal; zero infra cost.
- Cons: execution latency is human; not unattended; operator must be
  present during session hours.
- This is the RECOMMENDED starting mode until a strategy passes the
  U9 paper soak.

## Option 2 — All-Windows single VPS

MT5 terminal + bridge + the whole platform on one Windows VPS.

- Pros: one machine, no network hop, native MetaTrader5 package.
- Cons: Windows VPS costs more; research/backtests contend with live
  trading for resources.

## Option 3 — Ubuntu platform + Windows bridge (split)

Ubuntu VPS runs the platform; a small Windows VPS runs one MT5
terminal + bridge per account (see deploy_windows_bridge.md).
TLS + token + IP allow-list between them.

- Pros: cheap heavy compute on Linux; clean separation.
- Cons: two machines to operate; bridge setup.

## Option 4 — Firm-provided VPS

Many prop firms offer a free VPS colocated with their matching
servers for funded accounts. Use it for the Windows/bridge side
(Option 2 or 3's execution half) for the lowest latency. Ask the
firm first — this is part of the U4 onboarding questions.

## Option 5 — Copy-trading from a master account

The platform trades a master (demo) account via the bridge; prop
accounts copy trades with a local copier EA.

- Pros: prop terminal needs no strategy logic.
- Cons: a copier EA still runs in the prop terminal (same permission
  question); extra slippage layer.

## Recommended staging

1. NOW: Option 1 (semi-auto) — start collecting paper data while the
   strategy problem is being solved.
2. After a strategy passes U9: ask the firm about a free VPS
   (Option 4) and deploy the bridge (checklist in
   deploy_windows_bridge.md).
3. At scale (multiple accounts): Option 3 split.
