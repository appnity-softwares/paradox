# Windows Bridge Deployment Guide (U8 step 4)

Target: the Windows VPS that runs one MetaTrader 5 terminal + one
bridge process per account, feeding the platform.

## Verified facts (do not guess — re-verify before install)

- Package: `MetaTrader5` on PyPI, latest release **5.0.6231**
  (published 2026-09-27, verified via pypi.org/pypi/MetaTrader5/5.0.6231/json).
- **Windows only** (`platform: Windows`; all wheels are
  `*-win_amd64.whl`).
- **Wheels verified for CPython 3.10, 3.11, 3.12, 3.13, 3.14**
  (cp310–cp314). Python 3.6–3.9 wheels also exist but are legacy.
- `requires_python: >=3.6,<4`; dependency `numpy>=1.7`.
- RECOMMENDATION: install **CPython 3.12 or 3.13** on the VPS (both
  have verified wheels; 3.14 is newest and supported by the package
  but 3.12/3.13 have the longest track record with numpy stacks).

## Install outline

1. Windows Server 2022/2025 VPS, RDP restricted to your IP.
2. Install MT5 terminal from the FIRM's download link (not a generic
   installer) and log into the funded/demo account.
3. Install Python 3.12 x64 + `pip install MetaTrader5 numpy pyarrow`.
4. Run the bridge: `.venv\Scripts\python -m bridge.server --config bridge.toml`
   (the bridge exposes quotes/orders over TLS to the platform).
5. Register as a Windows service (`sc create DSHBridge binPath= ...`)
   with automatic restart; or use NSSM with restart-on-failure.

## Security requirements (all mandatory)

- **TLS**: bridge listens on TLS only; self-signed cert pinned on the
  platform side or a proper cert. No plaintext listeners.
- **Token auth**: every request carries a per-account bridge token
  (env var `DSH_BRIDGE_TOKEN`, never in the repo).
- **IP allow-list**: Windows Firewall inbound rule — only the
  platform's egress IP may reach the bridge port. Block all else.
- **One terminal, one bridge per account** (prop-firm rule + process
  isolation). Never share a terminal across accounts.
- No RDP password reuse; enable account lockout; keep the VPS patched.

## Operations

- **Log rotation**: bridge writes `logs/bridge.log` with daily rotation
  and 14-day retention (configure in bridge.toml `log_rotation`).
- **Restart behaviour**: service restarts on failure (NSSM
  `restart=5000`); on start it reconnects and re-syncs symbol specs
  before accepting orders.
- **Health checks**: `GET /health` on the bridge returns terminal
  connection state, last quote age, and account login; the platform's
  monitor treats last-quote-age > 60 s as `stale_feed` (U6 operational
  pause) and > 5 min disconnect as pause trigger.
- Time sync: Windows time service must stay synced (NTP) — tick
  timestamps drive session and news logic.

## Deployment checklist

Run `python scripts/deploy_checklist.py` ON the VPS. It prints
PASS/FAIL per item and exits non-zero on any FAIL:

1. Python version has a verified MT5 wheel (3.10–3.14).
2. `import MetaTrader5` works.
3. Terminal is installed and `mt5.initialize()` succeeds.
4. Bridge config: TLS enabled, token set (length >= 32), not default.
5. Firewall: bridge port restricted (checked via `netsh` output).
6. Log rotation configured and the log directory writable.
7. Service registered with automatic restart.
8. Health endpoint responds with a fresh quote.
9. Time sync: `w32tm /query /status` shows source, offset < 1 s.
10. One terminal per account: no duplicate terminal paths in config.
