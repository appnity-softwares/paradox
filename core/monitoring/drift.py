"""Drift detection to auto-pause (Update Blueprint v2.1, U6).

A DETERMINISTIC monitor (no LLM) that compares live per-deployment
statistics with the stored out-of-sample validation band for that
strategy version. Critical drift pauses the deployment (mode -> paper),
writes an audit entry and a Notifier alert. Resuming requires an
explicit authenticated human action (the API endpoint is trader-only).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

SEVERITIES = ("info", "warning", "critical")


@dataclass
class ValidationBand:
    """Stored out-of-sample validation band for one strategy version."""
    strategy_version_id: str
    win_rate: float                    # 0..1
    win_rate_tol: float = 0.10         # absolute tolerance
    avg_r: float = 0.0
    avg_r_tol: float = 0.25
    trades_per_day: float = 0.0
    trade_freq_tol: float = 1.0        # absolute trades/day tolerance
    max_drawdown_pct: float = 0.0
    max_dd_tol_pct: float = 5.0
    avg_slippage: float = 0.0
    slippage_tol: float = 0.05
    avg_spread_at_entry: float = 0.0
    spread_tol: float = 0.10

    def to_dict(self) -> dict:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationBand":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class DriftThresholds:
    warn_factor: float = 1.0      # deviation > warn_factor * tol  -> warning
    crit_factor: float = 2.0      # deviation > crit_factor * tol  -> critical
    streak_warn_prob: float = 0.05
    streak_crit_prob: float = 0.01
    stale_feed_ms: int = 60_000
    bridge_disconnect_ms: int = 300_000


@dataclass
class DriftAlert:
    ts_ms: int
    account_id: str
    deployment_id: str
    strategy_version_id: str
    severity: str
    findings: list[dict]
    action: str                     # "none" | "paused_to_paper"

    def to_dict(self) -> dict:
        return {"ts_ms": self.ts_ms, "account_id": self.account_id,
                "deployment_id": self.deployment_id,
                "strategy_version_id": self.strategy_version_id,
                "severity": self.severity, "findings": self.findings,
                "action": self.action}


class DriftAlertStore:
    """Append-only JSONL of alerts (persisted drift_alerts)."""

    def __init__(self, path: str | Path = "state/drift_alerts.jsonl"):
        self.path = Path(path)

    def append(self, alert: DriftAlert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(alert.to_dict(), sort_keys=True) + "\n")

    def entries(self, account_id: str | None = None,
                min_severity: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rank = {s: i for i, s in enumerate(SEVERITIES)}
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if account_id and rec.get("account_id") != account_id:
                continue
            if min_severity and rank.get(rec.get("severity", "info"), 0) \
                    < rank[min_severity]:
                continue
            out.append(rec)
        return out


class DriftMonitor:
    """Deterministic band comparison. No LLM anywhere in this class."""

    def __init__(self, mode_gate, audit_log, notifier,
                 alert_store: DriftAlertStore,
                 thresholds: DriftThresholds | None = None):
        self.mode_gate = mode_gate
        self.audit = audit_log
        self.notifier = notifier
        self.alert_store = alert_store
        self.thresholds = thresholds or DriftThresholds()
        self.bands: dict[str, ValidationBand] = {}

    # ── configuration ────────────────────────────────────────────────────

    def set_band(self, band: ValidationBand) -> None:
        self.bands[band.strategy_version_id] = band

    # ── evaluation ───────────────────────────────────────────────────────

    def evaluate(self, account_id: str, deployment_id: str,
                 strategy_version_id: str, live_stats: dict,
                 now_ms: int | None = None) -> DriftAlert | None:
        """Compare live stats with the stored band. Returns the alert
        (also persisted + audited); None when no band is configured or
        everything is inside it."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        band = self.bands.get(strategy_version_id)
        if band is None:
            return None
        th = self.thresholds
        findings: list[dict] = []

        def dev(metric: str, live: float, expected: float, tol: float,
                lower_is_worse: bool = False) -> None:
            delta = (expected - live) if lower_is_worse else (live - expected)
            if tol <= 0:
                tol = 1e-9
            ratio = delta / tol
            # Inside the band (ratio <= warn_factor) -> no finding at
            # all; the alert ladder starts at 'warning'.
            if ratio <= th.warn_factor:
                return
            severity = ("critical" if ratio > th.crit_factor
                        else "warning")
            findings.append({"metric": metric, "expected": expected,
                             "live": live, "tolerance": tol,
                             "deviation_ratio": round(ratio, 3),
                             "severity": severity})

        if "win_rate" in live_stats:
            dev("win_rate", float(live_stats["win_rate"]), band.win_rate,
                band.win_rate_tol, lower_is_worse=True)
        if "avg_r" in live_stats:
            dev("avg_r", float(live_stats["avg_r"]), band.avg_r,
                band.avg_r_tol, lower_is_worse=True)
        if "trades_per_day" in live_stats:
            dev("trade_frequency",
                float(live_stats["trades_per_day"]), band.trades_per_day,
                band.trade_freq_tol)
        if "max_drawdown_pct" in live_stats:
            dev("max_drawdown_pct", float(live_stats["max_drawdown_pct"]),
                band.max_drawdown_pct, band.max_dd_tol_pct)
        if "avg_slippage" in live_stats:
            dev("avg_slippage", float(live_stats["avg_slippage"]),
                band.avg_slippage, band.slippage_tol)
        if "avg_spread_at_entry" in live_stats:
            dev("avg_spread_at_entry",
                float(live_stats["avg_spread_at_entry"]),
                band.avg_spread_at_entry, band.spread_tol)
        # Losing streak vs its probability under the VALIDATED win rate:
        # P(>= k consecutive losses) = (1 - p)^k.
        streak = live_stats.get("losing_streak")
        if streak:
            p = min(max(band.win_rate, 0.0), 1.0)
            prob = (1.0 - p) ** int(streak)
            if prob >= th.streak_warn_prob:
                pass                      # inside the band
            elif prob < th.streak_crit_prob:
                findings.append({"metric": "losing_streak",
                                 "expected_prob_min": th.streak_crit_prob,
                                 "live_prob": prob, "streak": int(streak),
                                 "severity": "critical"})
            elif prob < th.streak_warn_prob:
                findings.append({"metric": "losing_streak",
                                 "expected_prob_min": th.streak_warn_prob,
                                 "live_prob": prob, "streak": int(streak),
                                 "severity": "warning"})

        severity = max((f["severity"] for f in findings),
                       key=SEVERITIES.index, default="info")
        if not findings:
            return None
        action = "none"
        if severity == "critical":
            # Pause: mode back to paper. Audited + notified.
            self.mode_gate.set_mode(account_id, "paper")
            action = "paused_to_paper"
        alert = DriftAlert(ts_ms=now, account_id=account_id,
                           deployment_id=deployment_id,
                           strategy_version_id=strategy_version_id,
                           severity=severity, findings=findings,
                           action=action)
        self.alert_store.append(alert)
        self.audit.append({"event": "drift_alert",
                           "account_id": account_id,
                           "deployment_id": deployment_id,
                           "severity": severity, "action": action,
                           "findings": findings})
        self.notifier.notify("drift_alert",
                             f"{severity}: {deployment_id} "
                             f"({len(findings)} findings) -> {action}")
        return alert

    # ── operational pauses (U6 item 3) ───────────────────────────────────

    def check_operational_pause(self, account_id: str, deployment_id: str,
                                *, drawdown_hard_stop: bool = False,
                                daily_loss_stop: bool = False,
                                reconciliation_mismatch: bool = False,
                                feed_stale_ms: int = 0,
                                bridge_disconnected_ms: int = 0,
                                now_ms: int | None = None) -> DriftAlert | None:
        """Pause triggers beyond statistical drift. Any hit pauses the
        deployment to paper with an audit entry and alert."""
        th = self.thresholds
        hits: list[str] = []
        if drawdown_hard_stop:
            hits.append("drawdown_hard_stop")
        if daily_loss_stop:
            hits.append("daily_loss_stop")
        if reconciliation_mismatch:
            hits.append("reconciliation_mismatch")
        if feed_stale_ms > th.stale_feed_ms:
            hits.append("stale_feed")
        if bridge_disconnected_ms > th.bridge_disconnect_ms:
            hits.append("bridge_disconnect")
        if not hits:
            return None
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        self.mode_gate.set_mode(account_id, "paper")
        alert = DriftAlert(ts_ms=now, account_id=account_id,
                           deployment_id=deployment_id,
                           strategy_version_id="", severity="critical",
                           findings=[{"metric": h, "severity": "critical"}
                                     for h in hits],
                           action="paused_to_paper")
        self.alert_store.append(alert)
        self.audit.append({"event": "operational_pause",
                           "account_id": account_id,
                           "deployment_id": deployment_id,
                           "hits": hits})
        self.notifier.notify("operational_pause",
                             f"{deployment_id} paused: {hits}")
        return alert

    # ── reports (deterministic numbers only) ─────────────────────────────

    def report_data(self, from_ms: int, to_ms: int) -> dict:
        """Daily/weekly report DATA. The performance_analyst agent may
        write the NARRATIVE from these numbers — never the reverse."""
        alerts = [a for a in self.alert_store.entries()
                  if from_ms <= a.get("ts_ms", 0) <= to_ms]
        by_sev = {s: sum(1 for a in alerts if a.get("severity") == s)
                  for s in SEVERITIES}
        pauses = [a for a in alerts if a.get("action") == "paused_to_paper"]
        return {"from_ms": from_ms, "to_ms": to_ms, "total_alerts":
                len(alerts), "by_severity": by_sev,
                "pauses": len(pauses),
                "accounts": sorted({a.get("account_id") for a in alerts
                                    if a.get("account_id")})}
