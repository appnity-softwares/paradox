"""Outbound notification sinks (Part 14.4 alerting): a generic webhook
sink (JSON POST, tested against a real local HTTP server) and an SMTP
sink (message construction tested; delivery requires a real relay and
is therefore gated behind configuration).

No secrets in code: webhook URL comes from DSH_WEBHOOK_URL (optionally
with a bearer token in DSH_WEBHOOK_TOKEN); SMTP from DSH_SMTP_* vars.
A sink that fails must never break the trading path — errors are
swallowed after logging to stderr.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


class WebhookSink:
    """POSTs every notification as JSON to a configured URL."""

    def __init__(self, url: str | None = None, token: str | None = None,
                 timeout_s: float = 5.0):
        self.url = url or os.environ.get("DSH_WEBHOOK_URL", "")
        self.token = token or os.environ.get("DSH_WEBHOOK_TOKEN", "")
        self.timeout_s = timeout_s
        self.enabled = bool(self.url)

    def __call__(self, event: dict) -> None:
        if not self.enabled:
            return
        body = json.dumps(event).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, data=body, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s):
                pass
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            # Alerting must never break trading; surface, don't raise.
            print(f"[webhook-sink] delivery failed: {exc}", file=sys.stderr)


class SmtpSink:
    """Builds and sends an email per notification. Delivery requires a
    real relay (DSH_SMTP_HOST/PORT/USER/PASS/TO); without configuration
    the sink is disabled and messages are only formatted."""

    def __init__(self, host: str | None = None, to_addr: str | None = None):
        self.host = host or os.environ.get("DSH_SMTP_HOST", "")
        self.to_addr = to_addr or os.environ.get("DSH_SMTP_TO", "")
        self.user = os.environ.get("DSH_SMTP_USER", "")
        self.enabled = bool(self.host and self.to_addr)

    def format(self, event: dict) -> tuple[str, str]:
        subject = f"[DSH] {event.get('kind', 'event')}"
        body = event.get("message", "") + "\n" + "\n".join(
            f"{k}: {v}" for k, v in event.items()
            if k not in ("kind", "message"))
        return subject, body

    def __call__(self, event: dict) -> None:
        if not self.enabled:
            return
        import smtplib
        from email.message import EmailMessage
        subject, body = self.format(event)
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["To"] = self.to_addr
        msg.set_content(body)
        try:
            with smtplib.SMTP(self.host,
                              int(os.environ.get("DSH_SMTP_PORT", "25")),
                              timeout=10) as smtp:
                if self.user:
                    smtp.starttls()
                    smtp.login(self.user,
                               os.environ.get("DSH_SMTP_PASS", ""))
                smtp.send_message(msg)
        except (OSError, smtplib.SMTPException) as exc:
            print(f"[smtp-sink] delivery failed: {exc}", file=sys.stderr)


def attach_configured_sinks(notifier) -> list[str]:
    """Attach every sink that the environment configures; returns the
    names of the sinks actually attached."""
    attached = []
    webhook = WebhookSink()
    if webhook.enabled:
        notifier.add_sink(webhook)
        attached.append("webhook")
    smtp = SmtpSink()
    if smtp.enabled:
        notifier.add_sink(smtp)
        attached.append("smtp")
    return attached
