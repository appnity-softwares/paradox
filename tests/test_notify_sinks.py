"""Notification sink tests: webhook delivery against a real local HTTP
server, SMTP message formatting, fail-safe behaviour, and env wiring.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.execution import Notifier
from core.execution.notify_sinks import (SmtpSink, WebhookSink,
                                         attach_configured_sinks)


class _Collector(BaseHTTPRequestHandler):
    received: list[dict] = []
    auth_seen: list[str] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _Collector.received.append(json.loads(self.rfile.read(length)))
        _Collector.auth_seen.append(self.headers.get("Authorization", ""))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture()
def webhook_server():
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    _Collector.received = []
    _Collector.auth_seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _Collector
    server.shutdown()
    thread.join(timeout=5)


def test_webhook_sink_delivers_json_with_token(webhook_server):
    url, collector = webhook_server
    sink = WebhookSink(url=url, token="secret-token")
    notifier = Notifier()
    notifier.add_sink(sink)
    notifier.notify("kill_switch_engaged", "halted", scope="global")
    assert len(collector.received) == 1
    event = collector.received[0]
    assert event["kind"] == "kill_switch_engaged"
    assert event["message"] == "halted" and event["scope"] == "global"
    assert collector.auth_seen[0] == "Bearer secret-token"


def test_webhook_failure_never_breaks_the_notification_path():
    # Unroutable URL: the sink swallows the error, the in-app feed still
    # receives the event (alerting must never break trading).
    sink = WebhookSink(url="http://127.0.0.1:1/nope", timeout_s=0.2)
    notifier = Notifier()
    notifier.add_sink(sink)
    notifier.notify("trade_opened", "x")
    assert notifier.feed[-1]["kind"] == "trade_opened"


def test_disabled_sink_is_a_noop():
    sink = WebhookSink(url="")
    notifier = Notifier()
    notifier.add_sink(sink)
    notifier.notify("trade_opened", "x")     # no request attempted
    assert notifier.feed[-1]["kind"] == "trade_opened"


def test_smtp_message_format():
    sink = SmtpSink()
    subject, body = sink.format({"kind": "reconciliation_mismatch",
                                 "message": "position unknown",
                                 "account": "acc-1"})
    assert subject == "[DSH] reconciliation_mismatch"
    assert "position unknown" in body and "account: acc-1" in body


def test_smtp_disabled_without_config():
    sink = SmtpSink()
    assert sink.enabled is False
    sink({"kind": "x", "message": "y"})      # no-op, no crash


def test_attach_configured_sinks_reads_environment(monkeypatch,
                                                   webhook_server):
    url, _ = webhook_server
    monkeypatch.setenv("DSH_WEBHOOK_URL", url)
    notifier = Notifier()
    attached = attach_configured_sinks(notifier)
    assert attached == ["webhook"]
    notifier.notify("feed_recovered", "back online")
    assert notifier.feed[-1]["kind"] == "feed_recovered"
    # Deselect env for the second call: nothing attaches.
    monkeypatch.delenv("DSH_WEBHOOK_URL")
    assert attach_configured_sinks(Notifier()) == []
