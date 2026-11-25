"""OandaStreamSource: free real-data alternative (OANDA v20 practice).

Network is never touched in tests — the HTTP poll lives in __iter__;
message parsing is a pure, offline-testable function.
"""
from __future__ import annotations

import pytest

from core.execution.live import OandaStreamSource, Tick


PRICE_MSG = {
    "type": "PRICE",
    "time": "2026-11-25T10:00:00.123456Z",
    "bids": [{"price": "4031.5"}],
    "asks": [{"price": "4032.0"}],
}
HEARTBEAT = {"type": "HEARTBEAT", "time": "2026-11-25T10:00:01Z"}


def test_ctor_requires_credentials():
    with pytest.raises(ValueError):
        OandaStreamSource(account_id="", token="")
    with pytest.raises(ValueError):
        OandaStreamSource(account_id="a", token="")


def test_ctor_validates_poll_seconds():
    with pytest.raises(ValueError):
        OandaStreamSource(account_id="a", token="t", poll_seconds=0)
    src = OandaStreamSource(account_id="a", token="t", poll_seconds=2.5)
    assert src.poll_seconds == 2.5


def test_endpoint_practice_vs_trade():
    p = OandaStreamSource(account_id="a", token="t")._endpoint()
    assert "api-fxpractice.oanda.com" in p and "XAU_USD" in p
    q = OandaStreamSource(account_id="a", token="t", env="trade")._endpoint()
    assert "api-fxtrade.oanda.com" in q


def test_parse_price_message():
    src = OandaStreamSource(account_id="a", token="t")
    tick = src._parse_price(PRICE_MSG)
    assert tick == Tick(time_utc_ms=1795600800123, bid=4031.5, ask=4032.0)


def test_parse_ignores_heartbeat_and_empty():
    src = OandaStreamSource(account_id="a", token="t")
    assert src._parse_price(HEARTBEAT) is None
    assert src._parse_price({"type": "PRICE", "bids": [], "asks": []}) is None
    assert src._parse_price({}) is None


def test_parse_falls_back_to_wall_clock_on_bad_time():
    src = OandaStreamSource(account_id="a", token="t")
    msg = dict(PRICE_MSG, time="not-a-timestamp")
    tick = src._parse_price(msg)
    assert tick is not None and tick.bid == 4031.5 and tick.ask == 4032.0
    assert tick.time_utc_ms > 0
