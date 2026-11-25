import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from core.broker import MarketDataResult
from dashboard.app import app
from dashboard.state import DATA_MANAGER


def fake_snapshot(timeframe="15m", rows=10):
    index = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    close = pd.Series(range(rows), index=index, dtype=float) + 2000
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100,
        },
        index=index,
    )
    return MarketDataResult(
        data=df,
        provider="Test provider",
        provider_key="test",
        symbol="XAUUSD",
        timeframe=timeframe,
        real_data=True,
        spot=False,
        delayed=True,
        fetched_at=datetime.now(timezone.utc),
    )


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_data_response_contains_provenance(self):
        with patch("dashboard.routers.data.fetch_market_data", return_value=fake_snapshot(rows=5)):
            response = self.client.get("/api/data?tf=15m&bars=5")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["bars"], 5)
        self.assertEqual(len(payload["data"]), 5)
        self.assertTrue(payload["meta"]["real_data"])
        self.assertTrue(payload["meta"]["delayed"])
        self.assertFalse(payload["meta"]["synthetic"])

    def test_backtest_rejects_unsafe_risk(self):
        response = self.client.post(
            "/api/backtest",
            json={"strategy": "ema_cross", "risk_pct": 99},
        )
        self.assertEqual(response.status_code, 422)

    def test_websocket_reports_tick_feed_unavailable_without_mt5(self):
        with patch.object(DATA_MANAGER, "is_spot", return_value=False):
            with self.client.websocket_connect("/ws/market/XAUUSD") as websocket:
                message = websocket.receive_json()
        self.assertIn(message["type"], ("unavailable", "info"))


if __name__ == "__main__":
    unittest.main()
