import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import core.data as data_module
from core.broker import MarketDataResult
from core.broker import BrokerStatus
from core.data_manager import DataManager


class FakeProvider:
    def key(self):
        return "fake"


class FakeManager:
    active = FakeProvider()

    def refresh_primary(self):
        return None

    def fetch_snapshot(self, *args, **kwargs):
        raise AssertionError("fresh cache should avoid a provider call")


class FailingProvider:
    def name(self):
        return "Failing provider"

    def key(self):
        return "failing"

    def fetch_ohlcv(self, *args, **kwargs):
        raise RuntimeError("offline")


class DataCacheTests(unittest.TestCase):
    def test_cached_result_respects_requested_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            original = data_module.CACHE_DIR
            data_module.CACHE_DIR = Path(directory)
            try:
                index = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
                df = pd.DataFrame(
                    {
                        "open": range(20),
                        "high": range(1, 21),
                        "low": range(20),
                        "close": range(20),
                        "volume": [1] * 20,
                    },
                    index=index,
                )
                result = MarketDataResult(
                    data=df,
                    provider="Fake",
                    provider_key="fake",
                    symbol="XAUUSD",
                    timeframe="1m",
                    real_data=True,
                    spot=True,
                    delayed=False,
                    fetched_at=datetime.now(timezone.utc),
                )
                data_module._write_cached_result(result)
                with patch("core.data_manager.get_manager", return_value=FakeManager()):
                    tail = data_module.fetch_market_data("1m", 5)
                self.assertEqual(len(tail.data), 5)
                self.assertEqual(tail.data.index[-1], index[-1])
            finally:
                data_module.CACHE_DIR = original

    def test_synthetic_fallback_is_never_labeled_real(self):
        manager = DataManager()
        provider = FailingProvider()
        manager.providers = [provider]
        manager.active = provider
        manager.statuses = [(
            provider,
            BrokerStatus(
                name=provider.name(),
                available=True,
                real_data=True,
                spot=True,
                execution=True,
            ),
        )]
        result = manager.fetch_snapshot("XAUUSD", "15m", 20, allow_synthetic=True)
        self.assertTrue(result.synthetic)
        self.assertFalse(result.real_data)
        self.assertEqual(result.provider_key, "synthetic")


if __name__ == "__main__":
    unittest.main()
