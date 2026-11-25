import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROTOTYPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROTOTYPE_ROOT not in sys.path:
    sys.path.insert(0, PROTOTYPE_ROOT)

import numpy as np
import pandas as pd

from core.vertex_store import VERTEX_STORE, NoTickDataError, VertexStore
from core.tick_backtest import TICK_BACKTEST_ENGINE
from core.rag_engine import MARKET_RAG_ENGINE


def synthetic_tick_frame(rows: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Deterministic, clearly-labelled synthetic ticks for unit tests only."""
    rng = np.random.default_rng(seed)
    base_ms = 1_700_000_000_000
    last = np.round(3900.0 + np.cumsum(rng.normal(0, 0.05, rows)), 3)
    return pd.DataFrame(
        {
            "time_ms": (base_ms + np.arange(rows) * 1000).astype(np.int64),
            "bid": np.round(last - 0.15, 3),
            "ask": np.round(last + 0.15, 3),
            "last": last,
            "volume": rng.integers(1, 10, rows),
        }
    )


class VertexAndRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls._patches = [
            patch.object(VertexStore, "PARQUET_FILE", tmp / "ticks.parquet"),
            patch.object(VertexStore, "VECTORS_FILE", tmp / "vectors.npz"),
        ]
        for p in cls._patches:
            p.start()
        VERTEX_STORE.reset()
        VERTEX_STORE.write_ticks(synthetic_tick_frame(), source="synthetic_test")

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        VERTEX_STORE.reset()
        cls._tmp.cleanup()

    def test_vertex_store_compression_and_status(self):
        status = VERTEX_STORE.get_status()
        self.assertEqual(status["status"], "READY")
        self.assertGreater(status["tick_count"], 1000)
        self.assertGreater(status["parquet_size_kb"], 0)
        self.assertIn("x", status["compression_ratio"])
        self.assertGreater(status["vector_count"], 10)
        self.assertEqual(status["sources"], ["synthetic_test"])

    def test_production_build_never_fabricates_ticks(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            empty = Path(empty_dir)
            with (
                patch.object(VertexStore, "PARQUET_FILE", empty / "ticks.parquet"),
                patch.object(VertexStore, "VECTORS_FILE", empty / "vectors.npz"),
                patch("core.vertex_store.MT5_MAC_DIR", empty / "no-mt5"),
                patch("core.marketdata.store.TICK_ROOT", empty / "no-ticks"),
            ):
                store = VertexStore()
                with self.assertRaises(NoTickDataError):
                    store.build_and_compress_vertex_store(force_refresh=True)
            self.assertFalse((empty / "ticks.parquet").exists())

    def test_market_rag_similarity_search(self):
        matches = MARKET_RAG_ENGINE.search_similar_patterns(top_k=5, min_similarity=0.50)
        self.assertIsInstance(matches, list)
        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertIn("similarity_pct", top)
        self.assertIn("outcome", top)
        self.assertGreaterEqual(top["similarity_pct"], 50.0)

    def test_rag_context_generation(self):
        rag_data = MARKET_RAG_ENGINE.get_market_pattern_rag_context()
        self.assertTrue(rag_data["rag_available"])
        self.assertIn("summary", rag_data)
        self.assertGreaterEqual(rag_data["sample_size"], 1)

    def test_tick_backtest_execution(self):
        res = TICK_BACKTEST_ENGINE.run(
            strategy_key="smc_fvg",
            risk_pct=1.0,
            sl_points=3.0,
            tp_points=6.0,
            max_ticks=5000,
        )
        self.assertNotIn("error", res)
        self.assertEqual(res["strategy"], "smc_fvg")
        self.assertGreater(res["ticks_replayed"], 1000)
        self.assertIn("net_profit_usd", res)
        self.assertIn("equity_curve", res)
        self.assertIsInstance(res["trades"], list)


if __name__ == "__main__":
    unittest.main()
