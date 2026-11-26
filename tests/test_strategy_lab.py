"""Tests for the Strategy Lab, flat-by enforcement, and or_mid stops."""

from __future__ import annotations

import json
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtest import opening_range_mid, run_backtest
from core.paper_trading import PaperTradingEngine
from core.strategy_lab import StrategyLab
from strategies.base import StrategyConfig
from strategies.library import EMACross


# ---------------------------------------------------------------- helpers

def make_bars(n=1200, start="2026-06-01 00:00", freq="15min", seed=7, trend=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    drift = 0.05 if trend else 0.0
    close = 4000 + np.cumsum(rng.normal(drift, 2.0, n))
    o = close + rng.normal(0, 0.5, n)
    h = np.maximum(o, close) + np.abs(rng.normal(0, 1.0, n))
    l = np.minimum(o, close) - np.abs(rng.normal(0, 1.0, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "volume": np.abs(rng.normal(100, 10, n))}, index=idx)


@dataclass
class FakeSnapshot:
    data: pd.DataFrame
    def metadata(self):
        return {"source": "fake"}


# ---------------------------------------------------------------- Strategy Lab

class StrategyLabTests(unittest.TestCase):
    def setUp(self):
        self.df = make_bars(1200)
        self.tmp = Path(self.enterContext(_tmpdir()))
        self.seen = {}

        def fetch(timeframe, bars, allow_synthetic=True):
            self.seen["timeframe"] = timeframe
            self.seen["bars"] = bars
            return FakeSnapshot(self.df.iloc[:bars])

        self.lab = StrategyLab(fetch, {"ema_cross": EMACross}, results_dir=self.tmp)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            self.lab.start("does_not_exist")

    def test_search_runs_stores_and_ranks(self):
        run_id = self.lab.start("ema_cross", bars=1200, max_variants=4)
        self.assertIn("ema_cross", run_id)
        for _ in range(200):
            st = self.lab.status()
            if not st.get("running"):
                break
            time.sleep(0.05)
        st = self.lab.status()
        self.assertFalse(st["running"])
        self.assertIsNone(st.get("error"))
        self.assertEqual(st["done"], 4)
        self.assertEqual(len(st["results"]), 4)
        # ranking: verified first, then by OOS return desc
        order = {"verified": 0, "overfit_risk": 1, "unmeasured": 2}
        keys = [(order[r["status"]],
                 -float((r.get("oos_metrics") or {}).get("Return %") or 0.0))
                for r in st["results"]]
        self.assertEqual(keys, sorted(keys))
        # stored on disk and re-loadable
        runs = self.lab.list_runs()
        self.assertEqual(len(runs), 1)
        stored = self.lab.get_run(run_id)
        self.assertEqual(stored["run_id"], run_id)
        self.assertEqual(stored["summary"]["variants"], 4)

    def test_busy_lock(self):
        self.lab.start("ema_cross", bars=1200, max_variants=4)
        with self.assertRaises(RuntimeError):
            self.lab.start("ema_cross", bars=1200)
        for _ in range(200):
            if not self.lab.status().get("running"):
                break
            time.sleep(0.05)

    def test_insufficient_data_errors_cleanly(self):
        run_id = self.lab.start("ema_cross", bars=100)
        for _ in range(200):
            if not self.lab.status().get("running"):
                break
            time.sleep(0.05)
        st = self.lab.status()
        self.assertFalse(st["running"])
        self.assertIsNotNone(st.get("error"))


import contextlib
import tempfile


@contextlib.contextmanager
def _tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------- flat-by

class FlatByTests(unittest.TestCase):
    def _engine_with_position(self, flat_by):
        cfg = StrategyConfig(name="fb", risk_pct=1.0, sl_method="fixed",
                             sl_fixed=5.0, rr=2.0)
        eng = PaperTradingEngine()
        eng.start(EMACross(cfg), cfg, "XAUUSD", flat_by=flat_by)
        return eng

    def _feed_to_position(self, eng, day="2026-06-02"):
        bars = make_bars(200, start=f"{day} 00:00", seed=9)
        opened = None
        for ts, row in bars.iloc[:120].iterrows():
            t = eng.process_bar({
                "time": ts, "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"], "spread": 0.3,
            })
            if eng.positions and opened is None:
                opened = ts
        return opened

    def test_flat_by_closes_positions(self):
        eng = self._engine_with_position("12:00")
        opened = self._feed_to_position(eng)
        if not opened:
            self.skipTest("strategy did not open a position in the window")
        # feed bars past 12:00 and confirm everything is closed
        bars = make_bars(200, start="2026-06-02 00:00", seed=9)
        after = [ts for ts in bars.index if ts.hour * 60 + ts.minute >= 12 * 60]
        for ts in after[:20]:
            row = bars.loc[ts]
            eng.process_bar({
                "time": ts, "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"], "spread": 0.3,
            })
            if not eng.positions:
                break
        self.assertEqual(len(eng.positions), 0)
        reasons = [t.reason for t in eng.trades]
        self.assertIn("Flat by", reasons)

    def test_no_flat_by_holds(self):
        eng = self._engine_with_position(None)
        self._feed_to_position(eng)
        # without flat_by nothing is force-closed purely by the clock
        bars = make_bars(200, start="2026-06-02 00:00", seed=9)
        late = [ts for ts in bars.index if ts.hour == 23][:10]
        for ts in late:
            row = bars.loc[ts]
            eng.process_bar({
                "time": ts, "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"], "spread": 0.3,
            })
        # either still open, or closed by SL/TP — but no "Flat by" reason
        self.assertNotIn("Flat by", [t.reason for t in eng.trades])


# ---------------------------------------------------------------- or_mid

class OrMidTests(unittest.TestCase):
    def test_opening_range_mid_shape_and_values(self):
        bars = make_bars(400)
        mid = opening_range_mid(bars, 3)
        self.assertEqual(len(mid), len(bars))
        # first or_bars bars of the day have no completed range
        self.assertTrue(np.isnan(mid[:3]).all())
        # afterwards it sits between day-open extremes of the first bars
        day_first = bars.iloc[:3]
        lo = float(day_first["low"].min())
        hi = float(day_first["high"].max())
        self.assertTrue(lo - 1e-9 <= mid[3] <= hi + 1e-9)

    def test_config_accepts_or_mid(self):
        cfg = StrategyConfig(sl_method="or_mid", or_bars=4)
        self.assertEqual(cfg.sl_method, "or_mid")
        with self.assertRaises(ValueError):
            StrategyConfig(sl_method="or_mid", or_bars=0)
        with self.assertRaises(ValueError):
            StrategyConfig(sl_method="bogus")

    def test_bar_engine_or_mid_respects_structure(self):
        bars = make_bars(900)
        ohlc = bars[["open", "high", "low", "close", "volume"]]
        cfg = StrategyConfig(name="om", risk_pct=1.0, sl_method="or_mid",
                             or_bars=3, rr=2.0)
        eq, tr = run_backtest(ohlc, EMACross(cfg), cfg)
        for _, t in tr.iterrows():
            risk = abs(t["entry"] - t["sl"])
            # RR must be honored from the structural stop
            self.assertAlmostEqual(
                abs(t["tp"] - t["entry"]), risk * cfg.rr, delta=1e-4)

    def test_tick_engine_or_mid_runs(self):
        from core.tick_replay import run_tick_replay
        bars = make_bars(400)
        cfg = StrategyConfig(name="om", risk_pct=1.0, sl_method="or_mid",
                             or_bars=3, rr=2.0)
        res = run_tick_replay(bars, EMACross(cfg), cfg, start_equity=10_000,
                              tick_dir=Path("data/ticks/XAUUSD"))
        self.assertEqual(res["engine"], "tick_replay")
        for t in res["trades"]:
            self.assertGreater(t["entry"], 0)


if __name__ == "__main__":
    unittest.main()
