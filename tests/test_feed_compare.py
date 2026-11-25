"""U8 tests: feed comparison, provenance evidence, cost suggestion,
feed recorder isolation, and the deployment checklist logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.marketdata.compare import (FeedRecorder, compare_feeds,
                                     provenance_indicators,
                                     suggest_cost_model, write_report)


def ticks(n: int = 50_000, seed: int = 3, spread: float = 0.20,
          spacing_ms: int | None = None, px: float = 2600.0,
          start_ms: int = 1_760_000_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if spacing_ms is None:
        gaps = rng.integers(250, 5_000, size=n)      # irregular (live-like)
    else:
        gaps = np.full(n, spacing_ms)                # regular (suspicious)
    t = start_ms + np.cumsum(gaps)
    moves = rng.normal(0, 0.05, size=n)
    bid = px + np.cumsum(moves)
    return pd.DataFrame({"time_utc_ms": t, "bid": bid, "ask": bid + spread,
                         "bid_volume": 1.0, "ask_volume": 1.0,
                         "source": "test"})


# ── provenance evidence ─────────────────────────────────────────────────

def test_provenance_live_like_vs_interpolated():
    live = provenance_indicators(ticks())
    interp = provenance_indicators(ticks(spacing_ms=1_000, seed=4))
    assert live["verdict"] == "consistent_with_recorded_ticks"
    assert interp["verdict"] == "likely_bar_interpolated"
    assert live["certainty"] == "none — evidence only"
    # Evidence fields present for both, no certainty claim anywhere.
    for rep in (live, interp):
        assert {"regular_spacing_share", "max_identical_price_run",
                "zero_price_movement_share"} <= set(rep)


def test_provenance_insufficient_data():
    assert provenance_indicators(ticks(50))["verdict"] == \
        "insufficient_data"


def test_provenance_repeated_price_runs():
    df = ticks(20_000, seed=5)
    df["bid"] = 2600.0                     # frozen prices -> long runs
    df["ask"] = 2600.2
    rep = provenance_indicators(df)
    assert rep["max_identical_price_run"] > 50
    assert rep["verdict"] == "likely_bar_interpolated"


# ── recorder isolation ──────────────────────────────────────────────────

def test_recorder_writes_separate_store(tmp_path):
    rec = FeedRecorder("TestFirm", "Server-1", root=tmp_path)
    assert rec.source_name == "TestFirm_Server-1"
    out = rec.append([{"time_utc_ms": 1_760_000_000_000, "bid": 2600.0,
                       "ask": 2600.2}])
    assert str(tmp_path) in str(out)          # under the recorder root
    assert "data/ticks" not in str(out)       # research store untouched
    # Appending again merges and keeps ascending order.
    rec.append([{"time_utc_ms": 1_760_000_060_000, "bid": 2600.1,
                 "ask": 2600.3}])
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert df["time_utc_ms"].is_monotonic_increasing
    assert (df["source"] == "TestFirm_Server-1").all()


def test_recorder_rejects_empty():
    rec = FeedRecorder("F", "S", root=None)
    with pytest.raises(ValueError):
        rec.append([])


# ── comparison ──────────────────────────────────────────────────────────

def test_compare_feeds_metrics():
    research = ticks(spread=0.20)
    firm = research.copy()
    firm["ask"] = firm["bid"] + 0.40          # same market, wider spread
    firm["source"] = "firm"
    report = compare_feeds(research, firm)
    assert report["firm"]["spread"]["p50"] > \
        report["research"]["spread"]["p50"]
    ratio = report["spread_ratio_firm_over_research_p50"]
    assert ratio == pytest.approx(2.0, abs=0.05)
    assert report["price_offset"]["buckets"] > 0
    # Same underlying prices -> mid offset ~ 0 (half the spread delta).
    assert abs(report["price_offset"]["offset_mean"]) < 0.25
    assert "provenance" in report["research"] and "provenance" in report["firm"]


def test_compare_feeds_news_behaviour():
    firm = ticks(40_000, seed=11)
    # Double the spread inside a "news window".
    t = firm["time_utc_ms"].to_numpy()
    mid = (t[-1] + t[0]) // 2
    mask = t >= mid
    firm.loc[mask, "ask"] = firm.loc[mask, "bid"] + 0.8
    report = compare_feeds(firm, firm.copy(),
                           news_windows=[(int(mid), int(t[-1]))])
    nb = report["firm_spread_news_vs_normal"]
    assert nb["p50_inside"] > nb["p50_outside"]


# ── cost suggestion ─────────────────────────────────────────────────────

def test_cost_suggestion_scales_with_ratio():
    research = ticks(spread=0.20)
    firm = ticks(spread=0.40, seed=9)
    report = compare_feeds(research, firm)
    sug = suggest_cost_model(report)
    from core.config import get_cost_model
    base = get_cost_model("XAUUSD_ECN")
    assert sug["slippage_market_points_suggestion"] == pytest.approx(
        base.slippage.market_points * 2.0, abs=0.01)
    assert sug["spread_model"] == "recorded"
    assert "TODO(U8-manual)" in sug["commission_per_lot"]


def test_report_writer(tmp_path):
    report = compare_feeds(ticks(), ticks(spread=0.4, seed=9))
    sug = suggest_cost_model(report)
    out = write_report(report, sug, "TestFirm", "2026-11-25",
                       out_dir=tmp_path)
    assert out.name == "feed_comparison_TestFirm_2026-11-25.json"
    data = json.loads(out.read_text())
    assert data["firm"] == "TestFirm"
    assert "cost_model_suggestion" in data


import json  # noqa: E402  (used by test_report_writer)


# ── Dukascopy research-store protection (U8 step 3) ─────────────────────

def test_dukascopy_guard_refuses_foreign_partition(tmp_path, monkeypatch):
    import core.marketdata.store as store
    import core.marketdata.dukascopy_ingest as di
    monkeypatch.setattr(store, "TICK_ROOT", tmp_path)
    mt5 = pd.DataFrame({"time_utc_ms": [1_741_000_000_000],
                        "bid": [2600.0], "ask": [2600.2],
                        "bid_volume": [1.0], "ask_volume": [1.0],
                        "source": ["mt5_research"]})
    store.write_partition(mt5, "XAUUSD", 2025, 3)
    refusal = di.check_partition_sources("XAUUSD", 2025, 3)
    assert refusal is not None
    assert refusal["status"] == "refused_existing_source"
    assert refusal["conflicting_sources"] == ["mt5_research"]
    # A clean partition (or missing one) is safe.
    assert di.check_partition_sources("XAUUSD", 2030, 1) is None


def test_dukascopy_guard_allows_own_source(tmp_path, monkeypatch):
    import core.marketdata.store as store
    import core.marketdata.dukascopy_ingest as di
    monkeypatch.setattr(store, "TICK_ROOT", tmp_path)
    duk = pd.DataFrame({"time_utc_ms": [1_741_000_000_000],
                        "bid": [2600.0], "ask": [2600.2],
                        "bid_volume": [1.0], "ask_volume": [1.0],
                        "source": ["dukascopy"]})
    store.write_partition(duk, "XAUUSD", 2025, 3)
    assert di.check_partition_sources("XAUUSD", 2025, 3) is None
