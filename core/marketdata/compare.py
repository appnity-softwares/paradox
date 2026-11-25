"""Feed comparison (Update Blueprint v2.1, U8 step 1-2).

Records live quotes from a target firm's server into a SEPARATE parquet
store (never touching the research data), then compares it with the
research tick store: spread distributions, bid/ask differences, tick
density, price offsets, behaviour around news. Also reports DATA
PROVENANCE evidence (recorded ticks vs bar-interpolated) without
asserting certainty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEED_ROOT = Path("data/feeds")          # separate from data/ticks (research)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

@dataclass
class FeedRecorder:
    """Append live quotes to data/feeds/<firm>_<server>/<YYYY>/<MM>.parquet.

    The source column names firm AND server so research data can never
    be confused with firm data."""
    firm: str
    server: str
    symbol: str = "XAUUSD"
    root: Path = FEED_ROOT

    @property
    def source_name(self) -> str:
        return f"{self.firm}_{self.server}".replace("/", "_").replace(" ", "_")

    def partition_dir(self, ts_ms: int) -> Path:
        import datetime as dt
        day = dt.datetime.fromtimestamp(ts_ms / 1000, dt.UTC)
        return self.root / self.source_name / self.symbol / \
            f"{day.year}" / f"{day.month:02d}"

    def append(self, quotes: list[dict]) -> Path:
        """quotes: [{time_utc_ms, bid, ask}] — sorted ascending."""
        if not quotes:
            raise ValueError("no quotes to record")
        df = pd.DataFrame(quotes)
        df["source"] = self.source_name
        df["bid_volume"] = 0.0
        df["ask_volume"] = 0.0
        first = int(df["time_utc_ms"].iloc[0])
        out = self.partition_dir(first) / "quotes.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            old = pd.read_parquet(out)
            df = pd.concat([old, df], ignore_index=True)
            df = df.sort_values("time_utc_ms").reset_index(drop=True)
        df.to_parquet(out, index=False)
        return out


# ---------------------------------------------------------------------------
# Provenance evidence (step 2) — evidence, never certainty
# ---------------------------------------------------------------------------

def provenance_indicators(df: pd.DataFrame) -> dict:
    """Evidence-based data-quality indicators.

    Recorded live ticks usually show irregular inter-tick gaps and few
    exactly-repeated prices. Bar-interpolated history shows long runs
    of identical prices and suspiciously regular spacing. We report the
    evidence and a soft verdict; we never claim certainty."""
    t = df["time_utc_ms"].to_numpy(dtype=np.int64)
    bid = df["bid"].to_numpy(dtype=np.float64)
    if len(t) < 100:
        return {"verdict": "insufficient_data", "n_ticks": len(t)}
    gaps = np.diff(t)
    # Repeated-price runs: consecutive identical bids.
    same = bid[1:] == bid[:-1]
    run_max = 0
    run = 0
    for s in same:
        run = run + 1 if s else 0
        run_max = max(run_max, run)
    # Regular-spacing evidence: share of gaps within +/-2ms of the modal gap.
    modal_gap = float(np.median(gaps))
    regular = float(np.mean(np.abs(gaps - modal_gap) <= 2.0))
    ticks_per_min = len(t) / max(1.0, (t[-1] - t[0]) / 60_000.0)
    zero_movement_share = float(np.mean(np.diff(bid) == 0.0))
    evidence = {
        "n_ticks": int(len(t)),
        "ticks_per_minute_avg": round(ticks_per_min, 1),
        "median_gap_ms": round(modal_gap, 1),
        "regular_spacing_share": round(regular, 3),
        "max_identical_price_run": int(run_max),
        "zero_price_movement_share": round(zero_movement_share, 3),
    }
    # Soft verdict from the evidence combination.
    suspicious = (regular > 0.6 or run_max > 50
                  or zero_movement_share > 0.5)
    evidence["verdict"] = ("likely_bar_interpolated" if suspicious
                           else "consistent_with_recorded_ticks")
    evidence["certainty"] = "none — evidence only"
    return evidence


# ---------------------------------------------------------------------------
# Comparison (step 1)
# ---------------------------------------------------------------------------

def _spread_stats(df: pd.DataFrame) -> dict:
    spread = df["ask"].to_numpy() - df["bid"].to_numpy()
    return {
        "p50": round(float(np.percentile(spread, 50)), 4),
        "p90": round(float(np.percentile(spread, 90)), 4),
        "p99": round(float(np.percentile(spread, 99)), 4),
        "max": round(float(spread.max()), 4),
        "mean": round(float(spread.mean()), 4),
    }


def _tick_density(df: pd.DataFrame) -> dict:
    t = df["time_utc_ms"].to_numpy(dtype=np.int64)
    per_min = np.bincount((t // 60_000).astype(np.int64))
    per_min = per_min[per_min > 0]
    return {"median_ticks_per_active_minute":
            round(float(np.median(per_min)), 1) if per_min.size else 0.0,
            "active_minutes": int(per_min.size)}


def _price_offset(research: pd.DataFrame, firm: pd.DataFrame,
                  bucket_ms: int = 60_000) -> dict:
    """Mid-price offset per aligned minute bucket (firm minus research)."""
    r = research.assign(_b=research["time_utc_ms"] // bucket_ms)
    f = firm.assign(_b=firm["time_utc_ms"] // bucket_ms)
    rm = r.groupby("_b")[["bid", "ask"]].mean().mean(axis=1)
    fm = f.groupby("_b")[["bid", "ask"]].mean().mean(axis=1)
    joined = pd.concat([rm.rename("r"), fm.rename("f")], axis=1).dropna()
    if joined.empty:
        return {"buckets": 0, "offset_mean": None, "offset_p95_abs": None}
    off = joined["f"] - joined["r"]
    return {"buckets": int(len(joined)),
            "offset_mean": round(float(off.mean()), 4),
            "offset_p95_abs": round(float(np.percentile(np.abs(off), 95)),
                                    4)}


def compare_feeds(research: pd.DataFrame, firm: pd.DataFrame,
                  news_windows: tuple = ()) -> dict:
    """Compare the research tick store with a firm feed over the same
    period. news_windows: [(start_ms, end_ms), ...] for news behaviour."""
    report = {
        "research": {"n_ticks": int(len(research)),
                     "spread": _spread_stats(research),
                     "density": _tick_density(research),
                     "provenance": provenance_indicators(research)},
        "firm": {"n_ticks": int(len(firm)),
                 "spread": _spread_stats(firm),
                 "density": _tick_density(firm),
                 "provenance": provenance_indicators(firm)},
    }
    # Spread ratio firm/research (p50) — the cost-suggestion driver.
    rs = report["research"]["spread"]["p50"]
    fs = report["firm"]["spread"]["p50"]
    report["spread_ratio_firm_over_research_p50"] = (
        round(fs / rs, 3) if rs > 0 else None)
    report["price_offset"] = _price_offset(research, firm)
    # Behaviour around news: spread expansion inside vs outside windows.
    if news_windows:
        ft = firm["time_utc_ms"].to_numpy(dtype=np.int64)
        inside = np.zeros(len(ft), dtype=bool)
        for (ws, we) in news_windows:
            inside |= (ft >= ws) & (ft <= we)
        if inside.any() and (~inside).any():
            fsp = firm["ask"].to_numpy() - firm["bid"].to_numpy()
            report["firm_spread_news_vs_normal"] = {
                "p50_inside": round(float(np.percentile(fsp[inside], 50)), 4),
                "p50_outside": round(float(np.percentile(fsp[~inside], 50)),
                                     4)}
    return report


def suggest_cost_model(report: dict, base_key: str = "XAUUSD_ECN",
                       ) -> dict:
    """CostModel suggestion for the firm, derived from the comparison.

    TODO(U8-manual): confirm the firm's REAL commission schedule and
    any spread markups by account tier before live use."""
    from core.config import get_cost_model
    base = get_cost_model(base_key)
    ratio = report.get("spread_ratio_firm_over_research_p50") or 1.0
    firm_spread = report["firm"]["spread"]
    suggestion = {
        "spread_model": "recorded",
        "recorded_spread_p50_usd": firm_spread["p50"],
        "fixed_spread_points_suggestion": round(firm_spread["p50"] * 100.0,
                                                1),
        "slippage_market_points_suggestion":
            round(base.slippage.market_points * max(1.0, ratio), 4),
        "slippage_stop_points_suggestion":
            round(base.slippage.stop_points * max(1.0, ratio), 4),
        "commission_per_lot": "TODO(U8-manual): confirm with the firm",
        "basis": "spread ratio firm/research p50 = "
                 f"{report.get('spread_ratio_firm_over_research_p50')}",
    }
    return suggestion


def write_report(report: dict, suggestion: dict, firm: str, date: str,
                 out_dir: str | Path = "reports") -> Path:
    out = Path(out_dir) / f"feed_comparison_{firm}_{date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"firm": firm, "comparison": report,
                               "cost_model_suggestion": suggestion},
                              indent=1, sort_keys=True))
    return out
