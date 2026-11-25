#!/usr/bin/env python3
"""U7 library-strategy baseline screening on the tick engine.

The 15 existing strategies run with their DEFAULT parameters on each
session window (no re-tuning: they were already validated and FAILED
out-of-sample on H1-2025; this baseline re-tests them under the
campaign's session/tick/cost conditions). Top performers by PF with
enough trades go to the same walk-forward + gates as the new variants.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.research import (LONDON, NEW_YORK, OVERLAP, ClientProfile,
                           HoldoutGuard, TrialStore, derive_risk_pct,
                           run_trial)  # noqa: E402
from strategies.adapter import LegacyAdapter  # noqa: E402
from strategies.base import StrategyConfig  # noqa: E402
from strategies import library as lib  # noqa: E402

DAY = 86_400_000
TRAIN_START = 1_704_067_200_000            # 2024-01-01 00:00 UTC
SCREEN_END = TRAIN_START + 365 * DAY       # first 12 training months

LIBRARY = [lib.EMACross, lib.MACDCross, lib.RSIReversal, lib.VWAPEMA,
           lib.Supertrend, lib.SessionBreakout, lib.BollingerReversion,
           lib.FairValueGap, lib.SMCOrderBlock, lib.LiquiditySweep,
           lib.PSARTrend, lib.StochasticCross, lib.ZScoreReversion,
           lib.ADXTrend, lib.BreakOfStructure]

REPORT_DIR = Path("reports/campaign_2026-11-25")
OUT = REPORT_DIR / "library_baseline.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    from core.marketdata.store import read_range
    profile = ClientProfile()
    holdout = HoldoutGuard(1_778_995_200_000, 1_794_720_000_000,
                           log_path="state/holdout_openings.jsonl")
    holdout.check_range(TRAIN_START, SCREEN_END, purpose="library_screen")
    risk = derive_risk_pct(profile)
    df = read_range("XAUUSD", TRAIN_START, SCREEN_END)
    log(f"loaded {len(df):,} ticks; risk={risk}%")
    results = []
    for sess in (LONDON, NEW_YORK, OVERLAP):
        for cls in LIBRARY:
            cfg = StrategyConfig(risk_pct=risk)
            strat = LegacyAdapter(cls(cfg), cfg)
            t0 = time.perf_counter()
            try:
                m = run_trial(strat, df, start_ms=TRAIN_START,
                              end_ms=SCREEN_END, session=sess,
                              risk_pct=risk)
            except Exception as exc:            # logged, never hidden
                log(f"{cls.key}@{sess.name}: ERROR {exc!r}")
                results.append({"strategy": cls.key, "session": sess.name,
                                "error": repr(exc)})
                continue
            rec = {"strategy": cls.key, "session": sess.name,
                   "elapsed_s": round(time.perf_counter() - t0, 1),
                   "metrics": m}
            results.append(rec)
            log(f"{cls.key}@{sess.name}: trades={m['n_trades']} "
                f"pf={m['pf']:.2f} wr={m['win_rate_pct']:.0f}% "
                f"dd={m['max_dd_pct']:.2f}%")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, sort_keys=True))
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
