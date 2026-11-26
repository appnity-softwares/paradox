"""Strategy Lab — agent-driven strategy variant search with live status.

The Strategy Optimization Agent's workbench: it generates parameter
variants of a chosen strategy from the documented PARAM_GRID, backtests
each variant on IN-SAMPLE data, then validates the survivors on
OUT-OF-SAMPLE data (walk-forward honesty: IS-only winners are flagged as
overfit-risk, not celebrated).

Real-time: a background thread runs the search while `status()` reports
live progress (variants done / total, current variant, results so far) so
the dashboard can poll and show the search as it happens.

Storage: every finished run is saved to results/strategy_lab/<run_id>.json
so multiple searches accumulate and can be re-opened later.

Honesty rules:
- A variant is only "verified" if it also performs on OUT-OF-SAMPLE data.
- Variants with too few trades are "unmeasured", never scored as winners.
- Nothing is auto-traded: promoting a variant only reconfigures the paper
  engine, never a live order path.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.backtest import run_backtest
from core.metrics import compute_metrics
from core.optimize import PARAM_GRID, param_combos
from strategies.base import StrategyConfig

RESULTS_DIR = Path("results/strategy_lab")
MIN_OOS_TRADES = 5


class StrategyLab:
    """Background strategy-variant search with live status and storage."""

    def __init__(self, data_fetcher, strategies: dict, results_dir: Path = RESULTS_DIR):
        self._fetch = data_fetcher
        self._strategies = strategies
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current: dict | None = None   # live status of the running search
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ public API
    def start(self, strategy: str, timeframe: str = "15m", bars: int = 8000,
              risk_pct: float = 1.0, rr: float = 2.0, sl_method: str = "atr",
              sl_atr_mult: float = 1.5, sl_fixed: float = 5.0, session: str = "all",
              max_trades_per_day: int = 4, daily_loss_limit_pct: float = 3.0,
              max_variants: int | None = None, oos_fraction: float = 0.3) -> str:
        if strategy not in self._strategies:
            raise ValueError(f"unknown strategy: {strategy}")
        with self._lock:
            if self._current and self._current.get("running"):
                raise RuntimeError("a search is already running — check /api/lab/status")
            grid = PARAM_GRID.get(strategy, {})
            combos = list(param_combos(grid))
            if not combos:
                raise ValueError(f"no parameter grid for strategy: {strategy}")
            if max_variants:
                combos = combos[:max_variants]
            run_id = f"{strategy}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            self._current = {
                "run_id": run_id,
                "strategy": strategy,
                "timeframe": timeframe,
                "running": True,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "total_variants": len(combos),
                "done": 0,
                "current_params": None,
                "results": [],
                "error": None,
            }
        args = dict(
            strategy=strategy, timeframe=timeframe, bars=bars,
            risk_pct=risk_pct, rr=rr, sl_method=sl_method, sl_atr_mult=sl_atr_mult,
            sl_fixed=sl_fixed, session=session, max_trades_per_day=max_trades_per_day,
            daily_loss_limit_pct=daily_loss_limit_pct,
            combos=combos, run_id=run_id, oos_fraction=oos_fraction,
        )
        self._thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._thread.start()
        return run_id

    def status(self) -> dict:
        with self._lock:
            if self._current is None:
                return {"running": False, "note": "no search started yet"}
            return copy.deepcopy(self._current)

    def list_runs(self) -> list[dict]:
        runs = []
        for p in sorted(self._results_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text())
                best = data.get("summary", {}).get("best", {})
                runs.append({
                    "run_id": data.get("run_id", p.stem),
                    "strategy": data.get("strategy"),
                    "timeframe": data.get("timeframe"),
                    "finished_at": data.get("finished_at"),
                    "variants": data.get("summary", {}).get("variants", 0),
                    "verified_variants": data.get("summary", {}).get("verified", 0),
                    "best_oos_return_pct": best.get("oos_return_pct"),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return runs

    def get_run(self, run_id: str) -> dict:
        path = self._results_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"no stored run: {run_id}")
        return json.loads(path.read_text())

    # ------------------------------------------------------------- worker
    def _worker(self, **kw):
        run_id = kw["run_id"]
        try:
            snapshot = self._fetch(kw["timeframe"], kw["bars"], allow_synthetic=False)
            df = snapshot.data
            if len(df) < 200:
                raise ValueError(f"not enough data: {len(df)} bars")
            split = int(len(df) * (1.0 - kw["oos_fraction"]))
            is_df, oos_df = df.iloc[:split], df.iloc[split:]
            if len(oos_df) < 50:
                raise ValueError("out-of-sample slice too small — fetch more bars")

            base_cfg = StrategyConfig(
                name=f"lab_{run_id}", risk_pct=kw["risk_pct"], rr=kw["rr"],
                sl_method=kw["sl_method"], sl_atr_mult=kw["sl_atr_mult"],
                sl_fixed=kw["sl_fixed"], session=kw["session"],
                max_trades_per_day=kw["max_trades_per_day"],
                daily_loss_limit_pct=kw["daily_loss_limit_pct"],
            )
            cls = self._strategies[kw["strategy"]]
            results = []
            for n, params in enumerate(kw["combos"]):
                with self._lock:
                    self._current["current_params"] = params
                    self._current["done"] = n
                row = self._eval_variant(cls, base_cfg, params, is_df, oos_df)
                results.append(row)
                with self._lock:
                    self._current["results"] = self._ranked(results)

            verified = [r for r in results if r["status"] == "verified"]
            best = verified[0] if verified else None
            summary = {
                "variants": len(results),
                "verified": len(verified),
                "overfit_risk": sum(1 for r in results if r["status"] == "overfit_risk"),
                "unmeasured": sum(1 for r in results if r["status"] == "unmeasured"),
                "best": best,
            }
            payload = {
                "run_id": run_id,
                "strategy": kw["strategy"],
                "timeframe": kw["timeframe"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "results": results,
            }
            (self._results_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=1))
            with self._lock:
                self._current["running"] = False
                self._current["done"] = len(results)
                self._current["summary"] = summary
                self._current["finished_at"] = payload["finished_at"]
        except Exception as exc:
            with self._lock:
                self._current["running"] = False
                self._current["error"] = str(exc)

    def _eval_variant(self, cls, base_cfg, params, is_df, oos_df) -> dict:
        row = {"params": params, "status": "unmeasured",
               "is_metrics": None, "oos_metrics": None}
        try:
            cfg_is = copy.deepcopy(base_cfg)
            strat_is = cls(cfg_is, **params)
            eq_is, tr_is = run_backtest(is_df, strat_is, cfg_is)
            m_is = compute_metrics(eq_is, tr_is, cfg_is, "in-sample")
            row["is_metrics"] = m_is
            if int(m_is.get("Total Trades", 0)) < 3:
                row["status"] = "unmeasured"
                row["note"] = "too few in-sample trades to judge"
                return row

            cfg_oos = copy.deepcopy(base_cfg)
            strat_oos = cls(cfg_oos, **params)
            eq_oos, tr_oos = run_backtest(oos_df, strat_oos, cfg_oos)
            m_oos = compute_metrics(eq_oos, tr_oos, cfg_oos, "out-of-sample")
            row["oos_metrics"] = m_oos
            oos_trades = int(m_oos.get("Total Trades", 0))
            oos_ret = float(m_oos.get("Return %", 0.0))
            if oos_trades < MIN_OOS_TRADES:
                row["status"] = "unmeasured"
                row["note"] = f"only {oos_trades} out-of-sample trades"
            elif oos_ret <= 0:
                row["status"] = "overfit_risk"
                row["note"] = "in-sample looked fine but out-of-sample lost money"
            else:
                row["status"] = "verified"
        except (TypeError, ValueError) as exc:
            row["status"] = "unmeasured"
            row["note"] = f"variant failed to run: {exc}"
        return row

    def _ranked(self, results: list[dict]) -> list[dict]:
        def key(r):
            order = {"verified": 0, "overfit_risk": 1, "unmeasured": 2}
            oos_ret = 0.0
            if r.get("oos_metrics"):
                try:
                    oos_ret = float(r["oos_metrics"].get("Return %", 0.0) or 0.0)
                except (TypeError, ValueError):
                    oos_ret = 0.0
            return (order.get(r["status"], 3), -oos_ret)
        return sorted(results, key=key)
