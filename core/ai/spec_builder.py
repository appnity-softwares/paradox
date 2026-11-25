"""AI Strategy Builder (Part 11.3).

Preferred route: natural language -> structured StrategySpec (JSON,
validated) -> interpreted by a SAFE RULE ENGINE over the indicator
library -> backtest -> report. Free-form generated code goes only through
the sandbox (`core.ai.sandbox`). Every generated strategy is versioned
with the prompt that produced it and must pass Phase 1 validation before
paper trading.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from core.ai.models import ModelRouter


class EntryCondition(BaseModel):
    indicator: str                       # "ema_cross_up", "rsi_below", ...
    params: dict = Field(default_factory=dict)


class ExitSpec(BaseModel):
    sl_method: str = "atr"               # "atr" | "fixed"
    sl_atr_mult: float = Field(default=1.5, gt=0, le=10)
    sl_fixed: float = Field(default=5.0, gt=0)
    rr: float = Field(default=1.5, gt=0, le=10)


class StrategySpec(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    entry: EntryCondition
    exit: ExitSpec
    session: str = "all"                 # all | london | ny | london_ny | tokyo
    max_trades_per_day: int = Field(default=2, ge=1, le=20)

    @model_validator(mode="after")
    def _known_entry(self):
        from core.ai.spec_rules import SUPPORTED_ENTRIES
        if self.entry.indicator not in SUPPORTED_ENTRIES:
            raise ValueError(
                f"unsupported entry rule {self.entry.indicator!r}; "
                f"supported: {sorted(SUPPORTED_ENTRIES)}")
        return self


@dataclass(frozen=True)
class GeneratedStrategy:
    version_id: str
    spec: StrategySpec
    prompt: str                          # the prompt that produced it


def nl_to_spec(router: ModelRouter, prompt: str, *,
               user: str = "default") -> tuple[StrategySpec, str]:
    """NL -> validated StrategySpec. One retry on invalid output, then
    fail safe (raises after recording the failure)."""
    last_error = ""
    for _attempt in (1, 2):
        try:
            reply = router.complete("nl_to_spec",
                                    StrategySpec.model_json_schema(),
                                    {"request": prompt}, user=user)
            return StrategySpec.model_validate(reply.output), prompt
        except Exception as exc:      # ValidationError or ModelError
            last_error = str(exc)
    raise ValueError(f"strategy spec generation failed safely: {last_error}")


def version_spec(spec: StrategySpec, prompt: str) -> GeneratedStrategy:
    payload = json.dumps({"spec": spec.model_dump(), "prompt": prompt},
                         sort_keys=True)
    vid = "spec:" + hashlib.sha256(payload.encode()).hexdigest()[:16]
    return GeneratedStrategy(vid, spec, prompt)


def spec_to_signals(spec: StrategySpec, bars: pd.DataFrame) -> pd.Series:
    """The safe rule-spec route: declarative conditions interpreted by
    the deterministic rule engine — no generated code in the loop."""
    from core.ai.spec_rules import apply_entry
    sig = apply_entry(spec.entry.indicator, spec.entry.params, bars)
    return sig
