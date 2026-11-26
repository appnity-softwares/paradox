"""LLM intelligence layer - Gemini (Google AI Studio) provider.

The LLM is the *intelligence layer only*: it translates a natural-language
command into a structured Strategy Spec (JSON). Every deterministic engine
(risk sizing, validation, backtest, walk-forward) owns the numbers and decisions
afterwards - the LLM never places orders and never touches risk math.

The emitted spec uses the same schema as the rule-based parser in ``ai.builder``
so downstream engines stay unchanged:

    {
      "strategy": "ema_cross",
      "params": {"fast": 20, "slow": 50},
      "risk_pct": 1.0,
      "rr": 2.0,
      "sl_method": "atr",
      "sl_atr_mult": 1.5,
      "sl_fixed": 5.0,
      "session": "all",
      "max_trades_per_day": 4,
      "daily_loss_limit_pct": 3.0,
      "warnings": [],
      "notes": []
    }
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import httpx

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
# Google retires models without notice (gemini-2.0-flash started 404-ing in
# Nov 2026). Try the configured model first, then these, so the LLM layer
# degrades gracefully instead of silently falling back to regex parsing.
FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]

SYSTEM_PROMPT = (
    "You are the strategy spec generator for an AI quantitative trading platform "
    "for XAUUSD gold futures. Convert the user's trading idea into a structured "
    "Strategy Spec as raw JSON (no markdown fences, no commentary). "
    "Schema requirements: "
    '"strategy" is one of ["ema_cross","macd_cross","rsi_reversal","bollinger","vwap_ema","supertrend","session_breakout"]; '
    '"params" holds strategy-specific keys (e.g. {"fast":20,"slow":50} for ema_cross, '
    '{"length":14,"oversold":30,"overbought":70} for rsi_reversal); '
    '"risk_pct" float 0.1-3.0; "rr" float 0.5-5.0; "sl_method" in ["atr","fixed"]; '
    '"sl_atr_mult" float 0.1-10; "sl_fixed" float > 0; "session" in ["all","tokyo","london","ny","london_ny"]; '
    '"max_trades_per_day" int 1-10; "daily_loss_limit_pct" float 0.5-10. '
    'If the idea is ambiguous, choose sensible conservative defaults and add a note in "notes". '
    'Respond with JSON only.'
)


class BaseLLM(ABC):
    def generate_spec(self, text: str) -> dict | None:  # pragma: no cover - interface
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or ""
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.last_error: str | None = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _call(self, model: str, text: str) -> httpx.Response:
        return httpx.post(
            f"{GEMINI_BASE}/{model}:generateContent",
            params={"key": self.api_key},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": text}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
            },
            timeout=45.0,
        )

    def generate_spec(self, text: str) -> dict | None:
        if not self.api_key:
            self.last_error = "no api key"
            return None
        last_err = "unknown error"
        for model in [self.model] + FALLBACK_MODELS:
            try:
                resp = self._call(model, text)
                if resp.status_code == 404:
                    last_err = f"{model}: retired/unavailable"
                    continue
                if resp.status_code in (429, 503):
                    last_err = f"{model}: {resp.json().get('error', {}).get('message', 'unavailable')[:80]}"
                    continue
                if resp.status_code == 402:
                    self.last_error = "credits depleted (402)"
                    return None
                if resp.status_code != 200:
                    last_err = f"{model}: HTTP {resp.status_code}"
                    continue
                data = resp.json()
                out_text = ""
                for part in data.get("candidates") or []:
                    for p in part.get("content", {}).get("parts") or []:
                        out_text += p.get("text") or ""
                spec = json.loads(out_text)
                if isinstance(spec, dict) and spec.get("strategy"):
                    self.last_error = None
                    return spec
                last_err = f"{model}: response was not a valid spec"
            except Exception as exc:
                last_err = f"{model}: {exc}"
        self.last_error = last_err
        return None


def get_llm() -> BaseLLM | None:
    """Return a configured LLM or None (rule-based fallback).

    Resolution order: GEMINI_API_KEY env var first, then the admin panel
    config file (set from the admin UI). Explicit env always wins.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            from dashboard.admin_config import load_config
            key = load_config().get("gemini_api_key") or ""
        except Exception:
            key = ""
    if not key:
        return None
    return GeminiLLM(api_key=key)
