"""Strategy fingerprint report (Update Blueprint v2.1, U5).

Informational only: for each deployed strategy version, list its
indicator set and parameters, and flag whether it is purely a standard
public indicator setup or uses a prohibited style — so a human can
review payout risk. Never a compliance decision by itself.
"""
from __future__ import annotations

STANDARD_PUBLIC_INDICATORS = frozenset({
    "ema", "sma", "rsi", "atr", "bollinger", "macd", "supertrend",
    "vwap", "donchian", "stochastic", "adx", "cci", "momentum",
    "roc", "obv", "psar", "ichimoku", "keltner", "moving_average",
    "price_action", "engulfing", "pin_bar", "support_resistance",
})

PROHIBITED_STYLES = ("martingale", "grid", "averaging_down", "hedging",
                     "copy_trading", "tick_scalping", "latency_arbitrage")


def fingerprint(strategy_version_id: str, spec: dict) -> dict:
    """spec: {'indicators': [...], 'params': {...}, 'styles': [...]}.
    Classification:
    - 'contains_prohibited_style' — any prohibited style present.
    - 'standard_public' — all indicators are standard public ones and
      no prohibited style.
    - 'custom' — anything else (custom/proprietary logic)."""
    indicators = [str(i).lower() for i in spec.get("indicators", [])]
    styles = [str(s).lower() for s in spec.get("styles", [])]
    prohibited = [s for s in styles if s in PROHIBITED_STYLES]
    if prohibited:
        classification = "contains_prohibited_style"
    elif indicators and all(i in STANDARD_PUBLIC_INDICATORS
                            for i in indicators):
        classification = "standard_public"
    else:
        classification = "custom"
    return {"strategy_version_id": strategy_version_id,
            "indicators": sorted(set(indicators)),
            "params": dict(spec.get("params", {})),
            "styles": sorted(set(styles)),
            "prohibited_styles": prohibited,
            "classification": classification,
            "human_review_required": classification !=
            "standard_public"}


def fingerprint_report(specs: dict[str, dict]) -> list[dict]:
    """Deterministic report sorted by version id."""
    return [fingerprint(vid, specs[vid]) for vid in sorted(specs)]
