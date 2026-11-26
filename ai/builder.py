"""AI Strategy Builder (prototype) - the platform's core differentiator.

Pipeline (matches the master vision):
 1. UNDERSTAND   natural-language command (rule-based NLU; an LLM drops in here later
                 and outputs the exact same spec schema)
 2. SPEC         structured Strategy Spec (JSON) - same schema future visual builder
                 and code generator will consume
 3. VALIDATE     param ranges + notebook guardrails (risk 0.5-2%, etc.)
 4. BUILD        instantiate the strategy class from the spec
 5. BACKTEST     honest engine (next-bar fills, spread/slippage, risk sizing)
 6. VERIFY       fixed-param walk-forward on unseen folds (overfitting check)
 7. EXPLAIN      results + weaknesses + concrete improvement suggestions
"""
import re

from core.backtest import run_backtest
from core.llm import BaseLLM, get_llm
from core.metrics import compute_metrics
from core.optimize import fixed_walk_forward
from strategies import ALL_STRATEGIES, StrategyConfig

DEFAULTS = {
    "risk_pct": 1.0,
    "rr": 2.0,
    "sl_method": "atr",
    "sl_atr_mult": 1.5,
    "sl_fixed": 5.0,
    "session": "all",
    "max_trades_per_day": 4,
    "daily_loss_limit_pct": 3.0,
}

# Combo keywords the user might mention that the prototype library can't combine yet.
COMBO_KEYWORDS = {
    "rsi": "RSI confirmation/filter",
    "volume": "volume filter",
    "news": "news filter",
}


def parse_command(text: str) -> dict:
    t = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "

    spec = dict(DEFAULTS)
    spec["strategy"] = None
    spec["params"] = {}
    spec["warnings"] = []
    spec["notes"] = []

    def has(word: str) -> bool:
        return re.search(r"\b" + re.escape(word) + r"\b", t) is not None

    # ---- 1) strategy type (specific -> general) ----
    ema_mentioned = has("ema") or has("moving average")
    cross_language = has("crossover") or has("cross")

    if has("supertrend"):
        spec["strategy"] = "supertrend"
    elif has("vwap"):
        spec["strategy"] = "vwap_ema"
    elif has("bollinger") or has("boll"):
        spec["strategy"] = "bollinger"
    elif has("macd"):
        spec["strategy"] = "macd_cross"
    elif has("breakout") or has("range break"):
        spec["strategy"] = "session_breakout"
    elif ema_mentioned and cross_language:
        spec["strategy"] = "ema_cross"
    elif has("rsi"):
        spec["strategy"] = "rsi_reversal"
    elif ema_mentioned:
        spec["strategy"] = "ema_cross"
        spec["notes"].append("cross language nahi mili - single EMA trend filter maana")

    # unsupported combos the user mentioned
    for kw, label in COMBO_KEYWORDS.items():
        if has(kw) and spec["strategy"] != "rsi_reversal":
            spec["notes"].append(f"'{label}' combo abhi prototype library mein nahi - future phase mein aayega")

    num = r"(\d+(?:\.\d+)?)"

    # ---- 2) strategy params ----
    if spec["strategy"] == "ema_cross":
        m = (
            re.search(r"ema\s*(\d+)\s*(?:/|x|-|and|,|with|to)\s*(\d+)", t)
            or re.search(r"ema\s+(\d+)\s+(\d+)", t)
            or re.search(r"moving average\s*(\d+)\s*(?:/|and|,)?\s*(\d+)", t)
        )
        if m:
            spec["params"] = {"fast": int(m.group(1)), "slow": int(m.group(2))}
        else:
            m1 = re.search(r"ema\s*(\d+)", t)
            fast = int(m1.group(1)) if m1 else 20
            spec["params"] = {"fast": fast, "slow": 50}
            spec["notes"].append("sirf ek EMA length mili - slow EMA default 50 liya")

    elif spec["strategy"] == "vwap_ema":
        m = re.search(r"ema\s*(\d+)", t) or re.search(num + r"\s*ema", t)
        ema_length = int(m.group(1)) if m else 20
        slope = 1 if (has("slope") or has("rising") or has("falling")) else 0
        if not slope:
            spec["notes"].append("slope filter off hai - 'with slope' likh ke on kar sakte ho")
        spec["params"] = {"ema_length": ema_length, "slope": slope}

    elif spec["strategy"] == "macd_cross":
        m = re.search(r"macd\D{0,4}(\d+)\D{1,4}(\d+)\D{1,4}(\d+)", t)
        spec["params"] = (
            {"fast": int(m.group(1)), "slow": int(m.group(2)), "signal": int(m.group(3))}
            if m
            else {"fast": 12, "slow": 26, "signal": 9}
        )

    elif spec["strategy"] == "rsi_reversal":
        m = re.search(r"rsi\D{0,4}(\d+)", t)
        length = int(m.group(1)) if m else 14
        mo = re.search(r"oversold\D{0,6}" + num, t)
        mb = re.search(r"overbought\D{0,6}" + num, t)
        spec["params"] = {
            "length": length,
            "oversold": float(mo.group(1)) if mo else 30.0,
            "overbought": float(mb.group(1)) if mb else 70.0,
        }

    elif spec["strategy"] == "bollinger":
        m = re.search(r"bollinger\w*\D{0,6}?(\d+)", t)
        length = int(m.group(1)) if m else 20
        ms = re.search(num + r"\s*(?:sd|standard deviation|std)", t) or re.search(
            r"(?:sd|std)\D{0,4}" + num, t
        )
        num_std = float(ms.group(1)) if ms else 2.0
        spec["params"] = {"length": length, "num_std": num_std}

    elif spec["strategy"] == "supertrend":
        m = re.search(r"supertrend\s*(\d+)", t)
        length = int(m.group(1)) if m else 10
        mm = (
            re.search(r"multiplier\D{0,6}" + num, t)
            or re.search(num + r"\s*multiplier", t)
            or re.search(r"supertrend\s*\d+\s*(?:with|ke\s+saath)\s*" + num, t)
            or re.search(r"\bx\s*" + num, t)
        )
        mult = float(mm.group(1)) if mm else 3.0
        spec["params"] = {"length": length, "mult": mult}

    elif spec["strategy"] == "session_breakout":
        mb = re.search(r"buffer\D{0,6}" + num, t)
        mr = re.search(r"(?:range|asia)\D{0,12}?(\d{1,2})(?:\s*(?:am|h\b|hour))?", t)
        spec["params"] = {
            "buffer": float(mb.group(1)) if mb else 0.0,
            "range_end": int(mr.group(1)) if mr else 7,
        }

    # ---- 3) risk & money management ----
    mrr = re.search(r"1\s*:\s*" + num, t) or re.search(r"\brr\D{0,4}" + num, t)
    if mrr:
        spec["rr"] = float(mrr.group(1))

    mrisk = re.search(num + r"\s*%\s*(?:risk|per trade)", t) or re.search(
        r"risk\D{0,20}?" + num + r"\s*%", t
    )
    if mrisk:
        spec["risk_pct"] = float(mrisk.group(1))

    mdl = re.search(r"daily\D{0,15}?" + num + r"\s*%", t) or re.search(
        num + r"\s*%\s*daily", t
    )
    if mdl:
        spec["daily_loss_limit_pct"] = float(mdl.group(1))

    mmt = re.search(r"max\w*\D{0,6}?(\d+)\s*trade", t) or re.search(
        r"(\d+)\s*trades?\s*(?:per|a|/)\s*day", t
    )
    if mmt:
        spec["max_trades_per_day"] = int(mmt.group(1))

    # ---- 4) session ----
    ny = has("ny") or has("new york")
    if has("london") and ny:
        spec["session"] = "london_ny"
    elif has("london"):
        spec["session"] = "london"
    elif ny:
        spec["session"] = "ny"
    elif has("tokyo") or has("asia"):
        spec["session"] = "tokyo"

    # ---- 5) stop loss ----
    if has("fixed"):
        mf = re.search(r"fixed\D{0,10}?" + num, t)
        spec["sl_method"] = "fixed"
        if mf:
            spec["sl_fixed"] = float(mf.group(1))
    elif has("atr"):
        ma = (
            re.search(num + r"\s*x\s*atr", t)
            or re.search(r"atr\D{0,4}" + num + r"\s*x", t)
            or re.search(r"atr\D{0,4}" + num, t)
        )
        if ma:
            spec["sl_atr_mult"] = float(ma.group(1))

    return spec


def _clamp(val, lo, hi, name, warnings):
    if val < lo or val > hi:
        fixed = min(max(val, lo), hi)
        warnings.append(f"{name}={val:g} allowed range {lo}-{hi} se bahar - {fixed:g} kar diya")
        return fixed
    return val


def validate(spec: dict) -> dict:
    w = spec["warnings"]
    p = spec["params"]

    if spec["strategy"] is None:
        spec["strategy"] = "ema_cross"
        w.append("koi strategy detect nahi hui - default EMA cross (20/50) use kiya")

    if spec["strategy"] == "ema_cross":
        if p.get("fast", 0) >= p.get("slow", 1):
            p["fast"], p["slow"] = sorted([p["fast"], p["slow"]])
            w.append(f"fast >= slow tha - swap karke {p['fast']}/{p['slow']} kiya")
        p["fast"] = int(_clamp(p.get("fast", 20), 2, 200, "fast EMA", w))
        p["slow"] = int(_clamp(p.get("slow", 50), 5, 400, "slow EMA", w))

    if spec["strategy"] == "vwap_ema":
        p["ema_length"] = int(_clamp(p.get("ema_length", 20), 5, 200, "EMA length", w))
        p["slope"] = int(bool(p.get("slope", 0)))

    if spec["strategy"] == "rsi_reversal":
        p["length"] = int(_clamp(p.get("length", 14), 2, 50, "RSI length", w))
        p["oversold"] = _clamp(p.get("oversold", 30.0), 5, 45, "oversold", w)
        p["overbought"] = _clamp(p.get("overbought", 70.0), 55, 95, "overbought", w)

    if spec["strategy"] == "bollinger":
        p["length"] = int(_clamp(p.get("length", 20), 5, 100, "BB length", w))
        p["num_std"] = _clamp(p.get("num_std", 2.0), 1.0, 4.0, "BB std", w)

    if spec["strategy"] == "supertrend":
        p["length"] = int(_clamp(p.get("length", 10), 5, 50, "ATR length", w))
        p["mult"] = _clamp(p.get("mult", 3.0), 0.5, 6.0, "multiplier", w)

    if spec["strategy"] == "session_breakout":
        p["buffer"] = _clamp(p.get("buffer", 0.0), 0.0, 5.0, "buffer", w)
        p["range_end"] = int(_clamp(p.get("range_end", 7), 3, 12, "range end hour", w))

    if spec["strategy"] == "macd_cross":
        p["fast"] = int(_clamp(p.get("fast", 12), 2, 50, "MACD fast", w))
        p["slow"] = int(_clamp(p.get("slow", 26), 6, 100, "MACD slow", w))
        p["signal"] = int(_clamp(p.get("signal", 9), 2, 30, "MACD signal", w))
        if p["fast"] >= p["slow"]:
            p["fast"], p["slow"] = sorted((p["fast"], p["slow"]))
            w.append(f"MACD fast >= slow tha - swap karke {p['fast']}/{p['slow']} kiya")

    spec["rr"] = _clamp(spec["rr"], 0.5, 5.0, "RR", w)
    spec["risk_pct"] = _clamp(spec["risk_pct"], 0.1, 3.0, "risk %/trade", w)
    if spec["risk_pct"] > 2.0:
        w.append(f"risk {spec['risk_pct']}% notebook guardrail (0.5-2%) se upar hai")
    spec["max_trades_per_day"] = int(_clamp(spec["max_trades_per_day"], 1, 10, "max trades/day", w))
    spec["daily_loss_limit_pct"] = _clamp(spec["daily_loss_limit_pct"], 0.5, 10.0, "daily loss limit", w)

    return spec


def build(spec: dict):
    cfg = StrategyConfig(
        risk_pct=spec["risk_pct"],
        rr=spec["rr"],
        sl_method=spec["sl_method"],
        sl_atr_mult=spec["sl_atr_mult"],
        sl_fixed=spec["sl_fixed"],
        session=spec["session"],
        max_trades_per_day=spec["max_trades_per_day"],
        daily_loss_limit_pct=spec["daily_loss_limit_pct"],
    )
    cls = ALL_STRATEGIES[spec["strategy"]]
    strat = cls(cfg, **spec["params"])
    return strat, cfg


def explain(spec: dict, metrics: dict, wf, tf_label: str) -> list:
    lines = []
    wr = metrics["Win Rate %"]
    rr = spec["rr"]
    realized_rr = metrics.get("RR (realized)")
    breakeven = 100.0 / (1.0 + realized_rr) if isinstance(realized_rr, (int, float)) and realized_rr > 0 else None
    margin = wr - breakeven if breakeven is not None else None
    dd = metrics["Max DD %"]
    con_losses = metrics["Con. Losses"]
    n = metrics["Total Trades"]
    verdict = wf["Verdict"] if wf else "N/A"

    sl_txt = f"{spec['sl_atr_mult']:g}xATR" if spec["sl_method"] == "atr" else f"{spec['sl_fixed']:g} USD fixed"
    lines.append(
        f"🎯 Kya bana: {metrics['Strategy']} | session {spec['session']} | risk {spec['risk_pct']:g}%/trade "
        f"| SL {sl_txt} | TP 1:{rr:g} | max {spec['max_trades_per_day']} trades/day | daily loss limit {spec['daily_loss_limit_pct']:g}%"
    )
    lines.append(
        f"📈 Full-data ({tf_label}): {n} trades | win {wr}% | PF {metrics['Profit Factor']} "
        f"| Sharpe {metrics['Sharpe']} | DD {dd}% | net ${metrics['Net P&L $']}"
    )
    if wf:
        lines.append(
            f"🧪 Walk-forward (fixed params, unseen folds): {wf['Verdict']} | OOS ret {wf['OOS Ret %']}% "
            f"| OOS Sharpe {wf['OOS Sharpe (med)']} | OOS win {wf['OOS Win %']}%"
        )

    if margin is None:
        lines.append("⚠️ Realized RR unavailable - sample mein wins/losses dono chahiye")
    elif margin < 0:
        lines.append(f"❌ Win rate breakeven ({breakeven:.0f}%) se {abs(margin):.1f} pts NEECHE - is RR pe ye setup mathematically losing hai")
    elif margin < 5:
        lines.append(f"⚠️ Win rate breakeven se sirf {margin:.1f} pts upar - spread+slippage ke baad margin bahut tight hai")
    else:
        lines.append(f"✅ Win rate breakeven se {margin:.1f} pts upar - RR ke saath math theek hai")

    if dd < -20:
        lines.append(f"⚠️ Max DD {dd}% kaafi deep hai - position size chhota kar")
    if con_losses >= 8:
        lines.append(f"⚠️ {con_losses} consecutive losses ka sequence aaya - prop firm daily DD rules ke saath ye combo dangerous hai")
    if n < 40:
        lines.append(f"⚠️ Sirf {n} trades - sample size chhota, koi bhi number pe zyada bharosa mat kar")

    if verdict == "FAIL":
        lines.append("❌ OOS FAIL: unseen data pe loss - full-data wala number dhokha hai, is config ko deploy mat kar")
    elif verdict == "ROBUST":
        lines.append("✅ OOS ROBUST: unseen folds pe bhi positive - overfitting ke kam signs")
    elif verdict == "MIXED":
        lines.append("🟡 OOS MIXED: profit hai par Sharpe weak - confirm karne ke liye aur data/folds chahiye")

    suggestions = []
    if margin is not None and margin < 5:
        better_rr = max(1.0, round(rr * 0.75, 1))
        suggestions.append(f"RR 1:{better_rr:g} try kar - win rate uthega, margin improve hoga")
    if dd < -20:
        suggestions.append(f"--risk {max(0.25, spec['risk_pct'] / 2):g} aur --daily-loss-limit 2 se re-run kar")
    if spec["session"] == "all":
        suggestions.append("session filter try kar: 'london session' (gold London/NY hours mein sabse active hai)")
    if spec["strategy"] == "vwap_ema" and not spec["params"].get("slope", 0):
        suggestions.append("'with slope' add kar - EMA direction filter noise kaat ta hai")
    if verdict in ("FAIL", "MIXED"):
        suggestions.append("run_optimize.py chala ke is strategy ka best param set dekh, phir wahi params yahan use kar")
    for s in suggestions:
        lines.append(f"💡 {s}")

    return lines


# LLMs name parameters in natural language; map common aliases onto the
# exact keys the strategy classes accept, then drop anything unknown so a
# creative LLM can never crash the backtest with a bogus kwarg.
PARAM_ALIASES = {
    "std_dev": "num_std", "std": "num_std", "bands": "num_std",
    "period": "length", "len": "length",
    "overbought_level": "overbought", "oversold_level": "oversold",
    "fast_period": "fast", "slow_period": "slow", "signal_period": "signal",
    "atr_multiplier": "sl_atr_mult", "atr_mult": "sl_atr_mult",
    "multiplier": "mult", "stop_mult": "sl_atr_mult",
}


def _normalize_spec(raw: dict) -> dict:
    """Fill defaults for any missing keys before validation (LLM or parser output)."""
    out = dict(DEFAULTS)
    out["strategy"] = raw.get("strategy") or "ema_cross"
    params = {}
    from core.optimize import PARAM_GRID
    allowed = set(PARAM_GRID.get(out["strategy"], {}).keys())
    for k, v in (raw.get("params") or {}).items():
        k2 = PARAM_ALIASES.get(k, k)
        if k2 in allowed:
            params[k2] = v
        elif k not in allowed and k2 not in allowed:
            # keep unknown keys out of the constructor; surface them as notes
            out.setdefault("notes", []).append(f"ignored unknown parameter '{k}'")
    out["params"] = params
    out["risk_pct"] = float(raw.get("risk_pct") or DEFAULTS["risk_pct"])
    out["rr"] = float(raw.get("rr") or DEFAULTS["rr"])
    out["sl_method"] = raw.get("sl_method") or DEFAULTS["sl_method"]
    out["sl_atr_mult"] = float(raw.get("sl_atr_mult") or DEFAULTS["sl_atr_mult"])
    out["sl_fixed"] = float(raw.get("sl_fixed") or DEFAULTS["sl_fixed"])
    out["session"] = raw.get("session") or DEFAULTS["session"]
    out["max_trades_per_day"] = int(raw.get("max_trades_per_day") or DEFAULTS["max_trades_per_day"])
    out["daily_loss_limit_pct"] = float(raw.get("daily_loss_limit_pct") or DEFAULTS["daily_loss_limit_pct"])
    out["warnings"] = list(raw.get("warnings") or [])
    out["notes"] = list(raw.get("notes") or [])
    return out


def run_pipeline(
    text: str,
    df,
    tf_label: str,
    folds: int = 4,
    start_equity: float = 10_000.0,
    llm: BaseLLM | None = None,
):
    engine_llm = llm if llm is not None else get_llm()
    if engine_llm is not None and engine_llm.is_available():
        try:
            llm_spec = engine_llm.generate_spec(text)
            if llm_spec and llm_spec.get("strategy"):
                spec = validate(_normalize_spec(llm_spec))
            else:
                spec = validate(parse_command(text))
        except Exception:
            spec = validate(parse_command(text))
    else:
        spec = validate(parse_command(text))
    strat, cfg = build(spec)

    equity, trades = run_backtest(df, strat, cfg, start_equity=start_equity)
    metrics = compute_metrics(equity, trades, cfg, tf_label)

    wf = None
    wf_folds = None
    try:
        wf_folds, _, wf = fixed_walk_forward(
            df, type(strat), spec["params"], cfg, tf_label, n_folds=folds,
        )
    except ValueError as e:
        spec["notes"].append(f"walk-forward skip: {e}")

    explanation = explain(spec, metrics, wf, tf_label)
    return spec, metrics, wf, explanation, trades, wf_folds, equity
