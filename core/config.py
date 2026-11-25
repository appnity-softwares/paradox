"""Symbol, broker and cost configuration.

All trading constants that were previously hardcoded in backtest.py,
execution_engine.py, tick_backtest.py etc. now live here as typed
Pydantic models. Add a new entry per symbol or broker as needed.

Rule: no other module may import a raw numeric constant for contract size,
slippage, spread or commission. Always use get_symbol_spec() or get_cost_model().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class SymbolSpec:
    """Static specification for a traded instrument."""
    symbol: str
    digits: int              # decimal places
    point: float             # smallest price move (e.g. 0.01 for XAUUSD)
    tick_size: float         # minimum price increment
    tick_value: float        # USD P&L per tick per standard lot
    contract_size: float     # units per standard lot (e.g. 100 oz for XAUUSD)
    min_lot: float = 0.01
    max_lot: float = 500.0
    lot_step: float = 0.01
    profit_currency: str = "USD"
    margin_currency: str = "USD"
    # Trading sessions (UTC hours, inclusive)
    session_open_utc: int = 22   # Sunday 22:00
    session_close_utc: int = 21  # Friday 21:59


@dataclass(frozen=True)
class SlippageModel:
    """Configurable slippage per order type, expressed in PRICE units.

    Engines subtract/add these directly to bid/ask, so 0.05 means $0.05 on
    XAUUSD (5 broker points). Naming is kept for compatibility.
    """
    # Slippage added to market-order fills, in price units
    market_points: float = 0.05
    # Stop orders (stops fill through the price)
    stop_points: float = 0.10
    # Limit orders fill at or better than requested
    limit_points: float = 0.0
    # Volatility multiplier applied during news windows
    news_multiplier: float = 3.0


@dataclass
class CostModel:
    """Full cost model for a symbol on a specific broker."""
    symbol: str
    # Commission: 0 = ECN raw (spread only), or USD per standard lot charged
    # at entry AND at exit (both engines; a round trip therefore costs 2x).
    commission_per_lot: float = 0.0
    # Spread model: "recorded" uses actual bid/ask, "fixed" uses fixed_spread_points
    spread_model: str = "recorded"       # "recorded" | "fixed"
    # Broker points (1 point = SymbolSpec.point). Used only when
    # spread_model == "fixed"; spread_price() converts to price units.
    fixed_spread_points: float = 30.0
    slippage: SlippageModel = field(default_factory=SlippageModel)
    # Swap: USD per standard lot per night charged at each UTC-midnight
    # rollover while a position is open (3x on the triple-swap weekday).
    swap_long: float = -0.30
    swap_short: float = -0.30
    swap_triple_day: int = 3             # 0=Mon…6=Sun; 3=Thu for most FX/gold
    # Latency to simulate (ms) — fills use first tick after this delay
    fill_latency_ms: int = 100

    def spread_price(self) -> float:
        """Documented fallback spread in price units (bar data has no quotes)."""
        spec = get_symbol_spec(self.symbol)
        return self.fixed_spread_points * spec.point

    def describe(self) -> dict:
        """Report-ready snapshot of every cost assumption."""
        return {
            "symbol": self.symbol,
            "spread_model": self.spread_model,
            "fixed_spread_price": round(self.spread_price(), 5),
            "commission_per_lot": self.commission_per_lot,
            "slippage_market": self.slippage.market_points,
            "slippage_stop": self.slippage.stop_points,
            "slippage_limit": self.slippage.limit_points,
            "news_multiplier": self.slippage.news_multiplier,
            "swap_long": self.swap_long,
            "swap_short": self.swap_short,
            "swap_triple_day": self.swap_triple_day,
            "fill_latency_ms": self.fill_latency_ms,
        }


# ── Defaults and broker registry ───────────────────────────────────────────────

DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_COST_MODEL = "XAUUSD_ECN"
DEFAULT_BROKER = "mt5_bridge"

# Symbol -> default cost model key, so engines can ask by symbol.
SYMBOL_DEFAULT_COST: Dict[str, str] = {
    "XAUUSD": "XAUUSD_ECN",
    "EURUSD": "XAUUSD_ECN",
    "GBPUSD": "XAUUSD_ECN",
}


# ── Symbol Registry ────────────────────────────────────────────────────────────

SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD",
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=1.0,       # $1 per 0.01 move per 1 lot (100 oz)
        contract_size=100.0,  # 1 standard lot = 100 troy oz
        min_lot=0.01,
        max_lot=500.0,
        lot_step=0.01,
        profit_currency="USD",
        margin_currency="USD",
    ),
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        digits=5,
        point=0.00001,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100_000.0,
        min_lot=0.01,
        max_lot=500.0,
        lot_step=0.01,
    ),
    "GBPUSD": SymbolSpec(
        symbol="GBPUSD",
        digits=5,
        point=0.00001,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100_000.0,
        min_lot=0.01,
        max_lot=500.0,
        lot_step=0.01,
    ),
}

# ── Cost Model Registry ────────────────────────────────────────────────────────

COST_MODELS: Dict[str, CostModel] = {
    "XAUUSD_ECN": CostModel(
        symbol="XAUUSD",
        commission_per_lot=0.0,
        spread_model="recorded",
        fixed_spread_points=30.0,
        slippage=SlippageModel(market_points=0.05, stop_points=0.10, limit_points=0.0),
        swap_long=-0.30,
        swap_short=-0.30,
        swap_triple_day=3,
        fill_latency_ms=100,
    ),
    "XAUUSD_STANDARD": CostModel(
        symbol="XAUUSD",
        commission_per_lot=0.0,
        spread_model="fixed",
        fixed_spread_points=40.0,
        slippage=SlippageModel(market_points=0.10, stop_points=0.20, limit_points=0.0),
        swap_long=-0.50,
        swap_short=-0.50,
        swap_triple_day=3,
        fill_latency_ms=150,
    ),
}


def get_symbol_spec(symbol: str) -> SymbolSpec:
    """Return the SymbolSpec for the given symbol. Raises KeyError if unknown."""
    key = symbol.upper()
    if key not in SYMBOL_SPECS:
        raise KeyError(
            f"Symbol '{symbol}' not found in registry. "
            f"Add it to core/config.py SYMBOL_SPECS first."
        )
    return SYMBOL_SPECS[key]


def get_cost_model(key: str = DEFAULT_COST_MODEL) -> CostModel:
    """Return a CostModel by registry key, or by symbol (uses its default model).

    Raises KeyError if unknown — engines must never silently fall back to a
    guessed cost model.
    """
    if key in COST_MODELS:
        return COST_MODELS[key]
    alias = SYMBOL_DEFAULT_COST.get(key.upper())
    if alias is not None:
        return COST_MODELS[alias]
    raise KeyError(
        f"Cost model '{key}' not found. Available: {list(COST_MODELS)}"
    )


def pnl_per_point(symbol: str, lots: float) -> float:
    """Calculate USD P&L per 1-point (tick_size) move for the given lot size."""
    spec = get_symbol_spec(symbol)
    return spec.tick_value * lots


def round_lot(lots: float, symbol: str = "XAUUSD") -> float:
    """Round lots down to the nearest lot_step and clamp to [min_lot, max_lot]."""
    spec = get_symbol_spec(symbol)
    if lots < spec.min_lot:
        return 0.0  # below minimum — caller should reject
    stepped = int(lots / spec.lot_step) * spec.lot_step
    return round(min(stepped, spec.max_lot), 10)


def required_margin(symbol: str, lots: float, price: float, leverage: int = 100) -> float:
    """Return margin in USD required to open a position."""
    spec = get_symbol_spec(symbol)
    notional = lots * spec.contract_size * price
    return notional / leverage


# ── Broker registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BrokerSpec:
    """Broker-side constraints. Leverage and stops levels are not symbol facts."""
    name: str
    server: str = ""
    leverage: int = 100
    commission_per_lot: float = 0.0
    stops_level_points: int = 0
    freeze_level_points: int = 0
    execution_mode: str = "market"   # "market" | "instant" | "request"
    margin_call_level_pct: float = 100.0
    stop_out_level_pct: float = 50.0


BROKER_SPECS: Dict[str, BrokerSpec] = {
    "mt5_bridge": BrokerSpec(name="mt5_bridge", leverage=100, execution_mode="market"),
    "paper": BrokerSpec(name="paper", server="simulated", leverage=100, execution_mode="instant"),
}


def get_broker_spec(name: str = DEFAULT_BROKER) -> BrokerSpec:
    """Return the BrokerSpec. Raises KeyError if unknown."""
    if name not in BROKER_SPECS:
        raise KeyError(f"Broker '{name}' not found. Available: {list(BROKER_SPECS)}")
    return BROKER_SPECS[name]


def register_broker(spec: BrokerSpec) -> None:
    BROKER_SPECS[spec.name] = spec


def register_symbol(spec: SymbolSpec) -> None:
    SYMBOL_SPECS[spec.symbol.upper()] = spec


def register_cost_model(key: str, model: CostModel) -> None:
    COST_MODELS[key] = model
