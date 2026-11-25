"""Strategy configuration - mirrors the notebook spec: SL/TP methods, risk limits, sessions."""
from dataclasses import dataclass

SESSIONS = {
    "all": None,
    "tokyo": (0, 9),
    "london": (7, 16),
    "ny": (12, 21),
    "london_ny": (7, 21),
}


@dataclass
class StrategyConfig:
    name: str = "XAUUSD"
    # --- Risk (notebook: "Max risk per trade = 0.5% - 2%") ---
    risk_pct: float = 1.0
    # --- SL methods (notebook: "Fixed points/pips/% | structural | dynamic (ex = 1.5 X ATR)") ---
    sl_method: str = "atr"       # "atr" | "fixed"
    sl_atr_mult: float = 1.5
    sl_fixed: float = 5.0        # price units (USD on gold)
    # --- TP methods (notebook: "1:/RR | ATR expansion | fixed") ---
    rr: float = 2.0
    # --- Session filter (notebook: "Only trades = Tokyo or London or NY or Sydney session") ---
    session: str = "all"
    # --- Trade limits (notebook: "Trade limits = 1 trade a day / 2 3 4 trades a day") ---
    max_trades_per_day: int = 4
    # --- Loss limits (notebook: "Loss limits = Max loss 2% a day / 3%") ---
    daily_loss_limit_pct: float = 3.0

    def __post_init__(self):
        if not 0 < self.risk_pct <= 3.0:
            raise ValueError("risk_pct must be > 0 and <= 3")
        if self.sl_method not in {"atr", "fixed"}:
            raise ValueError("sl_method must be 'atr' or 'fixed'")
        if self.sl_atr_mult <= 0 or self.sl_fixed <= 0:
            raise ValueError("stop-loss distance must be positive")
        if self.rr <= 0:
            raise ValueError("rr must be positive")
        if self.session not in SESSIONS:
            raise ValueError(f"unknown session: {self.session}")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be at least 1")
        if self.daily_loss_limit_pct <= 0:
            raise ValueError("daily_loss_limit_pct must be positive")

    @property
    def session_hours(self):
        return SESSIONS.get(self.session)
