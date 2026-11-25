from strategies.base import StrategyConfig, SESSIONS
from strategies.library import (
    EMACross,
    MACDCross,
    RSIReversal,
    BollingerReversion,
    VWAPEMA,
    Supertrend,
    SessionBreakout,
    FairValueGap,
    SMCOrderBlock,
    LiquiditySweep,
)

ALL_STRATEGIES = {
    EMACross.key: EMACross,
    MACDCross.key: MACDCross,
    RSIReversal.key: RSIReversal,
    BollingerReversion.key: BollingerReversion,
    VWAPEMA.key: VWAPEMA,
    Supertrend.key: Supertrend,
    SessionBreakout.key: SessionBreakout,
    FairValueGap.key: FairValueGap,
    SMCOrderBlock.key: SMCOrderBlock,
    LiquiditySweep.key: LiquiditySweep,
}

__all__ = [
    "StrategyConfig",
    "SESSIONS",
    "EMACross",
    "MACDCross",
    "RSIReversal",
    "BollingerReversion",
    "VWAPEMA",
    "Supertrend",
    "SessionBreakout",
    "FairValueGap",
    "SMCOrderBlock",
    "LiquiditySweep",
    "ALL_STRATEGIES",
]
