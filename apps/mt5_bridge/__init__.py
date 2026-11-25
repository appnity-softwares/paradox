"""MT5 bridge (Part 12.4): authenticated service beside the terminal."""
from apps.mt5_bridge.client import MT5BridgeAdapter, MT5BridgeError
from apps.mt5_bridge.mt5_backend import FakeMT5Backend, MT5Unavailable

__all__ = ["FakeMT5Backend", "MT5BridgeAdapter", "MT5BridgeError",
           "MT5Unavailable"]
