"""Multi-symbol MT5-Mac support: per-symbol history sync, per-symbol bar
files, per-symbol quote spreads — and the honesty guard that a non-gold
symbol can never be served gold's data."""
from __future__ import annotations

import pandas as pd
import pytest

from core.brokers import mac_mt5_broker as mac
from core.brokers import mt5_mac


@pytest.fixture
def fake_mt5(tmp_path, monkeypatch):
    """Fake MT5 install with EURUSD + XAUUSD history caches."""
    mt5_root = tmp_path / "mt5"
    for sym in ("EURUSD", "XAUUSD"):
        cache = mt5_root / "bases" / "Test" / "history" / sym / "cache"
        cache.mkdir(parents=True)
        (cache / "M1.hc").write_bytes(b"fake")
        (cache / "H1.hc").write_bytes(b"fake")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    idx = pd.date_range("2025-01-01", periods=120, freq="1min", tz="UTC")
    base = pd.DataFrame({
        "time": idx,
        "open": 1.05, "high": 1.0505, "low": 1.0495, "close": 1.05,
        "volume": 10.0,
    })

    def fake_decode(path):
        return base.copy()

    def fake_resample(df, rule):
        return base.copy()

    monkeypatch.setattr(mt5_mac, "MT5_MAC_DIR", mt5_root)
    monkeypatch.setattr(mt5_mac, "DATA_DIR", data_dir)
    monkeypatch.setattr(mt5_mac, "decode_mt5_hc_file", fake_decode)
    monkeypatch.setattr(mt5_mac, "resample_bars", fake_resample)
    monkeypatch.setattr(mac, "DATA_DIR", data_dir)
    return {"root": mt5_root, "data": data_dir}


def test_sync_writes_per_symbol_files(fake_mt5):
    res = mt5_mac.extract_and_sync_all_mt5_bars("EURUSD")
    assert res["success"] is True
    assert (fake_mt5["data"] / "eurusd_15m_real.csv").exists()
    assert (fake_mt5["data"] / "eurusd_1d_real.csv").exists()
    # gold files untouched by the EURUSD sync
    assert not (fake_mt5["data"] / "xauusd_15m_real.csv").exists()


def test_sync_gold_keeps_legacy_names(fake_mt5):
    res = mt5_mac.extract_and_sync_all_mt5_bars("XAUUSD")
    assert res["success"] is True
    assert (fake_mt5["data"] / "xauusd_15m_real.csv").exists()


def test_fetch_ohlcv_per_symbol(fake_mt5):
    df = mac.MacMT5Broker().fetch_ohlcv("EURUSD", "15m", 10)
    assert len(df) == 10
    assert float(df["close"].iloc[-1]) == 1.05


def test_fetch_ohlcv_unknown_symbol_never_serves_gold(fake_mt5, monkeypatch):
    # No USDJPY cache in the fake install: sync fails, file missing, and the
    # broker must raise rather than fall back to XAUUSD prices.
    with pytest.raises(FileNotFoundError):
        mac.MacMT5Broker().fetch_ohlcv("USDJPY", "15m", 10)


def test_fetch_tick_spreads_per_symbol(fake_mt5):
    broker = mac.MacMT5Broker()
    fx = broker.fetch_tick("EURUSD")
    assert fx is not None
    assert fx["ask"] - fx["bid"] == pytest.approx(0.0001, abs=1e-9)
    assert fx["bid"] < 2  # FX-scale price, not gold


def test_fetch_tick_gold_keeps_legacy_spread(fake_mt5):
    broker = mac.MacMT5Broker()
    gold = broker.fetch_tick("XAUUSD")
    assert gold is not None
    # gold-scale price (store tick or synthesized bar close), positive spread
    assert 1000 < gold["last"] < 10000
    assert gold["ask"] > gold["bid"]


def test_paxg_proxy_never_serves_other_symbols():
    from core.brokers.paxg_broker import PaxgBroker
    with pytest.raises(ValueError):
        PaxgBroker().fetch_ohlcv("EURUSD", "1h", 10)
    assert PaxgBroker().fetch_tick("EURUSD") is None


def test_yahoo_futures_never_serve_other_symbols():
    from core.brokers.yahoo_broker import YahooBroker
    with pytest.raises(ValueError):
        YahooBroker().fetch_ohlcv("EURUSD", "1h", 10)
