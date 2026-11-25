"""Auto-detecting data manager with explicit market-data provenance.

On startup the platform asks each provider if it's available and picks the
best one. MT5 (Windows) beats Yahoo (cross-platform) beats Synthetic.

The user never has to configure this - it just works.
"""
from datetime import datetime, timezone
import time

from core.broker import MarketDataResult
from core.brokers.mac_mt5_broker import MacMT5Broker
from core.brokers.mt5_broker import MT5Broker
from core.brokers.paxg_broker import PaxgBroker
from core.brokers.yahoo_broker import YahooBroker


class DataManager:
    def __init__(self):
        # Preference order: the broker's own feed first, then a labelled spot
        # proxy, then futures research data. Nothing is ever merged together.
        self.providers = [MacMT5Broker(), MT5Broker(), PaxgBroker(), YahooBroker()]
        self.active = None
        self.statuses = []
        self.last_result: MarketDataResult | None = None
        self._last_primary_probe = 0.0
        self.primary_retry_seconds = 30.0
        self._detect()

    def _detect(self):
        self.statuses = []
        for p in self.providers:
            st = p.is_available()
            self.statuses.append((p, st))
            if st.available and self.active is None:
                self.active = p
        # synthetic fallback if nothing else
        if self.active is None:
            self.active = None

    def summary(self):
        lines = []
        for p, st in self.statuses:
            mark = "✅ ACTIVE" if p is self.active else ("•" if st.available else "✗")
            lines.append(f"{mark} {p.name()}: {st.reason}")
        if self.active is None:
            lines.append("⚠️  No live source - using synthetic data (clearly labeled)")
        return "\n".join(lines)

    def _status_for(self, provider):
        return next((st for p, st in self.statuses if p is provider), None)

    def _probe_primary_if_due(self):
        """Recover MT5 automatically after a transient disconnect/startup delay."""
        if not self.providers or self.active is self.providers[0]:
            return
        now = time.monotonic()
        if now - self._last_primary_probe < self.primary_retry_seconds:
            return
        self._last_primary_probe = now
        primary = self.providers[0]
        status = primary.is_available()
        self.statuses = [
            (provider, status if provider is primary else old_status)
            for provider, old_status in self.statuses
        ]
        if status.available:
            self.active = primary

    def refresh_primary(self):
        self._probe_primary_if_due()

    def fetch_snapshot(
        self,
        symbol="XAUUSD",
        timeframe="15m",
        bars=5000,
        *,
        allow_synthetic=False,
    ) -> MarketDataResult:
        """Fetch data and preserve its true source/capabilities.

        Provider failures never masquerade as real data. Synthetic fallback is
        opt-in and is returned with ``synthetic=True``.
        """
        self._probe_primary_if_due()
        ordered = []
        if self.active is not None:
            ordered.append(self.active)
        ordered.extend(
            p for p, st in self.statuses
            if st.available and p is not self.active
        )

        errors = []
        for provider in ordered:
            status = self._status_for(provider)
            if status is None or not status.available:
                continue
            try:
                df = provider.fetch_ohlcv(symbol, timeframe, bars)
                if df is None or len(df) == 0:
                    raise RuntimeError("provider returned no rows")
                self.active = provider
                result = MarketDataResult(
                    data=df,
                    provider=provider.name(),
                    provider_key=provider.key(),
                    symbol=symbol,
                    timeframe=timeframe,
                    real_data=status.real_data,
                    spot=status.spot,
                    delayed=status.delayed,
                    proxy=getattr(status, "proxy", False),
                    fetched_at=datetime.now(timezone.utc),
                )
                self.last_result = result
                return result
            except Exception as exc:
                errors.append(f"{provider.name()}: {exc}")

        if allow_synthetic:
            from core.data import generate_synthetic, _interval_to_min

            df = generate_synthetic(n_bars=bars, interval_min=_interval_to_min(timeframe))
            result = MarketDataResult(
                data=df,
                provider="Synthetic generator",
                provider_key="synthetic",
                symbol=symbol,
                timeframe=timeframe,
                real_data=False,
                spot=False,
                delayed=False,
                synthetic=True,
                fetched_at=datetime.now(timezone.utc),
            )
            self.last_result = result
            return result

        detail = "; ".join(errors) if errors else "no provider available"
        raise RuntimeError(f"No market data for {symbol} {timeframe}: {detail}")

    def fetch_ohlcv(self, symbol="XAUUSD", timeframe="15m", bars=5000):
        """Compatibility helper. New code should use :meth:`fetch_snapshot`."""
        return self.fetch_snapshot(
            symbol,
            timeframe,
            bars,
            allow_synthetic=False,
        ).data

    @property
    def active_name(self):
        return self.active.name() if self.active else "No live data source"

    @property
    def is_real(self):
        return self.active is not None

    def active_status(self):
        return self._status_for(self.active) if self.active is not None else None

    def is_spot(self):
        st = self.active_status()
        return bool(st.spot) if st else False

    def has_execution(self):
        st = self.active_status()
        return bool(st.execution) if st else False

    def all_statuses(self):
        return [(p.name(), st) for p, st in self.statuses]


# singleton - import this everywhere
_manager = None


def get_manager():
    global _manager
    if _manager is None:
        _manager = DataManager()
    return _manager
