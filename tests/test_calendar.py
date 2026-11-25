"""U3 tests: economic calendar integration (Update Blueprint v2.1).

Covers: file providers (CSV/JSON), service staleness and fail-closed
policy, impact/currency filtering, router wiring, position policy at
window start, window edges (inclusive, documented), scheduler skip,
backtest parity, and the no-secrets guarantee of the HTTP skeleton.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from core.calendar import (CalendarEvent, CalendarService,
                           FileCalendarProvider, HttpCalendarProvider,
                           build_news_windows, windows_from_store)
from tests.test_execution_gate import env, intent, setup_auto  # noqa: F401

EVENT_MS = 1_780_329_600_000          # 2026-06-01 16:00:00 UTC (NFP-style)
DAY = 86_400_000
import time as _time
NOW_MS = int(_time.time() * 1000)     # real wall clock, for freshness only


def ruleset_with_news(**over):
    from core.compliance.rules import (FirmRuleSet, NewsRestriction,
                                       DailyLossRule, MaxLossRule)
    nr = NewsRestriction(enabled=True, window_before_min=2,
                         window_after_min=2, **over)
    return FirmRuleSet(
        firm="t", program="p", version="v", status="verified",
        verified_by="h", ea_policy="allowed",
        source_clause_refs={"daily_loss": ["3.2"], "max_loss": ["5.1"]},
        starting_balance=10_000.0, profit_target_pct=8.0,
        daily_loss=DailyLossRule(5.0, "balance", "17:00", "UTC", True),
        max_loss=MaxLossRule(10.0, "static", "starting_balance", True),
        news_restriction=nr)


# ── providers ───────────────────────────────────────────────────────────

def test_file_provider_csv(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("ts_ms,currency,impact,title\n"
                 f"{EVENT_MS},USD,high,Non-Farm Payrolls\n")
    prov = FileCalendarProvider(p)
    events = prov.fetch(EVENT_MS - DAY, EVENT_MS + DAY)
    assert len(events) == 1
    assert events[0].currency == "USD" and events[0].impact == "high"


def test_file_provider_json(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"events": [
        {"ts_ms": EVENT_MS, "currency": "USD", "impact": "high",
         "title": "NFP"},
        {"time_utc": "2026-06-01T09:30:00Z", "currency": "EUR",
         "impact": "medium", "title": "CPI"}]}))
    events = FileCalendarProvider(p).fetch(EVENT_MS - DAY, EVENT_MS + DAY)
    assert len(events) == 2
    assert any(e.currency == "EUR" for e in events)


def test_http_provider_skeleton_fails_closed_without_vendor(tmp_path):
    prov = HttpCalendarProvider("https://example.invalid/cal",
                                api_key_env="CAL_API_KEY",
                                cache_path=tmp_path / "c.json")
    with pytest.raises(NotImplementedError):
        prov.fetch(0, DAY)


def test_http_provider_key_only_from_env():
    prov = HttpCalendarProvider("https://x.invalid", api_key_env="CAL_API_KEY")
    # The instance stores only the ENV VAR NAME, never a key value.
    assert "CAL_API_KEY" in str(vars(prov))
    assert all("secret" not in k.lower() and "key_value" not in k.lower()
               for k in vars(prov))


# ── service ─────────────────────────────────────────────────────────────

def _svc(tmp_path, events=None, fetched_ms=None, error=False):
    class FakeProvider:
        name = "fake"
        def __init__(self, evts, fail):
            self.evts, self.fail = evts, fail
        def fetch(self, a, b):
            if self.fail:
                raise ConnectionError("provider down")
            return [CalendarEvent(**e, source="fake") for e in self.evts]
    evts = events if events is not None else [
        {"ts_ms": EVENT_MS, "currency": "USD", "impact": "high",
         "title": "NFP"}]
    svc = CalendarService(FakeProvider(evts, error),
                          store_path=tmp_path / "cal.json")
    if not error:
        # fetched_at is anchored to the REAL wall clock so status()
        # (which uses now_ms=None -> wall clock) reports fresh.
        svc.refresh(EVENT_MS - DAY, EVENT_MS + DAY,
                    now_ms=fetched_ms if fetched_ms is not None else NOW_MS)
    return svc


def test_service_status_fresh_stale_missing(tmp_path):
    assert _svc(tmp_path).status() == "ok"
    assert _svc(tmp_path, fetched_ms=NOW_MS - 2 * DAY).status() == "stale"
    svc = CalendarService(FileCalendarProvider(tmp_path / "nope.json"),
                          store_path=tmp_path / "x.json")
    assert svc.status() == "missing"


def test_service_provider_error_keeps_old_data_and_goes_stale(tmp_path):
    svc = _svc(tmp_path)                       # fresh store
    svc.provider = type("Broken", (), {
        "name": "broken",
        "fetch": lambda self, a, b: (_ for _ in ()).throw(
            ConnectionError("down"))})()
    st = svc.refresh(0, DAY, now_ms=EVENT_MS)
    assert st == "ok"                          # data still within limit
    assert svc.last_error is not None          # error recorded
    # After the staleness limit passes (relative to the real fetched_at
    # anchor): stale (fail closed upstream).
    assert svc.status(now_ms=NOW_MS + 2 * DAY) == "stale"


def test_impact_and_currency_filter(tmp_path):
    svc = _svc(tmp_path, events=[
        {"ts_ms": EVENT_MS, "currency": "USD", "impact": "high", "title": "NFP"},
        {"ts_ms": EVENT_MS + 1, "currency": "USD", "impact": "low", "title": "small"},
        {"ts_ms": EVENT_MS + 2, "currency": "EUR", "impact": "high", "title": "ECB"}])
    evts = svc.events_filtered(("USD",), ("high",), 0, EVENT_MS + DAY)
    assert [e.title for e in evts] == ["NFP"]


# ── router wiring ───────────────────────────────────────────────────────

def _router_with_news(env, svc, **nr_over):
    broker, router = env
    setup_auto(router)
    rs = ruleset_with_news(**nr_over)
    router.register_account(
        router.accounts["a1"], router.snapshots["a1"],
        router.profiles["a1"], rs, router.specs["a1"], 2600.0)
    if svc is not None:
        router.attach_calendar(svc)
    return broker, router


def test_news_window_blocks_entry_inside(env, tmp_path):  # noqa: F811
    broker, router = _router_with_news(env, _svc(tmp_path))
    # Event 16:00, window [15:58, 16:02] inclusive. Entry at 16:00.
    r = router.submit(intent(key="n1", created_ms=EVENT_MS))
    assert r.status == "rejected"
    assert "news_window" in r.reasons
    assert not broker.order_history


def test_calendar_missing_blocks_news_restricted_account(env):  # noqa: F811
    broker, router = _router_with_news(env, None)
    r = router.submit(intent(key="n2", created_ms=EVENT_MS - DAY))
    assert r.status == "rejected"
    assert "calendar_unavailable" in r.reasons


def test_calendar_stale_blocks_news_restricted_account(env, tmp_path):  # noqa: F811
    broker, router = _router_with_news(
        env, _svc(tmp_path, fetched_ms=NOW_MS - 2 * DAY))
    r = router.submit(intent(key="n3", created_ms=EVENT_MS - DAY))
    assert r.status == "rejected"
    assert "calendar_unavailable" in r.reasons


def test_no_news_restriction_unaffected_by_calendar(env, tmp_path):  # noqa: F811
    broker, router = env
    setup_auto(router)                         # default ruleset: news off
    r = router.submit(intent(key="n4", created_ms=EVENT_MS))
    assert r.status == "filled"                # no calendar attached at all


def test_entry_outside_window_allowed(env, tmp_path):  # noqa: F811
    broker, router = _router_with_news(env, _svc(tmp_path))
    r = router.submit(intent(key="n5", created_ms=EVENT_MS - 3 * 60_000))
    assert r.status == "filled", r.reasons


def test_low_impact_event_does_not_block(env, tmp_path):  # noqa: F811
    svc = _svc(tmp_path, events=[
        {"ts_ms": EVENT_MS, "currency": "USD", "impact": "low",
         "title": "small print"}])
    broker, router = _router_with_news(env, svc)   # default: high only
    r = router.submit(intent(key="n6", created_ms=EVENT_MS))
    assert r.status == "filled", r.reasons


# ── window edges (documented: BOTH ends inclusive) ──────────────────────

def test_window_edges_inclusive(env, tmp_path):  # noqa: F811
    broker, router = _router_with_news(env, _svc(tmp_path))
    before = 2 * 60_000
    # Exactly at window start (event - 2min): blocked.
    r = router.submit(intent(key="e1", created_ms=EVENT_MS - before))
    assert r.status == "rejected" and "news_window" in r.reasons
    # Exactly at window end (event + 2min): blocked.
    r = router.submit(intent(key="e2", created_ms=EVENT_MS + before))
    assert r.status == "rejected" and "news_window" in r.reasons
    # One ms beyond either edge: allowed.
    r = router.submit(intent(key="e3",
                             created_ms=EVENT_MS + before + 1))
    assert r.status == "filled", r.reasons


# ── position policy at window start ─────────────────────────────────────

def test_position_policy_flatten(env, tmp_path):  # noqa: F811
    broker, router = _router_with_news(
        env, _svc(tmp_path), position_policy="flatten")
    router.submit(intent(key="p1", created_ms=EVENT_MS - DAY))   # open first
    assert len(broker.positions_of("a1")) == 1
    actions = router.enforce_news_position_policy("a1", EVENT_MS)
    assert actions and actions[0]["protective"] is True
    assert not broker.positions_of("a1")


def test_position_policy_keep(env, tmp_path):  # noqa: F811
    broker, router = _router_with_news(env, _svc(tmp_path),
                                       position_policy="keep")
    router.submit(intent(key="p2", created_ms=EVENT_MS - DAY))
    actions = router.enforce_news_position_policy("a1", EVENT_MS)
    assert actions == []
    assert len(broker.positions_of("a1")) == 1


def test_position_policy_tighten_sl(env, tmp_path):  # noqa: F811
    broker, router = _router_with_news(
        env, _svc(tmp_path), position_policy="tighten_sl")
    router.submit(intent(key="p3", created_ms=EVENT_MS - DAY))
    actions = router.enforce_news_position_policy("a1", EVENT_MS)
    assert actions and actions[0]["action"] == "tighten_sl"
    events = [e for e in router.audit.entries()
              if e.get("event") == "news_tighten_sl"]
    assert events


# ── scheduler pre-check ─────────────────────────────────────────────────

def test_scheduler_skips_without_calendar(env, tmp_path):  # noqa: F811
    from core.session import SessionProfile, SessionScheduler, SessionRunStore
    broker, router = _router_with_news(env, None)
    sched = SessionScheduler(router,
                             run_store=SessionRunStore(tmp_path / "r.jsonl"))
    rec = sched.run_session("a1", SessionProfile(
        name="s", start="12:00", end="16:00", timezone_name="UTC"),
        date(2026, 6, 1), now_ms=EVENT_MS - DAY)
    assert rec["status"] == "skipped"
    assert "calendar_not_loaded" in rec["reason"]


def test_scheduler_runs_with_fresh_calendar(env, tmp_path):  # noqa: F811
    from core.session import SessionProfile, SessionScheduler, SessionRunStore
    broker, router = _router_with_news(env, _svc(tmp_path))
    broker.on_tick("XAUUSD", bid=2599.80, ask=2600.20, ts_ms=EVENT_MS - DAY)
    router.feed_monitor = __import__(
        "core.execution.monitoring", fromlist=["FeedMonitor"]).FeedMonitor(
        broker, max_staleness_ms=DAY)
    sched = SessionScheduler(router,
                             run_store=SessionRunStore(tmp_path / "r.jsonl"))
    # 12:00-16:00 UTC on May 31 contains EVENT_MS - DAY (16:00 UTC is
    # the window end; use a 13:00 check instant inside it).
    rec = sched.run_session("a1", SessionProfile(
        name="s", start="12:00", end="16:00", timezone_name="UTC"),
        date(2026, 5, 31), now_ms=EVENT_MS - DAY - 3 * 3_600_000)
    assert rec["status"] == "active", rec


# ── backtest parity ─────────────────────────────────────────────────────

def test_build_news_windows_and_store_roundtrip(tmp_path):
    svc = _svc(tmp_path)
    wins = windows_from_store(tmp_path / "cal.json", ("USD",), ("high",),
                              2, 2)
    assert wins == ((EVENT_MS - 120_000, EVENT_MS + 120_000),)
    # Currency filter excludes EUR-only events.
    svc2 = _svc(tmp_path / "e2", events=[
        {"ts_ms": EVENT_MS, "currency": "EUR", "impact": "high",
         "title": "ECB"}])
    svc2.refresh(0, DAY, now_ms=EVENT_MS)
    from core.calendar import windows_from_service as _wfs
    assert _wfs(svc2, ("USD",), ("high",), 2, 2, 0, DAY) == ()


def test_backtest_blocks_entries_inside_news_window():
    from core.tick_engine import BacktestConfig, TickEngine
    from core.strategy_api import Signal, SIDE_LONG, SIDE_CLOSE
    from tests.test_tick_engine import FixedSignalStrategy, make_ticks
    MIN = 60_000
    ticks = make_ticks([(i * MIN, 2000.0 + i, 2000.5 + i)
                        for i in range(10)])
    # News window covers minute 5 (bar 5's decision lands inside it).
    win_start = 5 * MIN
    cfg = BacktestConfig(symbol="XAUUSD", initial_balance=10_000.0,
                         risk_pct=1.0, dataset_hash="cal-test",
                         news_windows=((win_start - 30_000,
                                        win_start + 30_000),),
                         block_entries_in_news_windows=True)
    strat = FixedSignalStrategy({
        5: [Signal(SIDE_LONG, sl=1950.0)],
        8: [Signal(SIDE_CLOSE)]})
    res = TickEngine().run(strat, ticks, cfg)
    assert res["behaviour_blocks"]["news_window"] == 1
    assert res["total_trades"] == 0
    # Control: without the block flag the entry goes through (fresh
    # strategy instance — signal state is consumed per run).
    strat2 = FixedSignalStrategy({
        5: [Signal(SIDE_LONG, sl=1950.0)],
        8: [Signal(SIDE_CLOSE)]})
    cfg2 = BacktestConfig(symbol="XAUUSD", initial_balance=10_000.0,
                          risk_pct=1.0, dataset_hash="cal-test",
                          news_windows=((win_start - 30_000,
                                         win_start + 30_000),))
    res2 = TickEngine().run(strat2, ticks, cfg2)
    assert res2["total_trades"] == 1
