"""U2 tests: session profiles, gate enforcement, scheduler, backtest
parity and the zero-LLM guarantee (Update Blueprint v2.1)."""
from __future__ import annotations

import datetime as dt
from datetime import date, datetime, timezone

import pytest

from core.session import SessionProfile, SessionRunStore, SessionScheduler
from tests.test_execution_gate import env, intent, setup_auto  # noqa: F401


def utc_ms(y, mo, d, h, mi=0, s=0):
    return int(datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
               .timestamp() * 1000)


def utc_profile(**over) -> SessionProfile:
    base = dict(name="london_ny", start="12:00", end="16:00",
                timezone_name="UTC", days_of_week=tuple(range(7)),
                max_duration_minutes=240)
    base.update(over)
    return SessionProfile(**base)


# ── construction rules ──────────────────────────────────────────────────

def test_max_duration_241_raises_240_ok():
    with pytest.raises(ValueError, match="max_duration_minutes"):
        utc_profile(max_duration_minutes=241)
    assert utc_profile(max_duration_minutes=240).name == "london_ny"
    # A window longer than the cap is rejected too.
    with pytest.raises(ValueError, match="exceeds"):
        utc_profile(start="00:00", end="05:00", max_duration_minutes=240)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        utc_profile(start="25:00")
    with pytest.raises(ValueError):
        utc_profile(timezone_name="Not/AZone")
    with pytest.raises(ValueError):
        utc_profile(days_of_week=())
    with pytest.raises(ValueError, match="differ"):
        utc_profile(start="09:00", end="09:00")


# ── DST-safe conversion ─────────────────────────────────────────────────

def test_london_spring_forward_and_fall_back():
    # 2026-03-29: London jumps 01:00 GMT -> 02:00 BST. A 07:00-11:00
    # session is entirely BST: 06:00-10:00 UTC (4h, unchanged).
    p = SessionProfile(name="ldn", start="07:00", end="11:00",
                       timezone_name="Europe/London")
    s, e = p.window_utc(date(2026, 3, 29))
    assert (s, e) == (utc_ms(2026, 3, 29, 6), utc_ms(2026, 3, 29, 10))
    # 2026-10-25: fall back 02:00 BST -> 01:00 GMT. 07:00-11:00 is GMT:
    # 07:00-11:00 UTC.
    s, e = p.window_utc(date(2026, 10, 25))
    assert (s, e) == (utc_ms(2026, 10, 25, 7), utc_ms(2026, 10, 25, 11))


def test_new_york_spring_forward():
    # 2026-03-08: NY jumps 02:00 EST -> 03:00 EDT. 07:00-11:00 EST pre-
    # jump would be 12:00 UTC; on the jump day 07:00 is EDT = 11:00 UTC.
    p = SessionProfile(name="ny", start="07:00", end="11:00",
                       timezone_name="America/New_York")
    s, e = p.window_utc(date(2026, 3, 8))
    assert (s, e) == (utc_ms(2026, 3, 8, 11), utc_ms(2026, 3, 8, 15))
    # A normal winter day: 07:00 EST = 12:00 UTC.
    s, e = p.window_utc(date(2026, 2, 2))
    assert (s, e) == (utc_ms(2026, 2, 2, 12), utc_ms(2026, 2, 2, 16))


def test_session_crossing_midnight_utc():
    # 22:00-02:00 UTC window: end lands on the next UTC day.
    p = utc_profile(start="22:00", end="02:00")
    s, e = p.window_utc(date(2026, 6, 1))
    assert (s, e) == (utc_ms(2026, 6, 1, 22), utc_ms(2026, 6, 2, 2))
    # A tick at 01:00 on Jun 2 belongs to the Jun 1 session.
    assert p.in_window(utc_ms(2026, 6, 2, 1))
    assert not p.in_window(utc_ms(2026, 6, 2, 3))


# ── the three gate functions ────────────────────────────────────────────

def test_entries_allowed_boundaries():
    p = utc_profile(no_new_entries_after_minutes_before_end=30)
    # In window, before cutoff: allowed.
    assert p.entries_allowed(utc_ms(2026, 6, 1, 15, 29))
    # Exactly at the cutoff (15:30): blocked.
    assert not p.entries_allowed(utc_ms(2026, 6, 1, 15, 30))
    # One second after window end (16:00:01): blocked.
    assert not p.entries_allowed(utc_ms(2026, 6, 1, 16, 0, 1))
    # Exactly at window end: blocked.
    assert not p.entries_allowed(utc_ms(2026, 6, 1, 16, 0))


def test_must_flatten():
    p = utc_profile(flatten_by="15:45")
    assert not p.must_flatten(utc_ms(2026, 6, 1, 15, 44, 59))
    assert p.must_flatten(utc_ms(2026, 6, 1, 15, 45))
    # Default flatten instant is the window end.
    assert utc_profile().must_flatten(utc_ms(2026, 6, 1, 16, 0))


def test_days_of_week_gate():
    p = utc_profile(days_of_week=(0,))       # Monday only
    # 2026-06-01 is a Monday, 2026-06-02 a Tuesday.
    assert p.in_window(utc_ms(2026, 6, 1, 13))
    assert not p.in_window(utc_ms(2026, 6, 2, 13))


# ── router gate enforcement ─────────────────────────────────────────────

def _bind(router, profile):
    router.set_session_profile("a1", profile)


def test_entry_one_second_after_window_rejected(env):  # noqa: F811
    broker, router = env
    setup_auto(router)
    _bind(router, utc_profile())
    # 16:00:01 UTC — one second after the window end.
    r = router.submit(intent(created_ms=utc_ms(2026, 6, 1, 16, 0, 1)))
    assert r.status == "rejected"
    assert "outside_session_window" in r.reasons
    assert not broker.order_history


def test_entry_inside_window_fills_and_management_untouched(env):  # noqa: F811
    broker, router = env
    setup_auto(router)
    p = utc_profile()
    _bind(router, p)
    router.set_session_window("a1", p, date(2026, 6, 1))
    r = router.submit(intent(created_ms=utc_ms(2026, 6, 1, 13, 0)))
    assert r.status == "filled", r.reasons


def test_entry_after_cutoff_rejected(env):  # noqa: F811
    broker, router = env
    setup_auto(router)
    p = utc_profile(no_new_entries_after_minutes_before_end=30)
    _bind(router, p)
    router.set_session_window("a1", p, date(2026, 6, 1))
    r = router.submit(intent(created_ms=utc_ms(2026, 6, 1, 15, 30)))
    assert r.status == "rejected"
    assert "session_no_new_entries" in r.reasons


def test_no_window_open_means_no_entries(env):  # noqa: F811
    """Fail closed: a bound profile with no scheduler-opened window
    refuses entries."""
    broker, router = env
    setup_auto(router)
    _bind(router, utc_profile())
    r = router.submit(intent(created_ms=utc_ms(2026, 6, 1, 13, 0)))
    assert r.status == "rejected"
    assert "outside_session_window" in r.reasons


def test_max_trades_in_session_cap(env):  # noqa: F811
    broker, router = env
    setup_auto(router)
    _bind(router, utc_profile(max_trades_in_session=1))
    router.set_session_window("a1", utc_profile(max_trades_in_session=1),
                              date(2026, 6, 1))
    r1 = router.submit(intent(key="t1", created_ms=utc_ms(2026, 6, 1, 13)))
    assert r1.status == "filled"
    r2 = router.submit(intent(key="t2", created_ms=utc_ms(2026, 6, 1, 13, 5)))
    assert r2.status == "rejected"
    assert "session_max_trades" in r2.reasons


def test_riskconfig_session_reference_blocks(env):  # noqa: F811
    """Defence in depth: the risk engine consults the profile too."""
    from core.risk import AccountSnapshot, DeterministicRiskEngine, RiskConfig
    from core.risk.engine import OrderProposal
    eng = DeterministicRiskEngine(RiskConfig(version="v1",
                                             session_profile=utc_profile()))
    snap = AccountSnapshot(account_id="a1", equity=10_000.0, balance=10_000.0,
                           used_margin=0.0, open_positions=0,
                           exposure_lots_by_symbol={},
                           daily_realized_pnl=0.0, daily_floating_pnl=0.0,
                           equity_high_watermark=10_000.0,
                           last_tick_ms={"XAUUSD": utc_ms(2026, 6, 1, 17)},
                           spread={"XAUUSD": 0.20},
                           server_time_ms=utc_ms(2026, 6, 1, 17))
    prop = OrderProposal(account_id="a1", symbol="XAUUSD", side=1, lot=0.08,
                         price=2600.0, sl_price=2588.0,
                         time_ms=utc_ms(2026, 6, 1, 17))
    d = eng.evaluate(prop, snap)
    assert d.outcome == "rejected"
    assert "outside_session_window" in d.reasons


# ── scheduler ───────────────────────────────────────────────────────────

def _scheduler(tmp_path, router):
    from core.execution.monitoring import FeedMonitor
    router.feed_monitor = FeedMonitor(router.broker, max_staleness_ms=10_000)
    store = SessionRunStore(tmp_path / "runs.jsonl")
    return SessionScheduler(router, run_store=store), store


def test_stale_feed_skips_session(env, tmp_path):  # noqa: F811
    broker, router = env
    sched, store = _scheduler(tmp_path, router)
    # The env broker's last tick is at ts=1000 — ancient vs 2026 → stale.
    rec = sched.run_session("a1", utc_profile(), date(2026, 6, 1),
                            now_ms=utc_ms(2026, 6, 1, 12))
    assert rec["status"] == "skipped"
    assert "feed_stale_XAUUSD" in rec["reason"]
    assert store.entries()[-1]["status"] == "skipped"
    # No window opened: entries refused.
    r = router.submit(intent(created_ms=utc_ms(2026, 6, 1, 12, 30)))
    assert r.status == "rejected"


def test_healthy_session_opens_and_flatten_works(env, tmp_path):  # noqa: F811
    broker, router = env
    setup_auto(router)
    sched, store = _scheduler(tmp_path, router)
    # Fresh feed at the session's start instant.
    broker.on_tick("XAUUSD", bid=2599.80, ask=2600.20,
                   ts_ms=utc_ms(2026, 6, 1, 12))
    rec = sched.run_session("a1", utc_profile(), date(2026, 6, 1),
                            now_ms=utc_ms(2026, 6, 1, 12))
    assert rec["status"] == "active"
    # Fresh tick at submit time (the router pauses entries on stale feeds).
    broker.on_tick("XAUUSD", bid=2599.80, ask=2600.20,
                   ts_ms=utc_ms(2026, 6, 1, 12, 30))
    r = router.submit(intent(key="s1", created_ms=utc_ms(2026, 6, 1, 12, 30)))
    assert r.status == "filled"
    # Flatten at 15:45 closes the position protectively and audits it.
    closed = sched.flatten_account("a1", utc_ms(2026, 6, 1, 15, 45))
    assert len(closed) == 1 and closed[0]["protective"] is True
    assert not broker.positions_of("a1")
    events = [e for e in router.audit.entries()
              if e.get("event") == "protective_close"]
    assert events and events[-1]["reason"] == "session_flatten"


def test_missing_ruleset_skips(env, tmp_path):  # noqa: F811
    broker, router = env
    sched, store = _scheduler(tmp_path, router)
    broker.on_tick("XAUUSD", bid=2599.80, ask=2600.20,
                   ts_ms=utc_ms(2026, 6, 1, 12))
    rec = sched.run_session("ghost", utc_profile(), date(2026, 6, 1),
                            now_ms=utc_ms(2026, 6, 1, 12))
    assert rec["status"] == "skipped"
    assert "account_snapshot_missing" in rec["reason"]


def test_zero_llm_calls_in_session_loop(env, tmp_path, monkeypatch):  # noqa: F811
    """The whole session loop must never touch the LLM layer."""
    import core.llm as llm

    def _boom(*a, **k):
        raise AssertionError("LLM called during the session loop")
    monkeypatch.setattr(llm, "get_llm", _boom)
    broker, router = env
    setup_auto(router)
    sched, _ = _scheduler(tmp_path, router)
    broker.on_tick("XAUUSD", bid=2599.80, ask=2600.20,
                   ts_ms=utc_ms(2026, 6, 1, 12))
    sched.run_session("a1", utc_profile(), date(2026, 6, 1),
                      now_ms=utc_ms(2026, 6, 1, 12))
    router.submit(intent(key="s2", created_ms=utc_ms(2026, 6, 1, 12, 30)))
    sched.flatten_account("a1", utc_ms(2026, 6, 1, 15, 45))
    sched.close_session("a1", utc_profile(), date(2026, 6, 1),
                        now_ms=utc_ms(2026, 6, 1, 16))


# ── backtest parity ─────────────────────────────────────────────────────

def _ticks():
    from tests.test_tick_engine import make_ticks
    MIN = 60_000
    rows = []
    price = 2000.0
    for day in range(2):                     # two days
        for i in range(24):                  # 24 minutes per day
            base = day * 86_400_000 + i * MIN
            if i < 12:
                price = 2000.0 + i * 2.0
            else:
                price = 2022.0 - (i - 12) * 3.0
            rows.append((base, price, price + 0.5))
    return make_ticks(rows)


def _run(over: dict, script: dict):
    from core.tick_engine import BacktestConfig, TickEngine
    from core.strategy_api import Signal, SIDE_CLOSE, SIDE_LONG
    from tests.test_tick_engine import FixedSignalStrategy
    out = {}
    for bar, what in script.items():
        out[bar] = [Signal(SIDE_LONG, sl=1950.0)] if what == "ENTER" \
            else [Signal(SIDE_CLOSE)]
    cfg = BacktestConfig(symbol="XAUUSD", initial_balance=10_000.0,
                         risk_pct=1.0, dataset_hash="sess-test", **over)
    return TickEngine().run(FixedSignalStrategy(out), _ticks(), cfg)


def test_backtest_session_filter_prefilters_and_flattens():
    # Window = minutes 0-10 of each day. Day 1: enter at bar 2, close
    # signal at bar 15 lands OUTSIDE the prefiltered data → never seen;
    # the position flattens at the START of day 2's window instead.
    res = _run({"session_start_min": 0, "session_end_min": 10},
               {2: "ENTER", 15: "CLOSE"})
    assert res["valid"]
    assert res["behaviour_blocks"]["session_no_new_entries"] == 0
    # The day-2 window's first tick closes the carried position.
    closes = [t for t in res["trades"] if "session_flatten" in str(t.get("exit_reason", ""))]
    assert len(closes) == 1
    # Only window ticks were processed: 2 days x 10 minutes.
    assert res["ticks_replayed"] == 20


def test_backtest_entry_cutoff_blocks():
    # Window 0-10, cutoff 5 min before end: entries at minute >= 5 are
    # blocked. Enter at bar 7 → blocked.
    res = _run({"session_start_min": 0, "session_end_min": 10,
                "no_new_entries_before_end_min": 5},
               {7: "ENTER", 20: "CLOSE"})
    assert res["behaviour_blocks"]["session_no_new_entries"] >= 1
    assert res["total_trades"] == 0


def test_backtest_cutoff_boundary_exact():
    # Entry at bar 5 (minute 5) with cutoff 5: dist == cutoff → blocked.
    res = _run({"session_start_min": 0, "session_end_min": 10,
                "no_new_entries_before_end_min": 5},
               {5: "ENTER", 20: "CLOSE"})
    assert res["behaviour_blocks"]["session_no_new_entries"] >= 1
    # Entry at bar 4 (minute 4): dist 6 > cutoff 5 → allowed.
    res2 = _run({"session_start_min": 0, "session_end_min": 10,
                 "no_new_entries_before_end_min": 5},
                {4: "ENTER", 20: "CLOSE"})
    assert res2["behaviour_blocks"]["session_no_new_entries"] == 0
    assert res2["total_trades"] >= 1


def test_backtest_no_flatten_flag_keeps_position():
    # flatten_at_session_end=False: the position carries into day 2 and
    # the close signal (bar 15 → day 1 minute 15 is outside; use bar 26
    # = day 2 minute 2) closes it inside the window.
    res = _run({"session_start_min": 0, "session_end_min": 10,
                "flatten_at_session_end": False},
               {2: "ENTER", 26: "CLOSE"})
    assert res["valid"]
    assert not any("session_flatten" in str(t.get("exit_reason", ""))
                   for t in res["trades"])
