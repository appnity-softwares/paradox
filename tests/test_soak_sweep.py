"""Soak sweep tests: the month-by-month live-pipeline replay must
aggregate cleanly and record which safety systems fired.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_dataset

pytestmark = requires_dataset


@pytest.fixture(scope="module")
def sweep_report():
    from scripts.soak_sweep import sweep
    return sweep(max_months=2)


def test_sweep_labels_itself_honestly(sweep_report):
    assert sweep_report["source_label"] == "REPLAY_SOAK_SWEEP"
    assert "not live trading" in sweep_report["source_detail"]
    assert sweep_report["n_months"] == 2


def test_sweep_months_are_clean(sweep_report):
    assert sweep_report["n_problem"] == 0
    for row in sweep_report["months"]:
        assert row["integrity_problems"] == []
        assert row["ticks"] > 1_000_000
        assert row["orders"] > 0


def test_sweep_persists_report(sweep_report):
    path = Path("reports/soak_sweep.json")
    body = json.loads(path.read_text())
    assert body["source_label"] == "REPLAY_SOAK_SWEEP"
    assert len(body["months"]) >= sweep_report["n_months"]


def test_sweep_month_bounds_are_calendar_exact():
    from scripts.soak_sweep import month_bounds
    start, end = month_bounds(2024, 2)      # leap year, 29 days
    import datetime as dt
    assert dt.datetime.fromtimestamp(start / 1000, tz=dt.timezone.utc) \
        == dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc)
    assert dt.datetime.fromtimestamp(end / 1000, tz=dt.timezone.utc) \
        == dt.datetime(2024, 2, 29, 23, 59, 59,
                       tzinfo=dt.timezone.utc)
