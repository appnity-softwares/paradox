"""Replay-soak smoke tests (Part 14.4): the live pipeline over real
ticks must stay internally consistent and deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_dataset

pytestmark = requires_dataset


@pytest.fixture(scope="module")
def soak_report():
    from scripts.soak_replay import run_soak
    return run_soak(days=2)


def test_soak_labels_itself_honestly(soak_report):
    assert soak_report["source_label"] == "REPLAY_SOAK"
    assert "not live trading" in soak_report["source_detail"]
    assert soak_report["window"]["ticks"] > 100_000


def test_soak_integrity_is_clean(soak_report):
    assert soak_report["integrity_problems"] == []
    assert soak_report["end_equity"] > 0
    assert soak_report["stats"]["orders_submitted"] == \
        soak_report["stats"]["filled"] + soak_report["stats"]["rejected"] \
        + soak_report["stats"]["pending"]


def test_soak_rejections_are_recorded_with_reasons(soak_report):
    # The risk engine blocks entries on real spread blowouts; every
    # rejection must carry a reason.
    total = sum(soak_report["rejection_reasons"].values())
    assert total == soak_report["stats"]["rejected"]
    assert all(isinstance(k, str) and k for k in
               soak_report["rejection_reasons"])


def test_soak_closes_carry_reasons(soak_report):
    for c in soak_report["closed_trades"]:
        assert c["reason"] in ("stop_loss", "take_profit", "signal",
                               "session_end", "max_hold", "eod",
                               "partial_close", "manual")
        assert isinstance(c["pnl"], float)


def test_soak_report_persisted_and_deterministic(soak_report, tmp_path):
    from scripts.soak_replay import run_soak
    again = run_soak(days=2)
    for key in ("end_equity", "stats", "rejection_reasons",
                "integrity_problems"):
        assert again[key] == soak_report[key]
    # The report file exists and parses.
    path = Path("reports/soak_replay.json")
    assert json.loads(path.read_text())["source_label"] == "REPLAY_SOAK"
