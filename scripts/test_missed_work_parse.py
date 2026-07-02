"""Tier-1 missed-work detection and parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent import is_missed_work_request, parse_missed_work_spec


def test_rejects_snooze():
    assert not is_missed_work_request("snooze bcg until thursday")


def test_all_missed_today():
    msg = "i didnt work on anything today can you reschedule"
    assert is_missed_work_request(msg)
    assert parse_missed_work_spec(msg) == {"all_missed_today": True}


def test_kayak_window():
    msg = "i was kayaking for 2 hours, reschedule what i had"
    assert is_missed_work_request(msg)
    spec = parse_missed_work_spec(msg)
    assert spec == {"overlap_since_minutes": 120}


def test_single_task_missed():
    msg = "i lowk didnt work on any BCG prep today can you reschedule"
    assert is_missed_work_request(msg)
    spec = parse_missed_work_spec(msg)
    assert spec is not None
    assert "bcg" in spec.get("task_query", "").lower()


def test_skipped_with_target():
    msg = "i skipped BCG move it to tomorrow morning"
    spec = parse_missed_work_spec(msg)
    assert spec is not None
    assert "bcg" in spec.get("task_query", "").lower()
    assert spec.get("snooze_until_natural") == "tomorrow morning"


def test_week_bcg_task():
    msg = "i missed all my bcg work this week can you reschedule"
    assert is_missed_work_request(msg)
    spec = parse_missed_work_spec(msg)
    assert spec is not None
    assert spec.get("period") == "week"
    assert "bcg" in spec.get("task_query", "").lower()


def test_week_all_missed():
    msg = "i missed everything this week"
    spec = parse_missed_work_spec(msg)
    assert spec == {"period": "week", "all_missed_today": True}


def test_vague_missed_returns_none_for_tier1():
    assert parse_missed_work_spec("hey whats up") is None


def main():
    test_rejects_snooze()
    test_all_missed_today()
    test_kayak_window()
    test_single_task_missed()
    test_skipped_with_target()
    test_week_bcg_task()
    test_week_all_missed()
    test_vague_missed_returns_none_for_tier1()
    print("All missed-work parse tests passed.")


if __name__ == "__main__":
    main()
