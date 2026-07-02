"""Unit tests for unified reschedule_missed_work (mocked Reclaim planner)."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import _fold_missed_work_calls, _normalize_call_params
from reclaim import resolve_snooze_target


def test_resolve_snooze_default():
    now = datetime(2026, 7, 1, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    target, option = resolve_snooze_target(None, now)
    assert target is None
    assert option == "FROM_NOW_1H"


def test_resolve_snooze_tomorrow():
    now = datetime(2026, 7, 1, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    target, option = resolve_snooze_target("tomorrow morning", now)
    assert target is None
    assert option == "TOMORROW"


def test_resolve_snooze_explicit_time():
    now = datetime(2026, 7, 1, 20, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    target, option = resolve_snooze_target("tomorrow morning", now)
    target2, option2 = resolve_snooze_target("friday", now)
    assert option == "TOMORROW"
    assert target2 is not None or option2 is not None


def test_fold_legacy_multiple_call():
    calls = [
        {
            "function": "reschedule_multiple_missed_work",
            "params": {
                "all_missed_today": True,
                "snooze_until_natural": "tomorrow",
            },
        }
    ]
    _fold_missed_work_calls(calls)
    assert calls[0]["function"] == "reschedule_missed_work"
    assert calls[0]["params"]["all_missed_today"] is True
    assert calls[0]["params"]["snooze_until_natural"] == "tomorrow"
    assert _normalize_call_params(calls) == []


async def test_reschedule_missed_work_single_task():
    from composites import reschedule_missed_work

    block = {
        "eventId": "evt1",
        "calendarId": 890919,
        "title": "BCG prep",
        "eventStart": "2026-07-01T10:00:00-07:00",
        "eventEnd": "2026-07-01T10:30:00-07:00",
        "assist": {"taskId": 1},
    }

    async def fake_get_task_by_query(query, **kwargs):
        return {"id": 1, "title": "BCG prep"}

    with (
        patch("composites.get_task_by_query", side_effect=fake_get_task_by_query),
        patch("composites.get_all_events", new_callable=AsyncMock, return_value=[block]),
        patch(
            "composites.find_missed_task_blocks",
            new_callable=AsyncMock,
            return_value=[block],
        ),
        patch(
            "composites.reschedule_missed_event",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_resched,
    ):
        result = await reschedule_missed_work(task_query="bcg")

    assert result["ok"] is True
    assert result["moved_count"] == 1
    mock_resched.assert_awaited_once()


async def test_reschedule_missed_work_all_today():
    from composites import reschedule_missed_work

    block = {
        "eventId": "evt1",
        "calendarId": 890919,
        "title": "BCG prep",
        "eventStart": "2026-07-01T10:00:00-07:00",
        "eventEnd": "2026-07-01T10:30:00-07:00",
        "assist": {"taskId": 1},
    }

    with (
        patch("composites.get_all_events", new_callable=AsyncMock, return_value=[block]),
        patch(
            "composites.task_ids_with_missed_blocks",
            new_callable=AsyncMock,
            return_value=[1],
        ),
        patch(
            "composites.find_missed_task_blocks",
            new_callable=AsyncMock,
            return_value=[block],
        ),
        patch(
            "composites.reschedule_missed_event",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        result = await reschedule_missed_work(all_missed_today=True)

    assert result["ok"] is True
    assert result["moved_count"] == 1


async def test_reschedule_missed_work_week_task():
    from composites import reschedule_missed_work

    monday_block = {
        "eventId": "evt_mon",
        "calendarId": 890919,
        "title": "BCG prep",
        "eventStart": "2026-06-30T10:00:00-07:00",
        "eventEnd": "2026-06-30T10:30:00-07:00",
        "assist": {"taskId": 1},
    }
    today_block = {
        "eventId": "evt_today",
        "calendarId": 890919,
        "title": "BCG prep",
        "eventStart": "2026-07-01T10:00:00-07:00",
        "eventEnd": "2026-07-01T10:30:00-07:00",
        "assist": {"taskId": 1},
    }

    async def fake_get_task_by_query(query, **kwargs):
        return {"id": 1, "title": "BCG prep"}

    with (
        patch("composites.get_task_by_query", side_effect=fake_get_task_by_query),
        patch(
            "composites.get_all_events",
            new_callable=AsyncMock,
            return_value=[monday_block, today_block],
        ),
        patch(
            "composites.find_missed_task_blocks",
            new_callable=AsyncMock,
            return_value=[monday_block, today_block],
        ) as mock_find,
        patch(
            "composites.reschedule_missed_event",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        result = await reschedule_missed_work(task_query="bcg", period="week")

    assert result["ok"] is True
    assert result["moved_count"] == 2
    mock_find.assert_awaited()
    assert mock_find.await_args.kwargs.get("period") == "week"


def main():
    test_resolve_snooze_default()
    test_resolve_snooze_tomorrow()
    test_resolve_snooze_explicit_time()
    test_fold_legacy_multiple_call()
    import asyncio

    asyncio.run(test_reschedule_missed_work_single_task())
    asyncio.run(test_reschedule_missed_work_all_today())
    asyncio.run(test_reschedule_missed_work_week_task())
    print("All reschedule_missed tests passed.")


if __name__ == "__main__":
    main()
