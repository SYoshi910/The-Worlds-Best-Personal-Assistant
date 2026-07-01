"""Unit tests for queries.get_schedule_for_window (no live API)."""

import asyncio
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIMEZONE
from queries import (
    _format_window,
    _full_week_window,
    _resolve_day,
    _window_bounds,
    get_schedule_for_window,
    parse_schedule_read_request,
)

TZ = ZoneInfo(TIMEZONE)
TARGET = date(2026, 6, 30)


def _iso(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=TZ).isoformat()


FAKE_EVENTS = [
    {
        "title": "Morning standup",
        "eventStart": _iso(2026, 6, 30, 9, 0),
        "eventEnd": _iso(2026, 6, 30, 9, 30),
    },
    {
        "title": "Lunch",
        "eventStart": _iso(2026, 6, 30, 12, 30),
        "eventEnd": _iso(2026, 6, 30, 13, 30),
    },
    {
        "title": "Evening study",
        "eventStart": _iso(2026, 6, 30, 19, 0),
        "eventEnd": _iso(2026, 6, 30, 21, 0),
    },
    {
        "title": "Late night wrap",
        "eventStart": _iso(2026, 6, 30, 23, 30),
        "eventEnd": _iso(2026, 7, 1, 0, 30),
    },
]


def test_resolve_day():
    now = datetime(2026, 6, 30, 10, 0, tzinfo=TZ)
    assert _resolve_day("today", now) == date(2026, 6, 30)
    assert _resolve_day("tomorrow", now) == date(2026, 7, 1)
    assert _resolve_day("yesterday", now) == date(2026, 6, 29)
    assert _resolve_day(None, now) == date(2026, 6, 30)


def test_window_bounds_all_day():
    start, end, label = _window_bounds(TARGET, None, TZ)
    assert label == "all day"
    assert start == datetime.combine(TARGET, time(0, 0), tzinfo=TZ)
    assert end.hour == 23 and end.minute == 59


def test_window_bounds_afternoon():
    start, end, label = _window_bounds(TARGET, "afternoon", TZ)
    assert label == "afternoon"
    assert start.time() == time(12, 0)
    assert end.hour == 16 and end.minute == 59 and end.second == 59


def test_window_bounds_night_wraps_midnight():
    start, end, label = _window_bounds(TARGET, "night", TZ)
    assert label == "night"
    assert start.date() == TARGET and start.time() == time(23, 0)
    assert end.date() == TARGET + timedelta(days=1)
    assert end.hour == 3 and end.minute == 59


def test_window_bounds_unknown_period_falls_back():
    _, _, label = _window_bounds(TARGET, "banana", TZ)
    assert label == "all day"


def test_format_window_empty():
    out = _format_window([], TARGET, "afternoon")
    assert "Tuesday Jun 30" in out
    assert "(afternoon)" in out
    assert "(nothing scheduled)" in out


def test_format_window_lists_events():
    hits = [
        (datetime(2026, 6, 30, 12, 30, tzinfo=TZ), datetime(2026, 6, 30, 13, 30, tzinfo=TZ), "Lunch"),
    ]
    out = _format_window(hits, TARGET, "afternoon")
    assert "Lunch" in out
    assert "12:30 PM" in out


async def test_get_schedule_for_window_afternoon():
    fixed_now = datetime(2026, 6, 30, 10, 0, tzinfo=TZ)

    with (
        patch("queries.get_all_events", new_callable=AsyncMock, return_value=FAKE_EVENTS),
        patch("queries.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = fixed_now
        mock_dt.combine = datetime.combine
        mock_dt.fromisoformat = datetime.fromisoformat
        out = await get_schedule_for_window("today", "afternoon")

    assert "Lunch" in out
    assert "Morning standup" not in out
    assert "Evening study" not in out
    assert "(afternoon)" in out


async def test_get_schedule_for_window_includes_non_task_events():
    """Lunch/commute blocks must appear — not filtered to Reclaim tasks only."""
    fixed_now = datetime(2026, 6, 30, 10, 0, tzinfo=TZ)

    with (
        patch("queries.get_all_events", new_callable=AsyncMock, return_value=FAKE_EVENTS),
        patch("queries.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = fixed_now
        mock_dt.combine = datetime.combine
        mock_dt.fromisoformat = datetime.fromisoformat
        out = await get_schedule_for_window("today", None)

    assert "Lunch" in out
    assert "Morning standup" in out
    assert "Evening study" in out


async def test_get_schedule_for_window_night_includes_late_block():
    fixed_now = datetime(2026, 6, 30, 10, 0, tzinfo=TZ)

    with (
        patch("queries.get_all_events", new_callable=AsyncMock, return_value=FAKE_EVENTS),
        patch("queries.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = fixed_now
        mock_dt.combine = datetime.combine
        mock_dt.fromisoformat = datetime.fromisoformat
        out = await get_schedule_for_window("today", "night")

    assert "Late night wrap" in out
    assert "(night)" in out


def test_full_week_window_monday_to_sunday():
    # Tuesday Jun 30 2026 -> week ends Sunday Jul 5
    now = datetime(2026, 6, 30, 15, 0, tzinfo=TZ)
    start, end, week_start, week_end = _full_week_window(now, TZ)
    assert start == now
    assert week_start == date(2026, 6, 29)
    assert week_end == date(2026, 7, 5)
    assert end.date() == date(2026, 7, 5)
    assert end.hour == 23 and end.minute == 59


async def test_get_schedule_for_window_full_week():
    fixed_now = datetime(2026, 6, 30, 10, 0, tzinfo=TZ)
    week_event = {
        "title": "Friday review",
        "eventStart": _iso(2026, 7, 3, 14, 0),
        "eventEnd": _iso(2026, 7, 3, 15, 0),
    }

    with (
        patch(
            "queries.get_all_events",
            new_callable=AsyncMock,
            return_value=FAKE_EVENTS + [week_event],
        ),
        patch("queries.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = fixed_now
        mock_dt.combine = datetime.combine
        mock_dt.fromisoformat = datetime.fromisoformat
        out = await get_schedule_for_window(full_week=True)

    assert "Jun 29" in out and "Jul 05" in out
    assert "Friday review" in out
    assert "Lunch" in out


def test_parse_schedule_read_request():
    assert parse_schedule_read_request("what are my tasks this week") == {
        "full_week": True,
    }
    assert parse_schedule_read_request("what does my afternoon look like") == {
        "day": "today",
        "period": "afternoon",
    }
    assert parse_schedule_read_request("undo") is None


def main():
    test_resolve_day()
    test_window_bounds_all_day()
    test_window_bounds_afternoon()
    test_window_bounds_night_wraps_midnight()
    test_window_bounds_unknown_period_falls_back()
    test_format_window_empty()
    test_format_window_lists_events()
    asyncio.run(test_get_schedule_for_window_afternoon())
    asyncio.run(test_get_schedule_for_window_includes_non_task_events())
    asyncio.run(test_get_schedule_for_window_night_includes_late_block())
    test_full_week_window_monday_to_sunday()
    asyncio.run(test_get_schedule_for_window_full_week())
    test_parse_schedule_read_request()
    print("All schedule window tests passed.")


if __name__ == "__main__":
    main()
