"""
Manual test: move past BCG prep Reclaim task block(s) to tomorrow via GCal patch.

Usage (from project root):
  python scripts/test_move_bcg.py              # dry-run (default)
  python scripts/test_move_bcg.py --apply      # actually patch GCal
  python scripts/test_move_bcg.py --event-id EVENT_ID --apply
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIMEZONE
from gcal import get_event, move_event
from reclaim import (
    get_all_events,
    is_past_block,
    is_task_assignment,
    _parse_event_time,
)
from zoneinfo import ZoneInfo

DEFAULT_BCG_EVENT_ID = "e9im6r31d5miqobjedkn6t1dehgn6qpqeoojkc9j68p30chh64t32"


def _safe_print(s: str) -> None:
    print(s.encode("ascii", errors="replace").decode("ascii"))


def _get_event_by_id(events: list[dict], event_id: str) -> dict | None:
    for event in events:
        if event.get("eventId") == event_id:
            return event
    return None


def _shift_to_tomorrow_same_clock(
    start: datetime, end: datetime
) -> tuple[datetime, datetime]:
    """Move an event interval to tomorrow at the same local clock times."""
    tz = start.tzinfo or ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    target_date = now.date() + timedelta(days=1)
    duration = end - start
    new_start = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        start.hour,
        start.minute,
        start.second,
        tzinfo=tz,
    )
    return new_start, new_start + duration


def _is_bcg_block(event: dict) -> bool:
    if not is_task_assignment(event):
        return False
    return "bcg" in (event.get("title") or "").lower()


def _find_bcg_targets(events: list[dict], event_id: str | None) -> list[dict]:
    now = datetime.now(ZoneInfo(TIMEZONE))
    if event_id:
        match = _get_event_by_id(events, event_id)
        return [match] if match else []
    return [e for e in events if _is_bcg_block(e) and is_past_block(e, now)]


async def main():
    parser = argparse.ArgumentParser(description="Test moving BCG block to tomorrow via GCal")
    parser.add_argument("--apply", action="store_true", help="Actually patch GCal")
    parser.add_argument("--event-id", default=None, help="Reclaim/GCal eventId")
    parser.add_argument("--all-past-bcg", action="store_true", help="Move all past BCG blocks")
    args = parser.parse_args()

    eid = args.event_id or DEFAULT_BCG_EVENT_ID
    print("[...] Fetching Reclaim events...")
    events = await get_all_events()
    print(f"   {len(events)} events total")

    targets = _find_bcg_targets(events, args.event_id)
    if not targets and args.event_id is None:
        print(f"   Auto-find missed — trying default eventId {eid}")
        targets = _find_bcg_targets(events, eid)

    if not targets:
        print("[FAIL] No BCG task block found. Pass --event-id explicitly.")
        sys.exit(1)

    if not args.all_past_bcg and len(targets) > 1:
        targets = [max(targets, key=lambda e: _parse_event_time(e["eventStart"]))]

    for event in targets:
        event_id = event["eventId"]
        start = _parse_event_time(event["eventStart"])
        end = _parse_event_time(event["eventEnd"])
        new_start, new_end = _shift_to_tomorrow_same_clock(start, end)
        task_id = (event.get("assist") or {}).get("taskId")

        print()
        print("-" * 60)
        _safe_print(f"Title:      {event.get('title')}")
        print(f"taskId:     {task_id}")
        print(f"eventId:    {event_id}")
        print(f"Current:    {start.isoformat()} -> {end.isoformat()}")
        print(f"New:        {new_start.isoformat()} -> {new_end.isoformat()}")
        print(f"lockState:  {(event.get('assist') or {}).get('lockState')}")

        try:
            gcal_before = await get_event(event_id)
            _safe_print(
                f"GCal OK:    {gcal_before.get('summary')} @ "
                f"{gcal_before['start'].get('dateTime')}"
            )
        except Exception as e:
            _safe_print(f"[FAIL] Cannot read GCal event {event_id}: {e}")
            continue

        if not args.apply:
            print("[DRY-RUN] Re-run with --apply to patch")
            continue

        try:
            updated, _snapshot = await move_event(event_id, new_start, new_end)
            _safe_print(f"[OK] Patched:  {updated.get('summary')}")
            print(f"   Start:    {updated['start'].get('dateTime')}")
            print(f"   End:      {updated['end'].get('dateTime')}")
        except Exception as e:
            print(f"[FAIL] GCal patch failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
