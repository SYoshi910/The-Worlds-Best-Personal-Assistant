"""
Print a calendar event from earlier today with full Reclaim + GCal API payloads.

Shows every field returned by the APIs and which ones this codebase can edit.

Usage (from project root):
  python scripts/inspect_past_event.py
  python scripts/inspect_past_event.py --event-id EVENT_ID
  python scripts/inspect_past_event.py --all-today   # list all ended-today events
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIMEZONE
from gcal import get_event
from reclaim import (
    _parse_event_time,
    event_on_local_day,
    get_all_events,
    is_past_block,
    is_task_assignment,
)
from zoneinfo import ZoneInfo


def _safe_print(s: str) -> None:
    print(s.encode("ascii", errors="replace").decode("ascii"))


def _fmt_local(iso: str) -> str:
    dt = _parse_event_time(iso)
    return dt.astimezone(ZoneInfo(TIMEZONE)).strftime("%A %b %d %I:%M %p %Z")


def _ended_today_events(events: list[dict], now: datetime) -> list[dict]:
    today = now.astimezone(ZoneInfo(TIMEZONE)).date()
    ended = []
    for event in events:
        if not event_on_local_day(event, today):
            continue
        if not is_past_block(event, now):
            continue
        ended.append(event)
    return sorted(ended, key=lambda e: _parse_event_time(e["eventStart"]), reverse=True)


def _editable_cheatsheet(reclaim_event: dict, gcal_event: dict | None) -> str:
    assist = reclaim_event.get("assist") or {}
    lines = [
        "── Editable via this codebase ──",
        "",
        "GCal events().patch (gcal.move_event / create_buffer_event):",
        "  summary     ← create_event(name=...) / event title",
        "  start.dateTime, start.timeZone",
        "  end.dateTime, end.timeZone",
        "",
        "GCal events().insert (gcal.create_buffer_event):",
        "  summary, start, end  (commute/lunch/meeting blocks only)",
        "",
        "GCal events().delete (gcal.delete_event):",
        "  event id only",
        "",
        "Reclaim POST /planner/log-work/task/{id} (reclaim.log_work):",
        "  minutes, end (query params; end UTC Zulu)",
        "",
        "Reclaim PATCH /tasks/{id} (reclaim.update_task, move_due_date, etc.):",
        "  due, priority, eventCategory, timeChunksRequired, snoozeUntil, status, ...",
        "",
        "Reclaim planner snooze (reclaim.reschedule_task / snooze_task):",
        "  snoozeUntil on task_id",
        "",
        "This event:",
        f"  eventId (GCal id):  {reclaim_event.get('eventId')}",
        f"  title:              {reclaim_event.get('title')!r}",
        f"  eventStart:         {reclaim_event.get('eventStart')}",
        f"  eventEnd:           {reclaim_event.get('eventEnd')}",
        f"  reclaimEventType:   {reclaim_event.get('reclaimEventType')}",
        f"  is_task_assignment: {is_task_assignment(reclaim_event)}",
    ]
    if assist:
        lines.append("  assist.taskId:      " + str(assist.get("taskId")))
        lines.append("  assist.taskIndex:   " + str(assist.get("taskIndex")))
        lines.append("  assist.lockState:   " + str(assist.get("lockState")))
    if gcal_event:
        lines.extend(
            [
                "",
                "GCal mirror:",
                f"  id:                 {gcal_event.get('id')}",
                f"  status:             {gcal_event.get('status')}",
                f"  created:            {gcal_event.get('created')}",
                f"  updated:            {gcal_event.get('updated')}",
            ]
        )
    return "\n".join(lines)


def _print_event_block(reclaim_event: dict, gcal_event: dict | None) -> None:
    print()
    print("=" * 72)
    _safe_print(f"  {reclaim_event.get('title')}  ({_fmt_local(reclaim_event['eventStart'])})")
    print("=" * 72)
    print(_editable_cheatsheet(reclaim_event, gcal_event))
    print()
    print("── Reclaim GET /events (full object) ──")
    print(json.dumps(reclaim_event, indent=2, default=str))
    if gcal_event is not None:
        print()
        print("── GCal events().get (full object) ──")
        print(json.dumps(gcal_event, indent=2, default=str))
    else:
        print()
        print("── GCal events().get ──")
        print("  (not fetched or unavailable)")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a past-today event with full API payloads"
    )
    parser.add_argument("--event-id", help="Specific Reclaim/GCal eventId")
    parser.add_argument(
        "--all-today",
        action="store_true",
        help="Print every event that already ended today",
    )
    args = parser.parse_args()

    now = datetime.now(ZoneInfo(TIMEZONE))
    print(f"Now: {now.strftime('%A %b %d %I:%M %p %Z')}")
    print("Fetching Reclaim events...")
    events = await get_all_events(force_refresh=True)
    print(f"  {len(events)} events total")

    if args.event_id:
        targets = [e for e in events if e.get("eventId") == args.event_id]
        if not targets:
            print(f"[FAIL] No event with eventId={args.event_id!r}")
            return 1
    else:
        targets = _ended_today_events(events, now)
        if not targets:
            print("[FAIL] No events that already ended today.")
            print("       Pass --event-id or try again later in the day.")
            return 1
        if not args.all_today:
            targets = [targets[0]]
            print(f"  Using most recent ended-today event ({len(_ended_today_events(events, now))} total today)")

    for reclaim_event in targets:
        event_id = reclaim_event["eventId"]
        gcal_event = None
        try:
            gcal_event = await get_event(event_id)
        except Exception as e:
            _safe_print(f"[WARN] GCal fetch failed for {event_id}: {e}")
        _print_event_block(reclaim_event, gcal_event)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
