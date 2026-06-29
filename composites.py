"""Multi-step calendar operations built on gcal + reclaim primitives."""

from datetime import datetime, timedelta, timezone

from zoneinfo import ZoneInfo

import gcal
from cal_helper import get_task_by_query
from config import TIMEZONE
from inference import parse_to_iso
from reclaim import (
    _parse_event_time,
    find_missed_task_blocks,
    reschedule_task,
)


def _ok(summary: str, snapshots: list | None = None, **extra) -> dict:
    return {"ok": True, "summary": summary, "snapshots": snapshots or [], **extra}


def _fail(msg: str, **extra) -> dict:
    return {"ok": False, "failed": [msg], "snapshots": [], **extra}


def _compute_new_times(
    start: datetime,
    end: datetime,
    snooze_until: datetime | None,
) -> tuple[datetime, datetime]:
    duration = end - start
    if snooze_until is not None:
        if snooze_until.tzinfo is None:
            snooze_until = snooze_until.replace(tzinfo=ZoneInfo(TIMEZONE))
        return snooze_until, snooze_until + duration
    return gcal.shift_to_tomorrow_same_clock(start, end)


async def reschedule_missed_work(
    task_query: str,
    snooze_until_natural: str | None = None,
) -> dict:
    """Move today's missed task blocks to tomorrow or a parsed snooze time."""
    task = await get_task_by_query(task_query)
    if not task:
        return _fail(f"Could not find a task matching '{task_query}'")

    blocks = await find_missed_task_blocks(task["id"])
    if not blocks:
        return _fail(f"No missed blocks today for '{task.get('title')}'")

    base_time = datetime.now(ZoneInfo(TIMEZONE))
    snooze_dt = None
    if snooze_until_natural:
        iso = parse_to_iso(snooze_until_natural, base_time)
        if iso:
            snooze_dt = datetime.fromisoformat(iso)

    snapshots = []
    moved = []
    errors = []

    for event in blocks:
        event_id = event["eventId"]
        start = _parse_event_time(event["eventStart"])
        end = _parse_event_time(event["eventEnd"])
        new_start, new_end = _compute_new_times(start, end, snooze_dt)
        try:
            _updated, snap = await gcal.move_event(event_id, new_start, new_end)
            snapshots.append(snap)
            moved.append(event.get("title", event_id))
        except Exception as e:
            errors.append(f"Failed to move {event.get('title')}: {e}")

    if not moved:
        return _fail("; ".join(errors) if errors else "No blocks moved")

    summary = (
        f"Moved {len(moved)} block(s) for '{task.get('title')}' "
        f"({', '.join(moved)})"
    )
    result = _ok(summary, snapshots=snapshots, moved_count=len(moved))
    if errors:
        result["failed"] = errors
    return result


async def switch_active_task(
    new_task_query: str,
    current_task: dict | None = None,
) -> dict:
    """Bump the current block and schedule the new task now."""
    if not current_task or not current_task.get("event_id"):
        return _fail("No active calendar block to switch away from")

    new_task = await get_task_by_query(new_task_query)
    if not new_task:
        return _fail(f"Could not find a task matching '{new_task_query}'")

    snapshots = []
    now = datetime.now(ZoneInfo(TIMEZONE))

    event_id = current_task["event_id"]
    start = _parse_event_time(current_task["start_time"])
    end = _parse_event_time(current_task.get("end_time", current_task["start_time"]))
    duration = end - start
    bump_start = now + timedelta(minutes=30)
    bump_end = bump_start + duration

    try:
        _updated, snap = await gcal.move_event(event_id, bump_start, bump_end)
        snapshots.append(snap)
    except Exception as e:
        return _fail(f"Could not move current block: {e}")

    old_snooze = new_task.get("snoozeUntil")
    snapshots.append(
        {
            "type": "reschedule_task",
            "task_id": new_task["id"],
            "snoozeUntil": old_snooze,
        }
    )

    ok = await reschedule_task(new_task["id"], snooze_until=now.astimezone(timezone.utc).isoformat())
    if not ok:
        try:
            start = _parse_event_time(snap["start"])
            end = _parse_event_time(snap["end"])
            await gcal.move_event(event_id, start, end)
        except Exception as rollback_err:
            return _fail(
                f"Moved current block but could not schedule '{new_task.get('title')}' now "
                f"(rollback failed: {rollback_err})"
            )
        return _fail(f"Moved current block but could not schedule '{new_task.get('title')}' now")

    return _ok(
        f"Switched to '{new_task.get('title')}' — current block pushed to "
        f"{bump_start.strftime('%I:%M %p')}",
        snapshots=snapshots,
    )


async def extend_current_block(
    additional_minutes: int,
    task_query: str | None = None,
    current_task: dict | None = None,
) -> dict:
    """Create a GCal buffer for a short extension during an active block."""
    ctx = current_task or {}
    now = datetime.now(ZoneInfo(TIMEZONE))
    snapshots = []

    ongoing = bool(ctx.get("is_ongoing"))
    same_task = True
    if task_query and ctx.get("title"):
        same_task = task_query.lower() in (ctx.get("title") or "").lower()

    if ongoing and not same_task and ctx.get("event_id"):
        return await switch_active_task(task_query or "", current_task)

    end_time = now + timedelta(minutes=additional_minutes)
    name = ctx.get("title") or task_query or "Focus buffer"
    if task_query and not ongoing:
        task = await get_task_by_query(task_query)
        if task:
            name = task.get("title", name)
            old_snooze = task.get("snoozeUntil")
            snapshots.append(
                {"type": "reschedule_task", "task_id": task["id"], "snoozeUntil": old_snooze}
            )
            await reschedule_task(
                task["id"], snooze_until=now.astimezone(timezone.utc).isoformat()
            )

    try:
        _result, snap = await gcal.create_buffer_event(
            name,
            now.isoformat(),
            end_time.isoformat(),
        )
        snapshots.append(snap)
        return _ok(
            f"Added {additional_minutes}-min buffer until {end_time.strftime('%I:%M %p')}",
            snapshots=snapshots,
        )
    except Exception as e:
        return _fail(f"Could not create buffer: {e}")


async def route_ping_calls(
    intent: str,
    message: str,
    current_task: dict | None,
) -> list[dict] | None:
    """Build tool calls for extend/switch intents during an active ping block."""
    from intent import extract_extend_task_call, extract_minutes, extract_switch_task_query
    from reclaim import refresh_ongoing_state

    named_extend = extract_extend_task_call(message)
    if named_extend:
        return named_extend

    if not current_task or not current_task.get("title"):
        return None

    current_task = refresh_ongoing_state(current_task)

    if intent == "extend_time":
        minutes = extract_minutes(message) or 20
        if current_task.get("is_ongoing") and minutes >= 30:
            return [
                {
                    "function": "extend_task_instance",
                    "params": {
                        "task_query": current_task.get("title"),
                        "additional_minutes": minutes,
                    },
                }
            ]
        return [
            {
                "function": "extend_current_block",
                "params": {
                    "additional_minutes": minutes,
                    "task_query": current_task.get("title"),
                },
            }
        ]

    if intent == "switch_task":
        new_q = extract_switch_task_query(message) or message
        return [
            {
                "function": "switch_active_task",
                "params": {"new_task_query": new_q},
            }
        ]

    return None
