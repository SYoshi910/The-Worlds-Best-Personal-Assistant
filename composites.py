"""Multi-step calendar operations built on gcal + reclaim primitives.

Ping-mismatch rule (spec 4): when the user is NOT working on the pinged Reclaim
task, snooze that task 1 hour (planner ``FROM_NOW_1H``) and then schedule what
they *are* working on via ``plan_work`` (schedule-now). We never GCal-bump the
pinged block — that auto-locks and fights Reclaim.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import gcal
import reclaim
import schedule_cache
from cal_helper import get_task_by_query
from config import TIMEZONE
from config import now_local as _now
from inference import parse_to_iso
from queries import format_clock
from reclaim import (
    _parse_event_time,
    find_missed_blocks_in_window,
    find_missed_task_blocks,
    get_all_events,
    plan_work,
    resolve_snooze_target,
    reschedule_missed_event,
    snooze_task,
    task_ids_with_missed_blocks,
    task_ids_with_missed_blocks_today,
)


def _ok(summary: str, snapshots: list | None = None, **extra) -> dict:
    return {"ok": True, "summary": summary, "snapshots": snapshots or [], **extra}


def _fail(msg: str, **extra) -> dict:
    return {"ok": False, "failed": [msg], "snapshots": [], **extra}


def _duration_minutes(
    now: datetime,
    work_duration_minutes: int | None,
    work_until_natural: str | None,
) -> int | None:
    """Resolve how long the user will work: explicit minutes or an 'until' time."""
    if work_duration_minutes:
        try:
            return int(work_duration_minutes)
        except (TypeError, ValueError):
            pass
    if work_until_natural:
        iso = parse_to_iso(work_until_natural, now)
        if iso:
            end = datetime.fromisoformat(iso)
            if end.tzinfo is None:
                end = end.replace(tzinfo=ZoneInfo(TIMEZONE))
            mins = int((end - now).total_seconds() // 60)
            if mins > 0:
                return mins
    return None


async def _snooze_pinged(current_task: dict | None, snapshots: list) -> bool:
    """Snooze the pinged task 1 hour off the current ping; record undo snapshot.

    Returns True if a pinged task was found and snoozed (or there was nothing to
    snooze — a no-op is not a failure).
    """
    task_id = (current_task or {}).get("task_id")
    if not task_id:
        return True

    existing = await reclaim.get_task(task_id)
    if existing is not None:
        snapshots.append(
            {
                "type": "reschedule_task",
                "task_id": task_id,
                "snoozeUntil": existing.get("snoozeUntil"),
            }
        )
    return await snooze_task(task_id, hours=1)


def _compute_new_times(
    start: datetime,
    end: datetime,
    target: datetime | None,
) -> tuple[datetime, datetime]:
    """Place a block at ``target`` (default now + 1 hour, spec 9b), keeping length."""
    duration = end - start
    if target is None:
        target = _now() + timedelta(hours=1)
    if target.tzinfo is None:
        target = target.replace(tzinfo=ZoneInfo(TIMEZONE))
    return target, target + duration


def _dedupe_events(blocks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for event in blocks:
        eid = event.get("eventId")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(event)
    return out


async def _blocks_for_scope(
    *,
    task_query: str | None,
    task_queries: list[str] | None,
    all_missed_today: bool,
    overlap_since_minutes: int | None,
    period: str = "today",
) -> tuple[list[dict], str | None]:
    """Resolve missed blocks; return (blocks, error_message)."""
    events = await get_all_events()
    now = _now()
    scope = "week" if period == "week" else "today"

    if overlap_since_minutes:
        win_start = now - timedelta(minutes=int(overlap_since_minutes))
        return _dedupe_events(
            await find_missed_blocks_in_window(win_start, now, events=events)
        ), None

    if all_missed_today:
        blocks: list[dict] = []
        for tid in await task_ids_with_missed_blocks(events=events, period=scope):
            blocks.extend(
                await find_missed_task_blocks(tid, events=events, period=scope)
            )
        return _dedupe_events(blocks), None

    queries: list[str] = []
    if task_query:
        queries.append(task_query)
    if task_queries:
        queries.extend(task_queries)
    queries = list(dict.fromkeys(q for q in queries if q))
    if not queries:
        return [], "No task specified for missed-work reschedule"

    blocks = []
    for query in queries:
        task = await get_task_by_query(query)
        if not task:
            return [], f"Could not find a task matching '{query}'"
        blocks.extend(
            await find_missed_task_blocks(task["id"], events=events, period=scope)
        )
    return _dedupe_events(blocks), None


async def reschedule_missed_work(
    task_query: str | None = None,
    task_queries: list[str] | None = None,
    all_missed_today: bool = False,
    snooze_until_natural: str | None = None,
    overlap_since_minutes: int | None = None,
    period: str = "today",
    current_task: dict | None = None,
) -> dict:
    """Reschedule missed task block(s) via the Reclaim planner."""
    blocks, err = await _blocks_for_scope(
        task_query=task_query,
        task_queries=task_queries,
        all_missed_today=all_missed_today,
        overlap_since_minutes=overlap_since_minutes,
        period=period,
    )
    if err:
        return _fail(err)
    if not blocks:
        return _fail("No missed blocks to reschedule")

    now = _now()
    target, snooze_option = resolve_snooze_target(snooze_until_natural, now)
    snapshots: list[dict] = []
    moved: list[str] = []
    errors: list[str] = []

    for event in blocks:
        snapshots.append(
            {
                "type": "move_task_event",
                "event_id": event["eventId"],
                "calendar_id": event.get("calendarId"),
                "start": event.get("eventStart"),
                "end": event.get("eventEnd"),
            }
        )
        try:
            if await reschedule_missed_event(event, target, snooze_option):
                moved.append(event.get("title") or event["eventId"])
            else:
                errors.append(f"Failed to reschedule {event.get('title', 'block')}")
        except Exception as e:
            errors.append(f"{event.get('title', 'block')}: {e}")

    if not moved:
        return _fail("; ".join(errors) if errors else "No blocks moved")

    when = (
        target.strftime("%a %I:%M %p")
        if target
        else (snooze_until_natural or "+1 hour")
    )
    span = "this week" if period == "week" else "today"
    n = len(moved)
    summary = (
        f"I've rescheduled {n} missed block{'s' if n != 1 else ''} "
        f"from {span} to {when}"
    )
    result = _ok(summary, snapshots=snapshots, moved_count=n)
    if errors:
        result["failed"] = errors
    return result


async def switch_active_task(
    new_task_query: str,
    work_duration_minutes: int | None = None,
    work_until_natural: str | None = None,
    current_task: dict | None = None,
) -> dict:
    """Snooze the pinged task 1h and schedule the task the user is actually doing.

    Step 1: snooze the pinged Reclaim task off the ping (``FROM_NOW_1H``). Step 2:
    ``plan_work`` the new task starting now for the given duration (spec 4).
    """
    new_task = await get_task_by_query(new_task_query)
    if not new_task:
        return _fail(f"Could not find a task matching '{new_task_query}'")

    now = _now()
    duration = _duration_minutes(now, work_duration_minutes, work_until_natural)
    snapshots: list = []

    if not await _snooze_pinged(current_task, snapshots):
        return _fail("Could not snooze the current task off the ping")

    ok = await plan_work(new_task["id"], now.isoformat(), duration)
    if not ok:
        return _fail(f"Snoozed the ping but could not schedule '{new_task.get('title')}' now")

    tail = f" for {duration} min" if duration else ""
    return _ok(
        f"I've switched you to '{new_task.get('title')}' starting now{tail}",
        snapshots=snapshots,
    )


async def resume_previous_task(
    work_duration_minutes: int | None = None,
    work_until_natural: str | None = None,
    previous_task_query: str | None = None,
    current_task: dict | None = None,
) -> dict:
    """User is still on the previous task when a new ping fired (spec 5).

    Snooze the freshly pinged task 1h, then schedule the previous task (from the
    warm schedule cache, or an explicit query) starting now.
    """
    now = _now()

    previous_task = None
    if previous_task_query:
        previous_task = await get_task_by_query(previous_task_query)
    if previous_task is None:
        _current_block, previous_block = schedule_cache.current_and_previous(
            now.astimezone(ZoneInfo("UTC"))
        )
        prev_id = (previous_block or {}).get("task_id")
        if prev_id:
            previous_task = await reclaim.get_task(prev_id)

    if not previous_task:
        return _fail("I couldn't find the previous task you were working on")

    duration = _duration_minutes(now, work_duration_minutes, work_until_natural)
    snapshots: list = []

    if not await _snooze_pinged(current_task, snapshots):
        return _fail("Could not snooze the current ping")

    ok = await plan_work(previous_task["id"], now.isoformat(), duration)
    if not ok:
        return _fail(
            f"Snoozed the ping but could not resume '{previous_task.get('title')}'"
        )

    tail = f" for {duration} min" if duration else ""
    return _ok(
        f"I've kept you on '{previous_task.get('title')}' starting now{tail}",
        snapshots=snapshots,
    )


async def extend_current_gcal_block(
    additional_minutes: int,
    task_query: str | None = None,
    current_task: dict | None = None,
) -> dict:
    """Create a GCal buffer to extend a non-task block (lunch/commute) during a ping.

    Task blocks use ``extend_task_instance``/``extend_task_total``; this is only
    for fixed-time Google Calendar blocks, so there is no Reclaim snooze or switch
    delegation here.
    """
    ctx = current_task or {}
    now = _now()
    name = ctx.get("title") or task_query or "Focus buffer"
    end_time = now + timedelta(minutes=additional_minutes)

    try:
        _result, snap = await gcal.create_buffer_event(
            name,
            now.isoformat(),
            end_time.isoformat(),
        )
        return _ok(
            f"I've added a {additional_minutes}-min buffer until "
            f"{format_clock(end_time)}",
            snapshots=[snap],
        )
    except Exception as e:
        return _fail(f"Could not create buffer: {e}")
