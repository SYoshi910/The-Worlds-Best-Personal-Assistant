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
from inference import parse_upcoming_time
from reclaim import (
    _parse_event_time,
    find_missed_task_blocks,
    plan_work,
    snooze_task,
)


def _ok(summary: str, snapshots: list | None = None, **extra) -> dict:
    return {"ok": True, "summary": summary, "snapshots": snapshots or [], **extra}


def _fail(msg: str, **extra) -> dict:
    return {"ok": False, "failed": [msg], "snapshots": [], **extra}


def _now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


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
        iso = parse_upcoming_time(work_until_natural, now)
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


async def reschedule_missed_work(
    task_query: str,
    snooze_until_natural: str | None = None,
    current_task: dict | None = None,
) -> dict:
    """Move today's missed task blocks to a target time (default now + 1 hour)."""
    task = await get_task_by_query(task_query)
    if not task:
        return _fail(f"Could not find a task matching '{task_query}'")

    blocks = await find_missed_task_blocks(task["id"])
    if not blocks:
        return _fail(f"No missed blocks today for '{task.get('title')}'")

    now = _now()
    target = None
    if snooze_until_natural:
        iso = parse_upcoming_time(snooze_until_natural, now)
        if iso:
            target = datetime.fromisoformat(iso)

    snapshots = []
    moved = []
    errors = []

    for event in blocks:
        event_id = event["eventId"]
        start = _parse_event_time(event["eventStart"])
        end = _parse_event_time(event["eventEnd"])
        new_start, new_end = _compute_new_times(start, end, target)
        try:
            _updated, snap = await gcal.move_event(event_id, new_start, new_end)
            snapshots.append(snap)
            moved.append(event.get("title", event_id))
        except Exception as e:
            errors.append(f"Failed to move {event.get('title')}: {e}")

    if not moved:
        return _fail("; ".join(errors) if errors else "No blocks moved")

    when = "+1 hour" if target is None else target.strftime("%a %I:%M %p")
    summary = (
        f"I've moved {len(moved)} missed block(s) for '{task.get('title')}' to {when}"
    )
    result = _ok(summary, snapshots=snapshots, moved_count=len(moved))
    if errors:
        result["failed"] = errors
    return result


async def reschedule_multiple_missed_work(
    task_queries: list[str] | None = None,
    snooze_until_natural: str | None = None,
    all_missed_today: bool = False,
    current_task: dict | None = None,
) -> dict:
    """Reschedule missed blocks for multiple tasks (one ``reschedule_missed_work`` each).

  ``all_missed_today=True``: discover every task with missed blocks today (e.g.
  "I didn't work on anything today"). The LLM may also emit several
  ``reschedule_missed_work`` calls in one turn instead of using this composite.
    """
    queries_to_run: list[str] = list(task_queries or [])

    if all_missed_today:
        task_ids = await reclaim.task_ids_with_missed_blocks_today()
        for tid in task_ids:
            task = await reclaim.get_task(tid)
            if task and task.get("title"):
                title = task["title"]
                if title not in queries_to_run:
                    queries_to_run.append(title)

    if not queries_to_run:
        return _fail("No missed tasks found to reschedule")

    summaries = []
    snapshots = []
    failed = []
    moved_total = 0

    for query in queries_to_run:
        result = await reschedule_missed_work(
            query,
            snooze_until_natural=snooze_until_natural,
            current_task=current_task,
        )
        if result.get("ok"):
            summaries.append(result.get("summary", query))
            snapshots.extend(result.get("snapshots", []))
            moved_total += result.get("moved_count", 0)
        else:
            failed.extend(result.get("failed", [f"Failed for '{query}'"]))

    if not summaries:
        return _fail("; ".join(failed) if failed else "No blocks moved")

    summary = f"Rescheduled missed work for {len(summaries)} task(s): " + "; ".join(
        summaries
    )
    out = _ok(summary, snapshots=snapshots, moved_count=moved_total)
    if failed:
        out["failed"] = failed
    return out


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
            f"{end_time.strftime('%I:%M %p').lstrip('0')}",
            snapshots=[snap],
        )
    except Exception as e:
        return _fail(f"Could not create buffer: {e}")
