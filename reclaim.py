"""Reclaim API client, caching, and task/event helpers."""

import time

import httpx
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from config import RECLAIM_API_KEY, CALENDAR_ID, TIMEZONE
from duration_parser import minutes_to_chunks
import gcal

BASE_URL = "https://api.app.reclaim.ai/api"
HEADERS = {
    "Authorization": f"Bearer {RECLAIM_API_KEY}",
    "Content-Type": "application/json",
}
VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}
VALID_EVENT_CATEGORIES = {"WORK", "PERSONAL"}


def normalize_event_category(value: str) -> str:
    """Normalize to WORK or PERSONAL; default to WORK on invalid input."""
    cat = str(value).strip().upper()
    if cat not in VALID_EVENT_CATEGORIES:
        print(f"⚠️ Invalid event_category '{value}', defaulting to WORK")
        return "WORK"
    return cat

CACHE_TTL_SECONDS = 8

_client: httpx.AsyncClient | None = None
_tasks_cache: list | None = None
_tasks_cache_at: float = 0
_tasks_cache_instances: list | None = None
_tasks_cache_instances_at: float = 0
_events_cache: list | None = None
_events_cache_at: float = 0


def invalidate_reclaim_cache() -> None:
    """Clear in-memory task and event caches after writes."""
    global _tasks_cache, _events_cache, _tasks_cache_instances
    _tasks_cache = None
    _tasks_cache_instances = None
    _events_cache = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(headers=HEADERS, timeout=30.0)
    return _client


def _parse_event_time(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def _cache_valid(cache_at: float) -> bool:
    return cache_at > 0 and (time.monotonic() - cache_at) < CACHE_TTL_SECONDS


# ─── GETTERS ────────────────────────────────────────────────────────────────


async def get_task(task_id: int | str):
    """Fetch a single task by id."""
    url = f"{BASE_URL}/tasks/{task_id}"
    response = await _get_client().get(url)
    if response.status_code == 200:
        return response.json()
    print(f"❌ Error fetching task {task_id}: {response.status_code} - {response.text}")
    return None


async def get_all_tasks(force_refresh: bool = False, instances: bool = False):
    """Fetch all tasks, optionally with instance metadata."""
    global _tasks_cache, _tasks_cache_at, _tasks_cache_instances, _tasks_cache_instances_at

    if instances:
        if (
            not force_refresh
            and _tasks_cache_instances is not None
            and _cache_valid(_tasks_cache_instances_at)
        ):
            return _tasks_cache_instances
    elif not force_refresh and _tasks_cache is not None and _cache_valid(_tasks_cache_at):
        return _tasks_cache

    url = f"{BASE_URL}/tasks"
    params = {"instances": "true"} if instances else None
    response = await _get_client().get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if instances:
            _tasks_cache_instances = data
            _tasks_cache_instances_at = time.monotonic()
            return _tasks_cache_instances
        _tasks_cache = data
        _tasks_cache_at = time.monotonic()
        return _tasks_cache
    print(f"❌ Error fetching tasks: {response.status_code} - {response.text}")
    return []


async def get_active_tasks(force_refresh: bool = False):
    """Return IN_PROGRESS and SCHEDULED tasks."""
    return [
        t
        for t in await get_all_tasks(force_refresh=force_refresh)
        if t["status"] in ("IN_PROGRESS", "SCHEDULED")
    ]


async def get_all_events(force_refresh: bool = False):
    """Fetch all Reclaim calendar events."""
    global _events_cache, _events_cache_at
    if not force_refresh and _events_cache is not None and _cache_valid(_events_cache_at):
        return _events_cache

    url = f"{BASE_URL}/events"
    response = await _get_client().get(url)
    if response.status_code == 200:
        _events_cache = response.json()
        _events_cache_at = time.monotonic()
        return _events_cache
    print(f"❌ Error fetching events: {response.status_code} - {response.text}")
    return []


async def get_active_events():
    """Return events that are ongoing, upcoming, or recently started."""
    now = datetime.now(timezone.utc)
    grace = timedelta(minutes=15)
    events = []
    for event in await get_all_events():
        start = _parse_event_time(event["eventStart"])
        end = _parse_event_time(event.get("eventEnd", event["eventStart"]))
        if start > now or start <= now < end or (now - start) < grace:
            events.append(event)
    return events


async def get_next_event(exclude_event_id: str | None = None):
    """Return the soonest active event, optionally excluding one id."""
    events = await get_active_events()
    if exclude_event_id:
        events = [e for e in events if e.get("eventId") != exclude_event_id]
    if not events:
        return None
    return min(events, key=lambda e: _parse_event_time(e["eventStart"]))


# ─── EVENT HELPERS ──────────────────────────────────────────────────────────


def is_task_assignment(event: dict) -> bool:
    assist = event.get("assist") or {}
    return (
        event.get("reclaimEventType") == "TASK_ASSIGNMENT"
        and assist.get("task") is True
    )


def is_past_block(event: dict, now: datetime | None = None) -> bool:
    if now is None:
        now = datetime.now(timezone.utc)
    assist = event.get("assist") or {}
    if assist.get("lockState") == "IN_THE_PAST":
        return True
    end = _parse_event_time(event.get("eventEnd", event["eventStart"]))
    return end < now


def is_ongoing_block(event: dict, now: datetime | None = None) -> bool:
    if now is None:
        now = datetime.now(timezone.utc)
    start = _parse_event_time(event["eventStart"])
    end = _parse_event_time(event.get("eventEnd", event["eventStart"]))
    return start <= now < end


def event_on_local_day(event: dict, day: date) -> bool:
    start = _parse_event_time(event["eventStart"])
    local = start.astimezone(ZoneInfo(TIMEZONE))
    return local.date() == day


def _is_missed_block_on_day(event: dict, day: date, now: datetime) -> bool:
    """Task-assignment block on ``day`` that is already past."""
    if not is_task_assignment(event):
        return False
    if not event_on_local_day(event, day):
        return False
    return is_past_block(event, now)


def _is_missed_block_today(event: dict, day: date, now: datetime) -> bool:
    return _is_missed_block_on_day(event, day, now)


def _week_range(now: datetime | None = None) -> tuple[date, date]:
    """Local Mon–today inclusive (current calendar week so far)."""
    if now is None:
        now = datetime.now(ZoneInfo(TIMEZONE))
    today = now.astimezone(ZoneInfo(TIMEZONE)).date()
    week_start = today - timedelta(days=today.weekday())
    return week_start, today


def _is_missed_block_in_period(
    event: dict, period: str, now: datetime, *, day: date | None = None
) -> bool:
    if period == "week":
        start_day, end_day = _week_range(now)
        if not is_task_assignment(event) or not is_past_block(event, now):
            return False
        local_day = _parse_event_time(event["eventStart"]).astimezone(ZoneInfo(TIMEZONE)).date()
        return start_day <= local_day <= end_day
    if day is None:
        day = datetime.now(ZoneInfo(TIMEZONE)).date()
    return _is_missed_block_on_day(event, day, now)


async def find_missed_task_blocks(
    task_id: int,
    day: date | None = None,
    events: list[dict] | None = None,
    *,
    period: str = "today",
) -> list[dict]:
    if day is None:
        day = datetime.now(ZoneInfo(TIMEZONE)).date()
    now = datetime.now(timezone.utc)
    if events is None:
        events = await get_all_events()
    blocks = []
    for event in events:
        assist = event.get("assist") or {}
        if assist.get("taskId") != task_id:
            continue
        if not _is_missed_block_in_period(event, period, now, day=day):
            continue
        blocks.append(event)
    return sorted(blocks, key=lambda e: _parse_event_time(e["eventStart"]))


async def task_ids_with_missed_blocks(
    events: list[dict] | None = None,
    *,
    period: str = "today",
) -> list[int]:
    """Distinct task ids with missed blocks in ``period`` (today or week)."""
    day = datetime.now(ZoneInfo(TIMEZONE)).date()
    now = datetime.now(timezone.utc)
    if events is None:
        events = await get_all_events()
    ids: set[int] = set()
    for event in events:
        assist = event.get("assist") or {}
        task_id = assist.get("taskId")
        if not task_id:
            continue
        if not _is_missed_block_in_period(event, period, now, day=day):
            continue
        ids.add(task_id)
    return sorted(ids)


async def task_ids_with_missed_blocks_today(
    events: list[dict] | None = None,
) -> list[int]:
    return await task_ids_with_missed_blocks(events, period="today")


def _overlaps(
    block_start: datetime, block_end: datetime, win_start: datetime, win_end: datetime
) -> bool:
    return block_start < win_end and block_end > win_start


async def find_missed_blocks_in_window(
    window_start: datetime,
    window_end: datetime,
    events: list[dict] | None = None,
) -> list[dict]:
    """Missed task blocks today that overlap ``[window_start, window_end)``."""
    day = datetime.now(ZoneInfo(TIMEZONE)).date()
    now = datetime.now(timezone.utc)
    if events is None:
        events = await get_all_events()
    hits: list[dict] = []
    for event in events:
        if not _is_missed_block_in_period(event, "today", now, day=day):
            continue
        start = _parse_event_time(event["eventStart"])
        end = _parse_event_time(event.get("eventEnd", event["eventStart"]))
        if _overlaps(start, end, window_start, window_end):
            hits.append(event)
    return sorted(hits, key=lambda e: _parse_event_time(e["eventStart"]))


def resolve_snooze_target(
    snooze_until_natural: str | None, now: datetime
) -> tuple[datetime | None, str | None]:
    """Return ``(move_target, snooze_option)`` — use move when natural parses to a time."""
    from inference import parse_to_iso

    if not snooze_until_natural:
        return None, "FROM_NOW_1H"

    text = snooze_until_natural.strip().lower()
    if text in ("tomorrow", "tomorrow morning", "tomorrow afternoon", "tomorrow evening"):
        return None, "TOMORROW"
    if "next week" in text:
        return None, "NEXT_WEEK"

    if text.startswith("in "):
        from duration_parser import parse_duration_to_minutes

        mins = parse_duration_to_minutes(text)
        if mins is not None:
            if mins <= 15:
                return None, "FROM_NOW_15M"
            if mins <= 30:
                return None, "FROM_NOW_30M"
            if mins <= 60:
                return None, "FROM_NOW_1H"
            if mins <= 120:
                return None, "FROM_NOW_2H"
            return None, "FROM_NOW_4H"

    iso = parse_to_iso(snooze_until_natural, now)
    if iso:
        target = datetime.fromisoformat(iso)
        if target.tzinfo is None:
            target = target.replace(tzinfo=ZoneInfo(TIMEZONE))
        return target, None

    return None, "FROM_NOW_1H"


def enrich_event_context(event: dict, now: datetime | None = None) -> dict:
    if now is None:
        now = datetime.now(timezone.utc)
    assist = event.get("assist") or {}
    return {
        "title": event.get("title"),
        "start_time": event.get("eventStart"),
        "end_time": event.get("eventEnd"),
        "task_id": assist.get("taskId"),
        "event_id": event.get("eventId"),
        "task_index": assist.get("taskIndex"),
        "is_ongoing": is_ongoing_block(event, now),
    }


# ─── ACTIONS ────────────────────────────────────────────────────────────────


async def complete_task(task_id: int):
    task = await get_task(task_id)
    if not task:
        print(f"❌ Could not find task {task_id}")
        return False
    chunks_spent = task["timeChunksSpent"]
    url = f"{BASE_URL}/tasks/{task_id}"
    payload = {"status": "COMPLETE", "timeChunksRequired": chunks_spent}
    response = await _get_client().patch(url, json=payload)
    if response.status_code in (200, 204):
        invalidate_reclaim_cache()
        print(f"✅ Marked task {task_id} as complete.")
        return True
    print(f"❌ Error completing task: {response.status_code} - {response.text}")
    return False


async def log_work(task_id: int, start: str, end: str):
    """Record time spent via Reclaim planner (not the deprecated /tasks/{id}/log path)."""
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=ZoneInfo(TIMEZONE))
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=ZoneInfo(TIMEZONE))

    minutes = max(1, int((end_dt - start_dt).total_seconds() // 60))
    end_utc = end_dt.astimezone(timezone.utc)
    end_param = end_utc.isoformat()[:-9] + "Z"

    url = f"{BASE_URL}/planner/log-work/task/{task_id}"
    response = await _get_client().post(
        url, params={"minutes": minutes, "end": end_param}
    )
    if response.status_code in (200, 201, 204):
        invalidate_reclaim_cache()
        print(f"✅ Logged work for task {task_id} ({minutes} min).")
        return True
    print(f"❌ Error logging work: {response.status_code} - {response.text}")
    return False


async def reschedule_task(task_id: int, snooze_until: str | None = None):
    if snooze_until is None:
        snooze_until = datetime.now(timezone.utc).isoformat()
    url = f"{BASE_URL}/tasks/{task_id}"
    payload = {"snoozeUntil": snooze_until}
    response = await _get_client().patch(url, json=payload)
    if response.status_code in (200, 204):
        invalidate_reclaim_cache()
        print(f"✅ Rescheduled task {task_id} to {snooze_until}.")
        return True
    print(f"❌ Error rescheduling: {response.status_code} - {response.text}")
    return False


# Planner enum options keyed by whole-hour snooze amounts.
_SNOOZE_OPTION_BY_HOURS = {1: "FROM_NOW_1H"}


async def reschedule_task_event(
    calendar_id: int,
    event_id: str,
    *,
    snooze_option: str = "FROM_NOW_1H",
) -> bool:
    """Reclaim planner: delete missed instance and reschedule (no GCal lock)."""
    url = f"{BASE_URL}/planner/task/{calendar_id}/{event_id}/reschedule"
    response = await _get_client().post(url, params={"snoozeOption": snooze_option})
    if response.status_code in (200, 201, 204):
        invalidate_reclaim_cache()
        print(f"✅ Rescheduled missed event {event_id} ({snooze_option}).")
        return True
    print(
        f"❌ Error rescheduling event {event_id}: "
        f"{response.status_code} - {response.text}"
    )
    return False


async def move_task_event(
    calendar_id: int,
    event_id: str,
    start: datetime,
    end: datetime,
) -> bool:
    """Reclaim planner: move a task event to explicit start/end (no raw GCal patch)."""
    url = f"{BASE_URL}/planner/event/{calendar_id}/{event_id}/move"
    params = {"start": start.isoformat(), "end": end.isoformat()}
    response = await _get_client().post(url, params=params)
    if response.status_code in (200, 201, 204):
        invalidate_reclaim_cache()
        print(f"✅ Moved task event {event_id} to {start.isoformat()}.")
        return True
    print(
        f"❌ Error moving event {event_id}: {response.status_code} - {response.text}"
    )
    return False


async def reschedule_missed_event(
    event: dict,
    target: datetime | None,
    snooze_option: str | None,
) -> bool:
    """Reschedule one missed task block via the Reclaim planner."""
    calendar_id = event.get("calendarId")
    event_id = event.get("eventId")
    if not calendar_id or not event_id:
        return False
    if target is not None:
        start = _parse_event_time(event["eventStart"])
        end = _parse_event_time(event.get("eventEnd", event["eventStart"]))
        if target.tzinfo is None:
            target = target.replace(tzinfo=ZoneInfo(TIMEZONE))
        return await move_task_event(calendar_id, event_id, target, target + (end - start))
    return await reschedule_task_event(
        calendar_id, event_id, snooze_option=snooze_option or "FROM_NOW_1H"
    )


async def snooze_task(task_id: int, *, hours: int = 1) -> bool:
    """Snooze a task off the current ping via the Reclaim planner.

    Default 1-hour snooze uses the planner ``FROM_NOW_1H`` option (the verified
    path for switch/resume step 1). Custom durations fall back to a ``snoozeUntil``
    PATCH at now + ``hours``.
    """
    option = _SNOOZE_OPTION_BY_HOURS.get(hours)
    if not option:
        snooze_until = (
            datetime.now(timezone.utc) + timedelta(hours=hours)
        ).isoformat()
        return await reschedule_task(task_id, snooze_until=snooze_until)

    url = f"{BASE_URL}/planner/task/{task_id}/snooze"
    response = await _get_client().post(url, params={"snoozeOption": option})
    if response.status_code in (200, 201, 204):
        invalidate_reclaim_cache()
        print(f"✅ Snoozed task {task_id} for {hours}h.")
        return True
    print(f"❌ Error snoozing task: {response.status_code} - {response.text}")
    return False


async def plan_work(
    task_id: int, date_time: str, duration_minutes: int | None = None
) -> bool:
    """Schedule work on a task at a specific time via the planner (schedule-now).

    Primary way to place what the user is actually working on after snoozing the
    pinged task (switch/resume step 2). ``date_time`` is an ISO string;
    ``duration_minutes`` optionally caps the planned block length.
    """
    url = f"{BASE_URL}/planner/plan-work/task/{task_id}"
    params: dict = {"dateTime": date_time}
    if duration_minutes is not None:
        params["durationMinutes"] = int(duration_minutes)
    response = await _get_client().post(url, params=params)
    if response.status_code in (200, 201, 204):
        invalidate_reclaim_cache()
        print(f"✅ Planned work for task {task_id} at {date_time}.")
        return True
    print(f"❌ Error planning work: {response.status_code} - {response.text}")
    return False


async def extend_task_total(task_id: int, additional_chunks: int):
    task = await get_task(task_id)
    if not task:
        return False
    new_chunks = task["timeChunksRequired"] + additional_chunks
    url = f"{BASE_URL}/tasks/{task_id}"
    payload = {"timeChunksRequired": new_chunks}
    response = await _get_client().patch(url, json=payload)
    if response.status_code in (200, 204):
        invalidate_reclaim_cache()
        print(f"✅ Extended task {task_id} by {additional_chunks * 15} min.")
        return True
    print(f"❌ Error extending task: {response.status_code} - {response.text}")
    return False


async def extend_task_instance(task_id: int, additional_minutes: int):
    additional_chunks = minutes_to_chunks(additional_minutes)
    print(
        f"ℹ️ extend_task_instance: +{additional_minutes} min → +{additional_chunks} chunks"
    )
    return await extend_task_total(task_id, additional_chunks)


async def update_task_fields(task_id: int, fields: dict) -> bool:
    url = f"{BASE_URL}/tasks/{task_id}"
    response = await _get_client().patch(url, json=fields)
    if response.status_code in (200, 204):
        invalidate_reclaim_cache()
        print(f"✅ Updated task {task_id} fields.")
        return True
    print(f"❌ Error updating task: {response.status_code} - {response.text}")
    return False


async def delete_task(task_id: int) -> bool:
    url = f"{BASE_URL}/tasks/{task_id}"
    response = await _get_client().delete(url)
    if response.status_code in (200, 204):
        invalidate_reclaim_cache()
        print(f"✅ Deleted task {task_id}.")
        return True
    print(f"❌ Error deleting task: {response.status_code} - {response.text}")
    return False


async def create_gcal_event(name: str, start: str, end: str, calendar_id: str = CALENDAR_ID):
    result, _snapshot = await gcal.create_buffer_event(name, start, end, calendar_id)
    return result


async def create_task(
    title: str,
    due_date: str,
    priority: str = "P1",
    event_category: str = "WORK",
    min_chunk_size: int = 4,
    max_chunk_size: int = 8,
    time_needed: float = 8,
):
    if priority not in VALID_PRIORITIES:
        print(f"⚠️ Invalid priority '{priority}', defaulting to P1")
        priority = "P1"
    event_category = normalize_event_category(event_category)

    url = f"{BASE_URL}/tasks"
    payload = {
        "title": title,
        "due": due_date,
        "priority": priority,
        "eventCategory": event_category,
        "minChunkSize": min_chunk_size,
        "maxChunkSize": max_chunk_size,
        "timeChunksRequired": int(time_needed),
    }
    response = await _get_client().post(url, json=payload)
    if response.status_code in (200, 201):
        invalidate_reclaim_cache()
        print(f"✅ Created task '{title}' due {due_date}.")
        return response.json()
    print(f"❌ Error creating task: {response.status_code} - {response.text}")
    return None


async def close_client():
    """Close the shared httpx async client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
