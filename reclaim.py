"""Reclaim API client, caching, and task/event helpers."""

import time

import httpx
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from config import RECLAIM_API_KEY, CALENDAR_ID, TIMEZONE
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


def minutes_to_chunks(minutes: int) -> int:
    """Convert minutes to 15-min Reclaim chunks (rounds up, minimum 1)."""
    return max(1, (minutes + 14) // 15)


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


def get_event_by_id(events: list[dict], event_id: str) -> dict | None:
    for event in events:
        if event.get("eventId") == event_id:
            return event
    return None


def refresh_ongoing_state(current_task: dict) -> dict:
    """Recompute is_ongoing from stored start/end times (ping may have been scheduled earlier)."""
    if not current_task.get("start_time"):
        return current_task
    now = datetime.now(timezone.utc)
    start = _parse_event_time(current_task["start_time"])
    end = _parse_event_time(current_task.get("end_time", current_task["start_time"]))
    updated = dict(current_task)
    updated["is_ongoing"] = start <= now < end
    return updated


async def find_missed_task_blocks(
    task_id: int, day: date | None = None, events: list[dict] | None = None
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
        if not is_task_assignment(event):
            continue
        if not event_on_local_day(event, day):
            continue
        if not is_past_block(event, now):
            continue
        blocks.append(event)
    return sorted(blocks, key=lambda e: _parse_event_time(e["eventStart"]))


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
    url = f"{BASE_URL}/tasks/{task_id}/log"
    payload = {"start": start, "end": end}
    response = await _get_client().post(url, json=payload)
    if response.status_code in (200, 201, 204):
        invalidate_reclaim_cache()
        print(f"✅ Logged work for task {task_id}.")
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


async def restore_task_fields(task_id: int, fields: dict) -> bool:
    url = f"{BASE_URL}/tasks/{task_id}"
    response = await _get_client().patch(url, json=fields)
    if response.status_code in (200, 204):
        invalidate_reclaim_cache()
        print(f"✅ Restored task {task_id} fields.")
        return True
    print(f"❌ Error restoring task: {response.status_code} - {response.text}")
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
