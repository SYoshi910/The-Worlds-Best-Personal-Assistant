"""Warm in-memory index of Reclaim task blocks on the calendar.

Only Reclaim ``TASK_ASSIGNMENT`` blocks are indexed (filtered via
``reclaim.is_task_assignment``), so this module is the source of truth for
"the current task" and "the last task" the user was working on. It is
populated at startup and refreshed on every Google Calendar webhook.
"""

from datetime import datetime, timezone

import reclaim

# event_id -> {title, start, end, task_id, task_index, is_ongoing}
# ``start``/``end`` are timezone-aware UTC datetimes for easy comparison.
_blocks: dict[str, dict] = {}
# event_ids sorted chronologically by start time
_chronological: list[str] = []


def _entry_from_event(event: dict, now: datetime) -> dict:
    assist = event.get("assist") or {}
    start = reclaim._parse_event_time(event["eventStart"])
    end = reclaim._parse_event_time(event.get("eventEnd", event["eventStart"]))
    return {
        "event_id": event.get("eventId"),
        "title": event.get("title"),
        "start": start,
        "end": end,
        "task_id": assist.get("taskId"),
        "task_index": assist.get("taskIndex"),
        "is_ongoing": start <= now < end,
    }


async def refresh() -> None:
    """Rebuild the task-block index from current Reclaim events."""
    global _blocks, _chronological
    now = datetime.now(timezone.utc)
    events = await reclaim.get_all_events()
    blocks: dict[str, dict] = {}
    for event in events:
        if not reclaim.is_task_assignment(event):
            continue
        event_id = event.get("eventId")
        if not event_id:
            continue
        blocks[event_id] = _entry_from_event(event, now)
    _blocks = blocks
    _chronological = sorted(_blocks, key=lambda eid: _blocks[eid]["start"])
    print(f"✅ Schedule cache refreshed: {len(_blocks)} task blocks indexed")


def get(event_id: str) -> dict | None:
    """Return the indexed block for an event id, if present."""
    return _blocks.get(event_id)


def chronological() -> list[dict]:
    """Return all indexed task blocks ordered by start time."""
    return [_blocks[eid] for eid in _chronological]


def current_and_previous(
    now: datetime | None = None,
) -> tuple[dict | None, dict | None]:
    """Return ``(current, previous)`` task blocks.

    ``current`` is the ongoing block, or the most recently started block if
    none is ongoing. ``previous`` is the most recent preceding block belonging
    to a *different* task — the "last task" the user was on before the current
    one (spec).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    ordered = chronological()
    if not ordered:
        return None, None

    current = None
    current_idx = None
    for i, block in enumerate(ordered):
        if block["start"] <= now < block["end"]:
            current = block
            current_idx = i
            break

    if current is None:
        started = [(i, b) for i, b in enumerate(ordered) if b["start"] <= now]
        if started:
            current_idx, current = started[-1]

    if current is None or current_idx is None:
        return None, None

    previous = None
    for i in range(current_idx - 1, -1, -1):
        if ordered[i]["task_id"] != current["task_id"]:
            previous = ordered[i]
            break
    return current, previous
