"""Read-only schedule snapshots injected per-turn (never in permanent system prompt)."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from config import TIMEZONE
from reclaim import (
    _parse_event_time,
    get_active_tasks,
    get_all_events,
    is_task_assignment,
)


def format_snapshot_for_prompt(body: str) -> str:
    """Wrap snapshot body with the read-only prefix for LLM injection."""
    return f"Schedule snapshot (read-only, answer from this data only):\n{body}"


def _chunks_remaining(task: dict) -> int:
    required = task.get("timeChunksRequired") or 0
    spent = task.get("timeChunksSpent") or 0
    return max(0, required - spent)


def _format_chunks(chunks: int) -> str:
    hours = chunks * 15 / 60
    if hours == int(hours):
        return f"{int(hours)}h"
    return f"{hours:.1f}h"


async def build_weekly_snapshot() -> str:
    """Build a read-only snapshot of tasks and blocks due in the next 7 days."""
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    week_end = now + timedelta(days=7)

    tasks = await get_active_tasks()
    # Includes overdue tasks (due <= week_end); intentional for visibility.
    week_tasks = []
    for task in tasks:
        due_raw = task.get("due")
        if not due_raw:
            continue
        due = _parse_event_time(due_raw)
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        if due.astimezone(tz) <= week_end:
            week_tasks.append(task)

    events = await get_all_events()
    week_events = []
    for event in events:
        if not is_task_assignment(event):
            continue
        start = _parse_event_time(event["eventStart"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        local_start = start.astimezone(tz)
        if now <= local_start <= week_end:
            week_events.append(event)

    lines = [f"As of {now.strftime('%A %b %d %I:%M %p %Z')} — next 7 days:", ""]

    lines.append("Tasks (due within 7 days):")
    if not week_tasks:
        lines.append("  (none)")
    else:
        for task in sorted(week_tasks, key=lambda t: t.get("due", "")):
            remaining = _chunks_remaining(task)
            due = task.get("due", "?")
            lines.append(
                f"  - {task.get('title')} | due {due} | "
                f"{_format_chunks(remaining)} left ({remaining} chunks)"
            )

    lines.append("")
    lines.append("Scheduled task blocks (next 7 days):")
    if not week_events:
        lines.append("  (none)")
    else:
        for event in sorted(week_events, key=lambda e: e["eventStart"]):
            lines.append(
                f"  - {event.get('title')} | {event.get('eventStart')} to {event.get('eventEnd')}"
            )

    return format_snapshot_for_prompt("\n".join(lines))


def format_schedule_reply(snapshot: str) -> str:
    """User-facing weekly schedule from a read-only snapshot (no LLM)."""
    prefix = "Schedule snapshot (read-only, answer from this data only):\n"
    body = snapshot[len(prefix):] if snapshot.startswith(prefix) else snapshot
    return f"Here's your week:\n\n{body.strip()}"
