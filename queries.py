"""Read-only schedule snapshots injected per-turn (never in permanent system prompt)."""

import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from buffer_analysis import chunks_remaining, format_hours
from config import TIME_OF_DAY_BANDS, TIMEZONE, WEEK_END_WEEKDAY
from inference import parse_to_iso
from reclaim import (
    _parse_event_time,
    get_active_tasks,
    get_all_events,
    is_task_assignment,
)


def format_snapshot_for_prompt(body: str) -> str:
    """Wrap snapshot body with the read-only prefix for LLM injection."""
    return f"Schedule snapshot (read-only, answer from this data only):\n{body}"


def format_clock(dt: datetime) -> str:
    """Format a datetime as a friendly local clock time (e.g. '12:00 PM')."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _format_countdown(start: datetime, now: datetime) -> str:
    secs = (start - now).total_seconds()
    if secs <= 0:
        return "now"
    mins = int(secs // 60)
    if mins < 60:
        return f"{mins} min"
    hrs, rem = divmod(mins, 60)
    return f"{hrs}h {rem}m" if rem else f"{hrs}h"


def _snapshot_status_lines(events: list, now: datetime, tz: ZoneInfo) -> list[str]:
    """Active block and next-upcoming countdown (server-computed; LLM must not guess)."""
    lines: list[str] = []
    upcoming: list[tuple[datetime, datetime, str]] = []

    for event in events:
        title = event.get("title") or "block"
        start = _parse_event_time(event["eventStart"])
        end = _parse_event_time(event.get("eventEnd", event["eventStart"]))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        local_start = start.astimezone(tz)
        local_end = end.astimezone(tz)
        if local_start <= now < local_end:
            lines.append(
                f"Active task block: {title} until {format_clock(local_end)}"
            )
        elif local_start > now:
            upcoming.append((local_start, local_end, title))

    if upcoming:
        local_start, _local_end, title = min(upcoming, key=lambda x: x[0])
        lines.append(
            f"Next task block: {title} at {format_clock(local_start)} "
            f"({_format_countdown(local_start, now)} away)"
        )
    return lines


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

    status = _snapshot_status_lines(week_events, now, tz)
    if status:
        lines.extend(status)
        lines.append("")

    lines.append("Tasks (due within 7 days):")
    if not week_tasks:
        lines.append("  (none)")
    else:
        for task in sorted(week_tasks, key=lambda t: t.get("due", "")):
            remaining = chunks_remaining(task)
            due = task.get("due", "?")
            lines.append(
                f"  - {task.get('title')} | due {due} | "
                f"{format_hours(remaining * 15)} left ({remaining} chunks)"
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


# ─── Windowed read (spec 17) ───────────────────────────────────────────────────

_SCHEDULE_READ_HINTS = re.compile(
    r"\b(?:"
    r"schedule|calendar|going on|looks? like|what do i have|what am i doing|"
    r"what's on|whats on|tasks? this week|my week|whole week|rest of (?:the )?week|"
    r"this morning|this afternoon|this evening|tonight|tomorrow morning|"
    r"tomorrow afternoon|overdue"
    r")\b",
    re.I,
)

_PERIOD_IN_MESSAGE = re.compile(
    r"\b(morning|noon|afternoon|evening|night)\b",
    re.I,
)

_FULL_WEEK_IN_MESSAGE = re.compile(
    r"\b(?:this week|my week|whole week|rest of (?:the )?week|tasks? this week)\b",
    re.I,
)

_DEFERRAL_REPLY = re.compile(
    r"\b(?:let me check|i'll check|i will check|let me look|checking your)\b",
    re.I,
)


def parse_schedule_read_request(message: str) -> dict | None:
    """Return ``get_schedule_for_window`` params when the user is asking to read the calendar.

    Single consolidated parser used by the post-LLM repair path (not a Tier-1 bypass).
    """
    text = (message or "").strip()
    if not text or not _SCHEDULE_READ_HINTS.search(text):
        return None

    if _FULL_WEEK_IN_MESSAGE.search(text):
        return {"full_week": True}

    params: dict = {}
    lower = text.lower()
    if "tomorrow" in lower:
        params["day"] = "tomorrow"
    elif re.search(r"\b(?:today|tonight|this morning|this afternoon|this evening)\b", lower):
        params["day"] = "today"
    else:
        for day_name in (
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ):
            if re.search(rf"\b{day_name}\b", lower):
                params["day"] = day_name
                break

    period_match = _PERIOD_IN_MESSAGE.search(lower)
    if period_match:
        params["period"] = period_match.group(1).lower()

    if params and "day" not in params and "full_week" not in params:
        params.setdefault("day", "today")

    return params if params else None


def is_deferral_schedule_reply(reply: str) -> bool:
    """True when the LLM stalled with a 'let me check…' style reply instead of acting."""
    return bool(_DEFERRAL_REPLY.search(reply or ""))


def _to_local(iso_str: str, tz: ZoneInfo) -> datetime:
    dt = _parse_event_time(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz)


def _resolve_day(day: str | None, now: datetime) -> date:
    """Resolve a natural day reference to a concrete date (defaults to today)."""
    if not day:
        return now.date()
    text = day.strip().lower()
    if text in ("today", "tonight", "now", "this morning", "this afternoon", "this evening"):
        return now.date()
    if text in ("tomorrow", "tmrw"):
        return (now + timedelta(days=1)).date()
    if text == "yesterday":
        return (now - timedelta(days=1)).date()
    iso = parse_to_iso(day, now)
    if iso:
        try:
            return datetime.fromisoformat(iso).date()
        except ValueError:
            pass
    return now.date()


def _end_of_week_sunday(from_day: date, tz: ZoneInfo) -> datetime:
    """End of the current Mon–Sun week (Sunday 23:59:59 local)."""
    days_until_sunday = WEEK_END_WEEKDAY - from_day.weekday()
    sunday = from_day + timedelta(days=days_until_sunday)
    return datetime.combine(sunday, time(23, 59, 59), tzinfo=tz)


def _full_week_window(now: datetime, tz: ZoneInfo) -> tuple[datetime, datetime, date, date]:
    """Window from now through end of Sunday (Mon–Sun week)."""
    window_start = now
    window_end = _end_of_week_sunday(now.date(), tz)
    week_start = now.date() - timedelta(days=now.weekday())
    week_end_date = window_end.date()
    return window_start, window_end, week_start, week_end_date


def _window_bounds(
    target: date, period: str | None, tz: ZoneInfo
) -> tuple[datetime, datetime, str]:
    """Return (start, end, label) for the requested day/period window."""
    full_start = datetime.combine(target, time(0, 0), tzinfo=tz)
    full_end = datetime.combine(target, time(23, 59, 59), tzinfo=tz)

    if not period:
        return full_start, full_end, "all day"

    key = period.strip().lower()
    band = TIME_OF_DAY_BANDS.get(key)
    if not band:
        return full_start, full_end, "all day"

    band_start, band_end = band
    start = datetime.combine(target, band_start, tzinfo=tz)
    end_date = target + timedelta(days=1) if band_end < band_start else target
    end = datetime.combine(end_date, band_end, tzinfo=tz).replace(second=59)
    return start, end, key


def _fmt_day_header(d: date) -> str:
    return d.strftime("%A %b %d")


def _format_window(
    hits: list[tuple[datetime, datetime, str]], target: date, label: str
) -> str:
    day_str = target.strftime("%A %b %d")
    header = f"Schedule for {day_str}"
    if label != "all day":
        header += f" ({label})"
    header += ":"
    if not hits:
        return f"{header}\n  (nothing scheduled)"
    lines = [header]
    for start, end, title in hits:
        lines.append(f"  - {title} | {format_clock(start)} to {format_clock(end)}")
    return "\n".join(lines)


def _format_full_week(
    hits: list[tuple[datetime, datetime, str]],
    week_start: date,
    week_end: date,
    now: datetime,
) -> str:
    header = (
        f"Schedule {_fmt_day_header(week_start)} – {_fmt_day_header(week_end)} "
        f"(from {now.strftime('%A %I:%M %p').lstrip('0')}):"
    )
    if not hits:
        return f"{header}\n  (nothing scheduled)"

    by_day: dict[date, list[tuple[datetime, datetime, str]]] = defaultdict(list)
    for start, end, title in hits:
        by_day[start.date()].append((start, end, title))

    lines = [header]
    day = week_start
    while day <= week_end:
        day_hits = sorted(by_day.get(day, []), key=lambda h: h[0])
        lines.append(f"\n{_fmt_day_header(day)}:")
        if not day_hits:
            lines.append("  (nothing scheduled)")
        else:
            for start, end, title in day_hits:
                lines.append(f"  - {title} | {format_clock(start)} to {format_clock(end)}")
        day += timedelta(days=1)
    return "\n".join(lines)


async def get_schedule_for_window(
    day: str | None = "today",
    period: str | None = None,
    full_week: bool = False,
) -> str:
    """Read-only schedule over ALL GCal events (tasks, lunch, commute, everything).

    ``full_week=True``: from now through end of Sunday (Mon–Sun week). Omit ``day``
    and ``period`` when using full week.
    """
    tz = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)

    if full_week:
        window_start, window_end, week_start, week_end = _full_week_window(now, tz)
        label = "full week"
    else:
        target = _resolve_day(day, now)
        window_start, window_end, label = _window_bounds(target, period, tz)
        week_start = week_end = target

    hits: list[tuple[datetime, datetime, str]] = []
    for event in await get_all_events():
        start_raw = event.get("eventStart")
        if not start_raw:
            continue
        start = _to_local(start_raw, tz)
        end = _to_local(event.get("eventEnd", start_raw), tz)
        if full_week:
            if end <= now:
                continue
            if start < window_end and end > window_start:
                hits.append((start, end, event.get("title") or "(untitled)"))
        elif start < window_end and end > window_start:
            hits.append((start, end, event.get("title") or "(untitled)"))

    hits.sort(key=lambda h: h[0])
    if full_week:
        return _format_full_week(hits, week_start, week_end, now)
    return _format_window(hits, _resolve_day(day, now), label)
