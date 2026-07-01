"""Work-hours envelope buffer math and break permission decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from config import (
    BREAK_HORIZON_DAYS,
    MIN_BUFFER_HOURS,
    SCHEDULABLE_WEEKDAYS,
    TIGHT_BUFFER_HOURS,
    TIMEZONE,
    WORK_WINDOWS,
)
from reclaim import (
    _parse_event_time,
    get_all_events,
    get_all_tasks,
    is_task_assignment,
)

Outcome = Literal["green", "yellow", "red"]
MIN_ALLOWANCE_MINUTES = MIN_BUFFER_HOURS * 60
TIGHT_ALLOWANCE_MINUTES = TIGHT_BUFFER_HOURS * 60


@dataclass
class ScheduleState:
    now: datetime
    horizon_end: datetime
    tasks: list[dict]
    events: list[dict]
    tz: ZoneInfo = field(default_factory=lambda: ZoneInfo(TIMEZONE))


@dataclass
class AllowanceSnapshot:
    schedulable_minutes: int
    work_remaining_minutes: int
    allowance_minutes: int
    scheduled_minutes: int
    scheduling_gap_minutes: int


@dataclass
class BreakSimulation:
    new_allowance_minutes: int
    at_risk_now: list[str]
    would_be_at_risk: list[str]
    outcome: Outcome
    break_overlap_minutes: int
    allowance_before: AllowanceSnapshot


@dataclass
class BreakAssessment:
    break_start: datetime
    break_end: datetime
    simulation: BreakSimulation
    allowance: AllowanceSnapshot
    partial_break_minutes: int = 0
    calls: list[dict] = field(default_factory=list)
    reply: str = ""
    clarification_required: bool = False
    action_required: bool = False


def _chunks_remaining(task: dict) -> int:
    remaining = task.get("timeChunksRemaining")
    if remaining is not None:
        return max(0, int(remaining))
    required = task.get("timeChunksRequired") or 0
    spent = task.get("timeChunksSpent") or 0
    return max(0, required - spent)


def _overlap_minutes(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0
    return int((end - start).total_seconds() // 60)


def _window_on_day(
    day: date, start_t: time, end_t: time, tz: ZoneInfo
) -> tuple[datetime, datetime]:
    w_start = datetime.combine(day, start_t, tzinfo=tz)
    w_end = datetime.combine(day, end_t, tzinfo=tz)
    return w_start, w_end


def _iter_work_windows(
    range_start: datetime, range_end: datetime, tz: ZoneInfo
):
    if range_end <= range_start:
        return
    day = range_start.date()
    last_day = range_end.date()
    while day <= last_day:
        if day.weekday() in SCHEDULABLE_WEEKDAYS:
            for start_t, end_t in WORK_WINDOWS:
                w_start, w_end = _window_on_day(day, start_t, end_t, tz)
                clip_start = max(w_start, range_start)
                clip_end = min(w_end, range_end)
                if clip_end > clip_start:
                    yield clip_start, clip_end
        day += timedelta(days=1)


def schedulable_minutes(
    range_start: datetime,
    range_end: datetime,
    tz: ZoneInfo | None = None,
) -> int:
    """Total work-window minutes in [range_start, range_end)."""
    tz = tz or ZoneInfo(TIMEZONE)
    total = 0
    for w_start, w_end in _iter_work_windows(range_start, range_end, tz):
        total += int((w_end - w_start).total_seconds() // 60)
    return total


def break_overlap_minutes(
    break_start: datetime,
    break_end: datetime,
    tz: ZoneInfo | None = None,
) -> int:
    tz = tz or break_start.tzinfo or ZoneInfo(TIMEZONE)
    total = 0
    for w_start, w_end in _iter_work_windows(break_start, break_end, tz):
        total += _overlap_minutes(break_start, break_end, w_start, w_end)
    return total


def scheduled_in_windows(
    events: list[dict],
    range_start: datetime,
    range_end: datetime,
    tz: ZoneInfo | None = None,
) -> int:
    tz = tz or ZoneInfo(TIMEZONE)
    total = 0
    for event in events:
        if not is_task_assignment(event):
            continue
        ev_start = _parse_event_time(event["eventStart"])
        ev_end = _parse_event_time(event.get("eventEnd", event["eventStart"]))
        if ev_end.tzinfo is None:
            ev_end = ev_end.replace(tzinfo=timezone.utc)
        if ev_start.tzinfo is None:
            ev_start = ev_start.replace(tzinfo=timezone.utc)
        ev_start = ev_start.astimezone(tz)
        ev_end = ev_end.astimezone(tz)
        if ev_end <= range_start or ev_start >= range_end:
            continue
        for w_start, w_end in _iter_work_windows(
            max(ev_start, range_start), min(ev_end, range_end), tz
        ):
            total += _overlap_minutes(ev_start, ev_end, w_start, w_end)
    return total


def _tasks_in_horizon(tasks: list[dict], horizon_end: datetime, tz: ZoneInfo) -> list[dict]:
    active = []
    for task in tasks:
        if task.get("status") not in ("IN_PROGRESS", "SCHEDULED"):
            continue
        due_raw = task.get("due")
        if not due_raw:
            continue
        due = _parse_event_time(due_raw)
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        if due.astimezone(tz) <= horizon_end:
            active.append(task)
    return active


def compute_allowance(state: ScheduleState) -> AllowanceSnapshot:
    tz = state.tz
    horizon_tasks = _tasks_in_horizon(state.tasks, state.horizon_end, tz)
    work_remaining = sum(_chunks_remaining(t) * 15 for t in horizon_tasks)
    schedulable = schedulable_minutes(state.now, state.horizon_end, tz)
    scheduled = scheduled_in_windows(
        state.events, state.now, state.horizon_end, tz
    )
    allowance = schedulable - work_remaining
    return AllowanceSnapshot(
        schedulable_minutes=schedulable,
        work_remaining_minutes=work_remaining,
        allowance_minutes=allowance,
        scheduled_minutes=scheduled,
        scheduling_gap_minutes=schedulable - scheduled,
    )


def _would_be_at_risk_tasks(
    state: ScheduleState,
    break_start: datetime,
    break_end: datetime,
) -> tuple[list[str], list[str]]:
    tz = state.tz
    at_risk_now: list[str] = []
    would_be_at_risk: list[str] = []

    for task in _tasks_in_horizon(state.tasks, state.horizon_end, tz):
        title = task.get("title") or f"task {task.get('id')}"
        if task.get("atRisk"):
            at_risk_now.append(title)

        due_raw = task.get("due")
        if not due_raw:
            continue
        due = _parse_event_time(due_raw)
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        due_local = due.astimezone(tz)

        remaining_min = _chunks_remaining(task) * 15
        if remaining_min <= 0:
            continue

        sched_before_due = schedulable_minutes(state.now, due_local, tz)
        break_before_due = break_overlap_minutes(
            break_start, min(break_end, due_local), tz
        )
        sched_after_break = sched_before_due - break_before_due

        if task.get("atRisk") or remaining_min > sched_after_break:
            if title not in would_be_at_risk:
                would_be_at_risk.append(title)

    return at_risk_now, would_be_at_risk


def _classify_outcome(
    new_allowance_minutes: int, would_be_at_risk: list[str]
) -> Outcome:
    if would_be_at_risk or new_allowance_minutes <= TIGHT_ALLOWANCE_MINUTES:
        return "red"
    if new_allowance_minutes >= MIN_ALLOWANCE_MINUTES:
        return "green"
    return "yellow"


def simulate_break(
    state: ScheduleState,
    break_start: datetime,
    break_end: datetime,
) -> BreakSimulation:
    allowance_before = compute_allowance(state)
    overlap = break_overlap_minutes(break_start, break_end, state.tz)
    new_allowance = allowance_before.allowance_minutes - overlap
    at_risk_now, would_be_at_risk = _would_be_at_risk_tasks(
        state, break_start, break_end
    )
    outcome = _classify_outcome(new_allowance, would_be_at_risk)
    return BreakSimulation(
        new_allowance_minutes=new_allowance,
        at_risk_now=at_risk_now,
        would_be_at_risk=would_be_at_risk,
        outcome=outcome,
        break_overlap_minutes=overlap,
        allowance_before=allowance_before,
    )


def find_max_safe_break(
    state: ScheduleState,
    break_start: datetime,
    max_end: datetime,
    min_allowance_minutes: int = TIGHT_ALLOWANCE_MINUTES,
) -> int:
    total = int((max_end - break_start).total_seconds() // 60)
    if total <= 0:
        return 0

    best = 0
    for minutes in range(15, total + 1, 15):
        test_end = break_start + timedelta(minutes=minutes)
        sim = simulate_break(state, break_start, test_end)
        if (
            not sim.would_be_at_risk
            and sim.new_allowance_minutes >= min_allowance_minutes
        ):
            best = minutes
    return best


def format_hours(minutes: int) -> str:
    hours = minutes / 60
    if abs(hours - round(hours)) < 0.05:
        return f"{int(round(hours))}h"
    return f"{hours:.1f}h"


def build_create_event_call(
    break_start: datetime, break_end: datetime, name: str = "Break"
) -> dict:
    return {
        "function": "create_event",
        "params": {
            "name": name,
            "start": break_start.isoformat(),
            "end": break_end.isoformat(),
        },
    }


async def load_schedule_state(
    now: datetime | None = None,
) -> ScheduleState:
    tz = ZoneInfo(TIMEZONE)
    now = now or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)
    horizon_end = now + timedelta(days=BREAK_HORIZON_DAYS)
    tasks = await get_all_tasks(instances=True)
    events = await get_all_events()
    return ScheduleState(
        now=now,
        horizon_end=horizon_end,
        tasks=tasks,
        events=events,
        tz=tz,
    )


async def get_break_allowance() -> str:
    """Read-only: how much break slack is available through the planning horizon."""
    state = await load_schedule_state()
    allowance = compute_allowance(state)
    slack = format_hours(allowance.allowance_minutes)
    booked = format_hours(allowance.scheduled_minutes)
    schedulable = format_hours(allowance.schedulable_minutes)
    work_left = format_hours(allowance.work_remaining_minutes)
    return (
        f"Break allowance (through {state.horizon_end.strftime('%A %b %d')}): "
        f"~{slack} slack after required work ({work_left} work left, "
        f"{booked} of {schedulable} schedulable time already booked)."
    )


_BREAK_ALLOWANCE_HINTS = re.compile(
    r"\b(?:how long (?:of )?a break|break can i take|break allowance|"
    r"how much break|time for a break)\b",
    re.I,
)


def parse_break_allowance_request(message: str) -> bool:
    """True when the user is asking a read-only break-capacity question."""
    return bool(_BREAK_ALLOWANCE_HINTS.search(message or ""))


def _format_risk_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def format_assessment_reply(
    assessment: BreakAssessment, *, partial: bool = False
) -> str:
    sim = assessment.simulation
    allowance = assessment.allowance
    new_h = format_hours(sim.new_allowance_minutes)

    if partial and assessment.partial_break_minutes:
        partial_h = format_hours(assessment.partial_break_minutes)
        risk = _format_risk_names(sim.would_be_at_risk)
        if risk:
            return (
                f"A full break would put {risk} at-risk. "
                f"A {partial_h} break keeps ~{new_h} slack — want that?"
            )
        return (
            f"A full break gets too tight (~{new_h} slack left). "
            f"How about {partial_h} off instead — still want it?"
        )

    if sim.outcome == "green":
        gap_note = ""
        if allowance.scheduled_minutes > 0:
            sched_h = format_hours(allowance.scheduled_minutes)
            pool_h = format_hours(allowance.schedulable_minutes)
            gap_note = f" ({sched_h} of {pool_h} booked this week)"
        return f"You've got ~{new_h} slack after clearing that{gap_note} — blocking it off now."

    if sim.outcome == "yellow":
        before_h = format_hours(allowance.allowance_minutes)
        return (
            f"Clearing that drops you from ~{before_h} to ~{new_h} slack — "
            "things get tighter. Still want me to do it?"
        )

    # red — full break not offered; partial handled via partial=True
    risk = _format_risk_names(sim.would_be_at_risk)
    if risk:
        return f"A full break would put {risk} at-risk — I can't clear that safely."
    return (
        f"That break would leave only ~{new_h} slack — too tight for me to auto-clear."
    )


async def assess_break_request(
    break_start: datetime,
    break_end: datetime,
    state: ScheduleState | None = None,
) -> BreakAssessment:
    """Evaluate a break window and return permission, calls, and user reply."""
    state = state or await load_schedule_state(break_start)
    tz = state.tz
    if break_start.tzinfo is None:
        break_start = break_start.replace(tzinfo=tz)
    else:
        break_start = break_start.astimezone(tz)
    if break_end.tzinfo is None:
        break_end = break_end.replace(tzinfo=tz)
    else:
        break_end = break_end.astimezone(tz)

    if break_end <= break_start:
        break_end = break_start + timedelta(minutes=15)

    allowance = compute_allowance(state)
    sim = simulate_break(state, break_start, break_end)

    assessment = BreakAssessment(
        break_start=break_start,
        break_end=break_end,
        simulation=sim,
        allowance=allowance,
    )

    if sim.outcome == "green":
        assessment.action_required = True
        assessment.calls = [build_create_event_call(break_start, break_end)]
        assessment.reply = format_assessment_reply(assessment)
        return assessment

    if sim.outcome == "yellow":
        assessment.clarification_required = True
        assessment.calls = [build_create_event_call(break_start, break_end)]
        assessment.reply = format_assessment_reply(assessment)
        return assessment

    partial_mins = find_max_safe_break(state, break_start, break_end)
    assessment.partial_break_minutes = partial_mins

    if partial_mins > 0:
        partial_end = break_start + timedelta(minutes=partial_mins)
        partial_sim = simulate_break(state, break_start, partial_end)
        assessment.simulation = partial_sim
        assessment.clarification_required = True
        assessment.calls = [build_create_event_call(break_start, partial_end)]
        assessment.reply = format_assessment_reply(assessment, partial=True)
        return assessment

    assessment.clarification_required = False
    assessment.action_required = False
    assessment.reply = format_assessment_reply(assessment)
    if sim.would_be_at_risk:
        assessment.reply += " No shorter break works either right now."
    else:
        assessment.reply += " Can't fit a safe break right now."
    return assessment


_NON_BREAK_EVENT_WORDS = (
    "commute",
    "commuting",
    "lunch",
    "dinner",
    "drive",
    "driving",
    "travel",
    "transit",
    "appointment",
    "meeting",
    "class",
)


def is_break_like_event_call(call: dict) -> bool:
    if call.get("function") != "create_event":
        return False
    params = call.get("params") or {}
    name = (params.get("name") or "").lower()
    if any(w in name for w in _NON_BREAK_EVENT_WORDS):
        return False
    if any(w in name for w in ("break", "rest", "off", "tired", "relax")):
        return True
    start = params.get("start") or params.get("start_time_natural") or ""
    end = params.get("end") or params.get("end_time_natural") or ""
    if start and end:
        try:
            s = _parse_event_time(start) if "T" in str(start) else None
            e = _parse_event_time(end) if "T" in str(end) else None
            if s and e and int((e - s).total_seconds()) >= 30 * 60:
                return True
        except (ValueError, TypeError):
            pass
    return False


async def gate_break_calls(
    calls: list[dict], now: datetime | None = None
) -> BreakAssessment | None:
    """Run buffer analysis on break-like create_event calls; None if no gate needed."""
    break_calls = [c for c in calls if is_break_like_event_call(c)]
    if not break_calls:
        return None

    tz = ZoneInfo(TIMEZONE)
    now = now or datetime.now(tz)
    state = await load_schedule_state(now)

    from inference import parse_to_iso

    params = break_calls[0].get("params") or {}
    start_raw = params.get("start") or params.get("start_time_natural") or "now"
    end_raw = params.get("end") or params.get("end_time_natural") or ""

    if "T" in str(start_raw):
        break_start = _parse_event_time(str(start_raw)).astimezone(tz)
    else:
        iso = parse_to_iso(str(start_raw), now)
        break_start = (
            datetime.fromisoformat(iso).astimezone(tz) if iso else now
        )

    if end_raw:
        if "T" in str(end_raw):
            break_end = _parse_event_time(str(end_raw)).astimezone(tz)
        else:
            iso = parse_to_iso(str(end_raw), now)
            break_end = (
                datetime.fromisoformat(iso).astimezone(tz)
                if iso
                else break_start + timedelta(hours=1)
            )
    else:
        from duration_parser import parse_duration_to_minutes

        combined = f"{start_raw} {end_raw}".strip()
        mins = parse_duration_to_minutes(combined) or 60
        break_end = break_start + timedelta(minutes=mins)

    return await assess_break_request(break_start, break_end, state=state)
