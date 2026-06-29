"""Unit tests for buffer_analysis (no live API)."""

import sys
from datetime import datetime, timedelta, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zoneinfo import ZoneInfo

from buffer_analysis import (
    ScheduleState,
    assess_break_request,
    compute_allowance,
    find_max_safe_break,
    schedulable_minutes,
    simulate_break,
)
from config import TIMEZONE

TZ = ZoneInfo(TIMEZONE)


def _monday_noon():
    # A Monday at noon local
    return datetime(2026, 6, 29, 12, 0, tzinfo=TZ)


def _task(
    title: str,
    chunks_remaining: int,
    due_days: int = 3,
    at_risk: bool = False,
) -> dict:
    due = _monday_noon() + timedelta(days=due_days)
    return {
        "id": hash(title) % 10000,
        "title": title,
        "status": "SCHEDULED",
        "timeChunksRemaining": chunks_remaining,
        "timeChunksRequired": chunks_remaining,
        "timeChunksSpent": 0,
        "due": due.isoformat(),
        "atRisk": at_risk,
    }


def test_schedulable_minutes_one_weekday():
    now = _monday_noon()
    end = datetime.combine(now.date(), time(22, 0), tzinfo=TZ)
    mins = schedulable_minutes(now, end, TZ)
    # 12-17 = 5h, 19-22 = 3h
    assert mins == 480, f"expected 480 got {mins}"


def test_allowance_green():
    now = _monday_noon()
    horizon = now + timedelta(days=7)
    # ~55h schedulable over 5 weekdays in 7 days... roughly 5*11=55h if full weeks
    # Use light workload: 6h remaining
    tasks = [_task("light work", chunks_remaining=24)]  # 6h
    state = ScheduleState(now=now, horizon_end=horizon, tasks=tasks, events=[], tz=TZ)
    allowance = compute_allowance(state)
    assert allowance.work_remaining_minutes == 360
    assert allowance.allowance_minutes > 3 * 60


def test_simulate_break_yellow():
    now = _monday_noon()
    horizon = now + timedelta(days=7)
    sched = schedulable_minutes(now, horizon, TZ)
    # workload leaving ~2.5h allowance (yellow band)
    work_chunks = max(1, (sched - 150) // 15)
    tasks = [_task("busy", chunks_remaining=work_chunks)]
    state = ScheduleState(now=now, horizon_end=horizon, tasks=tasks, events=[], tz=TZ)
    before = compute_allowance(state)
    assert 60 < before.allowance_minutes < 180

    break_end = now + timedelta(hours=4)
    sim = simulate_break(state, now, break_end)
    assert sim.outcome in ("yellow", "red")


def test_at_risk_blocks_green():
    now = _monday_noon()
    horizon = now + timedelta(days=7)
    tasks = [
        _task("easy", chunks_remaining=8),
        _task("doomed", chunks_remaining=40, due_days=1, at_risk=True),
    ]
    state = ScheduleState(now=now, horizon_end=horizon, tasks=tasks, events=[], tz=TZ)
    break_end = now + timedelta(hours=2)
    sim = simulate_break(state, now, break_end)
    assert "doomed" in sim.would_be_at_risk
    assert sim.outcome == "red"


def test_find_partial_break():
    now = _monday_noon()
    horizon = now + timedelta(days=7)
    sched = schedulable_minutes(now, horizon, TZ)
    work_chunks = max(1, (sched - 90) // 15)  # tight: ~1.5h allowance
    tasks = [_task("packed", chunks_remaining=work_chunks)]
    state = ScheduleState(now=now, horizon_end=horizon, tasks=tasks, events=[], tz=TZ)
    max_end = now + timedelta(hours=5)
    partial = find_max_safe_break(state, now, max_end)
    assert partial >= 0


async def _test_assess_green():
    now = _monday_noon()
    horizon = now + timedelta(days=7)
    tasks = [_task("light", chunks_remaining=16)]
    state = ScheduleState(now=now, horizon_end=horizon, tasks=tasks, events=[], tz=TZ)
    assessment = await assess_break_request(
        now, now + timedelta(hours=2), state=state
    )
    assert assessment.action_required
    assert assessment.simulation.outcome == "green"
    assert assessment.calls


def run_async(coro):
    import asyncio

    return asyncio.run(coro)


def main():
    test_schedulable_minutes_one_weekday()
    test_allowance_green()
    test_simulate_break_yellow()
    test_at_risk_blocks_green()
    test_find_partial_break()
    run_async(_test_assess_green())
    print("All buffer_analysis tests passed.")


if __name__ == "__main__":
    main()
