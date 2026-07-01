"""Focused unit tests for clarification.gate_task_queries (spec 21: never error,
always clarify on an unresolved or placeholder task reference).
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarification import TASK_QUERY_KEYS, TASK_TARGET_FUNCTIONS, gate_task_queries

BCG = {"id": 1, "title": "BCG prep"}
ORGO = {"id": 2, "title": "Orgo homework"}


async def _fake_resolver(known: dict[str, dict]):
    async def resolver(query, current_task=None, preferred_task_ids=None):
        q = (query or "").lower()
        for needle, task in known.items():
            if needle in q:
                return task
        return None

    return resolver


def test_task_target_functions_cover_all_writes():
    # Every function that patches/reads a specific Reclaim task must be gated.
    for fn in (
        "complete_task",
        "extend_task_total",
        "extend_task_instance",
        "reschedule_task",
        "reschedule_missed_work",
        "move_due_date",
        "update_task",
        "log_work",
        "switch_active_task",
        "resume_previous_task",
    ):
        assert fn in TASK_TARGET_FUNCTIONS, fn


def test_task_query_keys_cover_all_reference_params():
    assert set(TASK_QUERY_KEYS) == {"task_query", "new_task_query", "previous_task_query"}


async def test_gate_passes_multiple_valid_calls():
    resolver = await _fake_resolver({"bcg": BCG, "orgo": ORGO})
    calls = [
        {"function": "complete_task", "params": {"task_query": "bcg prep"}},
        {"function": "extend_task_total", "params": {"task_query": "orgo homework", "additional_chunks": 4}},
    ]
    with patch("cal_helper.get_task_by_query", side_effect=resolver):
        result = await gate_task_queries(calls)
    assert result.ok is True
    assert result.clarification_required is False
    assert result.calls == calls


async def test_gate_stops_at_first_unresolved_call():
    resolver = await _fake_resolver({"bcg": BCG})
    calls = [
        {"function": "complete_task", "params": {"task_query": "bcg prep"}},
        {"function": "reschedule_task", "params": {"task_query": "nonexistent widget"}},
    ]
    with patch("cal_helper.get_task_by_query", side_effect=resolver):
        result = await gate_task_queries(calls)
    assert result.ok is False
    assert result.clarification_required is True
    assert result.pending is not None
    assert result.pending.partial_params["attempted_query"] == "nonexistent widget"
    assert result.pending.partial_params["function"] == "reschedule_task"


async def test_gate_checks_new_task_query_key():
    async def always_none(query, **kwargs):
        return None

    calls = [{"function": "switch_active_task", "params": {"new_task_query": "something else"}}]
    with patch("cal_helper.get_task_by_query", side_effect=always_none):
        result = await gate_task_queries(calls)
    assert result.ok is False
    assert result.pending.partial_params["field"] == "new_task_query"


async def test_gate_checks_previous_task_query_key():
    async def always_bcg(query, **kwargs):
        return BCG

    calls = [
        {
            "function": "resume_previous_task",
            "params": {"previous_task_query": "my last task", "work_duration_minutes": 30},
        }
    ]
    with patch("cal_helper.get_task_by_query", side_effect=always_bcg):
        result = await gate_task_queries(calls)
    assert result.ok is False
    assert result.pending.partial_params["field"] == "previous_task_query"


async def test_gate_allows_resume_without_previous_task_query():
    """previous_task_query is optional — omitting it lets the system resolve it."""
    calls = [
        {"function": "resume_previous_task", "params": {"work_duration_minutes": 30}}
    ]
    result = await gate_task_queries(calls)
    assert result.ok is True


async def test_gate_skips_non_task_functions():
    for fn, params in (
        ("create_event", {"name": "lunch", "start": "now", "end": "1pm"}),
        ("create_task", {"title": "x", "due_date": "2026-07-01", "event_category": "WORK"}),
        ("extend_current_gcal_block", {"additional_minutes": 20, "task_query": "lunch"}),
        ("get_schedule_for_window", {"day": "today", "period": "afternoon"}),
    ):
        result = await gate_task_queries([{"function": fn, "params": params}])
        assert result.ok is True, fn


async def test_gate_ignores_blank_task_query():
    calls = [{"function": "complete_task", "params": {"task_query": "   "}}]
    result = await gate_task_queries(calls)
    assert result.ok is True


async def test_gate_forwards_current_task_and_preferred_ids():
    seen: dict = {}

    async def capturing_resolver(query, current_task=None, preferred_task_ids=None):
        seen["current_task"] = current_task
        seen["preferred_task_ids"] = preferred_task_ids
        return BCG

    calls = [{"function": "complete_task", "params": {"task_query": "bcg"}}]
    ctx = {"title": "BCG prep", "task_id": 1}
    with patch("cal_helper.get_task_by_query", side_effect=capturing_resolver):
        result = await gate_task_queries(calls, current_task=ctx, preferred_task_ids=[1, 2])
    assert result.ok is True
    assert seen["current_task"] == ctx
    assert seen["preferred_task_ids"] == [1, 2]


async def test_gate_reply_wording_differs_placeholder_vs_miss():
    async def always_none(query, **kwargs):
        return None

    with patch("cal_helper.get_task_by_query", side_effect=always_none):
        placeholder = await gate_task_queries(
            [{"function": "complete_task", "params": {"task_query": "something else"}}]
        )
        miss = await gate_task_queries(
            [{"function": "complete_task", "params": {"task_query": "quantum widget prep"}}]
        )
    assert "placeholder" in placeholder.reply.lower()
    assert "couldn't find" in miss.reply.lower()


def main():
    test_task_target_functions_cover_all_writes()
    test_task_query_keys_cover_all_reference_params()
    asyncio.run(test_gate_passes_multiple_valid_calls())
    asyncio.run(test_gate_stops_at_first_unresolved_call())
    asyncio.run(test_gate_checks_new_task_query_key())
    asyncio.run(test_gate_checks_previous_task_query_key())
    asyncio.run(test_gate_allows_resume_without_previous_task_query())
    asyncio.run(test_gate_skips_non_task_functions())
    asyncio.run(test_gate_ignores_blank_task_query())
    asyncio.run(test_gate_forwards_current_task_and_preferred_ids())
    asyncio.run(test_gate_reply_wording_differs_placeholder_vs_miss())
    print("All task query gate tests passed.")


if __name__ == "__main__":
    main()
