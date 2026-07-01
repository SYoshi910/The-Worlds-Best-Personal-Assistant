"""Tests for the generalized clarification engine."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarification import (
    PLACEHOLDER_QUERIES,
    apply_llm_clarification_fields,
    build_scope_options,
    clear_expired,
    compute_create_task_missing,
    compute_missing_fields,
    gate_task_queries,
    is_extend_scope_reply,
    is_placeholder_query,
    map_scope_to_function,
    merge_clarification_reply,
    merge_create_task_reply,
    new_pending_clarification,
    resolve_scope_clarification,
)
from zoneinfo import ZoneInfo

from config import TIMEZONE


def test_compute_missing_fields_create_task():
    assert compute_missing_fields("create_task", {}) == [
        "title",
        "event_category",
        "due_date_natural",
    ]
    complete = {
        "title": "groceries",
        "event_category": "PERSONAL",
        "due_date_natural": "tonight",
    }
    assert compute_missing_fields("create_task", complete) == []
    assert compute_create_task_missing(complete) == []


def test_compute_missing_fields_switch_task():
    assert compute_missing_fields("switch_task", {}) == [
        "new_task_query",
        "work_duration_natural",
    ]
    assert compute_missing_fields("switch_task", {"new_task_query": "orgo"}) == [
        "work_duration_natural",
    ]
    with_until = {"new_task_query": "orgo", "work_until_natural": "5pm"}
    assert compute_missing_fields("switch_task", with_until) == []


def test_compute_missing_fields_other_kinds():
    assert compute_missing_fields("missed_blocks", {}) == ["task_query"]
    assert compute_missing_fields("disambiguate_task", {}) == ["task_query"]
    assert compute_missing_fields("extend_scope", {}) == []


def test_placeholder_queries():
    assert is_placeholder_query("something else") is True
    assert is_placeholder_query("BCG prep") is False
    assert is_placeholder_query("") is True
    assert "something else" in PLACEHOLDER_QUERIES


def test_merge_create_task_category_and_title():
    pending = new_pending_clarification(
        kind="create_task",
        partial_params={"event_category": "PERSONAL"},
        missing_fields=["title", "due_date_natural"],
    )
    merged = merge_create_task_reply(pending, "update WD address")
    assert merged["title"] == "update WD address"
    assert merged["event_category"] == "PERSONAL"
    assert "title" not in compute_create_task_missing(merged)


def test_merge_create_task_due_and_duration():
    pending = new_pending_clarification(
        kind="create_task",
        partial_params={"title": "xyz", "event_category": "WORK"},
        missing_fields=["due_date_natural"],
    )
    merged = merge_create_task_reply(pending, "tonight, 15 minutes")
    assert merged.get("due_date_natural") == "tonight"
    assert merged.get("time_needed_natural")
    assert compute_create_task_missing(merged) == []


def test_merge_clarification_switch_task():
    pending = new_pending_clarification(
        kind="switch_task",
        partial_params={"new_task_query": "orgo"},
    )
    merged = merge_clarification_reply(pending, "until 5pm")
    assert merged.get("work_until_natural") == "5pm"
    assert compute_missing_fields("switch_task", merged) == []

    pending2 = new_pending_clarification(
        kind="switch_task",
        partial_params={"new_task_query": "orgo"},
    )
    merged2 = merge_clarification_reply(pending2, "30 minutes")
    assert merged2.get("work_duration_natural") == "30 minutes"


def test_merge_clarification_task_query():
    pending = new_pending_clarification(
        kind="disambiguate_task",
        partial_params={"field": "task_query"},
    )
    merged = merge_clarification_reply(pending, "BCG prep")
    assert merged["task_query"] == "BCG prep"
    assert compute_missing_fields("disambiguate_task", merged) == []


def test_extend_scope_reply_mapping():
    assert is_extend_scope_reply("whole task") == "task_total"
    assert is_extend_scope_reply("just this block") == "current_instance"
    assert is_extend_scope_reply("hello") is None


def test_build_scope_options_and_map():
    opts = build_scope_options("bcg", "30 minutes", {"title": "BCG prep"})
    assert set(opts) == {"task_total", "current_instance", "current_block"}
    assert opts["current_block"]["function"] == "extend_current_gcal_block"
    assert (
        map_scope_to_function(
            "task_total",
            {"task_query": "bcg", "additional_time_natural": "30 minutes"},
            None,
        )["function"]
        == "extend_task_total"
    )
    assert resolve_scope_clarification(opts, "task_total") == opts["task_total"]


def test_apply_llm_clarification_all_kinds():
    for kind in (
        "create_task",
        "switch_task",
        "extend_scope",
        "missed_blocks",
        "disambiguate_task",
    ):
        data = {
            "clarification_required": True,
            "clarification_kind": kind,
            "pending_params": {},
        }
        pending = apply_llm_clarification_fields(data)
        assert pending is not None, kind
        assert pending.kind == kind


def test_clear_expired_clarification():
    pending = new_pending_clarification(kind="create_task")
    pending.expires_at = datetime.now(ZoneInfo(TIMEZONE)) - timedelta(minutes=1)
    cleared, _ = clear_expired(pending, None)
    assert cleared is None


async def test_gate_task_queries_resolves():
    async def fake_get_task_by_query(query, current_task=None, preferred_task_ids=None):
        if "bcg" in (query or "").lower():
            return {"id": 1, "title": "BCG prep"}
        return None

    with patch("cal_helper.get_task_by_query", side_effect=fake_get_task_by_query):
        result = await gate_task_queries(
            [{"function": "complete_task", "params": {"task_query": "bcg"}}]
        )
    assert result.ok is True
    assert result.clarification_required is False


async def test_gate_task_queries_placeholder_clarifies():
    async def fake_get_task_by_query(query, **kwargs):
        return {"id": 1, "title": "BCG prep"}

    with patch("cal_helper.get_task_by_query", side_effect=fake_get_task_by_query):
        result = await gate_task_queries(
            [
                {
                    "function": "switch_active_task",
                    "params": {"new_task_query": "something else"},
                }
            ]
        )
    assert result.ok is False
    assert result.clarification_required is True
    assert result.pending is not None
    assert result.pending.kind == "disambiguate_task"


async def test_gate_task_queries_miss_clarifies():
    async def fake_get_task_by_query(query, **kwargs):
        return None

    with patch("cal_helper.get_task_by_query", side_effect=fake_get_task_by_query):
        result = await gate_task_queries(
            [{"function": "complete_task", "params": {"task_query": "nonexistent xyz"}}]
        )
    assert result.ok is False
    assert result.clarification_required is True
    assert "couldn't find" in result.reply.lower()


async def test_gate_skips_non_task_functions():
    result = await gate_task_queries(
        [{"function": "extend_current_gcal_block", "params": {"task_query": "lunch"}}]
    )
    assert result.ok is True


def main():
    test_compute_missing_fields_create_task()
    test_compute_missing_fields_switch_task()
    test_compute_missing_fields_other_kinds()
    test_placeholder_queries()
    test_merge_create_task_category_and_title()
    test_merge_create_task_due_and_duration()
    test_merge_clarification_switch_task()
    test_merge_clarification_task_query()
    test_extend_scope_reply_mapping()
    test_build_scope_options_and_map()
    test_apply_llm_clarification_all_kinds()
    test_clear_expired_clarification()
    asyncio.run(test_gate_task_queries_resolves())
    asyncio.run(test_gate_task_queries_placeholder_clarifies())
    asyncio.run(test_gate_task_queries_miss_clarifies())
    asyncio.run(test_gate_skips_non_task_functions())
    print("All clarification flow tests passed.")


if __name__ == "__main__":
    main()
