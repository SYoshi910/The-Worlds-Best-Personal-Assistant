"""Tests for LLM param normalization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from zoneinfo import ZoneInfo

from config import TIMEZONE
from inference import (
    _apply_date_parsing,
    _apply_duration_parsing,
    _extract_json_object,
    _normalize_llm_numeric_params,
    _normalize_llm_response,
    _parse_llm_json,
    _validate_calls,
    parse_to_iso,
)


def test_normalize_llm_response_clarification_fields():
    data = _normalize_llm_response(
        {
            "action_required": False,
            "clarification_required": True,
            "clarification_kind": "create_task",
            "pending_params": {"event_category": "WORK"},
            "missing_fields": ["title"],
            "reply": "What's it called?",
        }
    )
    assert data["calls"] == []
    assert data["clarification_kind"] == "create_task"
    assert data["pending_params"]["event_category"] == "WORK"


def test_normalize_additional_chunks():
    calls = [
        {
            "function": "extend_task_total",
            "params": {"task_query": "bcg", "additional_chunks": 8},
        }
    ]
    assert _normalize_llm_numeric_params(calls) == []
    assert calls[0]["params"]["additional_chunks"] == 8


def test_normalize_prefers_numeric_over_natural():
    calls = [
        {
            "function": "extend_task_total",
            "params": {
                "task_query": "bcg",
                "additional_chunks": 8,
                "additional_time_natural": "2 hours",
            },
        }
    ]
    _normalize_llm_numeric_params(calls)
    _apply_duration_parsing(calls)
    assert calls[0]["params"]["additional_chunks"] == 8
    assert "additional_time_natural" not in calls[0]["params"]


def test_normalize_minutes_to_chunks():
    calls = [
        {
            "function": "extend_task_total",
            "params": {"task_query": "bcg", "additional_minutes": 120},
        }
    ]
    _normalize_llm_numeric_params(calls)
    assert calls[0]["params"]["additional_chunks"] == 8


def test_natural_only_still_works():
    calls = [
        {
            "function": "extend_task_total",
            "params": {
                "task_query": "bcg",
                "additional_time_natural": "2 hours",
            },
        }
    ]
    _normalize_llm_numeric_params(calls)
    assert _apply_duration_parsing(calls) == []
    assert calls[0]["params"]["additional_chunks"] == 8


def test_extract_json_strips_gemma_thought_blocks():
    raw = (
        "<thought>User wants schedule for today.</thought>"
        '{"action_required": false, "reply": "Here is your day.", "calls": []}'
    )
    data = _extract_json_object(raw)
    assert data is not None
    assert data["reply"] == "Here is your day."
    assert data["calls"] == []


def test_tonight_means_midnight():
    tz = ZoneInfo("America/Los_Angeles")
    base = datetime(2026, 7, 1, 18, 0, tzinfo=tz)
    assert parse_to_iso("tonight", base) == "2026-07-02T00:00:00-07:00"


def test_bare_hour_parses_with_meridiem():
    tz = ZoneInfo("America/Los_Angeles")
    base = datetime(2026, 7, 1, 20, 1, tzinfo=tz)
    assert parse_to_iso("5", base) == "2026-07-01T17:00:00-07:00"


def test_create_event_allows_past_start():
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 7, 1, 20, 1, tzinfo=tz)
    calls = [
        {
            "function": "create_event",
            "params": {
                "name": "dinner",
                "start_time_natural": "5",
                "end_time_natural": "8:30",
            },
        }
    ]
    assert _apply_date_parsing(calls, now) == []
    assert _validate_calls(calls, now) == []


def test_qwen_loose_reply_recovery():
    raw = ':reply: "awww, i\'m always here to help you with that!"'
    data = _normalize_llm_response(_parse_llm_json(raw))
    assert data["reply"] == "awww, i'm always here to help you with that!"
    assert ":reply:" not in data["reply"]
    assert data.get("_parse_failed") is None


def test_reply_prefix_without_colon_stripped():
    data = _normalize_llm_response(
        {"reply": 'reply: "logged that for you!"', "action_required": False, "calls": []}
    )
    assert data["reply"] == "logged that for you!"
    assert not data["reply"].lower().startswith("reply:")


def test_log_work_allows_past_end_now():
    now = datetime(2026, 7, 1, 23, 28, 4, tzinfo=ZoneInfo(TIMEZONE))
    calls = [
        {
            "function": "log_work",
            "params": {
                "task_query": "humira",
                "start_natural": "7 pm",
                "end_natural": "now",
            },
        }
    ]
    assert _apply_date_parsing(calls, now) == []
    assert _validate_calls(calls, now) == []
    assert calls[0]["params"]["end"].startswith("2026-07-01T23:28:04")


def main():
    test_normalize_llm_response_clarification_fields()
    test_normalize_additional_chunks()
    test_normalize_prefers_numeric_over_natural()
    test_normalize_minutes_to_chunks()
    test_natural_only_still_works()
    test_extract_json_strips_gemma_thought_blocks()
    test_tonight_means_midnight()
    test_bare_hour_parses_with_meridiem()
    test_create_event_allows_past_start()
    test_qwen_loose_reply_recovery()
    test_reply_prefix_without_colon_stripped()
    test_log_work_allows_past_end_now()
    print("All inference normalization tests passed.")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
