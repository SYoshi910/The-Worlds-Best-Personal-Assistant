"""Tests for LLM param normalization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import _apply_duration_parsing, _normalize_llm_numeric_params, _normalize_llm_response


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


def main():
    test_normalize_llm_response_clarification_fields()
    test_normalize_additional_chunks()
    test_normalize_prefers_numeric_over_natural()
    test_normalize_minutes_to_chunks()
    test_natural_only_still_works()
    print("All inference normalization tests passed.")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
