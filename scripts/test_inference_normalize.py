"""Tests for LLM param normalization and extend-task fast path."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zoneinfo import ZoneInfo

from config import TIMEZONE
from inference import _apply_duration_parsing, _normalize_llm_numeric_params
from intent import extract_extend_task_call


def test_extract_extend_bcg():
    calls = extract_extend_task_call("extend bcg by 2 hours")
    assert calls is not None
    assert calls[0]["function"] == "extend_task_total"
    assert calls[0]["params"]["task_query"].lower() == "bcg"
    assert calls[0]["params"]["additional_chunks"] == 8


def test_extract_extend_hrs():
    calls = extract_extend_task_call("add orgo prep by 90 mins")
    assert calls is not None
    assert calls[0]["params"]["additional_chunks"] == 6


def test_normalize_additional_chunks():
    calls = [
        {
            "function": "extend_task_total",
            "params": {"task_query": "bcg", "additional_chunks": 8},
        }
    ]
    assert _normalize_llm_numeric_params(calls) == []
    assert calls[0]["params"]["additional_chunks"] == 8
    assert "additional_time_natural" not in calls[0]["params"]


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
    test_extract_extend_bcg()
    test_extract_extend_hrs()
    test_normalize_additional_chunks()
    test_normalize_prefers_numeric_over_natural()
    test_normalize_minutes_to_chunks()
    test_natural_only_still_works()
    print("All inference/extend routing tests passed.")


if __name__ == "__main__":
    main()
