"""Unit tests for Tier-1 bug log detection and persistence."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bug_log import BUG_STATUS_UNREVIEWED, append_bug_report
from intent import is_bug_log_request, parse_bug_log_body


def test_is_bug_log_request_matches():
    for msg in (
        "Log bug: extend misheard my task",
        "log bug:",
        "  Log bug:  spaced prefix",
        "LOG BUG: caps ok",
    ):
        assert is_bug_log_request(msg), msg


def test_is_bug_log_request_rejects():
    for msg in (
        "please log bug: foo",
        "undo",
        "snooze bcg for 2 hours",
        "im tired",
        "log work on bcg",
    ):
        assert not is_bug_log_request(msg), msg


def test_parse_bug_log_body():
    assert parse_bug_log_body("Log bug: extend misheard") == "extend misheard"
    assert parse_bug_log_body("log bug:") == ""
    assert parse_bug_log_body("  Log bug:  ") == ""
    assert parse_bug_log_body("undo") == ""


def test_append_bug_report_writes_jsonl(tmp_path):
    log_file = tmp_path / "bugs.jsonl"
    report_id = append_bug_report(
        "test body",
        user_id=42,
        raw_message="Log bug: test body",
        context={"current_task_title": "BCG prep"},
        path=log_file,
    )
    assert report_id
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == report_id
    assert record["user_id"] == 42
    assert record["body"] == "test body"
    assert record["raw_message"] == "Log bug: test body"
    assert record["context"] == {"current_task_title": "BCG prep"}
    assert record["status"] == BUG_STATUS_UNREVIEWED
    assert "logged_at" in record


def test_append_bug_report_appends_second_line(tmp_path):
    log_file = tmp_path / "bugs.jsonl"
    append_bug_report("first", user_id=1, raw_message="Log bug: first", path=log_file)
    append_bug_report("second", user_id=1, raw_message="Log bug: second", path=log_file)
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["body"] == "first"
    assert json.loads(lines[1])["body"] == "second"


if __name__ == "__main__":
    import tempfile

    test_is_bug_log_request_matches()
    test_is_bug_log_request_rejects()
    test_parse_bug_log_body()
    with tempfile.TemporaryDirectory() as d:
        test_append_bug_report_writes_jsonl(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_append_bug_report_appends_second_line(Path(d))
    print("All bug log tests passed.")
