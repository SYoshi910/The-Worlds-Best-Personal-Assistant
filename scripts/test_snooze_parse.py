"""Unit tests for Tier-1 snooze detection/parsing (spec 3/11): intent.is_snooze_request
and intent.parse_snooze_spec. No LLM, no live API — pure string parsing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent import is_snooze_request, parse_snooze_spec


def test_is_snooze_request_matches_verbs():
    for msg in (
        "snooze bcg for 2 hours",
        "postpone orgo until tomorrow",
        "push back the report until friday",
        "hold off on laundry for a bit",
        "delay bcg prep until thursday",
    ):
        assert is_snooze_request(msg), msg


def test_is_snooze_request_rejects_unrelated():
    for msg in (
        "undo",
        "im tired",
        "complete BCG prep",
        "extend by 30 minutes",
        "im snoozing on the couch",  # "snoozing" is not the leading verb "snooze"
    ):
        assert not is_snooze_request(msg), msg


def test_parse_relative_snooze():
    spec = parse_snooze_spec("snooze bcg for 2 hours")
    assert spec == {
        "task_query": "bcg",
        "snooze_until_natural": "in 2 hours",
        "relative": True,
    }


def test_parse_relative_snooze_already_prefixed():
    spec = parse_snooze_spec("snooze bcg for in 90 minutes")
    assert spec is not None
    assert spec["relative"] is True
    assert spec["snooze_until_natural"] == "in 90 minutes"


def test_parse_absolute_snooze():
    spec = parse_snooze_spec("snooze orgo until tomorrow morning")
    assert spec == {
        "task_query": "orgo",
        "snooze_until_natural": "tomorrow morning",
        "relative": False,
    }


def test_parse_absolute_snooze_till_and_to():
    for phrase, expected_until in (
        ("postpone bcg till thursday", "thursday"),
        ("postpone bcg to thursday", "thursday"),
    ):
        spec = parse_snooze_spec(phrase)
        assert spec is not None, phrase
        assert spec["relative"] is False
        assert spec["snooze_until_natural"] == expected_until


def test_parse_bare_snooze_no_target():
    """'snooze BCG' with no for/until — task only, no target (caller decides default)."""
    spec = parse_snooze_spec("snooze BCG prep")
    assert spec is not None
    assert spec["task_query"] == "BCG prep"
    assert spec["snooze_until_natural"] is None
    assert spec["relative"] is None


def test_parse_snooze_strips_trailing_punctuation():
    spec = parse_snooze_spec("snooze bcg for 2 hours.")
    assert spec["task_query"] == "bcg"
    assert spec["snooze_until_natural"] == "in 2 hours"


def test_parse_snooze_not_a_snooze_request_returns_none():
    assert parse_snooze_spec("undo") is None
    assert parse_snooze_spec("complete BCG prep") is None


def test_parse_snooze_empty_task_returns_none():
    # A bare verb with nothing else shouldn't produce a spec with an empty task.
    spec = parse_snooze_spec("snooze")
    assert spec is None


def main():
    test_is_snooze_request_matches_verbs()
    test_is_snooze_request_rejects_unrelated()
    test_parse_relative_snooze()
    test_parse_relative_snooze_already_prefixed()
    test_parse_absolute_snooze()
    test_parse_absolute_snooze_till_and_to()
    test_parse_bare_snooze_no_target()
    test_parse_snooze_strips_trailing_punctuation()
    test_parse_snooze_not_a_snooze_request_returns_none()
    test_parse_snooze_empty_task_returns_none()
    print("All snooze parse tests passed.")


if __name__ == "__main__":
    main()
