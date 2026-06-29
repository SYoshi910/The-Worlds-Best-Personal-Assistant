"""Unit tests for duration_parser abbreviations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from duration_parser import parse_duration_to_minutes

_CASES = {
    "20 min": 20,
    "20 mins": 20,
    "20 minutes": 20,
    "90m": 90,
    "2 hr": 120,
    "2 hrs": 120,
    "2 hours": 120,
    "2h": 120,
    "1.5 hrs": 90,
    "an hour": 60,
    "half hour": 30,
    "need 45 mins more": 45,
    "take a 3 hr break": 180,
}


def main():
    failed = []
    for phrase, expected in _CASES.items():
        got = parse_duration_to_minutes(phrase)
        if got != expected:
            failed.append(f"{phrase!r}: expected {expected}, got {got}")
    if failed:
        raise AssertionError("\n".join(failed))
    print(f"All {len(_CASES)} duration_parser tests passed.")


if __name__ == "__main__":
    main()
