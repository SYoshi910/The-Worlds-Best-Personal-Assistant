"""Deterministic natural-language duration → minutes/chunks (LLM never does math)."""

import re

_HOUR_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hr)\b|(\d+(?:\.\d+)?)h\b",
    re.I,
)
_MIN_RE = re.compile(
    r"(\d+)\s*(?:minutes?|mins?|min)\b|(\d+)m\b",
    re.I,
)


def minutes_to_chunks(minutes: int) -> int:
    """Convert minutes to 15-min chunks (rounds up, minimum 1)."""
    return max(1, (minutes + 14) // 15)


def parse_duration_to_minutes(text: str) -> int | None:
    """Parse phrases like '6 hours', '90 min', '2 hrs', 'an hour' into total minutes."""
    if not text or not str(text).strip():
        return None

    t = str(text).lower().strip()

    if re.search(r"\bhalf\s+(?:an?\s+)?hour\b", t):
        return 30

    if re.search(r"\ban?\s+hour\b", t) and not re.search(r"\d", t):
        return 60

    hour_match = _HOUR_RE.search(t)
    if hour_match:
        raw = hour_match.group(1) or hour_match.group(2)
        return int(float(raw) * 60)

    min_match = _MIN_RE.search(t)
    if min_match:
        raw = min_match.group(1) or min_match.group(2)
        return int(raw)

    return None


def parse_duration_to_chunks(text: str) -> int | None:
    minutes = parse_duration_to_minutes(text)
    if minutes is None:
        return None
    return minutes_to_chunks(minutes)
