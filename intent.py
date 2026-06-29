"""Intent detection: tier-1 substring intercepts and tier-2 difflib exemplars."""

import difflib
import re

TIER1_UNDO = ("undo", "cancel", "never mind", "nevermind", "wait stop")
TIER1_FUZZY_SINGLES = ("undo", "cancel")
TIER1_FUZZY_PHRASES = ("never mind", "nevermind", "wait stop")
TIER1_FUZZY_WORD_CUTOFF = 0.75
TIER1_FUZZY_PHRASE_CUTOFF = 0.72
TIER1_FUZZY_MAX_WORD_LEN = 10

INTENT_EXEMPLARS: dict[str, list[str]] = {
    "read_schedule": [
        "what are all my tasks for the week",
        "what are my tasks for the week",
        "what's on my schedule this week",
        "what do i have due this week",
        "show me my calendar",
        "what's my week look like",
        "what am i doing this week",
        "tasks for the week",
    ],
    "missed_work": [
        "i didn't work on",
        "i didnt work on",
        "didn't work on any",
        "didnt work on any",
        "i skipped",
        "i missed my blocks",
        "didn't get to",
        "didnt get to",
        "never got to",
        "didn't do any",
        "didnt do any",
    ],
    "extend_time": [
        "need 20 more minutes",
        "need more time",
        "running late",
        "give me more time",
        "more minutes",
        "extend this block",
    ],
    "switch_task": [
        "working on another task",
        "actually i'm doing",
        "actually im doing",
        "not working on this",
        "doing something else",
        "switched to",
    ],
    "correction": [
        "actually i wanted",
        "i meant",
        "wait i wanted",
        "not that one",
        "actually can you",
        "actually, can you",
        "can you make that",
        "change that",
        "make that",
        "instead",
        "make it personal",
        "make it work",
        "not friday",
        "not thursday",
    ],
    "take_break": [
        "i'm tired",
        "im tired",
        "can we clear my evening",
        "clear my evening",
        "clear the evening",
        "free up tonight",
        "need the evening off",
        "take a break",
        "need a break",
        "clear tonight",
        "evening off",
    ],
}

AMEND_PHRASES = (
    "actually",
    "change that",
    "make that",
    "can you make",
    "instead",
    "make it personal",
    "make it work",
    "not friday",
    "not thursday",
    "i meant",
    "wait i wanted",
)

MATCH_CUTOFF = 0.55
TIE_MARGIN = 0.1

_MISSED_PREFIXES = re.compile(
    r"^(?:i\s+)?(?:didn't|didnt|never|did not)\s+(?:work on|do|get to)\s*(?:any\s+)?",
    re.I,
)


def _normalize_tier1_text(message: str) -> str:
    text = message.lower().strip()
    return re.sub(r"[^\w\s]", "", text)


def _fuzzy_tier1_undo(text: str) -> bool:
    words = text.split()
    if not words:
        return False

    first = words[0]
    if len(first) <= TIER1_FUZZY_MAX_WORD_LEN and difflib.get_close_matches(
        first, TIER1_FUZZY_SINGLES, n=1, cutoff=TIER1_FUZZY_WORD_CUTOFF
    ):
        return True

    for phrase in TIER1_FUZZY_PHRASES:
        if difflib.SequenceMatcher(None, text, phrase).ratio() >= TIER1_FUZZY_PHRASE_CUTOFF:
            return True

    return False


def is_undo_or_cancel(message: str) -> bool:
    """Return True if the message requests undo or cancel."""
    text = _normalize_tier1_text(message)
    if any(phrase in text for phrase in TIER1_UNDO):
        return True
    return _fuzzy_tier1_undo(text)


def extract_minutes(message: str) -> int | None:
    """Extract a duration in minutes from free text, if present."""
    from duration_parser import parse_duration_to_minutes

    return parse_duration_to_minutes(message)


def extract_switch_task_query(message: str) -> str | None:
    """Extract the new task name from a switch-task message."""
    patterns = [
        r"(?:actually\s+)?(?:i'?m|im)\s+(?:doing|working on)\s+(.+)",
        r"working on\s+(.+?)(?:\s+instead)?\.?$",
        r"switched to\s+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, message.strip(), re.I)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def extract_task_hint(message: str) -> str:
    """Best-effort task name hint from missed-work phrasing."""
    text = message.strip()
    cleaned = _MISSED_PREFIXES.sub("", text).strip()
    cleaned = re.sub(r"\s+today\.?$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(
        r",?\s*(?:can you\s+)?(?:reschedule|move|push).*$", "", cleaned, flags=re.I
    ).strip()
    return cleaned or text


def extract_snooze_hint(message: str) -> str | None:
    """Optional natural-language reschedule target from missed-work messages."""
    patterns = [
        r"\buntil\s+(.+?)(?:\.|$)",
        r"\bto\s+(tomorrow(?:\s+(?:morning|evening|afternoon))?[^,.]*)",
        r"\bfor\s+(tomorrow(?:\s+(?:morning|evening|afternoon))?[^,.]*)",
    ]
    for pat in patterns:
        m = re.search(pat, message.strip(), re.I)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def extract_event_category_from_message(message: str) -> str | None:
    """Detect work/personal when user states category in the same message as the request."""
    text = message.lower()
    if re.search(r"\b(as\s+a\s+)?personal\s+task\b", text):
        return "PERSONAL"
    if re.search(r"\bfor\s+personal\b", text):
        return "PERSONAL"
    if re.search(r"\b(as\s+a\s+)?work\s+task\b", text):
        return "WORK"
    if re.search(r"\bfor\s+work\b", text) and "work on" not in text:
        return "WORK"
    return None


def is_category_clarification_reply(message: str) -> bool:
    """Return True for short work/personal clarification replies."""
    text = message.strip().lower()
    return text in ("work", "personal", "w", "p")


def is_amendment_message(message: str) -> bool:
    """Return True if the message looks like a correction to a recent action."""
    text = message.lower().strip()
    if is_category_clarification_reply(message):
        return False
    return any(phrase in text for phrase in AMEND_PHRASES)


def detect_intent(message: str) -> str | None:
    """Classify user intent via substring match or fuzzy exemplar scoring."""
    text = message.lower().strip()
    if not text:
        return None

    if is_undo_or_cancel(text):
        return "undo"

    if is_category_clarification_reply(message):
        return None

    for intent, exemplars in INTENT_EXEMPLARS.items():
        for ex in exemplars:
            if ex in text:
                return intent

    scores: list[tuple[float, str]] = []
    for intent, exemplars in INTENT_EXEMPLARS.items():
        best = max(difflib.SequenceMatcher(None, text, ex).ratio() for ex in exemplars)
        if best >= MATCH_CUTOFF:
            scores.append((best, intent))

    if not scores:
        return None

    scores.sort(reverse=True)
    if len(scores) > 1 and scores[0][0] - scores[1][0] < TIE_MARGIN:
        return None
    return scores[0][1]


_EVENING_PHRASES = re.compile(
    r"\b(?:clear(?:\s+my)?\s+evening|clear\s+tonight|free\s+up\s+tonight|"
    r"evening\s+off|rest\s+of\s+(?:the\s+)?(?:night|evening)|tonight)\b",
    re.I,
)

_BREAK_CONFIRM = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "do it",
        "go ahead",
        "please",
        "sounds good",
        "let's do it",
        "lets do it",
    }
)

_BREAK_REJECT = frozenset(
    {
        "no",
        "nah",
        "nope",
        "don't",
        "dont",
        "never mind",
        "nevermind",
        "not now",
        "skip it",
    }
)


def is_break_confirmation(message: str) -> bool:
    """Return True if the user confirms a pending break proposal."""
    text = message.lower().strip().rstrip(".!")
    if text in _BREAK_CONFIRM:
        return True
    return any(text.startswith(p + " ") or text == p for p in ("yes", "yeah", "yep"))


def is_break_rejection(message: str) -> bool:
    """Return True if the user declines a pending break proposal."""
    text = message.lower().strip().rstrip(".!")
    if text in _BREAK_REJECT:
        return True
    return text.startswith("no ") or text == "no"


def extract_extend_task_call(message: str) -> list[dict] | None:
    """Thin parser: 'extend bcg by 2 hours' → extend_task_total without LLM."""
    from duration_parser import parse_duration_to_chunks

    text = message.strip()
    if not parse_duration_to_chunks(text):
        return None

    patterns = [
        r"(?:extend|add|give)\s+(?:me\s+)?(?:more\s+time\s+(?:on|for|to)\s+)?(.+?)\s+by\s+.+$",
        r"(?:extend|add)\s+(.+?)\s+by\s+.+$",
        r"(?:extend|add)\s+(.+?)\s+(?:for\s+)?(?:another\s+)?\d",
    ]
    task_query = None
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            task_query = m.group(1).strip().rstrip(".,!")
            break

    if not task_query or len(task_query) < 2:
        return None

    # Don't steal ping-style "extend this block" phrasing
    if re.search(r"\b(?:this|current)\s+block\b", task_query, re.I):
        return None

    chunks = parse_duration_to_chunks(text)
    if not chunks:
        return None

    return [
        {
            "function": "extend_task_total",
            "params": {
                "task_query": task_query,
                "additional_chunks": chunks,
            },
        }
    ]


def extract_break_window(message: str, now) -> tuple:
    """Return (break_start, break_end) datetimes in local tz."""
    from datetime import datetime, timedelta, time
    from zoneinfo import ZoneInfo

    from config import DEFAULT_EVENING_END, TIMEZONE
    from inference import parse_to_iso

    tz = ZoneInfo(TIMEZONE)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    minutes = extract_minutes(message)
    if minutes and not _EVENING_PHRASES.search(message):
        return now, now + timedelta(minutes=minutes)

    if _EVENING_PHRASES.search(message):
        if DEFAULT_EVENING_END == time(0, 0):
            end = datetime.combine(
                now.date() + timedelta(days=1), time(0, 0), tzinfo=tz
            )
        else:
            end = datetime.combine(now.date(), DEFAULT_EVENING_END, tzinfo=tz)
            if end <= now:
                end += timedelta(days=1)
        return now, end

    for pat in (
        r"\buntil\s+(.+?)(?:\.|$)",
        r"\bfor\s+(\d+\s*(?:hours?|hrs?|minutes?|mins?))",
    ):
        m = re.search(pat, message.strip(), re.I)
        if m:
            iso = parse_to_iso(m.group(1).strip(), now)
            if iso:
                end = datetime.fromisoformat(iso).astimezone(tz)
                if end > now:
                    return now, end

    return now, now + timedelta(hours=2)
