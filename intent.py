"""Tier-1 safety intercepts and substring matchers for read-only / break paths."""

import difflib
import re

TIER1_UNDO = ("undo", "cancel", "never mind", "nevermind", "wait stop")
TIER1_FUZZY_SINGLES = ("undo", "cancel")
TIER1_FUZZY_PHRASES = ("never mind", "nevermind", "wait stop")
TIER1_FUZZY_WORD_CUTOFF = 0.75
TIER1_FUZZY_PHRASE_CUTOFF = 0.72
TIER1_FUZZY_MAX_WORD_LEN = 10

TAKE_BREAK_PHRASES = (
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
)

_SNOOZE_RE = re.compile(
    r"^\s*(?:snooze|postpone|push\s+back|hold\s+off\s+on|delay)\b",
    re.I,
)
# "snooze X for 2 hours" (relative) vs "snooze X until tomorrow" (absolute).
_SNOOZE_RELATIVE_RE = re.compile(
    r"^\s*(?:snooze|postpone|push\s+back|hold\s+off\s+on|delay)\s+"
    r"(?P<task>.+?)\s+for\s+(?P<amount>.+?)\s*$",
    re.I,
)
_SNOOZE_ABSOLUTE_RE = re.compile(
    r"^\s*(?:snooze|postpone|push\s+back|hold\s+off\s+on|delay)\s+"
    r"(?P<task>.+?)\s+(?:until|til|till|to)\s+(?P<until>.+?)\s*$",
    re.I,
)
_SNOOZE_BARE_RE = re.compile(
    r"^\s*(?:snooze|postpone|push\s+back|hold\s+off\s+on|delay)\s+(?P<task>.+?)\s*$",
    re.I,
)
_BUG_LOG_RE = re.compile(r"^\s*log\s+bug\s*:\s*", re.I)

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


def is_snooze_request(message: str) -> bool:
    """Return True for explicit snooze/postpone requests (Tier-1, no LLM)."""
    return bool(_SNOOZE_RE.match(message or ""))


def is_bug_log_request(message: str) -> bool:
    """Return True when the message starts with 'Log bug:' (Tier-1, no LLM)."""
    return bool(_BUG_LOG_RE.match(message or ""))


def parse_bug_log_body(message: str) -> str:
    """Return text after the 'Log bug:' prefix (may be empty)."""
    m = _BUG_LOG_RE.match(message or "")
    if not m:
        return ""
    return (message or "")[m.end() :].strip()


def parse_snooze_spec(message: str) -> dict | None:
    """Parse 'snooze X for Y' (relative) or 'snooze X until Z' (absolute).

    Returns a dict with the task reference and a natural-language snooze target
    that the caller resolves deterministically (spec 11):
      {"task_query": str, "snooze_until_natural": str, "relative": bool}
    ``snooze_until_natural`` is phrased so ``inference.parse_to_iso`` can parse it
    directly ("in 2 hours" for relative, the raw phrase for absolute). Returns
    None when the message is not a snooze request or has no task reference.
    """
    if not is_snooze_request(message):
        return None

    m = _SNOOZE_ABSOLUTE_RE.match(message)
    if m:
        task = m.group("task").strip().rstrip(".,!")
        until = m.group("until").strip().rstrip(".,!")
        if task and until:
            return {
                "task_query": task,
                "snooze_until_natural": until,
                "relative": False,
            }

    m = _SNOOZE_RELATIVE_RE.match(message)
    if m:
        task = m.group("task").strip().rstrip(".,!")
        amount = m.group("amount").strip().rstrip(".,!")
        if task and amount:
            natural = amount if amount.lower().startswith("in ") else f"in {amount}"
            return {
                "task_query": task,
                "snooze_until_natural": natural,
                "relative": True,
            }

    m = _SNOOZE_BARE_RE.match(message)
    if m:
        task = m.group("task").strip().rstrip(".,!")
        if task:
            return {
                "task_query": task,
                "snooze_until_natural": None,
                "relative": None,
            }

    return None


def is_take_break_request(message: str) -> bool:
    """Substring match for break / clear-evening requests."""
    text = message.lower().strip()
    return any(phrase in text for phrase in TAKE_BREAK_PHRASES)


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
    if is_category_clarification_reply(message):
        return "WORK" if message.strip().lower() in ("work", "w") else "PERSONAL"
    return None


def is_category_clarification_reply(message: str) -> bool:
    """Return True for short work/personal clarification replies."""
    text = message.strip().lower()
    return text in ("work", "personal", "w", "p")


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


def extract_break_window(message: str, now) -> tuple:
    """Return (break_start, break_end) datetimes in local tz."""
    from datetime import datetime, timedelta, time
    from zoneinfo import ZoneInfo

    from config import DEFAULT_EVENING_END, TIMEZONE
    from duration_parser import parse_duration_to_minutes
    from inference import parse_to_iso

    tz = ZoneInfo(TIMEZONE)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    minutes = parse_duration_to_minutes(message)
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
