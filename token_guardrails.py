"""Input/output caps and conversation buffer limits before LLM calls."""

from config import (
    LLM_MAX_CHARS_PER_MESSAGE,
    LLM_MAX_INPUT_CHARS,
    MAX_CONVERSATION_TURNS,
    SNAPSHOT_MAX_CHARS,
)

_TRUNC_SUFFIX = " (truncated)"


def truncate_text(text: str, max_chars: int, suffix: str = _TRUNC_SUFFIX) -> str:
    """Truncate text to max_chars, appending suffix when clipped."""
    if not text or len(text) <= max_chars:
        return text
    keep = max_chars - len(suffix)
    if keep < 1:
        return text[:max_chars]
    return text[:keep] + suffix


def cap_incoming_text(text: str, max_chars: int = LLM_MAX_INPUT_CHARS) -> str:
    """Cap a single user message before it enters the conversation buffer."""
    return truncate_text(text.strip(), max_chars)


def cap_snapshot(snapshot: str | None, max_chars: int = SNAPSHOT_MAX_CHARS) -> str:
    """Cap schedule snapshot text injected into the LLM prompt."""
    if not snapshot:
        return ""
    return truncate_text(snapshot, max_chars)


def trim_conversation_buffer(
    buffer: list,
    max_turns: int = MAX_CONVERSATION_TURNS,
    max_chars_per_message: int = LLM_MAX_CHARS_PER_MESSAGE,
) -> None:
    """Trim buffer in place to turn and per-message limits."""
    while len(buffer) > max_turns:
        buffer.pop(0)
    for msg in buffer:
        content = msg.get("content")
        if isinstance(content, str) and len(content) > max_chars_per_message:
            msg["content"] = truncate_text(content, max_chars_per_message)


def prepare_messages_for_llm(
    buffer: list,
    max_turns: int = MAX_CONVERSATION_TURNS,
    max_chars_per_message: int = LLM_MAX_CHARS_PER_MESSAGE,
) -> list[dict]:
    """Return a capped copy of the conversation buffer for the LLM API."""
    trimmed = [dict(m) for m in buffer]
    trim_conversation_buffer(trimmed, max_turns=max_turns, max_chars_per_message=max_chars_per_message)
    return trimmed


