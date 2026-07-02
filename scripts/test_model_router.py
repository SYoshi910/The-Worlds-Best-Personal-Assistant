"""Unit tests for model_router quota ledger (no live API)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODEL_CHAIN_IDLE_RESET_SEC, TIMEZONE
from model_router import (
    MODEL_CHAIN,
    UsageLedger,
    _google_answer_text,
    _google_contents,
    _google_system_instruction,
    _parse_limit_reason,
    _parse_retry_after_seconds,
)
from zoneinfo import ZoneInfo

TZ = ZoneInfo(TIMEZONE)


def test_ledger_rollover_clears_day_usage():
    ledger = UsageLedger(
        la_date="2020-01-01",
        minute_key="2020-01-01T00:00",
        active_model_id="qwen/qwen3-32b",
    )
    ledger.usage_day["llama-3.3-70b-versatile"] = ledger._day("llama-3.3-70b-versatile")
    ledger.usage_day["llama-3.3-70b-versatile"].tokens = 99_000
    ledger.exhausted_day.add("openai/gpt-oss-120b")
    now = datetime(2026, 7, 1, 10, 0, tzinfo=TZ)
    ledger.rollover_if_needed(now)
    assert ledger.la_date == "2026-07-01"
    assert ledger.usage_day == {}
    assert ledger.exhausted_day == set()
    assert ledger.active_model_id == ""


def test_chain_start_stays_sticky_during_active_conversation():
    import model_router as mr

    old_ledger = mr._ledger
    old_activity = mr._last_user_activity_at
    try:
        now = datetime.now(TZ)
        qwen_idx = next(i for i, s in enumerate(MODEL_CHAIN) if s.id == "qwen/qwen3-32b")
        mr._ledger = UsageLedger(
            la_date=now.date().isoformat(),
            active_model_id="qwen/qwen3-32b",
        )
        mr._last_user_activity_at = now - timedelta(seconds=30)
        assert mr._chain_start_index(now) == qwen_idx
    finally:
        mr._ledger = old_ledger
        mr._last_user_activity_at = old_activity


def test_chain_start_resets_after_idle_when_primary_available():
    import model_router as mr

    old_ledger = mr._ledger
    old_activity = mr._last_user_activity_at
    try:
        now = datetime.now(TZ)
        mr._ledger = UsageLedger(
            la_date=now.date().isoformat(),
            active_model_id="qwen/qwen3-32b",
        )
        mr._last_user_activity_at = now - timedelta(
            seconds=MODEL_CHAIN_IDLE_RESET_SEC + 1
        )
        assert mr._chain_start_index(now) == 0
    finally:
        mr._ledger = old_ledger
        mr._last_user_activity_at = old_activity


def test_chain_start_stays_on_failover_while_all_higher_exhausted():
    import model_router as mr

    old_ledger = mr._ledger
    old_activity = mr._last_user_activity_at
    try:
        now = datetime.now(TZ)
        ledger = UsageLedger(
            la_date=now.date().isoformat(),
            active_model_id="qwen/qwen3-32b",
        )
        qwen_idx = next(i for i, s in enumerate(MODEL_CHAIN) if s.id == "qwen/qwen3-32b")
        for i in range(qwen_idx):
            ledger.mark_exhausted(MODEL_CHAIN[i], "daily token", now=now)
        mr._ledger = ledger
        mr._last_user_activity_at = now - timedelta(
            seconds=MODEL_CHAIN_IDLE_RESET_SEC + 1
        )
        assert mr._chain_start_index(now) == qwen_idx
    finally:
        mr._ledger = old_ledger
        mr._last_user_activity_at = old_activity


def test_proactive_threshold_marks_daily_exhausted():
    spec = MODEL_CHAIN[0]  # gpt-oss 120b, tpd 200k
    ledger = UsageLedger(la_date=datetime.now(TZ).date().isoformat())
    ledger.usage_day[spec.id] = ledger._day(spec.id)
    ledger.usage_day[spec.id].tokens = int(spec.limits.tpd * 0.96)
    reason = ledger.check_proactive_thresholds(spec)
    assert reason == "daily token"
    assert spec.id in ledger.exhausted_day


def test_parse_limit_reason():
    err = "Rate limit reached for tokens per day (TPD): Limit 100000, Used 97639"
    assert _parse_limit_reason(err) == "daily token"
    assert _parse_limit_reason("requests per minute") == "requests per minute"


def test_parse_retry_after():
    err = "Please try again in 9m26.784s."
    assert _parse_retry_after_seconds(err) == 9 * 60 + 26


class _FakePart:
    def __init__(self, text: str = "", *, thought: bool = False):
        self.text = text
        self.thought = thought


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts):
        self.content = _FakeContent(parts)


class _FakeResponse:
    def __init__(self, candidates):
        self.candidates = candidates


def test_google_answer_text_skips_thought_parts():
    response = _FakeResponse(
        [
            _FakeCandidate(
                [
                    _FakePart("internal reasoning", thought=True),
                    _FakePart('{"reply": "ok", "calls": []}'),
                ]
            )
        ]
    )
    assert _google_answer_text(response) == '{"reply": "ok", "calls": []}'


def test_google_contents_maps_roles():
    contents = _google_contents(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert len(contents) == 2
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "hi"
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "hello"


def test_google_system_instruction_json_suffix():
    text = _google_system_instruction([{"role": "system", "content": "Be helpful."}])
    assert "Be helpful." in text
    assert "single JSON object only" in text


if __name__ == "__main__":
    test_ledger_rollover_clears_day_usage()
    test_chain_start_stays_sticky_during_active_conversation()
    test_chain_start_resets_after_idle_when_primary_available()
    test_chain_start_stays_on_failover_while_all_higher_exhausted()
    test_proactive_threshold_marks_daily_exhausted()
    test_parse_limit_reason()
    test_parse_retry_after()
    test_google_answer_text_skips_thought_parts()
    test_google_contents_maps_roles()
    test_google_system_instruction_json_suffix()
    print("All model router tests passed.")
