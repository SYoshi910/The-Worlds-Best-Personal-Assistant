"""Unit tests for model_router quota ledger (no live API)."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIMEZONE
from model_router import (
    MODEL_CHAIN,
    UsageLedger,
    _parse_limit_reason,
    _parse_retry_after_seconds,
)
from zoneinfo import ZoneInfo

TZ = ZoneInfo(TIMEZONE)


def test_ledger_rollover_clears_day_usage():
    ledger = UsageLedger(la_date="2020-01-01", minute_key="2020-01-01T00:00")
    ledger.usage_day["llama-3.3-70b-versatile"] = ledger._day("llama-3.3-70b-versatile")
    ledger.usage_day["llama-3.3-70b-versatile"].tokens = 99_000
    ledger.exhausted_day.add("openai/gpt-oss-120b")
    now = datetime(2026, 7, 1, 10, 0, tzinfo=TZ)
    ledger.rollover_if_needed(now)
    assert ledger.la_date == "2026-07-01"
    assert ledger.usage_day == {}
    assert ledger.exhausted_day == set()


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


if __name__ == "__main__":
    test_ledger_rollover_clears_day_usage()
    test_proactive_threshold_marks_daily_exhausted()
    test_parse_limit_reason()
    test_parse_retry_after()
    print("All model router tests passed.")
