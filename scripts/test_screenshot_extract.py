"""Unit tests for screenshot text extraction (no live API)."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LLM_MAX_INPUT_CHARS
from inference import ScreenshotNoTextError, extract_screenshot_text
from model_router import AllModelsExhausted, CompletionResult


def _result(raw: str) -> CompletionResult:
    return CompletionResult(
        raw=raw,
        model_id="gemma-4-26b-a4b-it",
        display_name="Gemma Vision",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )


async def _run(coro):
    return await coro


def test_extract_returns_trimmed_text():
    with patch(
        "inference.complete_google_vision",
        new_callable=AsyncMock,
        return_value=_result("  Submit Q3 report — due Friday  \n"),
    ):
        text = asyncio.run(extract_screenshot_text(b"img", "image/jpeg"))
    assert text == "Submit Q3 report — due Friday"


def test_extract_no_text_raises():
    with patch(
        "inference.complete_google_vision",
        new_callable=AsyncMock,
        return_value=_result("NO_TEXT_FOUND"),
    ):
        try:
            asyncio.run(extract_screenshot_text(b"img", "image/jpeg"))
            assert False, "expected ScreenshotNoTextError"
        except ScreenshotNoTextError:
            pass


def test_extract_caps_long_output():
    long_raw = "x" * (LLM_MAX_INPUT_CHARS + 500)
    with patch(
        "inference.complete_google_vision",
        new_callable=AsyncMock,
        return_value=_result(long_raw),
    ):
        text = asyncio.run(extract_screenshot_text(b"img", "image/jpeg"))
    assert len(text) <= LLM_MAX_INPUT_CHARS


def test_extract_propagates_quota_exhausted():
    with patch(
        "inference.complete_google_vision",
        new_callable=AsyncMock,
        side_effect=AllModelsExhausted("quota"),
    ):
        try:
            asyncio.run(extract_screenshot_text(b"img", "image/jpeg"))
            assert False, "expected AllModelsExhausted"
        except AllModelsExhausted:
            pass


if __name__ == "__main__":
    test_extract_returns_trimmed_text()
    test_extract_no_text_raises()
    test_extract_caps_long_output()
    test_extract_propagates_quota_exhausted()
    print("All screenshot extract tests passed.")
