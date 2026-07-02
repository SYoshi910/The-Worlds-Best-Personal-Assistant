"""Rigorous tests for task resolution: difflib keyword matches and BGE-M3 intent matches."""

import asyncio
import difflib
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cal_helper
from cal_helper import (
    EMBEDDING_AMBIGUITY_MARGIN,
    EMBEDDING_MATCH_THRESHOLD,
    MIN_EMBEDDING_QUERY_WORDS,
    get_task_by_query,
)

# Real active task titles from Reclaim (2026-07-01 snapshot).
REAL_TASKS = {
    13221893: "startup of the week pitch",
    13221901: "BCG prep",
    13221902: "YC startup school app",
    13221903: "daily briefing",
    13221904: "finalize data infra pres",
    13221905: "TCA Notion work",
    13221906: "research positron competitors",
    13221907: "daily brief work",
    13221908: "research polymarket",
}

TASK_CACHE = {
    tid: {"id": tid, "title": title, "status": "IN_PROGRESS"}
    for tid, title in REAL_TASKS.items()
}


def _reset_task_state() -> None:
    cal_helper.TASK_MAP = dict(REAL_TASKS)
    cal_helper.TASK_CACHE = dict(TASK_CACHE)
    cal_helper._embeddings.clear()
    cal_helper._embedded_as.clear()


async def _resolve(query: str, *, force_embedding: bool = False) -> dict | None:
    _reset_task_state()
    await cal_helper._sync_embeddings()
    with ExitStack() as stack:
        stack.enter_context(patch("cal_helper.build_task_map", new=AsyncMock()))
        stack.enter_context(
            patch(
                "cal_helper.reclaim.get_task",
                new=AsyncMock(side_effect=lambda tid: TASK_CACHE[tid]),
            )
        )
        if force_embedding:
            stack.enter_context(patch("cal_helper.difflib.get_close_matches", return_value=[]))
        return await get_task_by_query(query)


def _title(task: dict | None) -> str | None:
    return task.get("title") if task else None


# --- difflib path: explicit keyword / near-exact matches ---

DIFFLIB_CASES = [
    ("bcg prep", "BCG prep"),
    ("BCG prep", "BCG prep"),
    ("daily briefing", "daily briefing"),
    ("YC startup school app", "YC startup school app"),
    ("finalize data infra pres", "finalize data infra pres"),
    ("TCA Notion work", "TCA Notion work"),
    ("research positron competitors", "research positron competitors"),
    ("research polymarket", "research polymarket"),
    ("startup of the week pitch", "startup of the week pitch"),
    ("data infra pres", "finalize data infra pres"),
    ("positron competitors", "research positron competitors"),
]

# --- embedding path: intent phrasing difflib won't catch ---

EMBEDDING_CASES = [
    ("bcg interview", "BCG prep"),
    ("bcg case prep", "BCG prep"),
    ("positron competitor analysis", "research positron competitors"),
    ("polymarket prediction markets", "research polymarket"),
    ("notion TCA tasks", "TCA Notion work"),
    ("data infra presentation", "finalize data infra pres"),
    ("pitch for startup of the week", "startup of the week pitch"),
    ("YC application", "YC startup school app"),
    ("morning daily brief", "daily briefing"),
]

# --- must NOT resolve: vague or adjacent-task collisions ---

REJECT_CASES = [
    "research",
    "daily brief",
    "startup",
    "briefing",
    "notion",
    "competitors",
    "something completely unrelated",
    "quantum widget prep",
]


async def test_difflib_keyword_matches():
    failures = []
    for query, expected_title in DIFFLIB_CASES:
        task = await _resolve(query, force_embedding=False)
        got = _title(task)
        if got != expected_title:
            failures.append(f"  {query!r}: expected {expected_title!r}, got {got!r}")
        # Confirm difflib would have matched without embeddings.
        matches = difflib.get_close_matches(query, list(REAL_TASKS.values()), n=1, cutoff=0.6)
        if not matches:
            failures.append(f"  {query!r}: expected difflib hit, but cutoff miss")
    if failures:
        raise AssertionError("difflib keyword failures:\n" + "\n".join(failures))


async def test_embedding_intent_matches():
    failures = []
    for query, expected_title in EMBEDDING_CASES:
        task = await _resolve(query, force_embedding=True)
        got = _title(task)
        if got != expected_title:
            failures.append(f"  {query!r}: expected {expected_title!r}, got {got!r}")
    if failures:
        raise AssertionError("embedding intent failures:\n" + "\n".join(failures))


async def test_rejects_ambiguous_or_unrelated():
    failures = []
    for query in REJECT_CASES:
        task = await _resolve(query, force_embedding=True)
        if task is not None:
            failures.append(f"  {query!r}: should reject, got {_title(task)!r}")
    if failures:
        raise AssertionError("false-positive failures:\n" + "\n".join(failures))


async def _resolve_with_ledger(
    query: str, preferred_task_ids: list[int], *, force_embedding: bool = False
) -> dict | None:
    _reset_task_state()
    await cal_helper._sync_embeddings()
    with ExitStack() as stack:
        stack.enter_context(patch("cal_helper.build_task_map", new=AsyncMock()))
        stack.enter_context(
            patch(
                "cal_helper.reclaim.get_task",
                new=AsyncMock(side_effect=lambda tid: TASK_CACHE[tid]),
            )
        )
        if force_embedding:
            stack.enter_context(patch("cal_helper.difflib.get_close_matches", return_value=[]))
        return await get_task_by_query(query, preferred_task_ids=preferred_task_ids)


async def test_ledger_does_not_override_explicit_title():
    """Bug 21: explicit 'daily briefing' must not resolve to unrelated ledger task."""
    yc_id = 13221902
    task = await _resolve_with_ledger("daily briefing", [yc_id])
    assert _title(task) == "daily briefing"


async def test_ledger_resolves_vague_reference():
    """Vague 'it' with a single recent-action task should use the ledger."""
    yc_id = 13221902
    task = await _resolve_with_ledger("it", [yc_id], force_embedding=True)
    assert _title(task) == "YC startup school app"


async def test_ledger_resolves_overlap_with_recent_action():
    """Ledger may win when the query overlaps the recent task title."""
    yc_id = 13221902
    task = await _resolve_with_ledger("YC startup school", [yc_id], force_embedding=True)
    assert _title(task) == "YC startup school app"


def test_adjacent_tasks_exist_in_fixture():
    """Sanity: fixture includes known collision pairs for reject-case coverage."""
    titles = set(REAL_TASKS.values())
    assert "daily briefing" in titles and "daily brief work" in titles
    assert "research positron competitors" in titles and "research polymarket" in titles


def test_threshold_constants_sane():
    assert 0.5 <= EMBEDDING_MATCH_THRESHOLD <= 0.85, EMBEDDING_MATCH_THRESHOLD
    assert 0.03 <= EMBEDDING_AMBIGUITY_MARGIN <= 0.15, EMBEDDING_AMBIGUITY_MARGIN
    assert MIN_EMBEDDING_QUERY_WORDS >= 2


async def main():
    print(f"Threshold={EMBEDDING_MATCH_THRESHOLD}, margin={EMBEDDING_AMBIGUITY_MARGIN}")
    print("Loading BGE-M3 and syncing embeddings...")
    test_threshold_constants_sane()
    test_adjacent_tasks_exist_in_fixture()
    await test_difflib_keyword_matches()
    print(f"  difflib: {len(DIFFLIB_CASES)} cases passed")
    await test_embedding_intent_matches()
    print(f"  embedding: {len(EMBEDDING_CASES)} cases passed")
    await test_rejects_ambiguous_or_unrelated()
    print(f"  reject: {len(REJECT_CASES)} cases passed")
    await test_ledger_does_not_override_explicit_title()
    await test_ledger_resolves_vague_reference()
    await test_ledger_resolves_overlap_with_recent_action()
    print("  ledger: 3 cases passed")
    print("All task resolution tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
