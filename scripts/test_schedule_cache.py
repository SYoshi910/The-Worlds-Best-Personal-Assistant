"""
Live integration tests for schedule_cache against real Reclaim events.

Read-only: fetches events and validates the warm index. No calendar writes.

Usage (from project root):
  python scripts/test_schedule_cache.py
  python scripts/test_schedule_cache.py -v
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reclaim
import schedule_cache
from config import TIMEZONE
from zoneinfo import ZoneInfo

TZ = ZoneInfo(TIMEZONE)
_passed = 0
_failed = 0
_verbose = False


def _safe_print(s: str) -> None:
    print(s.encode("ascii", errors="replace").decode("ascii"))


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        if _verbose:
            _safe_print(f"  PASS  {name}")
    else:
        _failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        _safe_print(msg)


def _expected_task_blocks(events: list[dict]) -> dict[str, dict]:
    """Ground truth: task-assignment events with an eventId."""
    now = datetime.now(timezone.utc)
    blocks: dict[str, dict] = {}
    for event in events:
        if not reclaim.is_task_assignment(event):
            continue
        event_id = event.get("eventId")
        if not event_id:
            continue
        start = datetime.fromisoformat(event["eventStart"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(
            event.get("eventEnd", event["eventStart"]).replace("Z", "+00:00")
        )
        assist = event.get("assist") or {}
        blocks[event_id] = {
            "event_id": event_id,
            "title": event.get("title"),
            "start": start,
            "end": end,
            "task_id": assist.get("taskId"),
            "task_index": assist.get("taskIndex"),
            "is_ongoing": start <= now < end,
        }
    return blocks


def test_index_matches_live_events(expected: dict[str, dict]) -> None:
    cached = {b["event_id"]: b for b in schedule_cache.chronological()}
    _check(
        "cache size matches live task blocks",
        len(cached) == len(expected),
        f"cache={len(cached)} expected={len(expected)}",
    )
    _check(
        "cache keys match live event ids",
        set(cached.keys()) == set(expected.keys()),
        f"missing={set(expected) - set(cached)} extra={set(cached) - set(expected)}",
    )
    for event_id, exp in expected.items():
        got = cached.get(event_id)
        if not got:
            continue
        _check(
            f"title[{event_id[:8]}…]",
            got["title"] == exp["title"],
            f"got={got['title']!r} exp={exp['title']!r}",
        )
        _check(
            f"start[{event_id[:8]}…]",
            got["start"] == exp["start"],
            f"got={got['start']} exp={exp['start']}",
        )
        _check(
            f"task_id[{event_id[:8]}…]",
            got["task_id"] == exp["task_id"],
            f"got={got['task_id']} exp={exp['task_id']}",
        )
        _check(
            f"is_ongoing[{event_id[:8]}…]",
            got["is_ongoing"] == exp["is_ongoing"],
            f"got={got['is_ongoing']} exp={exp['is_ongoing']}",
        )


def test_chronological_order() -> None:
    blocks = schedule_cache.chronological()
    if len(blocks) < 2:
        _check("chronological order (skipped — <2 blocks)", True)
        return
    starts = [b["start"] for b in blocks]
    _check(
        "chronological order ascending",
        starts == sorted(starts),
        f"first={starts[0]} last={starts[-1]}",
    )


def test_get_roundtrip(expected: dict[str, dict]) -> None:
    if not expected:
        _check("get() roundtrip (skipped — no blocks)", True)
        return
    sample_id = next(iter(expected))
    got = schedule_cache.get(sample_id)
    _check("get() returns block for known id", got is not None)
    _check("get() unknown id returns None", schedule_cache.get("__nonexistent__") is None)
    if got:
        _check("get() title matches", got["title"] == expected[sample_id]["title"])


def _blocks_for_task(task_id, day=None) -> list[dict]:
    """Local re-implementation of the retired schedule_cache.blocks_for_task."""
    result = []
    for block in schedule_cache.chronological():
        if block["task_id"] != task_id:
            continue
        if day is not None and block["start"].astimezone(TZ).date() != day:
            continue
        result.append(block)
    return result


def _find_by_title(query: str) -> list[dict]:
    """Local re-implementation of the retired schedule_cache.find_by_title."""
    q = (query or "").strip().lower()
    if not q:
        return []
    return [b for b in schedule_cache.chronological() if q in (b["title"] or "").lower()]


def test_blocks_for_task(expected: dict[str, dict]) -> None:
    if not expected:
        _check("blocks_for_task (skipped — no blocks)", True)
        return
    sample = next(iter(expected.values()))
    task_id = sample["task_id"]
    from_cache = _blocks_for_task(task_id)
    from_expected = sorted(
        [b for b in expected.values() if b["task_id"] == task_id],
        key=lambda b: b["start"],
    )
    _check(
        f"blocks_for_task({task_id}) count",
        len(from_cache) == len(from_expected),
        f"cache={len(from_cache)} expected={len(from_expected)}",
    )
    if from_cache and from_expected:
        _check(
            "blocks_for_task first start matches",
            from_cache[0]["start"] == from_expected[0]["start"],
        )

    today = datetime.now(TZ).date()
    today_blocks = _blocks_for_task(task_id, day=today)
    for block in today_blocks:
        local_day = block["start"].astimezone(TZ).date()
        _check(
            f"blocks_for_task day filter ({local_day})",
            local_day == today,
            f"block on {local_day} slipped into today filter",
        )


def test_find_by_title(expected: dict[str, dict]) -> None:
    _check("find_by_title empty query", _find_by_title("") == [])
    _check("find_by_title whitespace", _find_by_title("   ") == [])
    if not expected:
        _check("find_by_title (skipped — no blocks)", True)
        return
    sample = next(iter(expected.values()))
    title = sample["title"] or ""
    if len(title) < 3:
        _check("find_by_title (skipped — title too short)", True)
        return
    needle = title[: max(3, len(title) // 2)].lower()
    hits = _find_by_title(needle)
    _check(
        f"find_by_title({needle!r}) finds sample",
        any(h["event_id"] == sample["event_id"] for h in hits),
        f"hits={len(hits)}",
    )
    hit_starts = [h["start"] for h in hits]
    _check(
        "find_by_title results chronological",
        hit_starts == sorted(hit_starts),
    )


def test_current_and_previous(expected: dict[str, dict]) -> None:
    now = datetime.now(timezone.utc)
    current, previous = schedule_cache.current_and_previous(now)
    if not expected:
        _check("current_and_previous empty cache", current is None and previous is None)
        return

    ordered = sorted(expected.values(), key=lambda b: b["start"])
    ongoing = [b for b in ordered if b["start"] <= now < b["end"]]

    if ongoing:
        _check("current is ongoing block", current is not None)
        if current:
            _check(
                "current spans now",
                current["start"] <= now < current["end"],
                f"{current['title']} {current['start']}–{current['end']}",
            )
    else:
        started = [b for b in ordered if b["start"] <= now]
        if started:
            _check("current is most recent started block", current is not None)
            if current:
                _check(
                    "current start <= now",
                    current["start"] <= now,
                    f"start={current['start']} now={now}",
                )
        else:
            _check("no started blocks — current is None", current is None)

    if current and previous:
        _check(
            "previous has different task_id",
            previous["task_id"] != current["task_id"],
            f"both task_id={current['task_id']}",
        )
        _check(
            "previous starts before current",
            previous["start"] < current["start"],
            f"prev={previous['start']} cur={current['start']}",
        )
    elif current and not previous:
        _check(
            "no previous when only one task in history",
            all(b["task_id"] == current["task_id"] for b in ordered if b["start"] <= current["start"]),
        )


def _print_summary(expected: dict[str, dict]) -> None:
    blocks = schedule_cache.chronological()
    now = datetime.now(timezone.utc)
    current, previous = schedule_cache.current_and_previous(now)
    ongoing = [b for b in blocks if b["is_ongoing"]]

    _safe_print("")
    _safe_print("--- Live schedule cache snapshot ---")
    _safe_print(f"  Total Reclaim events fetched: (see refresh log)")
    _safe_print(f"  Task blocks indexed:          {len(blocks)}")
    _safe_print(f"  Ongoing now:                  {len(ongoing)}")
    if current:
        _safe_print(
            f"  Current:  {current['title']} "
            f"({current['start'].astimezone(TZ).strftime('%Y-%m-%d %H:%M')} – "
            f"{current['end'].astimezone(TZ).strftime('%H:%M')} local)"
        )
    else:
        _safe_print("  Current:  (none)")
    if previous:
        _safe_print(
            f"  Previous: {previous['title']} "
            f"(task_id={previous['task_id']})"
        )
    else:
        _safe_print("  Previous: (none)")
    if blocks:
        _safe_print("  Upcoming (next 3):")
        shown = 0
        for b in blocks:
            if b["start"] > now:
                _safe_print(
                    f"    - {b['title']} @ "
                    f"{b['start'].astimezone(TZ).strftime('%Y-%m-%d %H:%M')} local"
                )
                shown += 1
                if shown >= 3:
                    break
        if shown == 0:
            _safe_print("    (none)")


async def run_tests(verbose: bool) -> int:
    global _verbose
    _verbose = verbose

    _safe_print("[...] Fetching live Reclaim events...")
    events = await reclaim.get_all_events(force_refresh=True)
    _safe_print(f"   {len(events)} events total")

    non_task = sum(1 for e in events if not reclaim.is_task_assignment(e))
    task_assignments = sum(1 for e in events if reclaim.is_task_assignment(e))
    _safe_print(f"   {task_assignments} task assignments, {non_task} non-task events")

    _safe_print("[...] Refreshing schedule_cache...")
    await schedule_cache.refresh()

    expected = _expected_task_blocks(events)
    _safe_print(f"   Expected {len(expected)} indexed blocks")

    test_index_matches_live_events(expected)
    test_chronological_order()
    test_get_roundtrip(expected)
    test_blocks_for_task(expected)
    test_find_by_title(expected)
    test_current_and_previous(expected)

    _print_summary(expected)

    _safe_print("")
    _safe_print(f"Results: {_passed} passed, {_failed} failed")
    return 0 if _failed == 0 else 1


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Live schedule_cache integration tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each assertion")
    args = parser.parse_args()
    code = asyncio.run(run_tests(args.verbose))
    if code == 0:
        _safe_print("All schedule_cache live tests passed.")
    else:
        _safe_print("Some schedule_cache live tests FAILED.")
    sys.exit(code)


if __name__ == "__main__":
    main()
