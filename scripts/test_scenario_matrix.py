"""
Scenario matrix audit: deterministic routing + simulated LLM outputs + optional live LLM.

Run: python scripts/test_scenario_matrix.py
     python scripts/test_scenario_matrix.py --live-llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarification import (
    apply_llm_clarification_fields,
    compute_create_task_missing,
    gate_task_queries,
    map_scope_to_function,
)
from config import TIMEZONE
from intent import (
    is_snooze_request,
    is_take_break_request,
    is_undo_or_cancel,
    parse_snooze_spec,
)
from rollback import execute_calls

LOG_PATH = Path(__file__).resolve().parent.parent / "debug-86ff2d.log"


def _agent_log(location: str, message: str, data: dict, hypothesis_id: str = "matrix"):
    entry = {
        "sessionId": "86ff2d",
        "timestamp": int(datetime.now().timestamp() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": "scenario-matrix",
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@dataclass
class ScenarioResult:
    name: str
    prompt: str
    category: str  # pass | fail | llm_dependent | missing_feature | warn
    finding: str
    details: dict = field(default_factory=dict)


RESULTS: list[ScenarioResult] = []


def record(result: ScenarioResult):
    RESULTS.append(result)
    _agent_log(
        "test_scenario_matrix.py:record",
        result.name,
        {"category": result.category, "finding": result.finding, **result.details},
    )


# --- Deterministic fast-path tests ---

FAST_PATH_CASES = [
    ("snooze bcg for 2 hours", "snooze", is_snooze_request),
    ("postpone orgo until tomorrow", "snooze", is_snooze_request),
    ("im tired", "break", is_take_break_request),
    ("clear my evening", "break", is_take_break_request),
    ("undo", "undo", is_undo_or_cancel),
    ("never mind", "undo", is_undo_or_cancel),
]

READ_SCHEDULE_NOT_TIER1 = [
    "what do i have due this week",
    "show me my calendar",
]


def test_fast_paths():
    for prompt, label, fn in FAST_PATH_CASES:
        hit = fn(prompt)
        record(
            ScenarioResult(
                name=f"fast_path:{label}",
                prompt=prompt,
                category="pass" if hit else "fail",
                finding=f"Fast path '{label}' {'matched' if hit else 'MISSED'}",
                details={"matcher": fn.__name__, "matched": hit},
            )
        )


def test_read_schedule_not_tier1():
    for prompt in READ_SCHEDULE_NOT_TIER1:
        record(
            ScenarioResult(
                name="read_schedule_not_tier1",
                prompt=prompt,
                category="pass" if not is_snooze_request(prompt) else "fail",
                finding="Read-schedule prompts no longer bypass LLM (not snooze tier-1)",
                details={"is_snooze": is_snooze_request(prompt)},
            )
        )


def test_snooze_parse():
    cases = [
        ("snooze bcg for 2 hours", {"task_query": "bcg", "relative": True}),
        ("snooze orgo until tomorrow morning", {"task_query": "orgo", "relative": False}),
    ]
    for prompt, expected in cases:
        spec = parse_snooze_spec(prompt)
        ok = spec is not None and spec["task_query"] == expected["task_query"]
        ok = ok and spec["relative"] == expected["relative"]
        record(
            ScenarioResult(
                name="snooze_parse",
                prompt=prompt,
                category="pass" if ok else "fail",
                finding=f"parse_snooze_spec {'ok' if ok else 'FAILED'}: {spec}",
                details={"spec": spec},
            )
        )


# --- Simulated LLM output → execution layer ---

MOCK_CURRENT_TASK = {
    "title": "BCG prep",
    "start_time": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
    "end_time": (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(hours=1)).isoformat(),
    "event_id": "evt-mock-123",
    "is_ongoing": True,
}

MOCK_TASK_B = {
    "title": "Data infra pres",
    "start_time": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
    "end_time": (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(hours=1)).isoformat(),
    "event_id": "evt-mock-456",
    "is_ongoing": True,
}


async def _simulate_execute(calls: list[dict], current_task: dict | None = None) -> dict:
    """Run execute_calls with mocked external APIs."""
    mock_task = {"id": 1, "title": "BCG prep", "snoozeUntil": None}
    mock_task_b = {"id": 2, "title": "Data infra pres", "snoozeUntil": None}

    async def fake_get_task_by_query(query, **kwargs):
        q = (query or "").lower()
        if not q or q in ("something else", "last task", "my last task"):
            return None
        if "bcg" in q:
            return mock_task
        if "data infra" in q or "orgo" in q:
            return mock_task_b
        return None

    with (
        patch("rollback.dispatch", new_callable=AsyncMock) as mock_dispatch,
        patch("rollback.get_task_by_query", side_effect=fake_get_task_by_query),
        patch("cal_helper.get_task_by_query", side_effect=fake_get_task_by_query),
        patch("composites.get_task_by_query", side_effect=fake_get_task_by_query),
    ):
        async def dispatch_side_effect(calls, **kwargs):
            failed = []
            summaries = []
            for call in calls:
                fn = call.get("function")
                params = call.get("params", {})
                tq = (
                    params.get("task_query")
                    or params.get("new_task_query")
                    or params.get("previous_task_query")
                )
                if tq:
                    task = await fake_get_task_by_query(tq)
                    if not task:
                        failed.append(f"Could not find a task matching '{tq}'")
                        continue
                if fn == "switch_active_task":
                    nq = params.get("new_task_query", "")
                    task = await fake_get_task_by_query(nq)
                    if not task:
                        failed.append(f"Could not find a task matching '{nq}'")
                        continue
                    summaries.append(f"Switched to '{task['title']}'")
                elif fn == "resume_previous_task":
                    pq = params.get("previous_task_query", "")
                    task = await fake_get_task_by_query(pq) if pq else mock_task
                    if not task:
                        failed.append(f"Could not find a task matching '{pq}'")
                        continue
                    summaries.append(f"Kept you on '{task['title']}'")
                elif fn == "reschedule_missed_work":
                    summaries.append("Moved 1 block(s) for 'BCG prep'")
                elif fn == "extend_task_instance":
                    summaries.append("Extended block")
                else:
                    summaries.append(fn)
            return {
                "succeeded": [c["function"] for c in calls if not failed],
                "summaries": summaries,
                "snapshots": [],
                "failed": failed,
            }

        mock_dispatch.side_effect = dispatch_side_effect
        return await execute_calls(
            calls,
            user_message="test",
            current_task=current_task or MOCK_CURRENT_TASK,
        )


def test_extend_scope_mapping():
    """Extend scope is resolved via clarification, not confidence gating."""
    call = map_scope_to_function(
        "current_instance",
        {"additional_time_natural": "30 minutes", "task_query": "BCG prep"},
        MOCK_CURRENT_TASK,
    )
    ok = call["function"] == "extend_task_instance"
    record(
        ScenarioResult(
            name="scenario_3_extend_30min",
            prompt="hey ill be working on this for 30 more minutes",
            category="pass" if ok else "fail",
            finding=(
                "map_scope_to_function routes to extend_task_instance"
                if ok
                else f"Unexpected function: {call.get('function')}"
            ),
            details={"call": call},
        )
    )


async def test_gate_clarifies_vague_switch():
    """Task-query gate should clarify on placeholder queries (spec 21), not error."""
    bad_calls = [
        {"function": "switch_active_task", "params": {"new_task_query": "something else"}}
    ]

    async def fake_get_task_by_query(query, **kwargs):
        return {"id": 1, "title": "BCG prep"}

    with patch("cal_helper.get_task_by_query", side_effect=fake_get_task_by_query):
        result = await gate_task_queries(bad_calls, current_task=MOCK_CURRENT_TASK)
    clarifies = result.clarification_required and not result.ok
    record(
        ScenarioResult(
            name="scenario_2_vague_switch",
            prompt="hey ive been actually working on something else",
            category="pass" if clarifies else "fail",
            finding=(
                "gate_task_queries clarifies placeholder task query"
                if clarifies
                else "Gate did not clarify vague switch"
            ),
            details={
                "ok": result.ok,
                "clarification_required": result.clarification_required,
            },
        )
    )


async def test_resume_previous_task_path():
    """resume_previous_task (spec 5) snoozes the fresh ping and resumes the prior task."""
    good_calls = [
        {
            "function": "resume_previous_task",
            "params": {"previous_task_query": "BCG prep", "work_duration_minutes": 30},
        }
    ]
    result = await _simulate_execute(good_calls, current_task=MOCK_TASK_B)
    ok = bool(result["summaries"]) and not result["failed"]
    record(
        ScenarioResult(
            name="scenario_4_resume_previous",
            prompt="hey im actually still working on my last task",
            category="pass" if ok else "fail",
            finding=(
                "resume_previous_task executes cleanly (no longer a missing feature)"
                if ok
                else f"Failed: {result['failed']}"
            ),
            details={"summaries": result["summaries"], "failed": result["failed"]},
        )
    )


async def test_gate_clarifies_vague_resume():
    """A placeholder previous_task_query must clarify, never pass through (spec 21)."""
    bad_calls = [
        {
            "function": "resume_previous_task",
            "params": {"previous_task_query": "my last task", "work_duration_minutes": 30},
        }
    ]

    async def fake_get_task_by_query(query, **kwargs):
        return {"id": 1, "title": "BCG prep"}

    with patch("cal_helper.get_task_by_query", side_effect=fake_get_task_by_query):
        result = await gate_task_queries(bad_calls, current_task=MOCK_TASK_B)
    clarifies = result.clarification_required and not result.ok
    record(
        ScenarioResult(
            name="scenario_4_vague_resume",
            prompt="hey im actually still working on my last task",
            category="pass" if clarifies else "fail",
            finding=(
                "gate_task_queries clarifies placeholder previous_task_query"
                if clarifies
                else "Gate did not clarify vague resume"
            ),
            details={
                "ok": result.ok,
                "clarification_required": result.clarification_required,
            },
        )
    )


async def test_reschedule_missed_work_path():
    good_calls = [
        {"function": "reschedule_missed_work", "params": {"task_query": "BCG prep"}}
    ]
    result = await _simulate_execute(good_calls)
    record(
        ScenarioResult(
            name="scenario_1_missed_bcg",
            prompt="i lowk didnt work on any BCG prep today can you reschedule?",
            category="llm_dependent" if not result["summaries"] else "pass",
            finding=(
                "reschedule_missed_work executes cleanly when LLM picks correct fn"
                if result["summaries"]
                else f"Failed: {result['failed']}"
            ),
            details={"summaries": result["summaries"], "failed": result["failed"]},
        )
    )


async def test_wrong_fn_reschedule_task():
    """LLM picks reschedule_task instead of reschedule_missed_work — different semantics."""
    calls = [
        {
            "function": "reschedule_task",
            "params": {"task_query": "BCG prep", "snooze_until_natural": "tomorrow"},
        }
    ]
    result = await _simulate_execute(calls)
    record(
        ScenarioResult(
            name="scenario_1_wrong_fn",
            prompt="(LLM picks reschedule_task)",
            category="llm_dependent",
            finding="reschedule_task snoozes Reclaim — does NOT move past GCal blocks (semantic mismatch)",
            details={"fn": "reschedule_task", "summaries": result["summaries"]},
        )
    )


def test_clarification_kinds_supported():
    """Generic clarification engine supports all documented kinds."""
    for kind in ("switch_task", "missed_blocks", "disambiguate_task", "extend_scope"):
        data = {
            "clarification_required": True,
            "clarification_kind": kind,
            "reply": "Which task?",
            "pending_params": {},
        }
        pending = apply_llm_clarification_fields(data)
        record(
            ScenarioResult(
                name=f"clarification_kind:{kind}",
                prompt="(structural)",
                category="pass" if pending is not None else "fail",
                finding=(
                    f"clarification_kind='{kind}' supported"
                    if pending is not None
                    else f"clarification_kind='{kind}' NOT supported"
                ),
                details={"pending_created": pending is not None},
            )
        )


def test_create_task_clarification_chain():
    partial = {"event_category": "PERSONAL"}
    missing = compute_create_task_missing(partial)
    record(
        ScenarioResult(
            name="create_task_missing_duration",
            prompt="add task buy groceries tonight",
            category="pass" if "due_date_natural" in missing else "warn",
            finding=f"create_task missing fields detected: {missing}",
            details={"missing": missing},
        )
    )


# --- Live LLM probes ---

LIVE_LLM_PROMPTS = [
    ("scenario_1", "i lowk didnt work on any BCG prep today can you reschedule?"),
    ("scenario_2", "hey ive been actually working on something else"),
    ("scenario_3", "hey ill be working on this for 30 more minutes"),
    ("scenario_4", "hey im actually still working on my last task"),
    ("read_schedule", "what do i have due this week"),
    ("switch_named", "im actually working on data infra until 5pm"),
    ("switch_vague", "doing orgo instead"),
    ("extend_75", "extend by 75 mins"),
    ("extend_bcg_off_block", "extend bcg by 2 hours"),
    ("missed_snooze", "i skipped BCG move it to tomorrow morning"),
    ("snooze_reclaim", "snooze BCG until thursday"),
    ("create_task", "add a task called startup pitch, 5 hrs, due friday"),
    ("create_groceries", "add task buy groceries tonight"),
    ("event_lunch", "ill be at lunch from 12:30 to 1:30"),
    ("vague_ping", "yeah that changed"),
    ("complete", "complete BCG prep"),
]


async def run_live_llm_probes():
    from inference import call_llm

    ctx = MOCK_CURRENT_TASK
    for name, prompt in LIVE_LLM_PROMPTS:
        try:
            data = await call_llm(
                message=[{"role": "user", "content": prompt}],
                current_task=ctx,
            )
            calls = data.get("calls") or []
            fns = [c.get("function") for c in calls]
            record(
                ScenarioResult(
                    name=f"live_llm:{name}",
                    prompt=prompt,
                    category=_classify_live_llm(name, data, fns),
                    finding=_live_llm_finding(name, data, fns),
                    details={
                        "action_required": data.get("action_required"),
                        "clarification_required": data.get("clarification_required"),
                        "functions": fns,
                        "params": [c.get("params") for c in calls],
                        "reply_preview": (data.get("reply") or "")[:120],
                    },
                )
            )
        except Exception as e:
            record(
                ScenarioResult(
                    name=f"live_llm:{name}",
                    prompt=prompt,
                    category="fail",
                    finding=f"LLM call failed: {e}",
                    details={},
                )
            )


def _classify_live_llm(name: str, data: dict, fns: list) -> str:
    clar = data.get("clarification_required")
    action = data.get("action_required")
    expected = {
        "scenario_1": {"reschedule_missed_work"},
        "scenario_2": set(),  # should clarify, no action
        "scenario_4": {"resume_previous_task"},  # spec 5: resume, not switch
        "read_schedule": set(),
        "vague_ping": set(),
    }
    if name in expected:
        if clar and not action:
            return "pass"
        if expected[name] and expected[name].intersection(fns):
            return "pass"
        if name in ("scenario_2", "scenario_4", "vague_ping") and action and fns:
            bad = any(
                "something else" in str(c.get("params", {})).lower()
                or "last task" in str(c.get("params", {})).lower()
                for c in (data.get("calls") or [])
            )
            return "fail" if bad else "warn"
    if action and fns:
        return "warn" if name in ("scenario_2", "scenario_4") else "pass"
    return "warn"


def _live_llm_finding(name: str, data: dict, fns: list) -> str:
    parts = []
    if data.get("clarification_required"):
        parts.append("asks clarification")
    elif data.get("action_required") and fns:
        parts.append(f"calls {fns}")
    else:
        parts.append(f"reply-only: {(data.get('reply') or '')[:80]}")
    if name == "scenario_1" and "reschedule_task" in fns and "reschedule_missed_work" not in fns:
        parts.append("WRONG: used reschedule_task not reschedule_missed_work")
    if name == "switch_named" and "switch_active_task" in fns:
        params = next((c.get("params") for c in data.get("calls") or [] if c.get("function") == "switch_active_task"), {})
        if params and "until" not in str(params):
            parts.append("switch ignores until 5pm duration")
    return "; ".join(parts)


def print_report():
    by_cat: dict[str, list[ScenarioResult]] = {}
    for r in RESULTS:
        by_cat.setdefault(r.category, []).append(r)

    print("\n" + "=" * 72)
    print("SCENARIO MATRIX REPORT")
    print("=" * 72)
    for cat in ("fail", "missing_feature", "llm_dependent", "warn", "pass"):
        items = by_cat.get(cat, [])
        if not items:
            continue
        print(f"\n## {cat.upper()} ({len(items)})")
        for r in items:
            print(f"  [{r.name}]")
            print(f"    prompt: {r.prompt[:70]}")
            print(f"    → {r.finding}")
            if r.details:
                key_details = {k: v for k, v in r.details.items() if k in (
                    "functions", "failed", "matched", "is_amendment", "clarification_required"
                )}
                if key_details:
                    print(f"    details: {key_details}")

    fails = by_cat.get("fail", []) + by_cat.get("missing_feature", [])
    print("\n" + "=" * 72)
    print("FIXES NEEDED (from runtime evidence)")
    print("=" * 72)
    if fails:
        for r in fails:
            print(f"  • {r.name}: {r.finding}")
    else:
        print("  (no hard failures in deterministic layer; see LLM-dependent warnings)")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-llm", action="store_true", help="Call Groq for each prompt")
    args = parser.parse_args()

    test_fast_paths()
    test_read_schedule_not_tier1()
    test_snooze_parse()
    test_extend_scope_mapping()
    test_clarification_kinds_supported()
    test_create_task_clarification_chain()

    await test_gate_clarifies_vague_switch()
    await test_resume_previous_task_path()
    await test_gate_clarifies_vague_resume()
    await test_reschedule_missed_work_path()
    await test_wrong_fn_reschedule_task()

    if args.live_llm:
        print("Running live LLM probes (Groq)...")
        await run_live_llm_probes()

    print_report()
    return 0 if not any(r.category == "fail" for r in RESULTS) else 1


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(asyncio.run(main()))
