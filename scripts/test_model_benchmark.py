"""
Multi-model LLM benchmark: routing, params, JSON format, token usage.

No calendar/Reclaim API calls — only exercises the same prompt + validation path
as production (call_llm_benchmark).

Run:
  python scripts/test_model_benchmark.py
  python scripts/test_model_benchmark.py --models qwen/qwen3-32b,llama-3.3-70b-versatile
  python scripts/test_model_benchmark.py --cases missed_bcg,vague_switch --delay 2.5
  python scripts/test_model_benchmark.py --list-cases
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zoneinfo import ZoneInfo

from clarification import PLACEHOLDER_QUERIES
from config import TIMEZONE
from inference import call_llm_benchmark

# Groq model IDs (free tier limits vary by model)
DEFAULT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

ExpectKind = Literal["clarify", "action", "reply_only", "fast_path_note"]


def _now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def _ping(title: str, ongoing: bool = True) -> dict:
    start = _now()
    return {
        "title": title,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(hours=1)).isoformat(),
        "event_id": f"mock-{title[:8].replace(' ', '-')}",
        "is_ongoing": ongoing,
    }


BCG_PING = _ping("BCG prep")
DATA_INFRA_PING = _ping("Data infra pres")


@dataclass
class CaseExpect:
    kind: ExpectKind
    functions_any: list[str] = field(default_factory=list)
    functions_forbidden: list[str] = field(default_factory=list)
    param_contains: dict[str, dict[str, str]] = field(default_factory=dict)
    param_forbidden_values: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    notes: str = ""


@dataclass
class BenchmarkCase:
    id: str
    prompt: str
    expect: CaseExpect
    current_task: dict | None = None
    amendment_context: str | None = None
    clarification_context: str | None = None
    tags: list[str] = field(default_factory=list)


def _all_cases() -> list[BenchmarkCase]:
    create_pending = (
        "Pending clarification (create_task):\n"
        "  Accumulated params: {'title': 'startup pitch', 'due_date_natural': 'Friday', "
        "'time_needed_natural': '5 hours'}\n"
        "  Still missing: ['event_category']\n"
        "  Merge any new user info into pending_params; ask for one missing field at a time "
        "or emit create_task when complete."
    )
    amend_ctx = (
        "Recent action (amendable for 4 more minutes):\n"
        "  User said: add a task called startup pitch, 5 hrs, due friday\n"
        "  Result: Created startup pitch\n"
        "  Entities:\n"
        "  - task 99: 'startup pitch' | due 2026-07-04 | category WORK | 20 chunks\n"
        "Rules: amend in place with update_task or reschedule_task on entities above."
    )

    return [
        # --- Core scenarios (matrix) ---
        BenchmarkCase(
            id="missed_bcg",
            prompt="i lowk didnt work on any BCG prep today can you reschedule?",
            expect=CaseExpect(
                kind="action",
                functions_any=["reschedule_missed_work"],
                functions_forbidden=["reschedule_task", "update_task"],
                param_contains={"reschedule_missed_work": {"task_query": "bcg"}},
                notes="Missed past blocks → reschedule_missed_work, not Reclaim snooze",
            ),
            tags=["core", "reschedule"],
        ),
        BenchmarkCase(
            id="vague_switch",
            prompt="hey ive been actually working on something else",
            expect=CaseExpect(
                kind="clarify",
                functions_forbidden=["switch_active_task"],
                notes="No task name → must clarify, never switch_active_task(something else)",
            ),
            current_task=BCG_PING,
            tags=["core", "switch"],
        ),
        BenchmarkCase(
            id="extend_30min_ping",
            prompt="hey ill be working on this for 30 more minutes",
            expect=CaseExpect(
                kind="action",
                functions_any=["extend_task_instance", "extend_current_gcal_block"],
                param_contains={
                    "extend_task_instance": {"task_query": "bcg"},
                },
                notes="Active ping + 30min → extend instance/block on current task",
            ),
            current_task=BCG_PING,
            tags=["core", "extend"],
        ),
        BenchmarkCase(
            id="still_on_last_task",
            prompt="hey im actually still working on my last task",
            expect=CaseExpect(
                kind="clarify",
                functions_forbidden=["switch_active_task"],
                param_forbidden_values={
                    "switch_active_task": {"new_task_query": list(PLACEHOLDER_QUERIES)},
                    "resume_previous_task": {"previous_task_query": list(PLACEHOLDER_QUERIES)},
                    "extend_task_instance": {"task_query": list(PLACEHOLDER_QUERIES)},
                },
                notes="Previous task vs new ping — resume_previous_task (no placeholder query) or clarify",
            ),
            current_task=DATA_INFRA_PING,
            tags=["core", "switch"],
        ),
        BenchmarkCase(
            id="resume_previous_with_duration",
            prompt="hey im actually still working on my last task for another 20 minutes",
            expect=CaseExpect(
                kind="action",
                functions_any=["resume_previous_task"],
                functions_forbidden=["switch_active_task"],
                notes="Explicit duration given — should confidently resume_previous_task (spec 5)",
            ),
            current_task=DATA_INFRA_PING,
            tags=["core", "switch"],
        ),
        # --- Schedule / read ---
        BenchmarkCase(
            id="read_schedule_week",
            prompt="what do i have due this week",
            expect=CaseExpect(
                kind="reply_only",
                functions_forbidden=[
                    "create_task",
                    "reschedule_task",
                    "reschedule_missed_work",
                    "switch_active_task",
                    "complete_task",
                ],
                notes="Prod uses fast path; LLM should reply-only with no writes",
            ),
            tags=["read"],
        ),
        BenchmarkCase(
            id="read_tomorrow_afternoon",
            prompt="what am i doing tomorrow afternoon",
            expect=CaseExpect(
                kind="action",
                functions_any=["get_schedule_for_window"],
                functions_forbidden=[
                    "create_task",
                    "reschedule_task",
                    "switch_active_task",
                    "reschedule_missed_work",
                    "complete_task",
                ],
                param_contains={
                    "get_schedule_for_window": {"period": "afternoon"},
                },
                notes="Windowed day/period read (spec 17a) → get_schedule_for_window",
            ),
            tags=["read"],
        ),
        # --- Create / clarify ---
        BenchmarkCase(
            id="create_startup_pitch",
            prompt="add a task called startup pitch, 5 hrs, due friday",
            expect=CaseExpect(
                kind="action",
                functions_any=["create_task"],
                functions_forbidden=[],
                notes="May clarify work/personal OR emit create_task with category",
            ),
            tags=["create"],
        ),
        BenchmarkCase(
            id="create_groceries",
            prompt="add task buy groceries tonight",
            expect=CaseExpect(
                kind="clarify",
                functions_forbidden=["create_task"],
                notes="Missing category and/or duration — should clarify first",
            ),
            tags=["create"],
        ),
        BenchmarkCase(
            id="create_personal_followup",
            prompt="personal",
            expect=CaseExpect(
                kind="action",
                functions_any=["create_task"],
                param_contains={"create_task": {"event_category": "PERSONAL"}},
                notes="Follow-up to pending create_task clarification",
            ),
            clarification_context=create_pending,
            tags=["create", "multi_turn"],
        ),
        # --- Events ---
        BenchmarkCase(
            id="event_lunch",
            prompt="ill be at lunch from 12:30 to 1:30",
            expect=CaseExpect(
                kind="action",
                functions_any=["create_event"],
                param_contains={"create_event": {"name": "lunch"}},
                notes="Fixed-time block, not a Reclaim task",
            ),
            tags=["event"],
        ),
        BenchmarkCase(
            id="event_commute_noon",
            prompt="commuting until noon",
            expect=CaseExpect(
                kind="action",
                functions_any=["create_event"],
                notes="Start=now, end=noon",
            ),
            current_task=BCG_PING,
            tags=["event"],
        ),
        # --- Extend ---
        BenchmarkCase(
            id="extend_75_ping",
            prompt="extend by 75 mins",
            expect=CaseExpect(
                kind="action",
                functions_any=["extend_task_instance", "extend_current_gcal_block"],
                notes="During active ping, >=30 min",
            ),
            current_task=BCG_PING,
            tags=["extend"],
        ),
        BenchmarkCase(
            id="extend_bcg_off_block",
            prompt="extend bcg by 2 hours",
            expect=CaseExpect(
                kind="action",
                functions_any=["extend_task_total", "extend_task_instance"],
                param_contains={
                    "extend_task_total": {"task_query": "bcg"},
                    "extend_task_instance": {"task_query": "bcg"},
                },
                notes="Not on BCG block — total vs instance scope ambiguous OK",
            ),
            current_task=DATA_INFRA_PING,
            tags=["extend"],
        ),
        BenchmarkCase(
            id="extend_until_5pm",
            prompt="extend this until 5pm",
            expect=CaseExpect(
                kind="action",
                functions_any=[
                    "extend_task_instance",
                    "extend_current_gcal_block",
                    "extend_task_total",
                ],
                notes="Until-time parsing — may pass or hit validation",
            ),
            current_task=BCG_PING,
            tags=["extend"],
        ),
        # --- Switch ---
        BenchmarkCase(
            id="switch_data_infra_until_5pm",
            prompt="im actually working on data infra until 5pm",
            expect=CaseExpect(
                kind="action",
                functions_any=["switch_active_task"],
                param_contains={"switch_active_task": {"new_task_query": "data infra"}},
                notes="Named switch; until 5pm may be ignored without composite",
            ),
            current_task=BCG_PING,
            tags=["switch"],
        ),
        BenchmarkCase(
            id="switch_orgo",
            prompt="doing orgo instead",
            expect=CaseExpect(
                kind="action",
                functions_any=["switch_active_task"],
                param_contains={"switch_active_task": {"new_task_query": "orgo"}},
            ),
            current_task=BCG_PING,
            tags=["switch"],
        ),
        BenchmarkCase(
            id="vague_ping_reply",
            prompt="yeah that changed",
            expect=CaseExpect(
                kind="clarify",
                functions_forbidden=["switch_active_task", "complete_task"],
                notes="Vague ping reply — should ask what changed",
            ),
            current_task=BCG_PING,
            tags=["switch", "ping"],
        ),
        # --- Missed / snooze ---
        BenchmarkCase(
            id="missed_bcg_tomorrow_morning",
            prompt="i skipped BCG move it to tomorrow morning",
            expect=CaseExpect(
                kind="action",
                functions_any=["reschedule_missed_work"],
                functions_forbidden=["reschedule_task"],
                param_contains={"reschedule_missed_work": {"task_query": "bcg"}},
            ),
            tags=["reschedule"],
        ),
        BenchmarkCase(
            id="snooze_bcg_thursday",
            prompt="snooze BCG until thursday",
            expect=CaseExpect(
                kind="action",
                functions_any=["reschedule_task"],
                functions_forbidden=["reschedule_missed_work"],
                param_contains={"reschedule_task": {"task_query": "bcg"}},
            ),
            tags=["reschedule"],
        ),
        # --- Amend / undo ---
        BenchmarkCase(
            id="amend_due_thursday",
            prompt="actually make that due thursday",
            expect=CaseExpect(
                kind="action",
                functions_any=["update_task", "reschedule_task"],
                notes="Amend recent create — update_task on startup pitch",
            ),
            amendment_context=amend_ctx,
            tags=["amend"],
        ),
        BenchmarkCase(
            id="complete_bcg",
            prompt="complete BCG prep",
            expect=CaseExpect(
                kind="action",
                functions_any=["complete_task"],
                param_contains={"complete_task": {"task_query": "bcg"}},
            ),
            tags=["complete"],
        ),
        BenchmarkCase(
            id="move_due_date_only",
            prompt="push BCG's due date to next Monday, nothing else about it changes",
            expect=CaseExpect(
                kind="action",
                functions_any=["move_due_date", "update_task"],
                param_contains={
                    "move_due_date": {"task_query": "bcg"},
                    "update_task": {"task_query": "bcg"},
                },
                notes="Due-date-only change should prefer move_due_date over update_task",
            ),
            tags=["update"],
        ),
        # --- Break (fast path in prod) ---
        BenchmarkCase(
            id="im_tired",
            prompt="im tired",
            expect=CaseExpect(
                kind="reply_only",
                functions_forbidden=[
                    "create_task",
                    "reschedule_task",
                    "switch_active_task",
                ],
                notes="Prod uses break fast path; LLM should not schedule writes",
            ),
            tags=["break"],
        ),
        BenchmarkCase(
            id="clear_evening",
            prompt="clear my evening",
            expect=CaseExpect(
                kind="reply_only",
                functions_forbidden=[
                    "create_task",
                    "reschedule_task",
                    "switch_active_task",
                ],
                notes="Prod uses break/buffer fast path",
            ),
            tags=["break"],
        ),
        # --- Short extend during ping ---
        BenchmarkCase(
            id="extend_15min_ping",
            prompt="give me 15 more minutes",
            expect=CaseExpect(
                kind="action",
                functions_any=["extend_current_gcal_block", "extend_task_instance"],
                notes="Short extension during ping",
            ),
            current_task=BCG_PING,
            tags=["extend"],
        ),
        # --- Log work edge ---
        BenchmarkCase(
            id="log_work_bcg",
            prompt="log 30 minutes on bcg prep",
            expect=CaseExpect(
                kind="action",
                functions_any=["log_work"],
                param_contains={"log_work": {"task_query": "bcg"}},
            ),
            tags=["log"],
        ),
    ]


CASES: list[BenchmarkCase] = _all_cases()


@dataclass
class Score:
    format_ok: bool
    validation_ok: bool
    intent_ok: bool
    params_ok: bool
    overall: bool
    reasons: list[str] = field(default_factory=list)


def _calls(data: dict) -> list[dict]:
    return data.get("calls") or []


def _fn_names(calls: list[dict]) -> list[str]:
    return [c.get("function", "") for c in calls]


def _params(call: dict) -> dict:
    return call.get("params") or {}


def _param_fails_placeholder(calls: list[dict], rules: dict[str, dict[str, list[str]]]) -> list[str]:
    fails = []
    for fn, field_rules in rules.items():
        for call in calls:
            if call.get("function") != fn:
                continue
            for key, banned in field_rules.items():
                val = str(_params(call).get(key, "")).lower().strip()
                if val in banned or any(b in val for b in banned if len(b) > 4):
                    fails.append(f"{fn}.{key}={val!r} is placeholder/banned")
    return fails


def score_case(data: dict, expect: CaseExpect, validation_errors: list[str], json_ok: bool) -> Score:
    reasons: list[str] = []
    calls = _calls(data)
    fns = _fn_names(calls)
    action = bool(data.get("action_required"))
    clarify = bool(data.get("clarification_required"))

    format_ok = json_ok and isinstance(data.get("reply"), str)
    if not format_ok:
        reasons.append("JSON/format invalid")

    validation_ok = not validation_errors
    if not validation_ok:
        reasons.append(f"validation: {validation_errors}")

    intent_ok = True
    params_ok = True

    if expect.kind == "clarify":
        if not clarify and action and calls:
            if expect.functions_forbidden and any(f in fns for f in expect.functions_forbidden):
                intent_ok = False
                reasons.append(f"should clarify but called {fns}")
            elif _param_fails_placeholder(calls, expect.param_forbidden_values):
                intent_ok = False
                reasons.extend(_param_fails_placeholder(calls, expect.param_forbidden_values))
        elif not clarify and not action:
            intent_ok = False
            reasons.append("expected clarification_required=true")
        elif clarify:
            pass  # good
        elif action and not expect.functions_forbidden:
            if expect.functions_any and not any(f in fns for f in expect.functions_any):
                intent_ok = False
                reasons.append(f"expected clarify or one of {expect.functions_any}, got {fns}")

    elif expect.kind == "reply_only":
        if action and calls:
            intent_ok = False
            reasons.append(f"reply_only expected, got calls {fns}")
        if any(f in fns for f in expect.functions_forbidden):
            intent_ok = False
            reasons.append(f"forbidden calls: {fns}")

    elif expect.kind == "action":
        if clarify and not calls:
            intent_ok = False
            reasons.append("expected action but only clarification")
        elif not action and not calls:
            intent_ok = False
            reasons.append("expected action_required with calls")
        else:
            if expect.functions_any and not any(f in fns for f in expect.functions_any):
                intent_ok = False
                reasons.append(f"expected one of {expect.functions_any}, got {fns}")
            for forbidden in expect.functions_forbidden:
                if forbidden in fns:
                    intent_ok = False
                    reasons.append(f"forbidden function {forbidden}")

            for fn, rules in expect.param_contains.items():
                if fn not in fns:
                    continue
                matched = False
                for call in calls:
                    if call.get("function") != fn:
                        continue
                    p = _params(call)
                    ok = all(
                        substr.lower() in str(p.get(k, "")).lower()
                        for k, substr in rules.items()
                    )
                    if ok:
                        matched = True
                        break
                if not matched:
                    params_ok = False
                    reasons.append(f"{fn} params missing {rules}")

            ph_fails = _param_fails_placeholder(calls, expect.param_forbidden_values)
            if ph_fails:
                params_ok = False
                reasons.extend(ph_fails)

    # Global: no placeholder task queries on switch
    for call in calls:
        fn = call.get("function")
        p = _params(call)
        for key in ("task_query", "new_task_query"):
            if key not in p:
                continue
            val = str(p[key]).lower().strip()
            if val in PLACEHOLDER_QUERIES:
                params_ok = False
                intent_ok = False
                reasons.append(f"placeholder {fn}.{key}={val!r}")

    overall = format_ok and validation_ok and intent_ok and params_ok
    return Score(
        format_ok=format_ok,
        validation_ok=validation_ok,
        intent_ok=intent_ok,
        params_ok=params_ok,
        overall=overall,
        reasons=reasons,
    )


@dataclass
class RunResult:
    case_id: str
    model: str
    prompt: str
    score: Score
    functions: list[str]
    params: list[dict]
    action_required: bool
    clarification_required: bool
    validation_errors: list[str]
    usage: dict
    reply_preview: str
    raw_preview: str = ""
    error: str | None = None
    latency_ms: int = 0


async def run_one(case: BenchmarkCase, model: str) -> RunResult:
    t0 = time.perf_counter()
    try:
        result = await call_llm_benchmark(
            message=[{"role": "user", "content": case.prompt}],
            current_task=case.current_task,
            amendment_context=case.amendment_context,
            clarification_context=case.clarification_context,
            model=model,
        )
        data = result["data"]
        sc = score_case(
            data, case.expect, result["validation_errors"], result["json_ok"]
        )
        calls = _calls(data)
        return RunResult(
            case_id=case.id,
            model=model,
            prompt=case.prompt,
            score=sc,
            functions=_fn_names(calls),
            params=[_params(c) for c in calls],
            action_required=bool(data.get("action_required")),
            clarification_required=bool(data.get("clarification_required")),
            validation_errors=result["validation_errors"],
            usage=result["usage"],
            reply_preview=(data.get("reply") or "")[:160],
            raw_preview=(result.get("raw") or "")[:300],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            err = f"RATE_LIMIT: {err[:200]}"
        return RunResult(
            case_id=case.id,
            model=model,
            prompt=case.prompt,
            score=Score(False, False, False, False, False, [err]),
            functions=[],
            params=[],
            action_required=False,
            clarification_required=False,
            validation_errors=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            reply_preview="",
            error=err,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )


def _model_slug(model: str) -> str:
    return re.sub(r"[^\w.-]", "_", model.replace("/", "_"))


def print_summary(results: list[RunResult], models: list[str]) -> None:
    by_model: dict[str, list[RunResult]] = {m: [] for m in models}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    print("\n" + "=" * 88)
    print("MODEL BENCHMARK SUMMARY")
    print("=" * 88)

    header = f"{'Model':<32} {'Pass':>6} {'Fmt':>5} {'Val':>5} {'Int':>5} {'Prm':>5} {'Tok':>8} {'ms':>6}"
    print(header)
    print("-" * 88)

    for model in models:
        rows = by_model.get(model, [])
        if not rows:
            continue
        n = len(rows)
        passed = sum(1 for r in rows if r.score.overall)
        fmt = sum(1 for r in rows if r.score.format_ok)
        val = sum(1 for r in rows if r.score.validation_ok)
        intent = sum(1 for r in rows if r.score.intent_ok)
        prm = sum(1 for r in rows if r.score.params_ok)
        tok = sum(r.usage.get("total_tokens", 0) for r in rows)
        ms = sum(r.latency_ms for r in rows)
        short = model.split("/")[-1][:30]
        print(
            f"{short:<32} {passed}/{n:>4} {fmt:>5} {val:>5} {intent:>5} {prm:>5} {tok:>8} {ms:>6}"
        )

    print("\n--- Failures by model ---")
    for model in models:
        fails = [r for r in by_model.get(model, []) if not r.score.overall]
        if not fails:
            continue
        print(f"\n{model}:")
        for r in fails:
            why = r.error or "; ".join(r.score.reasons) or "unknown"
            print(f"  [{r.case_id}] {why}")
            if r.functions:
                print(f"    calls: {r.functions} {r.params}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Groq models on calendar routing prompts")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Groq model IDs",
    )
    parser.add_argument("--cases", default="", help="Comma-separated case ids (default: all)")
    parser.add_argument("--delay", type=float, default=2.2, help="Seconds between API calls")
    parser.add_argument("--list-cases", action="store_true", help="List case ids and exit")
    parser.add_argument(
        "--output",
        default="",
        help="JSON report path (default: reports/model_benchmark_<timestamp>.json)",
    )
    args = parser.parse_args()

    if args.list_cases:
        for c in CASES:
            print(f"  {c.id:<28} [{', '.join(c.tags)}] {c.prompt[:50]}")
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.cases.strip():
        ids = {x.strip() for x in args.cases.split(",")}
        cases = [c for c in CASES if c.id in ids]
        missing = ids - {c.id for c in cases}
        if missing:
            print(f"Unknown case ids: {missing}")
            return 1
    else:
        cases = CASES

    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / (
        f"model_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    total = len(cases) * len(models)
    print(f"Running {len(cases)} cases × {len(models)} models = {total} LLM calls")
    print(f"Models: {models}")
    print(f"Delay: {args.delay}s between calls\n")

    results: list[RunResult] = []
    n = 0
    for case in cases:
        for model in models:
            n += 1
            print(f"[{n}/{total}] {case.id} @ {model.split('/')[-1]}...", flush=True)
            r = await run_one(case, model)
            results.append(r)
            status = "PASS" if r.score.overall else "FAIL"
            tok = r.usage.get("total_tokens", 0)
            print(f"  {status} | {r.functions or ('clarify' if r.clarification_required else 'reply')} | {tok} tok | {r.latency_ms}ms")
            if not r.score.overall and (r.score.reasons or r.error):
                print(f"  → {r.error or '; '.join(r.score.reasons[:2])}")
            if n < total:
                await asyncio.sleep(args.delay)

    report = {
        "run_at": datetime.now().isoformat(),
        "models": models,
        "case_count": len(cases),
        "results": [
            {
                **{k: v for k, v in asdict(r).items() if k != "score"},
                "score": asdict(r.score),
            }
            for r in results
        ],
        "totals_by_model": {
            m: {
                "passed": sum(1 for r in results if r.model == m and r.score.overall),
                "total": sum(1 for r in results if r.model == m),
                "tokens": sum(r.usage.get("total_tokens", 0) for r in results if r.model == m),
            }
            for m in models
        },
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    print_summary(results, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
