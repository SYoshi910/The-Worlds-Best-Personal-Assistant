"""Task resolution, embeddings, and function dispatch."""

import asyncio
import difflib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import gcal
import queries
import reclaim
import buffer_analysis
from config import EMBEDDING_MODEL_PATH, TIMEZONE
from embeddings import embed_documents, embed_text

_EMBED_CACHE = Path("data/task_embeddings.json")


def register_task_in_cache(task: dict) -> None:
    """Index a task from a write response (may not be in get_active_tasks yet)."""
    global TASK_MAP, TASK_CACHE
    tid = task.get("id")
    if not tid:
        return
    TASK_CACHE[tid] = task
    TASK_MAP[tid] = task.get("title", "")


async def update_task(
    task_id: int,
    due_date: str | None = None,
    event_category: str | None = None,
    time_needed: int | None = None,
    snooze_until: str | None = None,
) -> bool:
    """Patch selected fields on a Reclaim task."""
    fields = {}
    if due_date is not None:
        fields["due"] = due_date
    if event_category is not None:
        fields["eventCategory"] = reclaim.normalize_event_category(event_category)
    if time_needed is not None:
        fields["timeChunksRequired"] = int(time_needed)
    if snooze_until is not None:
        fields["snoozeUntil"] = snooze_until
    if not fields:
        return False
    return await reclaim.update_task_fields(task_id, fields)


async def move_due_date(task_id: int, due_date: str) -> bool:
    """Move only a task's due date (spec 13c); due_date is a parsed ISO string."""
    return await update_task(task_id, due_date=due_date)


FUNCTION_MAP = {
    "log_work": reclaim.log_work,
    "reschedule_task": reclaim.reschedule_task,
    "create_event": reclaim.create_gcal_event,
    "create_task": reclaim.create_task,
    "extend_task_total": reclaim.extend_task_total,
    "extend_task_instance": reclaim.extend_task_instance,
    "complete_task": reclaim.complete_task,
    "update_task": update_task,
    "move_due_date": move_due_date,
    "get_schedule_for_window": queries.get_schedule_for_window,
    "get_break_allowance": buffer_analysis.get_break_allowance,
}

# Read-only functions never touch tasks/events and shouldn't be tracked as writes.
READ_ONLY_FUNCTIONS = {"get_schedule_for_window", "get_break_allowance"}

COMPOSITE_FUNCTIONS: set[str] = set()

# Composites that resolve their own task reference internally; dispatch must NOT
# convert their task_query/new_task_query into a task_id before calling them.
COMPOSITE_INTERNAL_QUERY = {
    "reschedule_missed_work",
    "extend_current_gcal_block",
}

WRITE_FUNCTIONS = {
    "create_task",
    "complete_task",
    "extend_task_total",
    "extend_task_instance",
    "reschedule_task",
    "update_task",
    "move_due_date",
    "log_work",
    "reschedule_missed_work",
    "switch_active_task",
    "resume_previous_task",
    "extend_current_gcal_block",
}

TASK_MAP: dict[int, str] = {}
TASK_CACHE: dict[int, dict] = {}
_embeddings: dict[int, np.ndarray] = {}
_embedded_as: dict[int, str] = {}

MAX_QUERY_WORDS = 8
EMBEDDING_MATCH_THRESHOLD = 0.58
EMBEDDING_AMBIGUITY_MARGIN = 0.05
MIN_EMBEDDING_QUERY_WORDS = 2


def _embed_text(text: str) -> list[float]:
    return embed_text(text)


def _embed_documents(titles: list[str]) -> list[list[float]]:
    return embed_documents(titles)


def _load_embed_cache() -> None:
    if not _EMBED_CACHE.exists():
        return
    try:
        data = json.loads(_EMBED_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if data.get("model") != EMBEDDING_MODEL_PATH:
        return
    for tid_str, entry in (data.get("tasks") or {}).items():
        tid = int(tid_str)
        title = entry.get("title", "")
        if TASK_MAP.get(tid) != title:
            continue
        _embeddings[tid] = np.array(entry["v"])
        _embedded_as[tid] = title


def _save_embed_cache() -> None:
    if not _embeddings:
        _EMBED_CACHE.unlink(missing_ok=True)
        return
    _EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _EMBED_CACHE.write_text(
        json.dumps(
            {
                "model": EMBEDDING_MODEL_PATH,
                "tasks": {
                    str(tid): {"title": _embedded_as[tid], "v": vec.tolist()}
                    for tid, vec in _embeddings.items()
                },
            }
        ),
        encoding="utf-8",
    )


async def _sync_embeddings() -> None:
    global _embeddings, _embedded_as
    if not _embeddings and TASK_MAP:
        _load_embed_cache()

    for tid in list(_embeddings):
        if tid not in TASK_MAP:
            del _embeddings[tid]
            _embedded_as.pop(tid, None)

    to_embed = [
        (tid, title)
        for tid, title in TASK_MAP.items()
        if _embedded_as.get(tid) != title
    ]
    if to_embed:
        vectors = await asyncio.to_thread(_embed_documents, [t for _, t in to_embed])
        for (tid, title), vec in zip(to_embed, vectors):
            _embeddings[tid] = np.array(vec)
            _embedded_as[tid] = title

    if not TASK_MAP:
        _embeddings.clear()
        _embedded_as.clear()

    _save_embed_cache()


def _register_composites():
    import composites

    COMPOSITE_FUNCTIONS.update(
        {
            "reschedule_missed_work",
            "switch_active_task",
            "resume_previous_task",
            "extend_current_gcal_block",
        }
    )
    FUNCTION_MAP.update(
        {
            "reschedule_missed_work": composites.reschedule_missed_work,
            "switch_active_task": composites.switch_active_task,
            "resume_previous_task": composites.resume_previous_task,
            "extend_current_gcal_block": composites.extend_current_gcal_block,
        }
    )


def _normalize_query(query: str) -> str:
    query = query.strip()
    words = query.split()
    if len(words) > MAX_QUERY_WORDS:
        query = " ".join(words[-MAX_QUERY_WORDS:])
    return query


def _fmt_clock(iso: str | None) -> str | None:
    """Format an ISO datetime as a friendly local clock time (e.g. '12:00 PM')."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(ZoneInfo(TIMEZONE))
    return queries.format_clock(dt)


def build_action_confirmation(
    fn_name: str, params: dict, result=None, task: dict | None = None
) -> str:
    """First-person confirmation of a completed action (spec 21).

    e.g. "I've created a lunch block from 12:00 PM until 12:30 PM". Replaces the
    old terse ``_human_summary`` strings.
    """
    if fn_name == "get_schedule_for_window":
        return str(result) if result else "Nothing scheduled in that window."
    if fn_name == "get_break_allowance":
        return str(result) if result else "Couldn't compute break allowance."

    title = (task or {}).get("title") or params.get("title") or "that"

    if fn_name == "create_event":
        name = params.get("name", "block")
        start = _fmt_clock(params.get("start"))
        end = _fmt_clock(params.get("end"))
        if start and end:
            return f"I've created a {name} block from {start} until {end}"
        if end:
            return f"I've created a {name} block until {end}"
        return f"I've created a {name} block"
    if fn_name == "reschedule_task":
        return f"I've rescheduled '{title}'"
    if fn_name == "move_due_date":
        return f"I've moved the due date for '{title}'"
    if fn_name == "extend_task_total":
        mins = params.get("additional_chunks", 0) * 15
        return f"I've added {mins} min to '{title}'"
    if fn_name == "extend_task_instance":
        mins = params.get("additional_minutes", 0)
        return f"I've added {mins} min to today's '{title}' block"
    if fn_name == "complete_task":
        return f"I've marked '{title}' complete"
    if fn_name == "create_task":
        cat = (params.get("event_category") or "WORK").lower()
        return f"I've created the {cat} task '{params.get('title', title)}'"
    if fn_name == "update_task":
        return f"I've updated '{title}'"
    if fn_name == "log_work":
        return f"I've logged your work on '{title}'"
    return f"Done: {fn_name}"


async def build_task_map(force_refresh: bool = False):
    """Index active tasks and sync embedding vectors (incremental + disk cache)."""
    global TASK_MAP, TASK_CACHE

    tasks = await reclaim.get_active_tasks(force_refresh=force_refresh)
    TASK_MAP = {t["id"]: t["title"] for t in tasks}
    TASK_CACHE = {t["id"]: t for t in tasks}

    await _sync_embeddings()
    if TASK_MAP:
        print(f"✅ Task map built: {len(TASK_MAP)} tasks indexed")
    else:
        print("ℹ️ Task map empty — no active tasks to index")


async def upsert_task_in_map(task_id: int):
    """Refresh one task after a write."""
    task = await reclaim.get_task(task_id)
    if not task:
        return

    if task.get("status") not in ("IN_PROGRESS", "SCHEDULED"):
        TASK_MAP.pop(task_id, None)
        TASK_CACHE.pop(task_id, None)
    else:
        register_task_in_cache(task)

    await _sync_embeddings()


def _query_overlaps_title(query: str, title: str, *, cutoff: float = 0.6) -> bool:
    """True when query clearly names this task title (substring or difflib)."""
    q_lower = query.lower().strip()
    t_lower = (title or "").lower().strip()
    if not q_lower or not t_lower:
        return False
    if q_lower in t_lower or t_lower in q_lower:
        return True
    return difflib.SequenceMatcher(None, q_lower, t_lower).ratio() >= cutoff


async def _task_from_ledger(
    query: str, preferred_task_ids: list[int]
) -> dict | None:
    """Resolve via recent-action ledger only for vague refs or title overlap."""
    from clarification import is_placeholder_query

    candidates = []
    for tid in preferred_task_ids:
        task = TASK_CACHE.get(tid) or await reclaim.get_task(tid)
        if task:
            candidates.append(task)

    if not candidates:
        return None

    overlapping = [
        task
        for task in candidates
        if _query_overlaps_title(query, task.get("title", ""))
    ]
    if len(overlapping) == 1:
        task = overlapping[0]
        print(f"✅ ledger (overlap): '{query}' → '{task.get('title')}'")
        return task

    if is_placeholder_query(query) and len(candidates) == 1:
        task = candidates[0]
        print(f"✅ ledger (vague): '{query}' → '{task.get('title')}'")
        return task

    return None


async def get_task_by_query(
    query: str,
    current_task: dict | None = None,
    preferred_task_ids: list[int] | None = None,
) -> dict | None:
    """Resolve a natural-language task query via difflib, embeddings, then ledger."""
    query = _normalize_query(query)
    if not query:
        return None

    if not TASK_MAP:
        await build_task_map()

    if not TASK_MAP:
        return None

    if current_task and current_task.get("task_id") and current_task.get("title"):
        title = current_task["title"]
        q_lower = query.lower()
        t_lower = title.lower()
        if (
            q_lower in t_lower
            or t_lower in q_lower
            or difflib.SequenceMatcher(None, q_lower, t_lower).ratio() >= 0.6
        ):
            task_id = current_task["task_id"]
            print(f"✅ current_task hint: '{query}' → '{title}'")
            return TASK_CACHE.get(task_id) or await reclaim.get_task(task_id)

    titles = list(TASK_MAP.values())

    matches = difflib.get_close_matches(query, titles, n=3, cutoff=0.6)
    if matches:
        matched_title = matches[0]
        matching_ids = [tid for tid, title in TASK_MAP.items() if title == matched_title]
        if len(matching_ids) == 1:
            task_id = matching_ids[0]
            print(f"✅ difflib: '{query}' → '{matched_title}'")
            return TASK_CACHE.get(task_id) or await reclaim.get_task(task_id)

    print(f"⚡ difflib miss, falling back to embeddings for '{query}'")

    if len(query.split()) < MIN_EMBEDDING_QUERY_WORDS:
        print(f"⚠️ Query too short for embedding match: '{query}'")
    else:
        if not _embeddings:
            await _sync_embeddings()

        ids = [tid for tid in TASK_MAP if tid in _embeddings]
        if ids:
            query_embedding = np.array(await asyncio.to_thread(_embed_text, query))
            matrix = np.stack([_embeddings[tid] for tid in ids])
            similarities = np.dot(matrix, query_embedding) / (
                np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_embedding)
            )

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            runner_up = (
                float(np.partition(similarities, -2)[-2])
                if len(similarities) > 1
                else -1.0
            )

            if best_score >= EMBEDDING_MATCH_THRESHOLD and not (
                runner_up >= 0
                and (best_score - runner_up) < EMBEDDING_AMBIGUITY_MARGIN
            ):
                task_id = ids[best_idx]
                print(
                    f"✅ embedding: '{query}' → '{TASK_MAP[task_id]}' "
                    f"(score: {best_score:.2f})"
                )
                return TASK_CACHE.get(task_id) or await reclaim.get_task(task_id)

            if best_score < EMBEDDING_MATCH_THRESHOLD:
                print(f"⚠️ No confident match for '{query}' (best: {best_score:.2f})")
            else:
                print(
                    f"⚠️ Ambiguous embedding match for '{query}' "
                    f"(best: {best_score:.2f}, runner-up: {runner_up:.2f})"
                )

    if preferred_task_ids:
        ledger_task = await _task_from_ledger(query, preferred_task_ids)
        if ledger_task:
            return ledger_task

    return None


async def dispatch(
    calls: list,
    current_task: dict | None = None,
    preferred_task_ids: list[int] | None = None,
) -> dict:
    """Execute a list of LLM tool calls and return summaries, snapshots, and failures."""
    if not COMPOSITE_FUNCTIONS:
        _register_composites()

    results = {}
    succeeded = []
    summaries = []
    snapshots = []
    failed = []
    touched_task_ids: set[int] = set()
    needs_full_rebuild = False

    for call in calls:
        fn_name = call.get("function")
        params = call.get("params", {})
        alias = call.get("result_alias")

        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                ref = v[2:-2].strip()
                alias_name, field = ref.split(".", 1)
                resolved_params[k] = results.get(alias_name, {}).get(field)
            else:
                resolved_params[k] = v

        resolved_task = None
        if "task_query" in resolved_params and fn_name not in COMPOSITE_INTERNAL_QUERY:
            task_query = resolved_params.pop("task_query")
            resolved_task = await get_task_by_query(
                task_query,
                current_task=current_task,
                preferred_task_ids=preferred_task_ids,
            )
            if not resolved_task:
                msg = f"Could not find a task matching '{task_query}'"
                print(f"⚠️ {msg}, skipping {fn_name}")
                failed.append(msg)
                continue
            resolved_params["task_id"] = resolved_task["id"]

        if fn_name in COMPOSITE_FUNCTIONS:
            resolved_params.setdefault("current_task", current_task)

        fn = FUNCTION_MAP.get(fn_name)
        if not fn:
            msg = f"Unknown function: {fn_name}"
            print(f"⚠️ {msg}")
            failed.append(msg)
            continue

        try:
            result = await fn(**resolved_params)
        except Exception as e:
            msg = f"{fn_name} failed: {e}"
            print(f"❌ {msg}")
            failed.append(msg)
            continue

        if isinstance(result, dict) and "ok" in result:
            if result.get("ok"):
                label = result.get("summary", fn_name)
                succeeded.append(fn_name)
                summaries.append(label)
                snapshots.extend(result.get("snapshots", []))
                needs_full_rebuild = True
            else:
                failed.extend(result.get("failed", [f"{fn_name} did not succeed"]))
        elif result is False or result is None:
            failed.append(f"{fn_name} did not succeed")
        else:
            succeeded.append(fn_name)
            summaries.append(
                build_action_confirmation(
                    fn_name, resolved_params, result, resolved_task
                )
            )
            if fn_name == "create_event" and isinstance(result, dict) and result.get("id"):
                snapshots.append(gcal.snapshot_from_gcal_event(result, "create"))
            if fn_name == "create_task" and isinstance(result, dict) and result.get("id"):
                snapshots.append({"type": "create_task", "task_id": result["id"]})
                register_task_in_cache(result)
                touched_task_ids.add(result["id"])
            if fn_name in WRITE_FUNCTIONS:
                tid = resolved_params.get("task_id")
                if tid:
                    touched_task_ids.add(tid)

        if alias:
            results[alias] = result if isinstance(result, dict) else {}

    if needs_full_rebuild:
        await build_task_map(force_refresh=True)
    elif touched_task_ids:
        for tid in touched_task_ids:
            await upsert_task_in_map(tid)

    return {
        "succeeded": succeeded,
        "summaries": summaries,
        "snapshots": snapshots,
        "failed": failed,
    }
