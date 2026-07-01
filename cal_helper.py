"""Task resolution, embeddings, and function dispatch."""

import asyncio
import difflib
from datetime import datetime
from zoneinfo import ZoneInfo

import google.generativeai as genai
import numpy as np

import gcal
import queries
import reclaim
import buffer_analysis
from config import GEMINI_TOKEN, TIMEZONE

genai.configure(api_key=GEMINI_TOKEN)

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
    return await reclaim.update_task_fields(task_id, {"due": due_date})


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
    "reschedule_multiple_missed_work",
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
    "reschedule_multiple_missed_work",
    "switch_active_task",
    "resume_previous_task",
    "extend_current_gcal_block",
}

TASK_MAP: dict[int, str] = {}
TASK_CACHE: dict[int, dict] = {}
_embeddings: np.ndarray | None = None

MAX_QUERY_WORDS = 8
EMBEDDING_MATCH_THRESHOLD = 0.75
EMBEDDING_MODEL = "models/gemini-embedding-2"


def register_task_in_cache(task: dict) -> None:
    """Index a task from a write response (may not be in get_active_tasks yet)."""
    global TASK_MAP, TASK_CACHE
    tid = task.get("id")
    if not tid:
        return
    TASK_CACHE[tid] = task
    TASK_MAP[tid] = task.get("title", "")


async def _embed_titles(titles: list[str]) -> np.ndarray:
    """One embedding per title — gemini-embedding-2 aggregates list inputs into one vector."""
    if not titles:
        return np.array([])

    def _embed_all():
        vectors = []
        for title in titles:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=title,
            )
            vectors.append(result["embedding"])
        return np.array(vectors)

    return await asyncio.to_thread(_embed_all)


async def _rebuild_embeddings() -> None:
    global _embeddings
    titles = list(TASK_MAP.values())
    if not titles:
        _embeddings = None
        return
    _embeddings = await _embed_titles(titles)


def _register_composites():
    import composites

    COMPOSITE_FUNCTIONS.update(
        {
            "reschedule_missed_work",
            "reschedule_multiple_missed_work",
            "switch_active_task",
            "resume_previous_task",
            "extend_current_gcal_block",
        }
    )
    FUNCTION_MAP.update(
        {
            "reschedule_missed_work": composites.reschedule_missed_work,
            "reschedule_multiple_missed_work": composites.reschedule_multiple_missed_work,
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
    return dt.strftime("%I:%M %p").lstrip("0")


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


def event_end_delta(
    event_id: str, end_natural: str, now: datetime | None = None
) -> int | None:
    """Minutes to shift a task block's end so it ends at ``end_natural`` (spec 6).

    Reads the block's current end from the warm ``schedule_cache`` and parses the
    natural target time (e.g. "5pm"). Returns the signed minute delta between the
    target and the current end, or ``None`` when the block is unknown or the
    target time can't be parsed.
    """
    import schedule_cache
    from inference import parse_to_iso

    block = schedule_cache.get(event_id)
    if not block:
        return None
    tz = ZoneInfo(TIMEZONE)
    base = now or datetime.now(tz)
    iso = parse_to_iso(end_natural, base)
    if not iso:
        return None
    target = datetime.fromisoformat(iso)
    if target.tzinfo is None:
        target = target.replace(tzinfo=tz)
    current_end = block["end"]
    return int(round((target - current_end).total_seconds() / 60))


async def build_task_map(force_refresh: bool = False):
    """Index active tasks and rebuild embedding vectors for query matching."""
    global TASK_MAP, TASK_CACHE, _embeddings

    tasks = await reclaim.get_active_tasks(force_refresh=force_refresh)
    TASK_MAP = {t["id"]: t["title"] for t in tasks}
    TASK_CACHE = {t["id"]: t for t in tasks}

    if not TASK_MAP:
        _embeddings = None
        print("ℹ️ Task map empty — no active tasks to index")
        return

    await _rebuild_embeddings()
    print(f"✅ Task map built: {len(TASK_MAP)} tasks indexed")


async def upsert_task_in_map(task_id: int):
    """Refresh one task after a write; full re-embed only if task is new or title changed."""
    global TASK_MAP, TASK_CACHE

    task = await reclaim.get_task(task_id)
    if not task:
        return

    if task.get("status") not in ("IN_PROGRESS", "SCHEDULED"):
        if task_id in TASK_MAP:
            del TASK_MAP[task_id]
            TASK_CACHE.pop(task_id, None)
            await _rebuild_embeddings()
        return

    old_title = TASK_MAP.get(task_id)
    register_task_in_cache(task)

    if old_title is None or old_title != task["title"]:
        await _rebuild_embeddings()


async def _task_from_ledger(
    query: str, preferred_task_ids: list[int]
) -> dict | None:
    candidates = []
    for tid in preferred_task_ids:
        task = TASK_CACHE.get(tid) or await reclaim.get_task(tid)
        if task:
            candidates.append(task)

    if not candidates:
        return None

    if len(candidates) == 1:
        task = candidates[0]
        print(f"✅ ledger: '{query}' → '{task.get('title')}'")
        return task

    q_lower = query.lower()
    for task in candidates:
        title = (task.get("title") or "").lower()
        if q_lower in title or title in q_lower:
            print(f"✅ ledger: '{query}' → '{task.get('title')}'")
            return task
    return None


async def get_task_by_query(
    query: str,
    current_task: dict | None = None,
    preferred_task_ids: list[int] | None = None,
) -> dict | None:
    """Resolve a natural-language task query via ledger, difflib, or embeddings."""
    query = _normalize_query(query)
    if not query:
        return None

    if not TASK_MAP or _embeddings is None:
        await build_task_map()

    if not TASK_MAP:
        return None

    if preferred_task_ids:
        ledger_task = await _task_from_ledger(query, preferred_task_ids)
        if ledger_task:
            return ledger_task

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
    ids = list(TASK_MAP.keys())

    matches = difflib.get_close_matches(query, titles, n=3, cutoff=0.6)
    if matches:
        matched_title = matches[0]
        matching_ids = [tid for tid, title in TASK_MAP.items() if title == matched_title]
        if len(matching_ids) == 1:
            task_id = matching_ids[0]
            print(f"✅ difflib: '{query}' → '{matched_title}'")
            return TASK_CACHE.get(task_id) or await reclaim.get_task(task_id)

    print(f"⚡ difflib miss, falling back to Gemini for '{query}'")

    def _embed_query():
        return genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
        )

    result = await asyncio.to_thread(_embed_query)
    query_embedding = np.array(result["embedding"])

    embeddings = _embeddings
    if embeddings is None:
        return None
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if len(ids) != embeddings.shape[0]:
        print(
            f"⚠️ Embedding index mismatch ({len(ids)} tasks, "
            f"{embeddings.shape[0]} vectors); rebuilding"
        )
        await _rebuild_embeddings()
        embeddings = _embeddings
        ids = list(TASK_MAP.keys())
        if not ids or embeddings is None:
            return None
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    runner_up = float(np.partition(similarities, -2)[-2]) if len(similarities) > 1 else -1.0

    if best_score < EMBEDDING_MATCH_THRESHOLD:
        print(f"⚠️ No confident match for '{query}' (best: {best_score:.2f})")
        return None

    if runner_up >= 0 and (best_score - runner_up) < 0.05:
        print(
            f"⚠️ Ambiguous embedding match for '{query}' "
            f"(best: {best_score:.2f}, runner-up: {runner_up:.2f})"
        )
        return None

    task_id = ids[best_idx]
    print(f"✅ Gemini: '{query}' → '{TASK_MAP[task_id]}' (score: {best_score:.2f})")
    return TASK_CACHE.get(task_id) or await reclaim.get_task(task_id)


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
                await _rebuild_embeddings()
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
