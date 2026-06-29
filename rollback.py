"""Action ledger and undo for calendar operations."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import gcal
import reclaim
from cal_helper import build_task_map, dispatch, get_task_by_query
from config import TIMEZONE

UNDO_WINDOW_MINUTES = 5

_last_completed: "CompletedAction | None" = None


@dataclass
class CompletedAction:
    id: str
    created_at: datetime
    expires_at: datetime
    user_message: str
    calls_executed: list
    snapshots: list[dict]
    summary: str


def _now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def get_last_completed() -> CompletedAction | None:
    """Return the last completed action if still within the undo window."""
    if _last_completed and _last_completed.expires_at < _now():
        return None
    return _last_completed


def _task_ids_from_action(action: CompletedAction) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for snap in action.snapshots:
        tid = snap.get("task_id")
        if tid and tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids


def get_ledger_task_ids() -> list[int]:
    """Task ids from the last completed action, for amendment disambiguation."""
    action = get_last_completed()
    if not action:
        return []
    return _task_ids_from_action(action)


async def build_amendment_context() -> str | None:
    """Build LLM context describing the amendable recent action and its entities."""
    action = get_last_completed()
    if not action:
        return None

    remaining = action.expires_at - _now()
    mins = max(0, int(remaining.total_seconds() / 60))
    if mins <= 0:
        return None

    lines = [
        f"Recent action (amendable for {mins} more minutes):",
        f"  User said: {action.user_message}",
        f"  Result: {action.summary}",
        "  Entities:",
    ]

    task_ids = _task_ids_from_action(action)
    if not task_ids:
        lines.append("  (no task entities recorded)")
    else:
        for tid in task_ids:
            task = await reclaim.get_task(tid)
            if not task:
                lines.append(f"  - task {tid}: (not found)")
                continue
            lines.append(
                f"  - task {tid}: '{task.get('title')}' | due {task.get('due')} | "
                f"category {task.get('eventCategory')} | "
                f"{task.get('timeChunksRequired')} chunks | snooze {task.get('snoozeUntil')}"
            )

    lines.append(
        "Rules: amend in place with update_task or reschedule_task on entities above; "
        "use the exact task title from entities as task_query; "
        "do not create_task again for the same item."
    )
    return "\n".join(lines)


async def _resolve_task(
    task_query: str,
    current_task: dict | None,
    preferred_task_ids: list[int] | None,
) -> dict | None:
    return await get_task_by_query(
        task_query,
        current_task=current_task,
        preferred_task_ids=preferred_task_ids,
    )


async def _snapshot_before_call(
    call: dict,
    current_task: dict | None = None,
    preferred_task_ids: list[int] | None = None,
) -> dict | None:
    fn = call.get("function")
    params = call.get("params", {})

    if fn == "reschedule_task" and "task_query" in params:
        task = await _resolve_task(
            params["task_query"], current_task, preferred_task_ids
        )
        if task:
            return {
                "type": "reschedule_task",
                "task_id": task["id"],
                "snoozeUntil": task.get("snoozeUntil"),
            }

    if fn == "update_task" and "task_query" in params:
        task = await _resolve_task(
            params["task_query"], current_task, preferred_task_ids
        )
        if task:
            return {
                "type": "update_task",
                "task_id": task["id"],
                "due": task.get("due"),
                "eventCategory": task.get("eventCategory"),
                "timeChunksRequired": task.get("timeChunksRequired"),
                "snoozeUntil": task.get("snoozeUntil"),
            }

    if fn in ("extend_task_total", "extend_task_instance") and "task_query" in params:
        task = await _resolve_task(
            params["task_query"], current_task, preferred_task_ids
        )
        if task:
            return {
                "type": "extend_task_total",
                "task_id": task["id"],
                "timeChunksRequired": task.get("timeChunksRequired"),
            }

    if fn == "complete_task" and "task_query" in params:
        task = await _resolve_task(
            params["task_query"], current_task, preferred_task_ids
        )
        if task:
            return {
                "type": "complete_task",
                "task_id": task["id"],
                "status": task.get("status"),
                "timeChunksRequired": task.get("timeChunksRequired"),
                "timeChunksSpent": task.get("timeChunksSpent"),
            }

    return None


async def _apply_snapshot(snap: dict) -> bool:
    action = snap.get("action")
    snap_type = snap.get("type")

    if action == "move" and snap.get("event_id"):
        start = datetime.fromisoformat(snap["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(snap["end"].replace("Z", "+00:00"))
        await gcal.move_event(snap["event_id"], start, end)
        return True

    if action == "create" and snap.get("event_id"):
        await gcal.delete_event(snap["event_id"])
        return True

    if action == "delete" and snap.get("event_id"):
        await gcal.create_buffer_event(
            snap.get("summary") or "Restored event",
            snap["start"],
            snap["end"],
        )
        return True

    if snap_type == "reschedule_task":
        task_id = snap["task_id"]
        snooze = snap.get("snoozeUntil")
        if snooze is None:
            return await reclaim.restore_task_fields(task_id, {"snoozeUntil": None})
        await reclaim.reschedule_task(task_id, snooze_until=snooze)
        return True

    if snap_type == "update_task":
        fields = {}
        for key in ("due", "eventCategory", "timeChunksRequired", "snoozeUntil"):
            if key in snap:
                fields[key] = snap[key]
        if not fields:
            return False
        return await reclaim.restore_task_fields(snap["task_id"], fields)

    if snap_type == "extend_task_total":
        return await reclaim.restore_task_fields(
            snap["task_id"],
            {"timeChunksRequired": snap["timeChunksRequired"]},
        )

    if snap_type == "complete_task":
        return await reclaim.restore_task_fields(
            snap["task_id"],
            {
                "status": snap["status"],
                "timeChunksRequired": snap["timeChunksRequired"],
                "timeChunksSpent": snap.get("timeChunksSpent", 0),
            },
        )

    if snap_type == "create_task":
        return await reclaim.delete_task(snap["task_id"])

    return False


def format_action_summary(summaries: list[str]) -> str:
    """Join action summaries for the undo ledger."""
    if not summaries:
        return ""
    return "; ".join(summaries)


async def execute_calls(
    calls: list[dict],
    user_message: str,
    current_task: dict | None = None,
    preferred_task_ids: list[int] | None = None,
) -> dict:
    """Run tool calls with pre-snapshots and record the action for undo/amend."""
    global _last_completed

    ledger_ids = preferred_task_ids

    pre_snapshots = []
    for call in calls:
        snap = await _snapshot_before_call(
            call, current_task=current_task, preferred_task_ids=ledger_ids
        )
        if snap:
            pre_snapshots.append(snap)

    summaries = []
    all_snapshots = list(pre_snapshots)
    failed = []

    result = await dispatch(
        calls, current_task=current_task, preferred_task_ids=ledger_ids
    )
    failed.extend(result.get("failed", []))
    summaries.extend(result.get("summaries", []))
    all_snapshots.extend(result.get("snapshots", []))

    undo_hint = ""
    if calls and not failed:
        now = _now()
        _last_completed = CompletedAction(
            id=str(uuid.uuid4())[:8],
            created_at=now,
            expires_at=now + timedelta(minutes=UNDO_WINDOW_MINUTES),
            user_message=user_message,
            calls_executed=calls,
            snapshots=all_snapshots,
            summary=format_action_summary(summaries),
        )
        undo_hint = (
            f"\n\n(Say undo to reverse, or tell me what to change within "
            f"{UNDO_WINDOW_MINUTES} min.)"
        )

    return {
        "summaries": summaries,
        "failed": failed,
        "snapshots": all_snapshots,
        "undo_hint": undo_hint,
    }


async def handle_undo_or_cancel() -> str | None:
    """Reverse snapshots from the last completed action within the undo window."""
    global _last_completed

    action = get_last_completed()
    if not action:
        return "Nothing recent to undo."

    errors = []
    for snap in reversed(action.snapshots):
        try:
            ok = await _apply_snapshot(snap)
            if not ok:
                errors.append(str(snap.get("type") or snap.get("action")))
        except Exception as e:
            errors.append(str(e))

    await build_task_map(force_refresh=True)
    _last_completed = None

    if errors:
        return f"Partially undone ({action.summary}). Issues: {'; '.join(errors)}"
    return f"Undone: {action.summary}"
