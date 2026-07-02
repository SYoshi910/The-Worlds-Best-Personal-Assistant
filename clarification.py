"""Generic multi-turn clarification engine.

Covers every intent that can be under-specified: create_task, switch_task,
extend_scope, missed_blocks, disambiguate_task. Owns the "do I have enough
info?" logic (`compute_missing_fields`), the merge-until-complete loop
(`merge_clarification_reply`), and the task-query gate (`gate_task_queries`)
that turns unresolvable task references into a clarification instead of a
runtime error (spec 21).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from config import CLARIFICATION_TTL_MINUTES, EXTEND_SCOPE_TTL_MINUTES
from config import now_local as _now
from duration_parser import parse_duration_to_minutes
from intent import extract_event_category_from_message, is_category_clarification_reply
from inference import parse_to_iso

ClarificationKind = Literal[
    "create_task",
    "switch_task",
    "extend_scope",
    "missed_blocks",
    "disambiguate_task",
    "schedule_activity",
]

# Task references the LLM must never pass through to execution (spec 21). These
# are vague pronouns/aliases; when one shows up we clarify rather than guess.
PLACEHOLDER_QUERIES = frozenset(
    s.lower()
    for s in (
        "something else",
        "something",
        "other",
        "another",
        "last task",
        "my last task",
        "previous task",
        "the previous task",
        "that",
        "this",
        "it",
        "the task",
        "task",
    )
)

# Required fields per clarifiable intent. `switch_task` also needs a work
# duration, handled specially in `compute_missing_fields` (either an explicit
# duration OR an "until" time satisfies it).
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "create_task": ("title", "event_category", "due_date_natural"),
    "switch_task": ("new_task_query",),
    "missed_blocks": ("task_query",),
    "disambiguate_task": ("task_query",),
    "extend_scope": (),  # resolved by option choice, not field collection
    "schedule_activity": ("activity_kind",),
}

# schedule_activity partial_params also use: name, duration_natural, task_query,
# event_category, due_date_natural — filled based on activity_kind (gcal vs task).

# Functions whose task reference must resolve to a real Reclaim task before we
# execute. Non-task GCal blocks (lunch/commute) intentionally excluded.
TASK_TARGET_FUNCTIONS = frozenset(
    {
        "complete_task",
        "extend_task_total",
        "extend_task_instance",
        "reschedule_task",
        "reschedule_missed_work",
        "move_due_date",
        "update_task",
        "log_work",
        "switch_active_task",
        "resume_previous_task",
    }
)

TASK_QUERY_KEYS = ("task_query", "new_task_query", "previous_task_query")


# ─── Pending state ────────────────────────────────────────────────────────────


@dataclass
class PendingClarification:
    kind: ClarificationKind
    partial_params: dict = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    options: dict | None = None
    expires_at: datetime = field(default_factory=_now)

    def is_expired(self) -> bool:
        return _now() >= self.expires_at


def ttl_for_kind(kind: ClarificationKind) -> timedelta:
    if kind == "extend_scope":
        return timedelta(minutes=EXTEND_SCOPE_TTL_MINUTES)
    return timedelta(minutes=CLARIFICATION_TTL_MINUTES)


def new_pending_clarification(
    kind: ClarificationKind,
    partial_params: dict | None = None,
    missing_fields: list[str] | None = None,
    options: dict | None = None,
) -> PendingClarification:
    params = dict(partial_params or {})
    missing = list(missing_fields) if missing_fields is not None else compute_missing_fields(kind, params)
    return PendingClarification(
        kind=kind,
        partial_params=params,
        missing_fields=missing,
        options=options,
        expires_at=_now() + ttl_for_kind(kind),
    )


def clear_expired(
    pending_clarification: PendingClarification | None,
) -> PendingClarification | None:
    if pending_clarification and pending_clarification.is_expired():
        return None
    return pending_clarification


# ─── Missing-field computation (the while-loop condition) ──────────────────────


def compute_missing_fields(kind: str, params: dict | None) -> list[str]:
    """Return the still-missing required fields for `kind` given `params`.

    Spec 1a: ask for ALL missing fields at once, loop until none remain.
    """
    params = params or {}

    if kind == "switch_task":
        missing: list[str] = []
        if not params.get("new_task_query"):
            missing.append("new_task_query")
        if not (params.get("work_duration_natural") or params.get("work_until_natural")):
            missing.append("work_duration_natural")
        return missing

    if kind == "schedule_activity":
        missing: list[str] = []
        kind_val = (params.get("activity_kind") or "").lower()
        if kind_val not in ("gcal", "task"):
            missing.append("activity_kind")
            return missing
        if not (params.get("duration_natural") or params.get("work_until_natural")):
            missing.append("duration_natural")
        if kind_val == "gcal":
            if not params.get("name"):
                missing.append("name")
        else:
            if not params.get("task_query") and not params.get("title"):
                missing.append("title")
            elif not params.get("task_query"):
                if not params.get("event_category"):
                    missing.append("event_category")
                if not params.get("due_date_natural"):
                    missing.append("due_date_natural")
        return missing

    required = REQUIRED_FIELDS.get(kind, ())
    return [f for f in required if not params.get(f)]


# ─── Reply merging (accumulate user answers across turns) ──────────────────────


def _extract_title_from_message(message: str) -> str | None:
    text = message.strip()
    patterns = [
        r"(?:called|named)\s+(.+?)(?:\s+due\b|\s+tonight\b|$)",
        r"^(.+?)\s+due\s+(?:tonight|tomorrow|today|next\b)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            title = m.group(1).strip().rstrip(".,!")
            if len(title) >= 2 and not re.match(
                r"^(?:add|create|personal|work|another)\b", title, re.I
            ):
                return title
    if is_category_clarification_reply(message):
        return None
    if re.search(r"\b(?:add|create)\b", text, re.I):
        return None
    if len(text) >= 2 and len(text) <= 80 and not parse_duration_to_minutes(text):
        if not re.search(r"\b(?:tonight|tomorrow|friday|monday|due)\b", text, re.I):
            return text
    return None


def _extract_due_natural(message: str, base: datetime) -> str | None:
    text = message.strip()
    for pat in (
        r"\bdue\s+(.+?)(?:\.|$)",
        r"\b(tonight|tomorrow(?:\s+(?:morning|evening|afternoon))?|"
        r"next\s+\w+|friday|monday|tuesday|wednesday|thursday|saturday|sunday)",
    ):
        m = re.search(pat, text, re.I)
        if m:
            phrase = m.group(1).strip().rstrip(".,!")
            if parse_to_iso(phrase, base):
                return phrase
    if re.search(r"\btonight\b", text, re.I):
        return "tonight"
    if re.search(r"\btomorrow\b", text, re.I):
        m = re.search(r"\btomorrow(?:\s+\w+)?", text, re.I)
        return m.group(0) if m else "tomorrow"
    return None


def _merge_create_task(partial_params: dict, message: str) -> dict:
    params = dict(partial_params)
    now = _now()

    if is_category_clarification_reply(message):
        params["event_category"] = (
            "WORK" if message.strip().lower() in ("work", "w") else "PERSONAL"
        )
    else:
        cat = extract_event_category_from_message(message)
        if cat:
            params["event_category"] = cat

    minutes = parse_duration_to_minutes(message)
    if minutes:
        params["time_needed_natural"] = message.strip()

    due = _extract_due_natural(message, now)
    if due:
        params["due_date_natural"] = due

    title = _extract_title_from_message(message)
    if title and "title" not in params:
        params["title"] = title
    elif title and "title" in params and len(message.strip()) < 80:
        if not minutes and not due and not is_category_clarification_reply(message):
            params["title"] = title

    return params


def _merge_switch_task(partial_params: dict, message: str) -> dict:
    params = dict(partial_params)
    text = message.strip()

    minutes = parse_duration_to_minutes(text)
    until = re.search(r"\b(?:until|til|till|through|to)\s+(.+?)(?:\.|$)", text, re.I)
    if until and parse_to_iso(until.group(1).strip(), _now()):
        params["work_until_natural"] = until.group(1).strip().rstrip(".,!")
    elif minutes:
        params["work_duration_natural"] = text

    # Anything else that looks like a task name (not a bare duration/until reply)
    # becomes the new task query if we still need one.
    if not params.get("new_task_query") and not minutes and not until:
        candidate = text.rstrip(".,!")
        if 2 <= len(candidate) <= 80 and not is_placeholder_query(candidate):
            params["new_task_query"] = candidate

    return params


_GCAL_ACTIVITY_RE = re.compile(
    r"\b(?:meeting|call|commute|lunch|dinner|break|appointment|convention|travel|"
    r"interview|doctor|dentist|flight|drive)\b",
    re.I,
)


def _merge_schedule_activity(partial_params: dict, message: str) -> dict:
    """Accumulate GCal-vs-task activity params across clarification turns."""
    params = dict(partial_params)
    text = message.strip()
    lower = text.lower()

    if not params.get("activity_kind"):
        if _GCAL_ACTIVITY_RE.search(text):
            params["activity_kind"] = "gcal"
        elif re.search(r"\b(?:project|task|work on|prep|review|write|finish)\b", lower):
            params["activity_kind"] = "task"

    if lower in ("gcal", "calendar", "calendar block", "meeting", "event"):
        params["activity_kind"] = "gcal"
    elif lower in ("task", "reclaim", "work task"):
        params["activity_kind"] = "task"

    mins = parse_duration_to_minutes(text)
    if mins:
        params["duration_natural"] = text if re.search(r"\bfor\b", lower) else f"{mins} minutes"
    elif not params.get("work_until_natural"):
        until = parse_to_iso(text, _now())
        if until:
            params["work_until_natural"] = text

    if params.get("activity_kind") == "gcal":
        m = re.search(
            r"\b(?:meeting|call|lunch|dinner|appointment|commute|break)\b",
            text,
            re.I,
        )
        if m:
            params["name"] = m.group(0).lower()
        elif not params.get("name") and len(text) <= 40:
            params.setdefault("name", text)
    else:
        title = _extract_title_from_message(text)
        if title:
            params["title"] = title
            params.setdefault("task_query", title)
        cat = extract_event_category_from_message(text)
        if cat:
            params["event_category"] = cat
        due = _extract_due_natural(text, _now())
        if due:
            params["due_date_natural"] = due

    return params


def _merge_task_query(partial_params: dict, message: str) -> dict:
    """Fill a single task-reference field for missed_blocks/disambiguate_task."""
    params = dict(partial_params)
    field_name = params.get("field") or "task_query"
    candidate = message.strip().rstrip(".,!")
    if 2 <= len(candidate) <= 80 and not is_placeholder_query(candidate):
        params[field_name] = candidate
    return params


def merge_clarification_reply(pending: PendingClarification, message: str) -> dict:
    """Merge a user reply into pending params based on the clarification kind.

    Returns the updated partial_params (caller re-runs `compute_missing_fields`).
    """
    kind = pending.kind
    if kind == "create_task":
        return _merge_create_task(pending.partial_params, message)
    if kind == "switch_task":
        return _merge_switch_task(pending.partial_params, message)
    if kind == "schedule_activity":
        return _merge_schedule_activity(pending.partial_params, message)
    if kind in ("missed_blocks", "disambiguate_task"):
        return _merge_task_query(pending.partial_params, message)
    return dict(pending.partial_params)


# ─── Task-query gate (spec 21: never error → clarify) ──────────────────────────


def is_placeholder_query(query: str | None) -> bool:
    if not query:
        return True
    return query.strip().lower() in PLACEHOLDER_QUERIES


@dataclass
class TaskQueryGateResult:
    ok: bool
    calls: list[dict] = field(default_factory=list)
    clarification_required: bool = False
    reply: str = ""
    pending: PendingClarification | None = None


def _gate_clarify(fn: str, key: str, query: str) -> TaskQueryGateResult:
    if is_placeholder_query(query):
        reply = "Which task do you mean? A placeholder like that isn't specific enough."
    else:
        reply = f'I couldn\'t find a task matching "{query}". Which task did you mean?'
    pending = new_pending_clarification(
        kind="disambiguate_task",
        partial_params={"function": fn, "field": key, "attempted_query": query},
        missing_fields=["task_query"],
    )
    return TaskQueryGateResult(
        ok=False,
        clarification_required=True,
        reply=reply,
        pending=pending,
    )


async def gate_task_queries(
    calls: list[dict],
    current_task: dict | None = None,
    preferred_task_ids: list[int] | None = None,
) -> TaskQueryGateResult:
    """Resolve every task reference before execution.

    For each call that targets a Reclaim task, resolve its task_query /
    new_task_query. If the reference is a placeholder or cannot be resolved,
    return a clarification instead of letting execution error out (spec 21).
    """
    from cal_helper import get_task_by_query

    for call in calls:
        fn = call.get("function")
        if fn not in TASK_TARGET_FUNCTIONS:
            continue
        params = call.get("params") or {}
        for key in TASK_QUERY_KEYS:
            raw = params.get(key)
            if raw is None or str(raw).strip() == "":
                continue
            query = str(raw).strip()
            if is_placeholder_query(query):
                return _gate_clarify(fn, key, query)
            resolved = await get_task_by_query(
                query,
                current_task=current_task,
                preferred_task_ids=preferred_task_ids,
            )
            if resolved is None:
                return _gate_clarify(fn, key, query)

    return TaskQueryGateResult(ok=True, calls=calls)


# ─── LLM contract glue ─────────────────────────────────────────────────────────

_VALID_LLM_KINDS = {
    "create_task",
    "switch_task",
    "extend_scope",
    "missed_blocks",
    "disambiguate_task",
    "schedule_activity",
}


def build_clarification_context(pending: PendingClarification) -> str:
    """Describe pending clarification state for injection into the LLM prompt."""
    missing = pending.missing_fields or compute_missing_fields(
        pending.kind, pending.partial_params
    )

    if pending.kind == "create_task":
        return "\n".join(
            [
                "Pending clarification (create_task):",
                f"  Accumulated params: {pending.partial_params}",
                f"  Still missing: {missing}",
                "  Merge any new user info into pending_params; ask for ALL missing "
                "fields at once, or emit create_task when complete.",
            ]
        )

    if pending.kind == "switch_task":
        return "\n".join(
            [
                "Pending clarification (switch_task):",
                f"  Accumulated params: {pending.partial_params}",
                f"  Still missing: {missing}",
                "  Need the new task and how long they'll work on it (duration or "
                "an 'until' time). Ask for everything still missing at once.",
            ]
        )

    if pending.kind == "extend_scope":
        return "\n".join(
            [
                "Pending clarification (extend_scope):",
                "  User must choose: whole task (task_total) or today's block "
                "(current_instance).",
                f"  Shared params: {pending.partial_params}",
            ]
        )

    if pending.kind == "schedule_activity":
        return "\n".join(
            [
                "Pending clarification (schedule_activity):",
                f"  Accumulated params: {pending.partial_params}",
                f"  Still missing: {missing}",
                "  Decide GCal block (activity_kind=gcal) vs Reclaim task "
                "(activity_kind=task), then gather missing name/duration or "
                "title/category/due. Emit create_event or create_task/switch_active_task "
                "when complete.",
            ]
        )

    if pending.kind in ("missed_blocks", "disambiguate_task"):
        return "\n".join(
            [
                f"Pending clarification ({pending.kind}):",
                f"  Accumulated params: {pending.partial_params}",
                f"  Still missing: {missing}",
                "  Ask which task they mean; never guess a placeholder task query.",
            ]
        )

    return ""


def apply_llm_clarification_fields(data: dict) -> PendingClarification | None:
    """Build pending state from an LLM clarification response (any kind)."""
    if not data.get("clarification_required"):
        return None
    kind = data.get("clarification_kind")
    if kind not in _VALID_LLM_KINDS:
        if data.get("pending_params") or data.get("missing_fields"):
            kind = "create_task"
        else:
            return None
    return new_pending_clarification(
        kind=kind,
        partial_params=data.get("pending_params") or {},
        missing_fields=data.get("missing_fields") or None,
        options=data.get("extend_options"),
    )
