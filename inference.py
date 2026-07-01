"""LLM call path: parse natural dates/durations and validate tool calls."""

import json

import re

from datetime import datetime, timedelta



import parsedatetime as pdt

from groq import AsyncGroq

from zoneinfo import ZoneInfo



from config import DEV_RELOAD, GROQ_TOKEN, LLM_MAX_OUTPUT_TOKENS, TIMEZONE
from duration_parser import parse_duration_to_chunks, parse_duration_to_minutes
from intent import extract_event_category_from_message
from reclaim import minutes_to_chunks, normalize_event_category

from system_prompts import SYSTEM_PROMPT_CAL

from token_guardrails import cap_snapshot, prepare_messages_for_llm



client = AsyncGroq(api_key=GROQ_TOKEN)

cal = pdt.Calendar()



DATE_FIELD_MAP = {

    "create_task": [("due_date_natural", "due_date")],

    "create_event": [("start_time_natural", "start"), ("end_time_natural", "end")],

    "reschedule_task": [("snooze_until_natural", "snooze_until")],

    "move_due_date": [("due_date_natural", "due_date")],

    "update_task": [

        ("due_date_natural", "due_date"),

        ("snooze_until_natural", "snooze_until"),

    ],

    "log_work": [("start_natural", "start"), ("end_natural", "end")],

}

# (natural_key, target_key, kind) — kind: "chunks" | "minutes" | "add_chunks"
DURATION_FIELD_MAP: dict[str, list[tuple[str, str, str]]] = {
    "create_task": [("time_needed_natural", "time_needed", "chunks")],
    "update_task": [("time_needed_natural", "time_needed", "chunks")],
    "extend_task_total": [("additional_time_natural", "additional_chunks", "add_chunks")],
    "extend_task_instance": [("additional_time_natural", "additional_minutes", "minutes")],
    "extend_current_gcal_block": [("additional_time_natural", "additional_minutes", "minutes")],
    "switch_active_task": [("work_duration_natural", "work_duration_minutes", "minutes")],
    "resume_previous_task": [("work_duration_natural", "work_duration_minutes", "minutes")],
}

# LLM may emit numeric duration fields; normalized to canonical keys (prefer numeric over *_natural).
_NUMERIC_DURATION_NORMALIZERS: dict[str, list[tuple[str, str]]] = {
    "extend_task_total": [
        ("additional_chunks", "additional_chunks"),
        ("additional_minutes", "additional_chunks"),  # converted via minutes_to_chunks
    ],
    "extend_task_instance": [
        ("additional_minutes", "additional_minutes"),
        ("additional_chunks", "additional_minutes"),  # chunks * 15
    ],
    "extend_current_gcal_block": [
        ("additional_minutes", "additional_minutes"),
        ("additional_chunks", "additional_minutes"),
    ],
    "create_task": [("time_needed", "time_needed")],
    "update_task": [("time_needed", "time_needed")],
}





# A bare clock time with no am/pm marker (e.g. "12:30", "5", "5:15").
_BARE_TIME_RE = re.compile(r"^\s*\d{1,2}(?::\d{2})?\s*$")
_MERIDIEM_RE = re.compile(
    r"\b(?:am|pm|a\.m|p\.m|noon|midnight|o'?clock)\b", re.I
)


def _bump_to_upcoming(parsed: datetime, base_date: datetime, raw: str) -> datetime:
    """Resolve an am/pm-ambiguous clock time to its next future occurrence (spec 18/20).

    Only applies to a bare clock time with no meridiem marker: 11AM + "12:30" → 12:30 PM
    (same day), 11PM + "12:30" → 12:30 AM (next day). Any phrase with an explicit
    am/pm/noon/midnight token, or that isn't a bare clock time (dates, "in 2 hours",
    etc.), is returned unchanged.
    """
    if _MERIDIEM_RE.search(raw) or not _BARE_TIME_RE.match(raw):
        return parsed
    # Roll forward in 12-hour steps until the time is strictly in the future.
    for _ in range(2):
        if parsed > base_date:
            break
        parsed = parsed + timedelta(hours=12)
    return parsed


def parse_to_iso(natural_date_str: str, base_date: datetime) -> str | None:
    """Parse natural-language date/time to ISO string in local timezone."""

    if not natural_date_str:

        return None

    raw = natural_date_str.strip()
    time_struct, parse_status = cal.parse(raw, sourceTime=base_date)

    if parse_status > 0:

        parsed = datetime(*time_struct[:6], tzinfo=ZoneInfo(TIMEZONE))

        parsed = _bump_to_upcoming(parsed, base_date, raw)

        return parsed.isoformat()

    return None


def parse_upcoming_time(phrase: str, base_date: datetime) -> str | None:
    """Parse a time phrase to its next upcoming occurrence (spec 18/20).

    Thin wrapper over :func:`parse_to_iso`; for am/pm-ambiguous bare clock times
    it returns the next future occurrence rather than today's already-passed time.
    """
    return parse_to_iso(phrase, base_date)


def _parse_until_as_minutes(natural: str, base_date: datetime) -> int | None:
    """Parse 'until 5pm' / 'to 1:30' style phrases into minutes from now."""
    m = re.search(r"\b(?:until|to|through)\s+(.+?)(?:\.|$)", natural.strip(), re.I)
    if not m:
        return None
    iso = parse_to_iso(m.group(1).strip(), base_date)
    if not iso:
        return None
    end = datetime.fromisoformat(iso)
    if end.tzinfo is None:
        end = end.replace(tzinfo=ZoneInfo(TIMEZONE))
    if end <= base_date:
        return None
    return int((end - base_date).total_seconds() // 60)





def normalize_bool(val) -> bool:
    """Coerce LLM boolean fields from bool, str, or other truthy values."""

    if isinstance(val, bool):

        return val

    if isinstance(val, str):

        return val.strip().lower() in ("true", "yes", "1")

    return bool(val)





_OPEN_THINK = "<" + "think" + ">"
_CLOSE_THINK = "</" + "think" + ">"
_THINK_BLOCK = re.compile(
    "(?is)" + re.escape(_OPEN_THINK) + r".*?" + re.escape(_CLOSE_THINK)
)


def _strip_reasoning_wrappers(raw: str) -> str:
    """Remove markdown fences and reasoning tags before JSON parse."""
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return match.group(1).strip()
    if _THINK_BLOCK.search(text):
        text = _THINK_BLOCK.sub("", text).strip()
    return text


def _groq_model_kwargs(model: str) -> dict:
    """Per-model Groq options for reliable JSON tool output."""
    m = model.lower()
    extra: dict = {}
    if "qwen" in m:
        extra["reasoning_effort"] = "none"
    elif "gpt-oss" in m:
        extra["reasoning_effort"] = "low"
    if not extra:
        return {}
    return {"extra_body": extra}


def _extract_json_object(raw: str) -> dict | None:
    """Parse a JSON object from model output, including nested calls arrays."""
    text = _strip_reasoning_wrappers(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_llm_json(raw: str) -> dict:

    parsed = _extract_json_object(raw)

    if parsed is not None:

        return parsed

    return {

        "action_required": False,

        "clarification_required": False,

        "reply": raw,

        "calls": [],

        "_parse_failed": True,

    }





def _normalize_llm_response(data: dict) -> dict:

    action_required = normalize_bool(data.get("action_required", False))

    clarification_required = normalize_bool(data.get("clarification_required", False))

    if clarification_required:
        action_required = False
        data["calls"] = []
    elif not action_required:
        data["calls"] = []

    data["action_required"] = action_required
    data["clarification_required"] = clarification_required
    data.setdefault("reply", "...")
    data.setdefault("calls", [])
    data.setdefault("clarification_kind", None)
    data.setdefault("pending_params", {})
    data.setdefault("missing_fields", [])
    data.setdefault("validation_errors", [])
    return data





def _apply_date_parsing(calls: list, current_time: datetime) -> list[str]:

    errors = []

    for call in calls:

        fn = call.get("function")

        params = call.get("params", {})

        for natural_key, iso_key in DATE_FIELD_MAP.get(fn, []):

            if natural_key not in params:

                continue

            natural = params.pop(natural_key)

            iso = parse_to_iso(natural, current_time)

            if iso is None:

                errors.append(f"Could not parse date '{natural}' for {fn}")

            else:

                params[iso_key] = iso

                print(f"✅ {fn}: '{natural}' → '{iso}'")

    return errors




def _apply_duration_parsing(calls: list, current_time: datetime | None = None) -> list[str]:
    errors = []
    now = current_time or datetime.now(ZoneInfo(TIMEZONE))
    for call in calls:
        fn = call.get("function")
        params = call.get("params", {})
        for natural_key, target_key, kind in DURATION_FIELD_MAP.get(fn, []):
            if target_key in params:
                params.pop(natural_key, None)
                continue
            if natural_key not in params:
                continue
            natural = params.pop(natural_key)
            if kind == "minutes":
                value = parse_duration_to_minutes(natural)
                if value is None:
                    value = _parse_until_as_minutes(natural, now)
            else:
                value = parse_duration_to_chunks(natural)
                if value is None:
                    mins = _parse_until_as_minutes(natural, now)
                    if mins is not None:
                        value = minutes_to_chunks(mins)
            if value is None:
                errors.append(f"Could not parse duration '{natural}' for {fn}")
            else:
                params[target_key] = value
                print(f"✅ {fn}: duration '{natural}' → {target_key}={value}")
    return errors


def _coerce_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_llm_numeric_params(calls: list) -> list[str]:
    """Accept numeric duration fields from the LLM; convert to canonical param names."""
    errors = []
    for call in calls:
        fn = call.get("function")
        params = call.get("params", {})
        normalizers = _NUMERIC_DURATION_NORMALIZERS.get(fn, ())
        target_key = None
        target_value = None

        for source_key, dest_key in normalizers:
            if source_key not in params:
                continue
            raw = _coerce_int(params.pop(source_key))
            if raw is None:
                errors.append(f"{fn}: invalid numeric {source_key}")
                break
            if dest_key == "additional_chunks" and source_key == "additional_minutes":
                raw = minutes_to_chunks(raw)
            elif dest_key == "additional_minutes" and source_key == "additional_chunks":
                raw = raw * 15
            if target_key is None:
                target_key, target_value = dest_key, raw
            elif dest_key == target_key and raw != target_value:
                print(f"⚠️ {fn}: conflicting {source_key}, keeping first value")
        if target_key is not None:
            params[target_key] = target_value
            natural_keys = {
                "additional_chunks": "additional_time_natural",
                "additional_minutes": "additional_time_natural",
                "time_needed": "time_needed_natural",
            }
            params.pop(natural_keys.get(target_key, ""), None)
            print(f"✅ {fn}: normalized → {target_key}={target_value}")
    return errors




def _normalize_call_params(calls: list) -> list[str]:

    errors = []

    for call in calls:

        fn = call.get("function")

        params = call.get("params", {})

        if fn == "create_task":

            if "event_category" not in params:

                errors.append("create_task: event_category is required (ask work or personal first)")

            else:

                params["event_category"] = normalize_event_category(params["event_category"])

        if fn == "update_task" and "event_category" in params:

            params["event_category"] = normalize_event_category(params["event_category"])

    return errors





def _validate_calls(calls: list, current_time: datetime) -> list[str]:

    errors = []

    known_functions = {

        "create_task",

        "create_event",

        "complete_task",

        "extend_task_total",

        "extend_task_instance",

        "reschedule_task",

        "reschedule_missed_work",

        "reschedule_multiple_missed_work",

        "switch_active_task",

        "resume_previous_task",

        "extend_current_gcal_block",

        "move_due_date",

        "update_task",

        "log_work",

        "get_schedule_for_window",

        "get_break_allowance",

    }

    for call in calls:

        fn = call.get("function")

        if fn not in known_functions:

            errors.append(f"Unknown function: {fn}")

            continue

        params = call.get("params", {})

        if fn == "create_task" and "event_category" not in params:

            errors.append("create_task: missing event_category")

        if fn == "extend_task_total" and "additional_chunks" not in params:
            errors.append("extend_task_total: missing additional time (natural phrase or chunks)")

        if fn == "extend_task_instance" and "additional_minutes" not in params:
            errors.append("extend_task_instance: missing additional time")

        if fn == "extend_current_gcal_block" and "additional_minutes" not in params:
            errors.append("extend_current_gcal_block: missing additional time")

        if fn == "move_due_date" and "due_date" not in params:
            errors.append("move_due_date: missing due date")

        if fn == "update_task" and not any(

            k in params

            for k in ("due_date", "event_category", "time_needed", "snooze_until")

        ):

            errors.append("update_task: at least one field to update is required")

        for _, iso_key in DATE_FIELD_MAP.get(fn, []):

            if iso_key in params:

                try:

                    parsed = datetime.fromisoformat(params[iso_key])

                    if parsed.tzinfo is None:

                        parsed = parsed.replace(tzinfo=ZoneInfo(TIMEZONE))

                    if parsed < current_time:

                        errors.append(f"{fn}: {iso_key} is in the past")

                except ValueError:

                    errors.append(f"{fn}: invalid {iso_key}")

    return errors


READ_ONLY_DISPATCH_FUNCS = frozenset({"get_schedule_for_window", "get_break_allowance"})


def _last_user_text(message: list) -> str:
    for msg in reversed(message):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def repair_stalled_read_response(data: dict, user_text: str) -> dict:
    """Force read-only tool calls when the LLM stalled or skipped dispatch."""
    if data.get("clarification_required"):
        return data

    calls = data.get("calls") or []
    if data.get("action_required") and any(
        c.get("function") in READ_ONLY_DISPATCH_FUNCS for c in calls
    ):
        return data

    from buffer_analysis import parse_break_allowance_request
    from queries import is_deferral_schedule_reply, parse_schedule_read_request

    if parse_break_allowance_request(user_text):
        data["action_required"] = True
        data["calls"] = [{"function": "get_break_allowance", "params": {}}]
        data["validation_errors"] = []
        return data

    schedule_params = parse_schedule_read_request(user_text)
    if not schedule_params:
        return data

    stall = is_deferral_schedule_reply(data.get("reply", ""))
    if stall or not data.get("action_required") or not calls:
        data["action_required"] = True
        data["calls"] = [
            {"function": "get_schedule_for_window", "params": schedule_params}
        ]
        data["validation_errors"] = []
    return data


async def call_llm(

    message: list,

    current_task: dict | None = None,

    context_snapshot: str | None = None,

    amendment_context: str | None = None,

    clarification_context: str | None = None,

    model: str = "openai/gpt-oss-120b",

) -> dict:
    """Call the LLM with conversation context and normalize/validate any tool calls."""

    current_time = datetime.now(ZoneInfo(TIMEZONE))

    if current_task and current_task.get("title"):

        ctx = f"{current_task['title']} (started {current_task['start_time']})"

    else:

        ctx = "None"



    snapshot_block = cap_snapshot(context_snapshot)

    category_lines = []
    for msg in message:
        if msg.get("role") != "user":
            continue
        cat = extract_event_category_from_message(msg.get("content", ""))
        if cat:
            category_lines.append(
                f"User already specified event_category={cat}; do not ask work or personal again."
            )
    if category_lines:
        hint = "\n".join(dict.fromkeys(category_lines))
        snapshot_block = f"{snapshot_block}\n\n{hint}" if snapshot_block else hint

    if amendment_context:

        snapshot_block = (

            f"{snapshot_block}\n\n{amendment_context}"

            if snapshot_block

            else amendment_context

        )

    if clarification_context:

        snapshot_block = (

            f"{snapshot_block}\n\n{clarification_context}"

            if snapshot_block

            else clarification_context

        )

    snapshot_block = cap_snapshot(snapshot_block)



    formatted_prompt = SYSTEM_PROMPT_CAL.format(

        now=current_time.strftime("%A, %B %d, %Y at %I:%M %p %Z"),

        weekday=current_time.strftime("%A"),

        current_task_context=ctx,

        context_snapshot=snapshot_block,

    )



    full_messages = [{"role": "system", "content": formatted_prompt}] + prepare_messages_for_llm(

        message

    )



    from model_router import AllModelsExhausted, complete_chat

    try:
        llm_result = await complete_chat(
            full_messages,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            temperature=0.3,
        )
    except AllModelsExhausted as e:
        return {
            "action_required": False,
            "clarification_required": False,
            "reply": str(e),
            "calls": [],
        }

    raw = llm_result.raw

    data = _normalize_llm_response(_parse_llm_json(raw))

    user_text = _last_user_text(message)
    data = repair_stalled_read_response(data, user_text)

    if DEV_RELOAD:
        print(f"LLM model: {llm_result.display_name} ({llm_result.model_id})")
        print(f"LLM raw JSON:\n{raw}")
        print(f"LLM parsed: {json.dumps(data, ensure_ascii=False)}")



    if data["action_required"] and data["calls"]:

        date_errors = _apply_date_parsing(data["calls"], current_time)

        numeric_errors = _normalize_llm_numeric_params(data["calls"])

        duration_errors = _apply_duration_parsing(data["calls"], current_time)

        param_errors = _normalize_call_params(data["calls"])

        validation_errors = _validate_calls(data["calls"], current_time)

        all_errors = date_errors + duration_errors + numeric_errors + param_errors + validation_errors

        if all_errors:

            # Never surface raw errors to the user (spec 21). Drop the unsafe calls
            # and hand structured errors back so the bot can route to a clarification
            # instead of executing something wrong.

            data["action_required"] = False

            data["calls"] = []

            data["validation_errors"] = all_errors



    return data


async def call_llm_benchmark(
    message: list,
    current_task: dict | None = None,
    context_snapshot: str | None = None,
    amendment_context: str | None = None,
    clarification_context: str | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Like call_llm but returns usage, raw output, and validation errors for benchmarking."""
    current_time = datetime.now(ZoneInfo(TIMEZONE))

    if current_task and current_task.get("title"):
        ctx = f"{current_task['title']} (started {current_task['start_time']})"
    else:
        ctx = "None"

    snapshot_block = cap_snapshot(context_snapshot)
    category_lines = []
    for msg in message:
        if msg.get("role") != "user":
            continue
        cat = extract_event_category_from_message(msg.get("content", ""))
        if cat:
            category_lines.append(
                f"User already specified event_category={cat}; do not ask work or personal again."
            )
    if category_lines:
        hint = "\n".join(dict.fromkeys(category_lines))
        snapshot_block = f"{snapshot_block}\n\n{hint}" if snapshot_block else hint

    if amendment_context:
        snapshot_block = (
            f"{snapshot_block}\n\n{amendment_context}"
            if snapshot_block
            else amendment_context
        )

    if clarification_context:
        snapshot_block = (
            f"{snapshot_block}\n\n{clarification_context}"
            if snapshot_block
            else clarification_context
        )

    snapshot_block = cap_snapshot(snapshot_block)

    formatted_prompt = SYSTEM_PROMPT_CAL.format(
        now=current_time.strftime("%A, %B %d, %Y at %I:%M %p %Z"),
        weekday=current_time.strftime("%A"),
        current_task_context=ctx,
        context_snapshot=snapshot_block,
    )

    full_messages = [{"role": "system", "content": formatted_prompt}] + prepare_messages_for_llm(
        message
    )

    response = await client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
        temperature=0.3,
        **_groq_model_kwargs(model),
    )

    raw = response.choices[0].message.content or ""
    parsed = _parse_llm_json(raw)
    json_ok = "_parse_failed" not in parsed
    data = _normalize_llm_response(parsed)

    validation_errors: list[str] = []
    if data["action_required"] and data["calls"]:
        validation_errors = (
            _apply_date_parsing(data["calls"], current_time)
            + _normalize_llm_numeric_params(data["calls"])
            + _apply_duration_parsing(data["calls"], current_time)
            + _normalize_call_params(data["calls"])
            + _validate_calls(data["calls"], current_time)
        )
        if validation_errors:
            data["action_required"] = False
            data["calls"] = []
            data["reply"] = (
                data.get("reply", "")
                + "\n\n(I couldn't safely run that — "
                + "; ".join(validation_errors)
                + ")"
            )

    usage = response.usage
    return {
        "data": data,
        "raw": raw,
        "json_ok": json_ok,
        "validation_errors": validation_errors,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        },
        "system_prompt_chars": len(formatted_prompt),
    }


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe voice note bytes via Groq Whisper."""

    transcription = await client.audio.transcriptions.create(

        file=("voice.ogg", audio_bytes),

        model="whisper-large-v3-turbo",

    )

    return transcription.text

