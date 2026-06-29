"""LLM call path: parse natural dates/durations and validate tool calls."""

import json

import re

from datetime import datetime



import parsedatetime as pdt

from groq import AsyncGroq

from zoneinfo import ZoneInfo



from config import GROQ_TOKEN, LLM_MAX_OUTPUT_TOKENS, TIMEZONE
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
    "extend_current_block": [("additional_time_natural", "additional_minutes", "minutes")],
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
    "extend_current_block": [
        ("additional_minutes", "additional_minutes"),
        ("additional_chunks", "additional_minutes"),
    ],
    "create_task": [("time_needed", "time_needed")],
    "update_task": [("time_needed", "time_needed")],
}





def parse_to_iso(natural_date_str: str, base_date: datetime) -> str | None:
    """Parse natural-language date/time to ISO string in local timezone."""

    if not natural_date_str:

        return None



    time_struct, parse_status = cal.parse(natural_date_str, sourceTime=base_date)

    if parse_status > 0:

        parsed = datetime(*time_struct[:6], tzinfo=ZoneInfo(TIMEZONE))

        return parsed.isoformat()

    return None





def normalize_bool(val) -> bool:
    """Coerce LLM boolean fields from bool, str, or other truthy values."""

    if isinstance(val, bool):

        return val

    if isinstance(val, str):

        return val.strip().lower() in ("true", "yes", "1")

    return bool(val)





def _parse_llm_json(raw: str) -> dict:

    try:

        return json.loads(raw)

    except json.JSONDecodeError:

        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)

        if match:

            return json.loads(match.group())

        return {

            "action_required": False,

            "clarification_required": False,

            "reply": raw,

            "calls": [],

        }





def _normalize_llm_response(data: dict) -> dict:

    action_required = normalize_bool(data.get("action_required", False))

    clarification_required = normalize_bool(data.get("clarification_required", False))



    if clarification_required:

        action_required = False

        data["calls"] = []

    elif not action_required:

        clarification_required = False

        data["calls"] = []



    data["action_required"] = action_required

    data["clarification_required"] = clarification_required

    data.setdefault("reply", "...")

    data.setdefault("calls", [])

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




def _apply_duration_parsing(calls: list) -> list[str]:
    errors = []
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
            else:
                value = parse_duration_to_chunks(natural)
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

        "switch_active_task",

        "extend_current_block",

        "update_task",

        "log_work",

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

        if fn == "extend_current_block" and "additional_minutes" not in params:
            errors.append("extend_current_block: missing additional time")

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





async def call_llm(

    message: list,

    current_task: dict | None = None,

    context_snapshot: str | None = None,

    amendment_context: str | None = None,

    model: str = "llama-3.3-70b-versatile",

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

    )

    raw = response.choices[0].message.content

    data = _normalize_llm_response(_parse_llm_json(raw))



    if data["action_required"] and data["calls"]:

        date_errors = _apply_date_parsing(data["calls"], current_time)

        numeric_errors = _normalize_llm_numeric_params(data["calls"])

        duration_errors = _apply_duration_parsing(data["calls"])

        param_errors = _normalize_call_params(data["calls"])

        validation_errors = _validate_calls(data["calls"], current_time)

        all_errors = date_errors + duration_errors + numeric_errors + param_errors + validation_errors

        if all_errors:

            data["action_required"] = False

            data["calls"] = []

            data["reply"] = (

                data.get("reply", "")

                + "\n\n(I couldn't safely run that — "

                + "; ".join(all_errors)

                + ")"

            )



    return data





async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe voice note bytes via Groq Whisper."""

    transcription = await client.audio.transcriptions.create(

        file=("voice.ogg", audio_bytes),

        model="whisper-large-v3-turbo",

    )

    return transcription.text

