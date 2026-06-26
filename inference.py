from groq import AsyncGroq
from config import GROQ_TOKEN
from datetime import datetime, timezone
import json
from reclaim import active_tasks

client = AsyncGroq(api_key=GROQ_TOKEN)

SYSTEM_PROMPT = """You are ARIA, an AI chief of staff managing Sumedh's calendar and tasks via Reclaim.

Current time: {now}
Current task on calendar: {current_task_title} (started {current_task_start})
Upcoming tasks: {upcoming_tasks}

Before responding, reason through:
1. What was scheduled and what has Sumedh actually been doing?
2. Is any action required, or is this just an acknowledgment?
3. If action is required, which functions are needed and in what order?
4. Are there dependent calls (e.g. find a task before acting on it)?
If Sumedh's request isn't clearly mappable to available functions, explain the specific issue you are seeing.

Available functions:
- log_work(task_id, start, end): log a completed work session (ISO 8601 timestamps)
- reschedule_task(task_name, snooze_until): push a task to later, defaults to now
- create_gcal_event(name, start, end): create a calendar event (breaks, commutes, anything non-task)
- create_task(title, due_date, priority, min_chunk_size, max_chunk_size, time_needed): create a new Reclaim task
- extend_task_instance(task_name, additional_minutes): extend the current block right now
- extend_task_total(task_name, additional_chunks): add total time to a task (1 chunk = 15 min)
- complete_task(task_name): mark a task as done

Respond ONLY with a JSON object, no markdown, no backticks:
{{
    "reasoning": "brief explanation of what you inferred",
    "action_required": true or false,
    "reply": "short conversational message to send back to Sumedh",
    "calls": [
        {{
            "function": "function_name",
            "params": {{"param": "value"}},
            "result_alias": "optional_alias"
        }}
    ]
}}

If no action is required, calls should be [].
All timestamps must be ISO 8601 with timezone e.g. 2026-06-26T14:00:00-07:00."""

async def call_llm(
    messages: list,
    model: str = "llama-3.1-8b-instant",
    current_task: dict = None,
    upcoming_events: list = None,
) -> dict:
    current_task = current_task or {}
    upcoming_events = upcoming_events or []

    formatted_prompt = SYSTEM_PROMPT.format(
        now=datetime.now(timezone.utc).isoformat(),
        current_task_title=current_task.get("title", "Nothing scheduled"),
        current_task_start=current_task.get("start_time", "N/A"),
        upcoming_tasks=", ".join([e["title"] for e in upcoming_events]) or "None",
    )

    full_messages = [{"role": "system", "content": formatted_prompt}] + messages

    response = await client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=1024,
        temperature=0.3,
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback if model adds backticks or preamble
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"action_required": False, "reply": raw, "calls": [], "reasoning": "parse failed"}