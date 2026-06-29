"""Telegram bot handlers, scheduling pings, and message routing."""

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from composites import route_ping_calls
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID, MAX_VOICE_DURATION_SEC, TIMEZONE
from buffer_analysis import assess_break_request, gate_break_calls
from inference import call_llm, transcribe_audio
from intent import (
    detect_intent,
    extract_break_window,
    extract_extend_task_call,
    extract_snooze_hint,
    extract_task_hint,
    is_amendment_message,
    is_break_confirmation,
    is_break_rejection,
    is_category_clarification_reply,
    is_undo_or_cancel,
)
from queries import build_weekly_snapshot, format_schedule_reply
from reclaim import (
    enrich_event_context,
    get_all_tasks,
    get_next_event,
    refresh_ongoing_state,
)
from rollback import (
    build_amendment_context,
    execute_calls,
    get_last_completed,
    get_ledger_task_ids,
    handle_undo_or_cancel,
)
from token_guardrails import (
    cap_incoming_text,
    cap_snapshot,
    router_reply,
    trim_conversation_buffer,
)

scheduler = AsyncIOScheduler()
current_task: dict = {"title": None, "start_time": None}
conversation_buffer: list = []
pending_break: dict | None = None
_last_pinged_event_id: str | None = None

_prep_lock = asyncio.Lock()


async def send_with_retry(coro_factory, label: str = "send"):
    """Retry a Telegram send coroutine on transient network errors."""
    for attempt in range(3):
        try:
            return await coro_factory()
        except (TimedOut, NetworkError) as e:
            if attempt == 2:
                print(f"❌ Failed to {label} after 3 attempts: {e}")
                raise
            wait = 2**attempt
            print(f"⚠️ {label} failed (attempt {attempt + 1}), retrying in {wait}s...")
            await asyncio.sleep(wait)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start for the authorized user only."""
    user_id = update.effective_user.id
    if user_id != TELEGRAM_USER_ID:
        print(f"Blocked unauthorized access attempt from User ID: {user_id}")
        return
    await update.message.reply_text("ARIA is live...")


def _build_missed_work_calls(incoming_text: str) -> list[dict]:
    params: dict = {"task_query": extract_task_hint(incoming_text)}
    snooze = extract_snooze_hint(incoming_text)
    if snooze:
        params["snooze_until_natural"] = snooze
    return [{"function": "reschedule_missed_work", "params": params}]


async def _post_break_at_risk_note() -> str:
    tasks = await get_all_tasks(instances=True, force_refresh=True)
    names = [
        t.get("title", "task")
        for t in tasks
        if t.get("atRisk")
        and t.get("status") in ("IN_PROGRESS", "SCHEDULED")
    ]
    if not names:
        return ""
    if len(names) == 1:
        return f"\n\n⚠️ Reclaim flags {names[0]} as at-risk now."
    return f"\n\n⚠️ Reclaim flags these as at-risk: {', '.join(names)}."


async def _execute_break_calls(update, calls: list[dict], incoming_text: str, amendment_context):
    global current_task
    ledger_ids = get_ledger_task_ids() if amendment_context else None
    result = await execute_calls(
        calls,
        user_message=incoming_text,
        current_task=current_task,
        preferred_task_ids=ledger_ids,
    )
    reply = ""
    if result["failed"]:
        reply += "\n\n⚠️ Some actions failed: " + "; ".join(result["failed"])
    elif result["summaries"]:
        reply += "\n\n✅ " + "; ".join(result["summaries"])
    reply += result.get("undo_hint", "")
    if "create_event" in result.get("succeeded", []):
        reply += await _post_break_at_risk_note()
    await send_with_retry(
        lambda: update.message.reply_text(reply.lstrip("\n") or "Done."),
        label="break confirm reply",
    )


async def process_message(update: Update, incoming_text: str):
    """Route one user message through intent detection, LLM, and action execution."""
    global current_task, pending_break

    incoming_text = cap_incoming_text(incoming_text)
    conversation_buffer.append({"role": "user", "content": incoming_text})
    trim_conversation_buffer(conversation_buffer)

    if is_undo_or_cancel(incoming_text):
        conversation_buffer.clear()
        undo_reply = await handle_undo_or_cancel()
        await send_with_retry(
            lambda: update.message.reply_text(undo_reply or "Nothing to undo or cancel."),
            label="undo/cancel reply",
        )
        return

    if pending_break:
        if is_break_confirmation(incoming_text):
            calls = pending_break.get("calls", [])
            pending_break = None
            conversation_buffer.clear()
            if calls:
                await _execute_break_calls(
                    update, calls, incoming_text, amendment_context=None
                )
            else:
                await send_with_retry(
                    lambda: update.message.reply_text("On it!"),
                    label="break confirm empty",
                )
            return
        if is_break_rejection(incoming_text):
            pending_break = None
            conversation_buffer.clear()
            await send_with_retry(
                lambda: update.message.reply_text("Okay — no break then."),
                label="break reject reply",
            )
            return
        pending_break = None

    intent = detect_intent(incoming_text)

    amendment_context = None
    if get_last_completed() and (
        intent in ("correction", "amend")
        or is_amendment_message(incoming_text)
    ) and not is_category_clarification_reply(incoming_text):
        amendment_context = await build_amendment_context()

    context_snapshot = None
    if intent == "read_schedule":
        context_snapshot = cap_snapshot(await build_weekly_snapshot())

    router_calls = None
    data = None
    if intent == "missed_work":
        router_calls = _build_missed_work_calls(incoming_text)
    elif intent in ("extend_time", "switch_task") and current_task.get("title"):
        active = refresh_ongoing_state(current_task)
        router_calls = await route_ping_calls(intent, incoming_text, active)
    elif intent not in ("take_break", "read_schedule", "missed_work"):
        extend_calls = extract_extend_task_call(incoming_text)
        if extend_calls:
            router_calls = extend_calls
            intent = "extend_task"
    elif intent == "take_break":
        now = datetime.now(ZoneInfo(TIMEZONE))
        break_start, break_end = extract_break_window(incoming_text, now)
        assessment = await assess_break_request(break_start, break_end)
        if assessment.clarification_required:
            pending_break = {"calls": assessment.calls}
            data = {
                "action_required": False,
                "clarification_required": True,
                "reply": assessment.reply,
                "calls": [],
            }
        elif assessment.action_required:
            data = {
                "action_required": True,
                "clarification_required": False,
                "reply": assessment.reply,
                "calls": assessment.calls,
            }
        else:
            data = {
                "action_required": False,
                "clarification_required": False,
                "reply": assessment.reply,
                "calls": [],
            }

    if intent == "read_schedule" and context_snapshot is not None:
        data = {
            "action_required": False,
            "clarification_required": False,
            "reply": format_schedule_reply(context_snapshot),
            "calls": [],
        }
    elif router_calls:
        data = {
            "action_required": True,
            "clarification_required": False,
            "reply": router_reply(intent, router_calls),
            "calls": router_calls,
        }
    elif data is None:
        data = await call_llm(
            message=conversation_buffer,
            current_task=current_task,
            context_snapshot=context_snapshot,
            amendment_context=amendment_context,
        )
        if data.get("action_required") and data.get("calls"):
            gated = await gate_break_calls(data["calls"])
            if gated is not None:
                if gated.clarification_required:
                    pending_break = {"calls": gated.calls}
                    data = {
                        "action_required": False,
                        "clarification_required": True,
                        "reply": gated.reply,
                        "calls": [],
                    }
                elif gated.action_required:
                    data = {
                        "action_required": True,
                        "clarification_required": False,
                        "reply": gated.reply,
                        "calls": gated.calls,
                    }
                else:
                    data = {
                        "action_required": False,
                        "clarification_required": False,
                        "reply": gated.reply,
                        "calls": [],
                    }

    if data.get("clarification_required"):
        conversation_buffer.append({"role": "assistant", "content": data["reply"]})
        trim_conversation_buffer(conversation_buffer)
    else:
        conversation_buffer.clear()

    reply = data.get("reply", "...")

    if data.get("action_required") and data.get("calls") and not data.get("clarification_required"):
        ledger_ids = get_ledger_task_ids() if amendment_context else None
        result = await execute_calls(
            data["calls"],
            user_message=incoming_text,
            current_task=current_task,
            preferred_task_ids=ledger_ids,
        )
        if result["failed"]:
            reply += "\n\n⚠️ Some actions failed: " + "; ".join(result["failed"])
        elif result["summaries"]:
            reply += "\n\n✅ " + "; ".join(result["summaries"])
        reply += result.get("undo_hint", "")
        if "create_event" in result.get("succeeded", []):
            reply += await _post_break_at_risk_note()

    await send_with_retry(
        lambda: update.message.reply_text(reply),
        label="reply to user",
    )


async def message_master(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram handler for text messages."""
    if update.effective_user.id != TELEGRAM_USER_ID:
        return
    try:
        await process_message(update, update.message.text)
    except Exception as e:
        print(f"❌ message_master error: {e}")
        try:
            await send_with_retry(
                lambda: update.message.reply_text(
                    "Something went wrong on my end — try again?"
                ),
                label="error reply",
            )
        except Exception:
            pass


async def voice_master(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram handler for voice notes (transcribe then process as text)."""
    if update.effective_user.id != TELEGRAM_USER_ID:
        return
    try:
        voice_msg = update.message.voice
        if voice_msg.duration > MAX_VOICE_DURATION_SEC:
            await send_with_retry(
                lambda: update.message.reply_text(
                    f"Voice notes must be {MAX_VOICE_DURATION_SEC} seconds or shorter — "
                    "try a shorter message or type it out?"
                ),
                label="voice duration reply",
            )
            return
        voice = await voice_msg.get_file()
        audio_bytes = await voice.download_as_bytearray()
        incoming_text = await transcribe_audio(bytes(audio_bytes))
        incoming_text = cap_incoming_text(incoming_text)
        await process_message(update, incoming_text)
    except Exception as e:
        print(f"❌ voice_master error: {e}")
        try:
            await send_with_retry(
                lambda: update.message.reply_text(
                    "Couldn't process that voice note — try again or type it out?"
                ),
                label="voice error reply",
            )
        except Exception:
            pass


def build_bot():
    """Construct the Telegram Application with handlers registered."""
    request = HTTPXRequest(connect_timeout=15, read_timeout=30)
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .build()
    )
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_master))
    application.add_handler(MessageHandler(filters.VOICE, voice_master))
    return application


def schedule_ping(ctx: dict):
    """Schedule a one-shot job to ping when a calendar block starts."""
    start_time = ctx["start_time"]
    task_title = ctx.get("title", "block")
    fire_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if fire_time.tzinfo is None:
        fire_time = fire_time.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if fire_time < now:
        print(f"⏭️ Skipping past event '{task_title}' at {start_time}")
        return

    scheduler.add_job(
        send_block_start,
        trigger="date",
        run_date=fire_time,
        kwargs=ctx,
        id=f"block_start_{start_time}",
        replace_existing=True,
        misfire_grace_time=60,
    )
    print(f"✅ {task_title} will ping at {start_time}")


async def send_block_start(**kwargs):
    """Notify the user that a scheduled block is starting and queue the next ping."""
    global current_task, _last_pinged_event_id
    current_task = {k: v for k, v in kwargs.items() if v is not None}
    _last_pinged_event_id = current_task.get("event_id")

    title = current_task.get("title", "your block")
    try:
        await send_with_retry(
            lambda: bot_app.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=f"You have {title} scheduled for now, lmk if that changed",
            ),
            label="block start ping",
        )
    except Exception as e:
        print(f"❌ block start ping failed: {e}")
    finally:
        await prep_next_block()


async def prep_next_block():
    """Find the next upcoming event and schedule its start ping."""
    global _last_pinged_event_id
    async with _prep_lock:
        event = await get_next_event(exclude_event_id=_last_pinged_event_id)
        if event is None:
            print("ℹ️ No upcoming events to schedule")
            return
        ctx = enrich_event_context(event)
        schedule_ping(ctx)


bot_app = build_bot()
