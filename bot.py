"""Telegram bot handlers, scheduling pings, and message routing."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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

import schedule_cache
from bug_log import append_bug_report
import queries
from buffer_analysis import assess_break_request, gate_break_calls
from clarification import (
    PendingClarification,
    apply_llm_clarification_fields,
    build_clarification_context,
    clear_expired,
    compute_missing_fields,
    gate_task_queries,
    merge_clarification_reply,
)
from config import (
    DEFER_WINDOW_SEC,
    MAX_VOICE_DURATION_SEC,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_USER_ID,
    TIMEZONE,
)
from inference import call_llm, parse_upcoming_time, transcribe_audio
from intent import (
    extract_break_window,
    is_break_confirmation,
    is_break_rejection,
    is_bug_log_request,
    is_snooze_request,
    is_take_break_request,
    is_undo_or_cancel,
    parse_bug_log_body,
    parse_snooze_spec,
)
from reclaim import (
    enrich_event_context,
    get_all_tasks,
    get_next_event,
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
    trim_conversation_buffer,
)

# Compound actions that hold for a 30s deferred buffer before firing (spec 4/7).
DEFERRED_FUNCTIONS = frozenset({"switch_active_task", "resume_previous_task"})

scheduler = AsyncIOScheduler()
current_task: dict = {"title": None, "start_time": None}
previous_task: dict | None = None
conversation_buffer: list = []
pending_break: dict | None = None
pending_clarification: PendingClarification | None = None
pending_deferred: "PendingDeferred | None" = None

_prep_lock = asyncio.Lock()


@dataclass
class PendingDeferred:
    """A compound action queued to fire after the 30s deferred buffer."""

    calls: list
    user_message: str
    reply: str
    update: Update
    amendment_context: str | None = None
    task: asyncio.Task | None = None
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(ZoneInfo(TIMEZONE))
    )


def classify_deferral(calls: list[dict]) -> bool:
    """True if any call is a compound action that should be held for the buffer."""
    return any(c.get("function") in DEFERRED_FUNCTIONS for c in calls)


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


async def _execute_and_reply(
    update: Update,
    calls: list[dict],
    incoming_text: str,
    reply: str,
    amendment_context: str | None = None,
) -> str:
    global pending_clarification

    ledger_ids = get_ledger_task_ids() if amendment_context else None
    result = await execute_calls(
        calls,
        user_message=incoming_text,
        current_task=current_task,
        preferred_task_ids=ledger_ids,
    )
    if result["summaries"]:
        reply += "\n\n" + "; ".join(result["summaries"])
    if result["failed"]:
        # Never surface raw errors (spec 21); ask the user to clarify instead.
        reply += (
            "\n\nHmm, I couldn't finish part of that — which task did you mean?"
        )
    reply += result.get("undo_hint", "")
    if "create_event" in result.get("succeeded", []):
        reply += await _post_break_at_risk_note()
    if "create_task" in result.get("succeeded", []):
        pending_clarification = None
    await send_with_retry(
        lambda: update.message.reply_text(reply.lstrip("\n") or "Done."),
        label="reply to user",
    )
    return reply


def _cancel_pending_deferred() -> "PendingDeferred | None":
    """Cancel and clear any queued deferred compound action."""
    global pending_deferred
    pd = pending_deferred
    pending_deferred = None
    if pd is not None and pd.task is not None and not pd.task.done():
        pd.task.cancel()
    return pd


def _queue_deferred(
    update: Update,
    calls: list[dict],
    incoming_text: str,
    reply: str,
    amendment_context: str | None,
) -> None:
    """Hold a compound action for DEFER_WINDOW_SEC so the user can amend/cancel it."""
    global pending_deferred
    _cancel_pending_deferred()

    pd = PendingDeferred(
        calls=calls,
        user_message=incoming_text,
        reply=reply,
        update=update,
        amendment_context=amendment_context,
        expires_at=datetime.now(ZoneInfo(TIMEZONE)) + timedelta(seconds=DEFER_WINDOW_SEC),
    )

    async def _fire():
        try:
            await asyncio.sleep(DEFER_WINDOW_SEC)
        except asyncio.CancelledError:
            return
        await _fire_deferred(pd)

    pd.task = asyncio.create_task(_fire())
    pending_deferred = pd


async def _fire_deferred(pd: "PendingDeferred") -> None:
    """Execute a queued compound action once its buffer elapses."""
    global pending_deferred
    if pending_deferred is not pd:
        return
    pending_deferred = None
    try:
        await _execute_and_reply(
            pd.update,
            pd.calls,
            pd.user_message,
            pd.reply,
            amendment_context=pd.amendment_context,
        )
    except Exception as e:
        print(f"❌ deferred action failed: {e}")


async def _handle_snooze(update: Update, incoming_text: str, spec: dict) -> None:
    """Tier-1 snooze: resolve the task and reschedule it (no LLM, spec 3/11)."""
    natural = spec.get("snooze_until_natural")
    params: dict = {"task_query": spec["task_query"]}
    if natural:
        iso = parse_upcoming_time(natural, datetime.now(ZoneInfo(TIMEZONE)))
        if iso:
            params["snooze_until"] = iso
    call = {"function": "reschedule_task", "params": params}
    await _execute_and_reply(update, [call], incoming_text, "On it — snoozing that.")


async def process_message(update: Update, incoming_text: str):
    """Route one user message through LLM-first handling and action execution."""
    global current_task, pending_break, pending_clarification

    incoming_text = cap_incoming_text(incoming_text)
    pending_clarification, _ = clear_expired(pending_clarification, None)

    conversation_buffer.append({"role": "user", "content": incoming_text})
    trim_conversation_buffer(conversation_buffer)

    # ── Tier-1: undo / cancel (no LLM) ────────────────────────────────────────
    if is_undo_or_cancel(incoming_text):
        conversation_buffer.clear()
        pending_clarification = None
        # A queued compound action hasn't fired yet — just cancel it.
        cancelled = _cancel_pending_deferred()
        if cancelled is not None:
            await send_with_retry(
                lambda: update.message.reply_text("Okay — cancelled that."),
                label="cancel deferred reply",
            )
            return
        undo_reply = await handle_undo_or_cancel()
        await send_with_retry(
            lambda: update.message.reply_text(undo_reply or "Nothing to undo or cancel."),
            label="undo/cancel reply",
        )
        return

    # Any other message while a compound action is queued amends it: cancel the
    # queued action and reprocess this message fresh (spec 4/7).
    if pending_deferred is not None:
        _cancel_pending_deferred()

    # ── Tier-1: bug log (no LLM) ─────────────────────────────────────────────
    if is_bug_log_request(incoming_text):
        conversation_buffer.clear()
        pending_clarification = None
        body = parse_bug_log_body(incoming_text)
        append_bug_report(
            body,
            user_id=update.effective_user.id,
            raw_message=incoming_text,
            context={"current_task_title": (current_task or {}).get("title")},
        )
        await send_with_retry(
            lambda: update.message.reply_text("bug logged"),
            label="bug log reply",
        )
        return

    # ── Tier-1: snooze (no LLM) ───────────────────────────────────────────────
    if is_snooze_request(incoming_text):
        spec = parse_snooze_spec(incoming_text)
        if spec:
            conversation_buffer.clear()
            pending_clarification = None
            await _handle_snooze(update, incoming_text, spec)
            return

    if pending_break:
        if is_break_confirmation(incoming_text):
            calls = pending_break.get("calls", [])
            pending_break = None
            conversation_buffer.clear()
            if calls:
                await _execute_and_reply(update, calls, incoming_text, "On it!")
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

    clarification_context = None
    if pending_clarification:
        merged = merge_clarification_reply(pending_clarification, incoming_text)
        pending_clarification.partial_params = merged
        pending_clarification.missing_fields = compute_missing_fields(
            pending_clarification.kind, merged
        )
        clarification_context = build_clarification_context(pending_clarification)

    amendment_context = None
    if get_last_completed():
        amendment_context = await build_amendment_context()

    data = None

    if is_take_break_request(incoming_text):
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
    else:
        context_snapshot = await queries.build_weekly_snapshot()
        data = await call_llm(
            message=conversation_buffer,
            current_task=current_task,
            context_snapshot=context_snapshot,
            amendment_context=amendment_context,
            clarification_context=clarification_context,
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

    # Validation stripped the calls (bad params/dates): never error, just ask (spec 21).
    if data.get("validation_errors"):
        conversation_buffer.clear()
        reply = data.get("reply") or ""
        if reply.strip() and not data.get("action_required"):
            reply = (
                "I couldn't safely run that yet — can you give me a bit more detail?"
            )
        if not reply.strip():
            reply = "I couldn't quite act on that safely — can you give me a bit more detail?"
        await send_with_retry(
            lambda: update.message.reply_text(reply),
            label="validation clarify reply",
        )
        return

    if data.get("clarification_required"):
        new_pending = apply_llm_clarification_fields(data)
        if new_pending:
            if pending_clarification and pending_clarification.kind == new_pending.kind:
                new_pending.partial_params = {
                    **pending_clarification.partial_params,
                    **new_pending.partial_params,
                }
                new_pending.missing_fields = compute_missing_fields(
                    new_pending.kind, new_pending.partial_params
                )
            pending_clarification = new_pending
        conversation_buffer.append({"role": "assistant", "content": data["reply"]})
        trim_conversation_buffer(conversation_buffer)
        await send_with_retry(
            lambda: update.message.reply_text(data.get("reply", "...")),
            label="clarification reply",
        )
        return

    reply = data.get("reply", "...")

    if data.get("action_required") and data.get("calls"):
        ledger_ids = get_ledger_task_ids() if amendment_context else None
        # Task-query gate: unresolvable/placeholder references clarify, never error (spec 21).
        gate = await gate_task_queries(
            data["calls"],
            current_task=current_task,
            preferred_task_ids=ledger_ids,
        )
        if not gate.ok and gate.clarification_required:
            pending_clarification = gate.pending
            conversation_buffer.append({"role": "assistant", "content": gate.reply})
            trim_conversation_buffer(conversation_buffer)
            await send_with_retry(
                lambda: update.message.reply_text(gate.reply),
                label="gate clarify reply",
            )
            return

        conversation_buffer.clear()
        pending_clarification = None

        # Compound actions hold for a 30s buffer so the user can amend/cancel.
        if classify_deferral(data["calls"]):
            _queue_deferred(update, data["calls"], incoming_text, reply, amendment_context)
            await send_with_retry(
                lambda: update.message.reply_text(
                    (reply.rstrip() + f"\n\n(Locking that in {DEFER_WINDOW_SEC}s — "
                     "tell me if you want to change it.)").lstrip("\n")
                ),
                label="deferred queued reply",
            )
            return

        await _execute_and_reply(
            update,
            data["calls"],
            incoming_text,
            reply,
            amendment_context=amendment_context,
        )
        return

    conversation_buffer.clear()
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
    global current_task, previous_task
    # Remember the task we were pinged on before overwriting (the "last task").
    if current_task.get("task_id") or current_task.get("title"):
        previous_task = current_task
    current_task = {k: v for k, v in kwargs.items() if v is not None}
    await schedule_cache.refresh()

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
    async with _prep_lock:
        event = await get_next_event(exclude_event_id=current_task.get("event_id"))
        if event is None:
            print("ℹ️ No upcoming events to schedule")
            return
        ctx = enrich_event_context(event)
        schedule_ping(ctx)


bot_app = build_bot()


async def _notify_model_switch(from_name: str, to_name: str, reason: str) -> None:
    text = f"Switching to {to_name} due to {reason} limit."
    try:
        await send_with_retry(
            lambda: bot_app.bot.send_message(chat_id=TELEGRAM_USER_ID, text=text),
            label="model switch notice",
        )
    except Exception as e:
        print(f"⚠️ model switch notice failed: {e}")


from model_router import set_switch_notifier

set_switch_notifier(_notify_model_switch)
