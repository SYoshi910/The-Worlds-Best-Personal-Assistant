from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from config import TELEGRAM_USER_ID, TELEGRAM_BOT_TOKEN
from reclaim import upcoming_info, get_active_events
from cal_helper import dispatch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
import asyncio
from telegram.request import HTTPXRequest
from inference import call_llm, transcribe_audio

scheduler = AsyncIOScheduler()
current_task = {"title": None, "start_time": None}  # tracks active ping context
conversation_buffer = []

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_USER_ID:
        print(f'Blocked unauthorized access attempt from User ID: {user_id}')
        return
    await update.message.reply_text('ARIA is live...')

async def process_message(update: Update, incoming_text: str):
    global current_task
    conversation_buffer.append({"role": "user", "content": incoming_text})
    print(f'reply from user: {conversation_buffer}')

    data = await call_llm(message=conversation_buffer)
    print(data)

    if data.get("clarification_required"):
        conversation_buffer.append({"role": "assistant", "content": data["reply"]})
    else:
        conversation_buffer.clear()

    if data.get("action_required") and data.get("calls"):
        dispatch(data["calls"])

    for attempt in range(3):
        try:
            await update.message.reply_text(data.get("reply", "..."))
            break
        except (TimedOut, NetworkError) as e:
            if attempt == 2:
                print(f"Failed to reply after 3 attempts: {e}")
            await asyncio.sleep(2 ** attempt)

async def message_master(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_USER_ID:
        return
    await process_message(update, update.message.text)

async def voice_master(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_USER_ID:
        return
    voice = await update.message.voice.get_file()
    audio_bytes = await voice.download_as_bytearray()
    incoming_text = await transcribe_audio(bytes(audio_bytes))
    print(f"Transcribed: {incoming_text}")
    await process_message(update, incoming_text)
    
def build_bot():
    request = HTTPXRequest(connect_timeout=15, read_timeout=30)
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).request(request).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_master))
    application.add_handler(MessageHandler(filters.VOICE, voice_master))
    return application

def schedule_ping(task_title: str, start_time: str):
    fire_time = datetime.fromisoformat(start_time)
    scheduler.add_job(
        send_block_start,
        trigger='date',
        run_date=fire_time,
        kwargs={'title': task_title, 'start_time': start_time},
        id=f'block_start{start_time}',
        replace_existing=True,
        misfire_grace_time=60
    )
    print(f'{task_title} will send at {start_time}')

async def send_block_start(title: str, start_time: str):
    global current_task
    current_task = {"title": title, "start_time": start_time}  # update context

    print('message is about to send...')
    try:
        for attempt in range(3):
            try:
                await bot_app.bot.send_message(
                    chat_id=TELEGRAM_USER_ID,
                    text=f'You have {title} scheduled for now, lmk if that changed'
                )
                print('message sent')
                break
            except (TimedOut, NetworkError) as e:
                if attempt == 2:
                    print(f'Failed to send after 3 attempts: {e}')
                    raise
                wait = 2 ** attempt
                print(f'Send failed (attempt {attempt + 1}), retrying in {wait}s...')
                await asyncio.sleep(wait)
    except Exception as e:
        print(f'catastrophic failure {e}')
    finally:
        prep_next_block()

def prep_next_block():
    title, start = upcoming_info()
    schedule_ping(title, start)

bot_app = build_bot()