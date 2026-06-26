from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from config import TELEGRAM_USER_ID, TELEGRAM_BOT_TOKEN
from reclaim import upcoming_info
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
import asyncio
from telegram.request import HTTPXRequest

scheduler = AsyncIOScheduler()


async def start_command(update : Update, context : ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id != TELEGRAM_USER_ID:
        print(f'Blocked unauthorized access attempt from User ID: {user_id}')
        return
    
    await update.message.reply_text('ARIA is live...')
    print('Sent /start confirmation to telegram')

async def echo_message(update : Update, context : ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    incoming_text = update.message.text
    if user_id != TELEGRAM_USER_ID:
        return
    
    reply_text = f'Fym {incoming_text}??'

    await update.message.reply_text(reply_text)
    print(f'Message {incoming_text} is echoed')

def build_bot():
        
    request = HTTPXRequest(
    connect_timeout=15,
    read_timeout=30,
    )
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    application.add_handler(CommandHandler("start", start_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))

    return application


def schedule_ping(task_title: str, start_time: str):
    fire_time = datetime.fromisoformat(start_time)

    scheduler.add_job(
        send_block_start,
        trigger='date',
        run_date=fire_time,
        kwargs={'title' : task_title},
        id=f'block_start{start_time}',
        replace_existing=True,
        misfire_grace_time=60
    )
    print(f'{task_title} will send at {start_time}')

async def send_block_start(title: str):
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
                wait = 2 ** attempt  # 1s, then 2s
                print(f'Send failed (attempt {attempt + 1}), retrying in {wait}s...')
                await asyncio.sleep(wait)
    except Exception as e:
        print(f'catastropgic failure {e}')
    finally:
        prep_next_block()

def prep_next_block():
    title, start = upcoming_info()
    schedule_ping(title, start)
bot_app = build_bot()