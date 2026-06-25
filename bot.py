from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from config import TELEGRAM_USER_ID, TELEGRAM_BOT_TOKEN
from reclaim import upcoming_info
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

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
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))

    return application


def schedule_ping(task_title: str, start_time: str):
    fire_time = datetime.fromisoformat(start_time)

    scheduler.add_job(
        send_block_start,
        trigger='date',
        run_date=fire_time,
        kwargs={'title' : task_title}
    )

async def send_block_start(title: str):
    await bot_app.bot.send_message(
        chat_id=TELEGRAM_USER_ID,
        text=f'You have {title} scheduled for now. Do with that what you will'
    )

bot_app = build_bot()
# def task_start_notif(task=None):
#     url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

#     message_text = 'You have an event starting now!'

#     payload = {
#         "chat_id": TELEGRAM_USER_ID,
#         "text": message_text,
#         "parse_mode": "Markdown"
#     }

#     try:
#         requests.post(url, json=payload)
#         print("📲 Telegram notification sent successfully!")
#     except Exception as e:
#         print(f"❌ Failed to send Telegram message: {e}")