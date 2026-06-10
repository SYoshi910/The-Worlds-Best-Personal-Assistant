import os
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_USER_ID = os.getenv('TELEGRAM_USER_ID')
RECLAIM_API_KEY = os.getenv('RECLAIM_API_KEY')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_USER_ID or not RECLAIM_API_KEY:
    print(f'Error! {TELEGRAM_BOT_TOKEN}, {TELEGRAM_USER_ID}, {RECLAIM_API_KEY}')

else:
    TELEGRAM_USER_ID = int(TELEGRAM_USER_ID)