import os
import sys
from datetime import time

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")
RECLAIM_API_KEY = os.getenv("RECLAIM_API_KEY")
MY_CUSTOM_TOKEN = os.getenv("MY_CUSTOM_TOKEN")
GROQ_TOKEN = os.getenv("GROQ_TOKEN")
GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")

CALENDAR_ID = os.getenv("CALENDAR_ID", "syoshi910@gmail.com")
TIMEZONE = os.getenv("TIMEZONE", "America/Los_Angeles")
DEV_RELOAD = os.getenv("DEV_RELOAD", "false").lower() == "true"

# Token / input guardrails
LLM_MAX_OUTPUT_TOKENS = 512
LLM_MAX_INPUT_CHARS = 2000
LLM_MAX_CHARS_PER_MESSAGE = 2000
SNAPSHOT_MAX_CHARS = 4000
MAX_CONVERSATION_TURNS = 5
MAX_VOICE_DURATION_SEC = 30

# Break permission / buffer analysis
MIN_BUFFER_HOURS = 3
TIGHT_BUFFER_HOURS = 1
BREAK_HORIZON_DAYS = 7
DEFAULT_EVENING_END = time(0, 0)  # midnight (start of next calendar day)
WORK_WINDOWS = ((time(9, 0), time(17, 0)), (time(19, 0), time(22, 0)))
SCHEDULABLE_WEEKDAYS = (0, 1, 2, 3, 4)  # Mon–Fri

_REQUIRED = {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "TELEGRAM_USER_ID": TELEGRAM_USER_ID,
    "RECLAIM_API_KEY": RECLAIM_API_KEY,
    "MY_CUSTOM_TOKEN": MY_CUSTOM_TOKEN,
    "GROQ_TOKEN": GROQ_TOKEN,
    "GEMINI_TOKEN": GEMINI_TOKEN,
}
_MISSING = [name for name, val in _REQUIRED.items() if not val]
if _MISSING:
    print(f"❌ Missing required environment variables: {', '.join(_MISSING)}")
    sys.exit(1)

TELEGRAM_USER_ID = int(TELEGRAM_USER_ID)
