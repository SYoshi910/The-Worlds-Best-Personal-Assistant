import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

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
BUG_LOG_PATH = os.getenv("BUG_LOG_PATH", "data/bug_reports.jsonl")
MODEL_USAGE_PATH = os.getenv("MODEL_USAGE_PATH", "data/model_usage.jsonl")
MODEL_QUOTA_THRESHOLD = float(os.getenv("MODEL_QUOTA_THRESHOLD", "0.95"))
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemma-4-31b-it")
GOOGLE_VISION_MODEL = os.getenv("GOOGLE_VISION_MODEL", "gemma-4-26b-a4b-it")
MAX_PHOTO_BYTES = int(os.getenv("MAX_PHOTO_BYTES", str(10 * 1024 * 1024)))
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "models/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Token / input guardrails
LLM_MAX_OUTPUT_TOKENS = 512
LLM_MAX_INPUT_CHARS = 2000
LLM_MAX_CHARS_PER_MESSAGE = 2000
SNAPSHOT_MAX_CHARS = 4000
MAX_CONVERSATION_TURNS = 8
CLARIFICATION_TTL_MINUTES = 10
EXTEND_SCOPE_TTL_MINUTES = 5
MAX_VOICE_DURATION_SEC = 30

# Undo / defer windows (seconds)
UNDO_WINDOW_SEC = 30
DEFER_WINDOW_SEC = 30

# After this many seconds without a user message, retry higher-priority LLM models
# whose TPM/RPM limits have recovered (TPD/RPD stay exhausted until LA midnight).
MODEL_CHAIN_IDLE_RESET_SEC = int(os.getenv("MODEL_CHAIN_IDLE_RESET_SEC", "180"))

# Break permission / buffer analysis
MIN_BUFFER_HOURS = 3
TIGHT_BUFFER_HOURS = 1
BREAK_HORIZON_DAYS = 7
DEFAULT_EVENING_END = time(22, 59)  # evening ends 10:59 PM (spec 19c)
WORK_WINDOWS = ((time(9, 0), time(17, 0)), (time(19, 0), time(22, 0)))
SCHEDULABLE_WEEKDAYS = (0, 1, 2, 3, 4)  # Mon–Fri

# Calendar week boundaries (Monday start → Sunday end).
WEEK_START_WEEKDAY = 0  # Monday (datetime.weekday())
WEEK_END_WEEKDAY = 6  # Sunday

# Time-of-day bands for schedule reads (local time).
# Each band is (start, end) inclusive of start, inclusive of end.
# "night" wraps past midnight (23:00–03:59).
TIME_OF_DAY_BANDS = {
    "morning": (time(4, 0), time(11, 59)),
    "noon": (time(12, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(16, 59)),
    "evening": (time(17, 0), time(22, 59)),
    "night": (time(23, 0), time(3, 59)),
}

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


def now_local() -> datetime:
    """Current time in the configured local timezone — the one shared clock helper."""
    return datetime.now(ZoneInfo(TIMEZONE))
