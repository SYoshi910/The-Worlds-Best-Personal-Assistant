"""Google Calendar webhook registration and change notifications."""

import asyncio
import time
import uuid

import httpx
from fastapi import APIRouter, Header, Response

import schedule_cache
from bot import prep_next_block
from cal_helper import build_task_map
from config import CALENDAR_ID, MY_CUSTOM_TOKEN
from gcal import get_calendar_service, run_calendar_op
from reclaim import invalidate_reclaim_cache

router = APIRouter()

_last_webhook_time = 0.0
DEBOUNCE_SECONDS = 2
_webhook_lock = asyncio.Lock()
WATCH_RENEWAL_DAYS = 6


async def _get_ngrok_webhook_url() -> str | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:4040/api/tunnels", timeout=5.0)
            resp.raise_for_status()
            public_url = resp.json()["tunnels"][0]["public_url"]
            return f"{public_url}/gcal-webhook"
    except Exception:
        print("⚠️ Could not detect an active Ngrok tunnel on port 4040.")
        print("   Run 'ngrok http 5000' before starting the app for GCal webhooks.")
        return None


async def register_gcal_watch() -> bool:
    """Register a GCal push notification channel via the local ngrok tunnel."""
    webhook_url = await _get_ngrok_webhook_url()
    if not webhook_url:
        return False

    service = get_calendar_service(readonly=True)
    body = {
        "id": str(uuid.uuid4()),
        "type": "web_hook",
        "address": webhook_url,
        "token": MY_CUSTOM_TOKEN,
    }

    def _watch():
        return service.events().watch(calendarId=CALENDAR_ID, body=body).execute()

    try:
        print(f"📡 Registering Google Calendar webhook at: {webhook_url}")
        await run_calendar_op(_watch)
        print("✅ Google Calendar watch channel attached!")
        return True
    except FileNotFoundError:
        print("❌ 'google_creds.json' missing from project root.")
        return False
    except Exception as e:
        print(f"❌ GCal watch registration failed: {e}")
        return False


def schedule_watch_renewal(scheduler):
    """Schedule periodic renewal of the GCal watch channel."""
    scheduler.add_job(
        register_gcal_watch,
        trigger="interval",
        days=WATCH_RENEWAL_DAYS,
        id="gcal_watch_renewal",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    print(f"✅ GCal watch renewal scheduled every {WATCH_RENEWAL_DAYS} days")


@router.post("/gcal-webhook")
async def handle_gcal_notification(
    x_goog_resource_state: str = Header(None),
    x_goog_channel_token: str = Header(None),
):
    """Handle GCal push notifications: debounce, refresh cache, reschedule pings."""
    global _last_webhook_time

    if x_goog_resource_state == "sync":
        print("✅ Synchronized webhook channel with Google!")
        return Response(status_code=200)

    if x_goog_channel_token != MY_CUSTOM_TOKEN:
        print("⚠️ Unauthorized webhook attempt blocked.")
        return Response(status_code=403, content="Unauthorized token")

    if x_goog_resource_state == "exists":
        async with _webhook_lock:
            now = time.time()
            if now - _last_webhook_time < DEBOUNCE_SECONDS:
                return Response(status_code=200)
            _last_webhook_time = now

        print("🔄 Google Calendar change detected...")
        invalidate_reclaim_cache()
        await asyncio.sleep(10.0)
        await prep_next_block()
        await build_task_map()
        await schedule_cache.refresh()

    return Response(status_code=200)
