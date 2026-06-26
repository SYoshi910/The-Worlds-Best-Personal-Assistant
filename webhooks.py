# webhooks.py
import requests
import uuid
from fastapi import APIRouter, Header, Response
from google.oauth2 import service_account
from googleapiclient.discovery import build
from bot import prep_next_block
import time
import asyncio


from config import MY_CUSTOM_TOKEN

router = APIRouter()

_last_webhook_time = 0
DEBOUNCE_SECONDS = 2

# 2. Define the Webhook Listener Route
@router.post("/gcal-webhook")
async def handle_gcal_notification(
    x_goog_resource_state: str = Header(None),
    x_goog_channel_token: str = Header(None)
):
    global _last_webhook_time
    """
    Listens for webhook pings from Google Calendar.
    Triggers a Reclaim API check whenever a calendar change occurs.
    """
    # The Handshake: Acknowledge Google's initial setup ping
    if x_goog_resource_state == "sync":
        print("Successfully synchronized webhook channel with Google!")
        return Response(status_code=200)

    # The Security Bouncer: Block unauthorized requests
    if x_goog_channel_token != MY_CUSTOM_TOKEN:
        print("Warning: Unauthorized webhook attempt blocked.")
        return Response(status_code=403, content="Unauthorized token")

    # The Action Trigger: A real calendar event was updated
    if x_goog_resource_state == "exists":
        now = time.time()
        if now - _last_webhook_time < DEBOUNCE_SECONDS:
            return Response(status_code=200)  # silent drop
        _last_webhook_time = now
        print("🔄 Google Calendar sync event detected...")
        await asyncio.sleep(10.0)
        prep_next_block()

    return Response(status_code=200)



def auto_register_gcal_watch():
    """
    Automatically fetches the live local Ngrok URL and registers 
    the watch channel with Google Calendar at startup.
    """
    # Ask your local running Ngrok agent what its public URL is
    try:
        ngrok_api_response = requests.get("http://127.0.0.1:4040/api/tunnels").json()
        public_url = ngrok_api_response['tunnels'][0]['public_url']
        webhook_url = f"{public_url}/gcal-webhook"
    except Exception:
        print("Warning: Could not detect an active Ngrok tunnel on port 4040.")
        print("Make sure you ran 'ngrok http 5000' in a separate terminal before starting the app!")
        return

    # Load Google Service Account Credentials
    try:
        creds = service_account.Credentials.from_service_account_file(
            'google_creds.json', 
            scopes=['https://www.googleapis.com/auth/calendar.events.readonly']
        )
    except FileNotFoundError:
        print("Error: 'google_creds.json' missing from project root.")
        return

    # Build the Watch Request for Google
    service = build('calendar', 'v3', credentials=creds)
    body = {
        "id": str(uuid.uuid4()),
        "type": "web_hook",
        "address": webhook_url,
        "token": MY_CUSTOM_TOKEN,
    }
    
    try:
        print(f"📡 Auto-registering Google Calendar webhook at: {webhook_url}")
        service.events().watch(calendarId="syoshi910@gmail.com", body=body).execute()
        print("✅ Google Calendar watch channel successfully attached!")
    except Exception as e:
        print(f"❌ Auto-registration failed: {e}")