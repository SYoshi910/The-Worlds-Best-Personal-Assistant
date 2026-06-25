# webhooks.py
import requests
import uuid
from fastapi import APIRouter, Header, Response
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Import your custom configuration and functions
from config import MY_CUSTOM_TOKEN
#from reclaim import check_next_upcoming_task

# 1. Create a router instance (This acts as a plug-in for main.py)
router = APIRouter()

# 2. Define the Webhook Listener Route
@router.post("/gcal-webhook")
async def handle_gcal_notification(
    x_goog_resource_state: str = Header(None),
    x_goog_channel_token: str = Header(None)
):
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
        print("⚠️ Warning: Unauthorized webhook attempt blocked.")
        return Response(status_code=403, content="Unauthorized token")

    # The Action Trigger: A real calendar event was updated
    if x_goog_resource_state == "exists":
        print("🔄 Google Calendar sync event detected...")
      #  task_start_notif()
        # Pull the fresh task schedule from Reclaim
        #next_task = check_next_upcoming_task()
        
        #if next_task:
         #   print(f"🎯 Next task on deck: {next_task.get('title')}")
          #  # TODO: Add your Telegram bot alert function here later!

    # The Quick Receipt: Tell Google we received the message successfully
    return Response(status_code=200)


# 3. Define the Automatic Registration Tool
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
        print("⚠️ Warning: Could not detect an active Ngrok tunnel on port 4040.")
        print("Make sure you ran 'ngrok http 5000' in a separate terminal before starting the app!")
        return

    # Load Google Service Account Credentials
    try:
        creds = service_account.Credentials.from_service_account_file(
            'google_creds.json', 
            scopes=['https://www.googleapis.com/auth/calendar.events.readonly']
        )
    except FileNotFoundError:
        print("❌ Error: 'google_creds.json' missing from project root.")
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