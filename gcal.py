"""Google Calendar API helpers (async wrappers around sync client)."""

import asyncio
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import CALENDAR_ID, TIMEZONE

# google-auth 2.55+ background probe to IAM allowedLocations fails with
# FAILED_PRECONDITION for typical personal service accounts — harmless noise.
logging.getLogger("google.oauth2._client").setLevel(logging.ERROR)
logging.getLogger("google.auth._regional_access_boundary_utils").setLevel(logging.ERROR)

SCOPES_WRITE = ["https://www.googleapis.com/auth/calendar.events"]
SCOPES_READ = ["https://www.googleapis.com/auth/calendar.events.readonly"]

_CREDS_PATH = Path("google_creds.json")


@lru_cache(maxsize=2)
def get_calendar_service(readonly: bool = False):
    """Return a cached Google Calendar API service client."""
    scopes = SCOPES_READ if readonly else SCOPES_WRITE
    creds = service_account.Credentials.from_service_account_file(
        str(_CREDS_PATH), scopes=scopes
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


async def run_calendar_op(fn):
    """Run a sync Calendar API call in a thread with HttpError logging."""
    def _wrapped():
        try:
            return fn()
        except HttpError as e:
            print(
                f"❌ GCal API error {e.resp.status}: "
                f"{e.content.decode('utf-8', errors='replace')[:300]}"
            )
            raise

    return await asyncio.to_thread(_wrapped)


def _iso_from_gcal_field(field: dict) -> str | None:
    return field.get("dateTime") or field.get("date")


def snapshot_from_gcal_event(event: dict, action: str) -> dict:
    """Build a rollback snapshot from a GCal event dict."""
    return {
        "event_id": event["id"],
        "start": _iso_from_gcal_field(event["start"]),
        "end": _iso_from_gcal_field(event["end"]),
        "summary": event.get("summary"),
        "action": action,
    }


async def get_event(event_id: str, calendar_id: str = CALENDAR_ID) -> dict:
    """Fetch a single calendar event by id."""
    service = get_calendar_service(readonly=True)

    def _get():
        return service.events().get(calendarId=calendar_id, eventId=event_id).execute()

    return await run_calendar_op(_get)


async def move_event(
    event_id: str,
    new_start: datetime,
    new_end: datetime,
    calendar_id: str = CALENDAR_ID,
) -> tuple[dict, dict]:
    """Patch event start/end and return updated event plus rollback snapshot."""
    before = await get_event(event_id, calendar_id)
    snapshot = snapshot_from_gcal_event(before, action="move")

    service = get_calendar_service(readonly=False)
    body = {
        "start": {"dateTime": new_start.isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": new_end.isoformat(), "timeZone": TIMEZONE},
    }

    def _patch():
        return (
            service.events()
            .patch(calendarId=calendar_id, eventId=event_id, body=body)
            .execute()
        )

    updated = await run_calendar_op(_patch)
    print(f"✅ Moved GCal event '{updated.get('summary')}' to {new_start.isoformat()}")
    return updated, snapshot


async def delete_event(event_id: str, calendar_id: str = CALENDAR_ID) -> dict:
    """Delete a calendar event and return a rollback snapshot."""
    before = await get_event(event_id, calendar_id)
    snapshot = snapshot_from_gcal_event(before, action="delete")

    service = get_calendar_service(readonly=False)

    def _delete():
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()

    await run_calendar_op(_delete)
    print(f"✅ Deleted GCal event {event_id}")
    return snapshot


async def create_buffer_event(
    name: str,
    start: str,
    end: str,
    calendar_id: str = CALENDAR_ID,
) -> tuple[dict, dict]:
    """Insert a buffer/focus event and return result plus rollback snapshot."""
    service = get_calendar_service(readonly=False)
    body = {
        "summary": name,
        "start": {"dateTime": start, "timeZone": TIMEZONE},
        "end": {"dateTime": end, "timeZone": TIMEZONE},
    }

    def _insert():
        return service.events().insert(calendarId=calendar_id, body=body).execute()

    result = await run_calendar_op(_insert)
    snapshot = snapshot_from_gcal_event(result, action="create")
    print(f"✅ Created GCal event '{name}': {result.get('htmlLink')}")
    return result, snapshot
