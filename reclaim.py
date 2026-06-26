import requests
import difflib
from datetime import datetime, timezone
from config import RECLAIM_API_KEY
from google.oauth2 import service_account
from googleapiclient.discovery import build

BASE_URL = "https://api.app.reclaim.ai/api"

HEADERS = {
    "Authorization": f"Bearer {RECLAIM_API_KEY}",
    "Content-Type": "application/json"
}

# ─── GETTERS ────────────────────────────────────────────────────────────────

def get_task(task_id: str):
    url = f'{BASE_URL}/tasks/{task_id}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    print(f"❌ Error fetching task {task_id}: {response.status_code} - {response.text}")

def get_all_tasks():
    url = f"{BASE_URL}/tasks"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    print(f"❌ Error fetching tasks: {response.status_code} - {response.text}")
    return []

def get_active_tasks():
    return [t for t in get_all_tasks() if t['status'] in ["IN_PROGRESS", "SCHEDULED"]]

def get_all_events():
    url = f"{BASE_URL}/events"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    print(f"❌ Error fetching events: {response.status_code} - {response.text}")
    return []

def get_active_events():
    now = datetime.now(timezone.utc)
    return [
        e for e in get_all_events()
        if datetime.fromisoformat(e['eventStart']) > now
    ]

def get_next_event():
    return min(get_active_events(), key=lambda e: datetime.fromisoformat(e['eventStart']))

def get_event_time(event):
    return event['eventStart']

def upcoming_info():
    event = get_next_event()
    return event['title'], get_event_time(event)

def find_task_by_name(query: str):
    tasks = get_active_tasks()
    if not tasks:
        return None
    task_names = [t["title"] for t in tasks]
    matches = difflib.get_close_matches(query, task_names, n=1, cutoff=0.4)
    if matches:
        return next(t for t in tasks if t["title"] == matches[0])
    return None

def get_task_id(name: str):
    try:
        return find_task_by_name(name)['id']
    except Exception as e:
        print(f'Error getting id: {e}')


# ─── ACTIONS ────────────────────────────────────────────────────────────────

def complete_task(task_name: str):
    task = find_task_by_name(task_name)
    if not task:
        print(f"❌ Could not find task '{task_name}'")
        return False
    
    chunks_spent = task["timeChunksSpent"]
    url = f"{BASE_URL}/tasks/{task['id']}"
    payload = {"status": "COMPLETE", "timeChunksRequired": chunks_spent}
    response = requests.patch(url, headers=HEADERS, json=payload)
    if response.status_code in [200, 204]:
        print(f"✅ Marked '{task_name}' as complete.")
        return True
    print(f"❌ Error completing task: {response.status_code} - {response.text}")
    return False

def log_work(task_name: str, start: str, end: str):
    task = find_task_by_name(task_name)
    if not task:
        return False
    url = f"{BASE_URL}/tasks/{task['id']}/log"
    payload = {"start": start, "end": end}
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code in [200, 201, 204]:
        print(f"✅ Logged work for '{task_name}'.")
        return True
    print(f"❌ Error logging work: {response.status_code} - {response.text}")
    return False

def reschedule_task(task_name: str, snooze_until: str = None):
    task = find_task_by_name(task_name)
    if not task:
        return False
    if snooze_until is None:
        snooze_until = datetime.now(timezone.utc).isoformat()
    url = f"{BASE_URL}/tasks/{task['id']}"
    payload = {"snoozeUntil": snooze_until}
    response = requests.patch(url, headers=HEADERS, json=payload)
    if response.status_code in [200, 204]:
        print(f"✅ Rescheduled '{task_name}' from {snooze_until}.")
        return True
    print(f"❌ Error rescheduling: {response.status_code} - {response.text}")
    return False

def extend_task_total(task_name: str, additional_chunks: int):
    task = find_task_by_name(task_name)
    if not task:
        return False
    new_chunks = task["timeChunksRequired"] + additional_chunks
    url = f"{BASE_URL}/tasks/{task['id']}"
    payload = {"timeChunksRequired": new_chunks}
    response = requests.patch(url, headers=HEADERS, json=payload)
    if response.status_code in [200, 204]:
        print(f"✅ Extended '{task_name}' by {additional_chunks * 15} min.")
        return True
    print(f"❌ Error extending task: {response.status_code} - {response.text}")
    return False

def extend_task_instance(task_name: str, additional_minutes: int):
    task = find_task_by_name(task_name)
    if not task:
        return False
    url = f"{BASE_URL}/planner/extend/{task['id']}"
    payload = {"extendBy": additional_minutes}
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code in [200, 204]:
        print(f"✅ Extended current block of '{task_name}' by {additional_minutes} min.")
        return True
    print(f"❌ Error extending instance: {response.status_code} - {response.text}")
    return False

def create_gcal_event(name: str, start: str, end: str, calendar_id: str = "syoshi910@gmail.com"):
    creds = service_account.Credentials.from_service_account_file(
        'google_creds.json',
        scopes=['https://www.googleapis.com/auth/calendar.events']
    )
    service = build('calendar', 'v3', credentials=creds)
    
    event = {
        "summary": name,
        "start": {"dateTime": start, "timeZone": "America/Los_Angeles"},
        "end": {"dateTime": end, "timeZone": "America/Los_Angeles"},
    }
    
    result = service.events().insert(calendarId=calendar_id, body=event).execute()
    print(f"✅ Created GCal event '{name}': {result.get('htmlLink')}")
    return result

def create_task(
    title: str,
    due_date: str,
    priority: str = "P1",
    min_hours: int = 1,
    max_hours: int = 2,
    time_needed: float = 2.0
):
    """
    Create a new Reclaim task.
    priority: "CRITICAL", "HIGH", "MEDIUM", "LOW"
    min/max_chunk_size: in hours
    time_needed: total hours needed
    due_date: ISO 8601 string
    """
    url = f"{BASE_URL}/tasks"
    payload = {
        "title": title,
        "due": due_date,
        "priority": priority,
        "minChunkSize": min_hours * 4,   # Reclaim uses 15-min chunks
        "maxChunkSize": max_hours * 4,
        "timeChunksRequired": int(time_needed * 4)
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code in [200, 201]:
        print(f"✅ Created task '{title}' due {due_date}.")
        return response.json()
    print(f"❌ Error creating task: {response.status_code} - {response.text}")
    return None

# ─── DISPATCHER ─────────────────────────────────────────────────────────────

FUNCTION_MAP = {
    "find_task_by_name": find_task_by_name,
    "log_work": log_work,
    "reschedule_task": reschedule_task,
    "create_event": create_gcal_event,
    "create_task": create_task,
    "extend_task_total": extend_task_total,
    "complete_task": complete_task,
}

def dispatch(calls: list):
    """
    Execute a list of LLM-generated action calls in order.
    Handles result passing for dependent calls via {{alias.field}} syntax.
    """
    results = {}

    for call in calls:
        fn_name = call.get("function")
        params = call.get("params", {})
        alias = call.get("result_alias")

        # resolve any {{alias.field}} references from prior results
        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                ref = v[2:-2].strip()
                alias_name, field = ref.split(".")
                resolved_params[k] = results.get(alias_name, {}).get(field)
            else:
                resolved_params[k] = v

        fn = FUNCTION_MAP.get(fn_name)
        if not fn:
            print(f"⚠️ Unknown function: {fn_name}")
            continue

        result = fn(**resolved_params)

        if alias:
            results[alias] = result if isinstance(result, dict) else {}

    return results

active_tasks = get_active_tasks()['name']