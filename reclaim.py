import requests
import difflib
from datetime import datetime, timezone
from config import RECLAIM_API_KEY

BASE_URL = "https://api.app.reclaim.ai/api"
now = datetime.now(timezone.utc)

HEADERS = {
    "Authorization" : f"Bearer {RECLAIM_API_KEY}",
    "Content-Type" : "application/json"
}

def get_task(task_id: str):
    url = f'{BASE_URL}/tasks/{task_id}'
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.json()
    
    else:
        print(f'Error fetching task {task_id}: {response.status_code} - {response.text}')

def get_all_tasks():
    url = f"{BASE_URL}/tasks"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error fetching tasks: {response.status_code} - {response.text}")
        return []
    
def get_active_tasks():
    active = [t for t in get_all_tasks() if t['status'] in ["IN_PROGRESS", "SCHEDULED"]]
    return active
    
def get_all_events():
    url = f"{BASE_URL}/events"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error fetching events: {response.status_code} - {response.text}")
        return []
    
def get_active_events():
    now = datetime.now(timezone.utc)
    return [
        e for e in get_all_events()
        if datetime.fromisoformat(e['eventStart']) > now
    ]
    
def get_next_event():
    next_event = min(get_active_events(), key=lambda e: datetime.fromisoformat(e['eventStart']))
    return next_event

def get_event_time(event):
    return event['eventStart']

def upcoming_info():
    event = get_next_event()
    return event['title'], get_event_time(event)

def find_task_by_name(query: str):
    tasks = get_all_tasks()
    if not tasks:
        return None
    
    task_names = [t["title"] for t in tasks]
    matches = difflib.get_close_matches(query, task_names, n=1, cutoff=0.4)
    
    if matches:
        matched = next(t for t in tasks if t["title"] == matches[0])
        return matched
    return None

def snooze_task(task_id: str, snooze_until: str):
    url = f'{BASE_URL}/tasks/{task_id}'
    payload = {'snoozeUntil' : snooze_until}
    response = requests.patch(url, headers=HEADERS, json=payload)

    if response.status_code in [200, 204]:
        print(f"✅ Successfully pushed task {task_id} forward to {snooze_until}.")
        return True
    else:
        print(f"❌ Error pushing task {task_id}: {response.status_code} - {response.text}")
        return False
    
def extend_task(task_id: str, total_chunks_required: int):
    url = f"{BASE_URL}/tasks/{task_id}"
    payload = {'timeChunksRequired': total_chunks_required}
    
    response = requests.patch(url, headers=HEADERS, json=payload)
    
    if response.status_code in [200, 204]:
        print(f"Successfully updated task {task_id} total chunks to {total_chunks_required}.")
        return True
    else:
        print(f"Error extending task {task_id}: {response.status_code} - {response.text}")
        return False
    
