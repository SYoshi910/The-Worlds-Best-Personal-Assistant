"""Append-only bug report log (persists across bot restarts)."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import BUG_LOG_PATH, TIMEZONE

BUG_STATUS_UNREVIEWED = "unreviewed"
BUG_STATUS_REVIEWED = "reviewed"
BUG_STATUS_FIXED = "fixed"
BUG_STATUSES = frozenset(
    {BUG_STATUS_UNREVIEWED, BUG_STATUS_REVIEWED, BUG_STATUS_FIXED}
)


def append_bug_report(
    body: str,
    *,
    user_id: int,
    raw_message: str,
    context: dict | None = None,
    status: str = BUG_STATUS_UNREVIEWED,
    path: str | Path | None = None,
) -> str:
    """Append one bug report as a JSON line. Returns the new report id."""
    if status not in BUG_STATUSES:
        raise ValueError(f"status must be one of {sorted(BUG_STATUSES)}")
    report_id = str(uuid.uuid4())
    logged_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat()
    record = {
        "id": report_id,
        "logged_at": logged_at,
        "user_id": user_id,
        "body": body,
        "raw_message": raw_message,
        "status": status,
    }
    if context:
        record["context"] = context

    dest = Path(path or BUG_LOG_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return report_id
