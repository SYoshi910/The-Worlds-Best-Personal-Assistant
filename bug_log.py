"""Append-only bug report log (persists across bot restarts)."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import BUG_LOG_PATH, TIMEZONE


def append_bug_report(
    body: str,
    *,
    user_id: int,
    raw_message: str,
    context: dict | None = None,
    path: str | Path | None = None,
) -> str:
    """Append one bug report as a JSON line. Returns the new report id."""
    report_id = str(uuid.uuid4())
    logged_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat()
    record = {
        "id": report_id,
        "logged_at": logged_at,
        "user_id": user_id,
        "body": body,
        "raw_message": raw_message,
    }
    if context:
        record["context"] = context

    dest = Path(path or BUG_LOG_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return report_id
