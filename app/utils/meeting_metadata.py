from __future__ import annotations

import re
from datetime import datetime

from app.models.transcript import MeetingTranscript


def normalize_topic_key(value: str | None) -> str:
    text = (value or "").strip().casefold()
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ]+", "-", text)
    return text.strip("-") or "untitled"


def get_meeting_date(transcript: MeetingTranscript) -> str:
    return transcript.meeting_datetime[:10] if transcript.meeting_datetime else (transcript.meeting_date or "")


def get_meeting_datetime(transcript: MeetingTranscript) -> str:
    if transcript.meeting_datetime:
        return transcript.meeting_datetime
    if transcript.meeting_date:
        return f"{transcript.meeting_date}T00:00:00"
    return ""


def get_topic_key(transcript: MeetingTranscript) -> str:
    return normalize_topic_key(transcript.topic or transcript.title or transcript.meeting_id)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
