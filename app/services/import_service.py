from __future__ import annotations

import json
from pathlib import Path

from app.models.transcript import MeetingTranscript
from app.services.neo4j_service import ingest_transcript_to_graph
from app.services.qdrant_service import ingest_transcript

# Switch graph backend here:
# from app.services.graphiti_service import ingest_transcript_to_graphiti
from app.utils.meeting_metadata import get_meeting_date, get_meeting_datetime, get_topic_key


DATA_DIR = Path("data")
_IMPORTED: dict[str, MeetingTranscript] = {}


def list_data_transcripts() -> list[dict]:
    files = []
    for path in sorted(DATA_DIR.glob("*.json")):
        transcript = load_transcript(path.name)
        files.append(
            {
                "filename": path.name,
                "meeting_id": transcript.meeting_id,
                "title": transcript.title or transcript.meeting_id,
                "meeting_date": get_meeting_date(transcript),
                "segments": len(transcript.segments),
            }
        )
    return files


def load_transcript(filename: str) -> MeetingTranscript:
    safe_name = Path(filename).name
    path = DATA_DIR / safe_name
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data.get("transcript", data)
    transcript = MeetingTranscript.model_validate(payload)
    if not transcript.topic:
        transcript = transcript.model_copy(update={"topic": transcript.title or transcript.meeting_id})
    return transcript


async def import_transcript_file(filename: str) -> dict:
    transcript = load_transcript(filename)
    qdrant_points = await ingest_transcript(transcript)
    graph_counts = await ingest_transcript_to_graph(transcript)
    if isinstance(graph_counts, int):
        graph_counts = {"nodes": graph_counts, "relationships": 0}
    graph_summary = {
        "backend": "neo4j",
        "meeting_id": transcript.meeting_id,
        "nodes": graph_counts["nodes"],
        "relationships": graph_counts["relationships"],
        "nodes_created": graph_counts["nodes"],
    }

    # Graphiti backend:
    # graph_summary = await ingest_transcript_to_graphiti(transcript)
    _IMPORTED[transcript.meeting_id] = transcript
    return {
        "meeting_id": transcript.meeting_id,
        "title": transcript.title or transcript.meeting_id,
        "meeting_date": get_meeting_date(transcript),
        "meeting_datetime": get_meeting_datetime(transcript),
        "topic_key": get_topic_key(transcript),
        "qdrant_points": qdrant_points,
        "graph": graph_summary,
    }


def get_imported_meetings() -> list[dict]:
    return [
        {
            "meeting_id": transcript.meeting_id,
            "title": transcript.title or transcript.meeting_id,
            "meeting_date": get_meeting_date(transcript),
            "segments": len(transcript.segments),
        }
        for transcript in _IMPORTED.values()
    ]


def get_transcript_for_chat(meeting_id: str | None) -> MeetingTranscript:
    if meeting_id and meeting_id in _IMPORTED:
        return _IMPORTED[meeting_id]
    if _IMPORTED:
        return list(_IMPORTED.values())[-1]
    files = list_data_transcripts()
    if not files:
        raise ValueError("No transcript files found in data/")
    transcript = load_transcript(files[0]["filename"])
    _IMPORTED[transcript.meeting_id] = transcript
    return transcript
