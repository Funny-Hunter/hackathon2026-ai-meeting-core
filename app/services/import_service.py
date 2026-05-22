from __future__ import annotations

import json
import re
import urllib.request
import uuid
from pathlib import Path
from urllib.error import HTTPError

from app.core.config import get_settings
from app.models.transcript import ChatResponse, MeetingTranscript, SourceChunk
from app.services.neo4j_service import ingest_transcript_to_graph

# Qdrant embedding ingest is paused for Neo4j-only mode.
# Uncomment this import and the call in import_transcript_file() when an embedding key is available.
# from app.services.qdrant_service import ingest_transcript

# Switch graph backend here:
# from app.services.graphiti_service import ingest_transcript_to_graphiti
from app.utils.meeting_metadata import get_meeting_date, get_meeting_datetime, get_topic_key


DATA_DIR = Path("data")
QDRANT_URL = "http://127.0.0.1:6333"
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
    # qdrant_points = await ingest_transcript(transcript)
    qdrant_points = 0
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


def chat_with_imported_transcript(meeting_id: str | None, question: str) -> ChatResponse:
    transcript = get_transcript_for_chat(meeting_id)
    terms = _query_terms(question)
    scored = []
    for segment in transcript.segments:
        text = f"{segment.speaker} {segment.text}".casefold()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, segment))

    if not scored:
        scored = [(1, segment) for segment in transcript.segments[:3]]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [segment for _, segment in scored[:4]]

    context = " ".join(segment.text for segment in selected)
    answer = (
        f"I found this in meeting '{transcript.title or transcript.meeting_id}': "
        f"{context}"
    )
    return ChatResponse(
        answer=answer,
        route_used="mock_keyword_rag",
        sources=[
            SourceChunk(
                speaker=segment.speaker,
                timestamp=segment.timestamp,
                text=segment.text,
                score=float(score),
            )
            for score, segment in scored[:4]
        ],
    )


def ingest_qdrant_mock(transcript: MeetingTranscript) -> int:
    collection = get_settings().qdrant_collection
    _ensure_qdrant_collection(collection)
    _delete_qdrant_meeting(collection, transcript.meeting_id)

    points = []
    for index, segment in enumerate(transcript.segments):
        points.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{transcript.meeting_id}:{index}")),
                "vector": _mock_vector(segment.text, index),
                "payload": _segment_payload(transcript, segment, index),
            }
        )
    _qdrant_request("PUT", f"/collections/{collection}/points", {"points": points})
    return len(points)


def _qdrant_request(method: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{QDRANT_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _ensure_qdrant_collection(collection: str) -> None:
    try:
        _qdrant_request("GET", f"/collections/{collection}")
    except HTTPError as exc:
        if exc.code != 404:
            raise
        _qdrant_request("PUT", f"/collections/{collection}", {"vectors": {"size": 4, "distance": "Cosine"}})


def _delete_qdrant_meeting(collection: str, meeting_id: str) -> None:
    try:
        _qdrant_request(
            "POST",
            f"/collections/{collection}/points/delete",
            {
                "filter": {
                    "must": [
                        {"key": "meeting_id", "match": {"value": meeting_id}},
                    ]
                }
            },
        )
    except HTTPError as exc:
        if exc.code != 404:
            raise


def _segment_payload(transcript: MeetingTranscript, segment, index: int) -> dict:
    return {
        "meeting_id": transcript.meeting_id,
        "meeting_date": get_meeting_date(transcript),
        "meeting_datetime": get_meeting_datetime(transcript),
        "title": transcript.title or transcript.meeting_id,
        "topic": transcript.topic or transcript.title or "",
        "topic_key": get_topic_key(transcript),
        "speaker": segment.speaker,
        "timestamp": segment.timestamp,
        "segment_index": index,
        "text": segment.text,
    }


def _mock_vector(text: str, index: int) -> list[float]:
    return [
        1.0,
        float(index + 1),
        min(len(text) / 500.0, 1.0),
        float(sum(ord(char) for char in text) % 997) / 997.0,
    ]


def _query_terms(question: str) -> list[str]:
    terms = re.findall(r"[\wÀ-ỹ]{3,}", question.casefold())
    return [term for term in terms if term not in {"của", "là", "gì", "nào", "cho", "hỏi", "trong"}]
