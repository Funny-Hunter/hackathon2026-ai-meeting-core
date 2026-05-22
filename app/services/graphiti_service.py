from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.transcript import MeetingTranscript
from app.utils.meeting_metadata import get_meeting_datetime

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
except ImportError:  # pragma: no cover
    Graphiti = None
    EpisodeType = None


class Person(BaseModel):
    """A person mentioned in a meeting."""

    role: Optional[str] = Field(None, description="Role or responsibility mentioned in the meeting")


class Organization(BaseModel):
    """An organization, team, vendor, or customer."""

    category: Optional[str] = Field(None, description="Type of organization")


class Project(BaseModel):
    """A project or initiative discussed in the meeting."""

    status: Optional[str] = Field(None, description="Current project status")


class Product(BaseModel):
    """A product, service, or feature."""

    category: Optional[str] = Field(None, description="Product or service category")


class Task(BaseModel):
    """A task, action item, or work item."""

    status: Optional[str] = Field(None, description="Task status")


class Decision(BaseModel):
    """A decision or agreed outcome from the meeting."""

    outcome: Optional[str] = Field(None, description="Decision outcome")


class Issue(BaseModel):
    """A risk, blocker, concern, or open problem."""

    severity: Optional[str] = Field(None, description="Issue severity")


class Topic(BaseModel):
    """A subject discussed in the meeting."""

    category: Optional[str] = Field(None, description="Topic category")


class DateTime(BaseModel):
    """A date, time, deadline, or schedule expression."""

    value: Optional[str] = Field(None, description="Date or time expression")


class AssignedTo(BaseModel):
    """An assignment relationship between a task and a person."""

    evidence: Optional[str] = Field(None, description="Text evidence for assignment")


class Owns(BaseModel):
    """An ownership relationship."""

    evidence: Optional[str] = Field(None, description="Text evidence for ownership")


class Discussed(BaseModel):
    """A discussion relationship."""

    evidence: Optional[str] = Field(None, description="Text evidence for discussion")


class Decided(BaseModel):
    """A decision relationship."""

    evidence: Optional[str] = Field(None, description="Text evidence for decision")


class DependsOn(BaseModel):
    """A dependency relationship."""

    evidence: Optional[str] = Field(None, description="Text evidence for dependency")


class Blocks(BaseModel):
    """A blocking relationship."""

    evidence: Optional[str] = Field(None, description="Text evidence for blocking relationship")


class FollowUpFor(BaseModel):
    """A follow-up relationship."""

    evidence: Optional[str] = Field(None, description="Text evidence for follow-up")


class Mentioned(BaseModel):
    """A generic mention relationship."""

    evidence: Optional[str] = Field(None, description="Text evidence for mention")


ENTITY_TYPES = {
    "Person": Person,
    "Organization": Organization,
    "Project": Project,
    "Product": Product,
    "Task": Task,
    "Decision": Decision,
    "Issue": Issue,
    "Topic": Topic,
    "DateTime": DateTime,
}

EDGE_TYPES = {
    "AssignedTo": AssignedTo,
    "Owns": Owns,
    "Discussed": Discussed,
    "Decided": Decided,
    "DependsOn": DependsOn,
    "Blocks": Blocks,
    "FollowUpFor": FollowUpFor,
    "Mentioned": Mentioned,
}

EDGE_TYPE_MAP = {
    ("Person", "Task"): ["AssignedTo", "Owns", "FollowUpFor"],
    ("Task", "Person"): ["AssignedTo"],
    ("Task", "Task"): ["DependsOn", "Blocks", "FollowUpFor"],
    ("Issue", "Task"): ["Blocks", "FollowUpFor"],
    ("Decision", "Task"): ["Decided", "FollowUpFor"],
    ("Person", "Topic"): ["Discussed", "Mentioned"],
    ("Person", "Project"): ["Owns", "Discussed"],
    ("Project", "Task"): ["DependsOn", "Blocks", "FollowUpFor"],
    ("Entity", "Entity"): ["Mentioned", "Discussed"],
}


def format_transcript_episode_body(transcript: MeetingTranscript) -> str:
    return "\n".join(f"{segment.speaker}: {segment.text}" for segment in transcript.segments)


def _reference_time(transcript: MeetingTranscript) -> datetime:
    raw = get_meeting_datetime(transcript)
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now()


def _new_graphiti():
    if Graphiti is None:
        raise RuntimeError("graphiti-core is not installed")
    settings = get_settings()
    if "OPENAI_API_KEY" not in os.environ:
        if not settings.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    return Graphiti(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)


async def ingest_transcript_to_graphiti(transcript: MeetingTranscript) -> dict:
    if EpisodeType is None:
        raise RuntimeError("graphiti-core is not installed")

    graphiti = _new_graphiti()
    try:
        await graphiti.build_indices_and_constraints()
        await graphiti.add_episode(
            name=f"meeting:{transcript.meeting_id}",
            episode_body=format_transcript_episode_body(transcript),
            source=EpisodeType.message,
            source_description=f"Meeting transcript: {transcript.title or transcript.meeting_id}",
            reference_time=_reference_time(transcript),
            group_id=transcript.meeting_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )
    finally:
        await graphiti.close()

    return {
        "backend": "graphiti",
        "meeting_id": transcript.meeting_id,
        "episodes_added": 1,
        "source": "message",
    }


async def search_graphiti(meeting_id: str, query: str, limit: int = 8) -> str:
    graphiti = _new_graphiti()
    try:
        try:
            results = await graphiti.search(query=query, group_ids=[meeting_id], num_results=limit)
        except TypeError:
            results = await graphiti.search(query)
    finally:
        await graphiti.close()

    if not results:
        return "No graph facts found."

    lines = []
    for item in results[:limit]:
        fact = getattr(item, "fact", None) or getattr(item, "content", None) or str(item)
        lines.append(f"- {fact}")
    return "\n".join(lines)
