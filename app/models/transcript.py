from pydantic import BaseModel, Field
from typing import Optional


class TranscriptSegment(BaseModel):
    speaker: str = Field(..., description="Speaker name/ID, e.g. 'Speaker A' or 'Nguyen Van A'")
    timestamp: str = Field(..., description="Timestamp in HH:MM:SS or MM:SS format")
    text: str = Field(..., description="Spoken content")


class MeetingTranscript(BaseModel):
    meeting_id: str = Field(..., description="Unique meeting identifier")
    segments: list[TranscriptSegment]
    meeting_date: Optional[str] = None
    meeting_datetime: Optional[str] = Field(
        default=None,
        description="Meeting datetime in ISO-8601 format, e.g. 2026-05-21T09:30:00+07:00",
    )
    title: Optional[str] = None
    topic: Optional[str] = Field(
        default=None,
        description="Stable topic used to link related meetings over time",
    )


class NextAction(BaseModel):
    task: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None


class SummarizeOutput(BaseModel):
    meeting_id: str
    title: str
    topic: str
    attendees: list[str]
    results: list[str] = Field(..., description="Outcomes achieved during the meeting")
    next_actions: list[NextAction]
    concerns: list[str] = Field(..., description="Open issues or items that need resolution")
    summary_paragraph: str = Field(..., description="Short one-paragraph overall summary")



class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    meeting_id: str
    question: str
    chat_history: list[ChatMessage] = []


class SourceChunk(BaseModel):
    speaker: str
    timestamp: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []
    route_used: str = Field(..., description="'hybrid', 'vector_rag', or 'graph_rag'")



class IngestRequest(BaseModel):
    transcript: MeetingTranscript


class IngestResponse(BaseModel):
    meeting_id: str
    chunks_indexed: int
    nodes_created: int
    message: str
