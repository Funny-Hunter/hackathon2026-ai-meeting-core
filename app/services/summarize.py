import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import get_llm
from app.models.transcript import MeetingTranscript, SummarizeOutput, NextAction
from app.utils.transcript_formatter import format_full_transcript

logger = logging.getLogger(__name__)


SUMMARIZE_SYSTEM = """You are an AI specialized in analyzing meeting content.
Your task is to read the transcript and return a complete analytical JSON object.
Return only valid JSON, without markdown fences or extra explanation."""

SUMMARIZE_HUMAN = """Below is the meeting transcript:

{transcript}

Analyze it and return JSON using the following structure (keep the English keys unchanged):
{{
  "title": "meeting name/topic (string)",
  "topic": "concise main topic (string)",
  "attendees": ["list of attendees (array of strings)"],
  "results": ["achieved outcomes (array of strings)"],
  "next_actions": [
    {{
      "task": "task name",
      "assignee": "assigned person (null if unclear)",
      "due_date": "deadline if available (null if unclear)"
    }}
  ],
  "concerns": ["open issues or risks (array of strings)"],
  "summary_paragraph": "overall summary in 2-4 sentences"
}}

Notes:
- Extract people's names accurately from the transcript
- next_actions must be actionable and specific
- concerns are unresolved or worrying points
"""


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZE_SYSTEM),
    ("human", SUMMARIZE_HUMAN),
])



def _build_summarize_chain():
    llm = get_llm()
    return _PROMPT | llm | StrOutputParser()



_SPELLCHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Vietnamese spellchecker. Fix spelling and grammar while preserving the meaning, proper nouns, and technical terms. Return only the corrected text."),
    ("human", "{text}"),
])


async def spellcheck_transcript(transcript: MeetingTranscript) -> MeetingTranscript:
    """Use an LLM to spellcheck each segment."""
    llm = get_llm()
    chain = _SPELLCHECK_PROMPT | llm | StrOutputParser()

    corrected_segments = []
    for seg in transcript.segments:
        try:
            corrected_text = await chain.ainvoke({"text": seg.text})
            corrected_segments.append(seg.model_copy(update={"text": corrected_text.strip()}))
        except Exception as e:
            logger.warning(f"Spellcheck failed for segment [{seg.timestamp}]: {e}")
            corrected_segments.append(seg)

    return transcript.model_copy(update={"segments": corrected_segments})



async def summarize_meeting(transcript: MeetingTranscript) -> SummarizeOutput:

    chain = _build_summarize_chain()
    formatted = format_full_transcript(transcript)

    raw_json = await chain.ainvoke({"transcript": formatted})

    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        parts = raw_json.split("```")
        raw_json = parts[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()

    data = json.loads(raw_json)

    next_actions = [
        NextAction(
            task=item.get("task", ""),
            assignee=item.get("assignee"),
            due_date=item.get("due_date"),
        )
        for item in data.get("next_actions", [])
    ]

    return SummarizeOutput(
        meeting_id=transcript.meeting_id,
        title=data.get("title", transcript.title or transcript.meeting_id),
        topic=data.get("topic", ""),
        attendees=data.get("attendees", []),
        results=data.get("results", []),
        next_actions=next_actions,
        concerns=data.get("concerns", []),
        summary_paragraph=data.get("summary_paragraph", ""),
    )
