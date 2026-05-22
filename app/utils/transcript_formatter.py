from app.models.transcript import MeetingTranscript, TranscriptSegment
from collections import defaultdict


def format_full_transcript(transcript: MeetingTranscript) -> str:
    """Format the full transcript as readable text for the LLM."""
    lines = []
    if transcript.title:
        lines.append(f"Meeting: {transcript.title}")
    if transcript.meeting_date:
        lines.append(f"Date: {transcript.meeting_date}")
    lines.append("")

    for seg in transcript.segments:
        lines.append(f"[{seg.timestamp}] {seg.speaker}: {seg.text}")

    return "\n".join(lines)


def group_by_speaker(segments: list[TranscriptSegment]) -> dict[str, list[TranscriptSegment]]:
    """Group segments by speaker."""
    grouped = defaultdict(list)
    for seg in segments:
        grouped[seg.speaker].append(seg)
    return dict(grouped)


def format_speaker_block(speaker: str, segments: list[TranscriptSegment]) -> str:
    """Format everything a speaker said into one text block."""
    lines = [f"=== {speaker} ==="]
    for seg in segments:
        lines.append(f"[{seg.timestamp}] {seg.text}")
    return "\n".join(lines)


def make_chunk_id(meeting_id: str, speaker: str, timestamp: str) -> str:
    """Tạo unique ID cho 1 chunk trong vector store."""
    clean = timestamp.replace(":", "-")
    speaker_clean = speaker.replace(" ", "_").lower()
    return f"{meeting_id}__{speaker_clean}__{clean}"


def get_all_speakers(transcript: MeetingTranscript) -> list[str]:
    """Trả về danh sách unique speakers."""
    return list({seg.speaker for seg in transcript.segments})
