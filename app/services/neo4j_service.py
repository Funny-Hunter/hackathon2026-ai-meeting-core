from functools import lru_cache
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from neo4j import Driver, GraphDatabase

from app.core.config import get_settings
from app.core.llm import get_llm
from app.models.transcript import MeetingTranscript
from app.utils.meeting_metadata import get_meeting_date, get_meeting_datetime, get_topic_key
from app.utils.transcript_formatter import get_all_speakers


logger = logging.getLogger(__name__)


@lru_cache()
def get_neo4j_driver() -> Driver:
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (sp:Speaker) REQUIRE sp.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (seg:Segment) REQUIRE seg.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
]


def setup_schema():
    driver = get_neo4j_driver()
    with driver.session() as session:
        for query in CONSTRAINT_QUERIES:
            session.run(query)
    logger.info("Neo4j schema constraints created")


async def ingest_transcript_to_graph(transcript: MeetingTranscript) -> dict[str, int]:
    driver = get_neo4j_driver()
    meeting_id = transcript.meeting_id
    segments = transcript.segments
    meeting_date = get_meeting_date(transcript)
    meeting_datetime = get_meeting_datetime(transcript)
    topic_key = get_topic_key(transcript)

    with driver.session() as session:
        session.run(
            """
            MATCH (m:Meeting {id: $meeting_id})
            OPTIONAL MATCH (m)-[:HAS_SEGMENT]->(seg:Segment)
            DETACH DELETE seg, m
            """,
            meeting_id=meeting_id,
        )

        session.run(
            """
            MERGE (m:Meeting {id: $meeting_id})
            SET m.title = $title,
                m.topic = $topic,
                m.topic_key = $topic_key,
                m.date = $date,
                m.meeting_date = $date,
                m.meeting_datetime = $meeting_datetime
            """,
            meeting_id=meeting_id,
            title=transcript.title or meeting_id,
            topic=transcript.topic or transcript.title or "",
            topic_key=topic_key,
            date=meeting_date,
            meeting_datetime=meeting_datetime,
        )

        if meeting_datetime:
            session.run(
                """
                MATCH (current:Meeting {id: $meeting_id})
                MATCH (previous:Meeting {topic_key: $topic_key})
                WHERE previous.id <> $meeting_id
                  AND previous.meeting_datetime <> ""
                  AND previous.meeting_datetime < $meeting_datetime
                WITH current, previous
                ORDER BY previous.meeting_datetime DESC
                LIMIT 1
                MERGE (current)-[:RELATED_TO_PREVIOUS {
                    reason: "same_topic",
                    topic_key: $topic_key
                }]->(previous)
                """,
                meeting_id=meeting_id,
                topic_key=topic_key,
                meeting_datetime=meeting_datetime,
            )

        speakers = get_all_speakers(transcript)
        for speaker in speakers:
            speaker_id = f"{meeting_id}__{speaker}"
            session.run(
                """
                MERGE (sp:Speaker {id: $speaker_id})
                SET sp.name = $name, sp.meeting_id = $meeting_id
                """,
                speaker_id=speaker_id,
                name=speaker,
                meeting_id=meeting_id,
            )

        prev_seg_id = None
        for index, segment in enumerate(segments):
            seg_id = f"{meeting_id}__seg__{index}"
            speaker_id = f"{meeting_id}__{segment.speaker}"

            session.run(
                """
                CREATE (seg:Segment {
                    id: $seg_id,
                    meeting_id: $meeting_id,
                    speaker: $speaker,
                    timestamp: $timestamp,
                    text: $text,
                    index: $index
                })
                WITH seg
                MATCH (m:Meeting {id: $meeting_id})
                MERGE (m)-[:HAS_SEGMENT]->(seg)
                WITH seg
                MATCH (sp:Speaker {id: $speaker_id})
                MERGE (seg)-[:SPOKEN_BY]->(sp)
                """,
                seg_id=seg_id,
                meeting_id=meeting_id,
                speaker=segment.speaker,
                timestamp=segment.timestamp,
                text=segment.text,
                index=index,
                speaker_id=speaker_id,
            )

            if prev_seg_id:
                session.run(
                    """
                    MATCH (prev:Segment {id: $prev_id})
                    MATCH (curr:Segment {id: $curr_id})
                    MERGE (prev)-[:FOLLOWS]->(curr)
                    """,
                    prev_id=prev_seg_id,
                    curr_id=seg_id,
                )
            prev_seg_id = seg_id

        for index in range(len(segments) - 1):
            if segments[index].speaker != segments[index + 1].speaker:
                sp1_id = f"{meeting_id}__{segments[index].speaker}"
                sp2_id = f"{meeting_id}__{segments[index + 1].speaker}"
                session.run(
                    """
                    MATCH (sp1:Speaker {id: $sp1_id})
                    MATCH (sp2:Speaker {id: $sp2_id})
                    MERGE (sp1)-[:INTERACTED_WITH]->(sp2)
                    """,
                    sp1_id=sp1_id,
                    sp2_id=sp2_id,
                )

        topics_map = await _extract_topics_per_speaker(transcript)
        for speaker, topics in topics_map.items():
            speaker_id = f"{meeting_id}__{speaker}"
            for topic in topics:
                session.run(
                    """
                    MERGE (t:Topic {name: $topic})
                    WITH t
                    MATCH (sp:Speaker {id: $speaker_id})
                    MERGE (sp)-[:MENTIONED]->(t)
                    """,
                    topic=topic,
                    speaker_id=speaker_id,
                )
        graph_counts = _count_meeting_graph(session, meeting_id)

    logger.info(
        "Graph built for meeting %s: %s nodes, %s relationships",
        meeting_id,
        graph_counts["nodes"],
        graph_counts["relationships"],
    )
    return graph_counts


def _count_meeting_graph(session, meeting_id: str) -> dict[str, int]:
    row = session.run(
        """
        MATCH (m:Meeting {id: $meeting_id})
        OPTIONAL MATCH (m)-[:HAS_SEGMENT]->(seg:Segment)
        OPTIONAL MATCH (seg)-[:SPOKEN_BY]->(sp:Speaker)
        OPTIONAL MATCH (sp)-[:MENTIONED]->(t:Topic)
        WITH collect(DISTINCT m)
           + collect(DISTINCT seg)
           + collect(DISTINCT sp)
           + collect(DISTINCT t) AS graph_nodes
        MATCH (m:Meeting {id: $meeting_id})
        OPTIONAL MATCH (m)-[r1:HAS_SEGMENT]->(:Segment {meeting_id: $meeting_id})
        OPTIONAL MATCH (:Segment {meeting_id: $meeting_id})-[r2:SPOKEN_BY]->(:Speaker {meeting_id: $meeting_id})
        OPTIONAL MATCH (:Segment {meeting_id: $meeting_id})-[r3:FOLLOWS]->(:Segment {meeting_id: $meeting_id})
        OPTIONAL MATCH (:Speaker {meeting_id: $meeting_id})-[r4:INTERACTED_WITH]->(:Speaker {meeting_id: $meeting_id})
        OPTIONAL MATCH (:Speaker {meeting_id: $meeting_id})-[r5:MENTIONED]->(:Topic)
        RETURN
            size(graph_nodes) AS nodes,
            count(DISTINCT r1)
            + count(DISTINCT r2)
            + count(DISTINCT r3)
            + count(DISTINCT r4)
            + count(DISTINCT r5) AS relationships
        """,
        meeting_id=meeting_id,
    ).single()

    if not row:
        return {"nodes": 0, "relationships": 0}

    return {
        "nodes": int(row["nodes"] or 0),
        "relationships": int(row["relationships"] or 0),
    }


_TOPIC_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI meeting analyst. Below is what {speaker} said:

    {text}

    Extract 3-7 keywords/main topics mentioned by this person.
    Return a JSON array of strings. Return JSON only, without explanation.
    Example: ["deadline Q4", "marketing budget", "hiring"]
    """
)


async def _extract_topics_per_speaker(transcript: MeetingTranscript) -> dict[str, list[str]]:
    from app.utils.transcript_formatter import format_speaker_block, group_by_speaker

    llm = get_llm()
    chain = _TOPIC_PROMPT | llm

    grouped = group_by_speaker(transcript.segments)
    result = {}

    for speaker, segments in grouped.items():
        block = format_speaker_block(speaker, segments)
        try:
            response = await chain.ainvoke({"speaker": speaker, "text": block})
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            topics = json.loads(raw.strip())
            result[speaker] = topics if isinstance(topics, list) else []
        except Exception as exc:
            logger.warning("Topic extraction failed for %s: %s", speaker, exc)
            result[speaker] = []

    return result


def query_speaker_context(meeting_id: str, speaker_name: str) -> str:
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (sp:Speaker {name: $speaker, meeting_id: $meeting_id})
            OPTIONAL MATCH (seg:Segment)-[:SPOKEN_BY]->(sp)
            OPTIONAL MATCH (sp)-[:MENTIONED]->(t:Topic)
            OPTIONAL MATCH (sp)-[:INTERACTED_WITH]->(other:Speaker)
            RETURN
                collect(DISTINCT seg.text)[..10] AS texts,
                collect(DISTINCT t.name) AS topics,
                collect(DISTINCT other.name) AS interacted_with
            """,
            speaker=speaker_name,
            meeting_id=meeting_id,
        )
        row = result.single()
        if not row:
            return f"Can't find information about {speaker_name}"

        texts = row["texts"]
        topics = row["topics"]
        others = row["interacted_with"]

        return (
            f"Speaker: {speaker_name}\n"
            f"Interacted with: {', '.join(others) if others else 'no one'}\n"
            f"Topics mentioned: {', '.join(topics) if topics else 'unclear'}\n"
            "Spoken content (sample):\n" + "\n".join(f"- {text}" for text in texts)
        )


def query_topic_speakers(meeting_id: str, topic_keyword: str) -> str:
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (sp:Speaker {meeting_id: $meeting_id})-[:MENTIONED]->(t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic_keyword)
            OPTIONAL MATCH (seg:Segment)-[:SPOKEN_BY]->(sp)
            RETURN
                t.name AS topic,
                collect(DISTINCT sp.name) AS speakers,
                collect(DISTINCT seg.text)[..8] AS texts
            """,
            meeting_id=meeting_id,
            topic_keyword=topic_keyword,
        )
        rows = list(result)

    if not rows:
        return f"No one mentioned '{topic_keyword}'"

    parts = []
    for row in rows:
        parts.append(
            f"Topic: {row['topic']}\n"
            f"Speakers: {', '.join(row['speakers'])}\n"
            "Evidence:\n" + "\n".join(f"- {text}" for text in row["texts"])
        )
    return "\n\n".join(parts)


def query_speaker_relationship(meeting_id: str, speaker_a: str, speaker_b: str) -> str:
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH path = (a:Speaker {name: $speaker_a, meeting_id: $meeting_id})
                -[:INTERACTED_WITH*1..4]-
                (b:Speaker {name: $speaker_b, meeting_id: $meeting_id})
            RETURN [node IN nodes(path) | node.name] AS speakers
            LIMIT 5
            """,
            meeting_id=meeting_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        rows = list(result)

    if not rows:
        return f"{speaker_a} and {speaker_b} have no interaction in this meeting"

    paths = [" -> ".join(row["speakers"]) for row in rows]
    return "Interaction path:\n" + "\n".join(f"- {path}" for path in paths)


def query_interaction_path(meeting_id: str, speaker_a: str, speaker_b: str) -> str:
    """Backward-compatible name used by the original Neo4j service."""
    return query_speaker_relationship(meeting_id, speaker_a, speaker_b)


def query_all_segments_text(meeting_id: str) -> str:
    rows = query_all_segments(meeting_id)
    if not rows:
        return f"Can't find transcript for meeting {meeting_id}"

    return "\n".join(
        f"[{row['timestamp']}] {row['speaker']}: {row['text']}"
        for row in rows
    )


def query_all_segments(meeting_id: str) -> list[dict]:
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:Meeting {id: $meeting_id})-[:HAS_SEGMENT]->(seg:Segment)
            RETURN seg.timestamp AS timestamp, seg.speaker AS speaker, seg.text AS text
            ORDER BY seg.index
            """,
            meeting_id=meeting_id,
        )
        rows = list(result)

    return [
        {
            "timestamp": row["timestamp"],
            "speaker": row["speaker"],
            "text": row["text"],
            "score": 1.0,
        }
        for row in rows
    ]


def list_meetings(
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:Meeting)
            WHERE ($start_date IS NULL OR m.meeting_date >= $start_date)
              AND ($end_date IS NULL OR m.meeting_date <= $end_date)
            OPTIONAL MATCH (m)-[:HAS_SEGMENT]->(seg:Segment)
            RETURN
                m.id AS meeting_id,
                m.title AS title,
                m.meeting_date AS meeting_date,
                m.meeting_datetime AS meeting_datetime,
                count(seg) AS segments
            ORDER BY meeting_datetime DESC, meeting_date DESC, meeting_id
            """,
            start_date=start_date,
            end_date=end_date,
        )
        rows = list(result)

    return [
        {
            "meeting_id": row["meeting_id"],
            "title": row["title"] or row["meeting_id"],
            "meeting_date": row["meeting_date"] or "",
            "segments": row["segments"],
        }
        for row in rows
    ]

def query_speaker_context_on_topic(
    meeting_id: str,
    speaker: str,
    topic: str,
) -> str:
    driver = get_neo4j_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (sp:Speaker {name:$speaker, meeting_id:$meeting_id})
            MATCH (sp)-[:MENTIONED]->(t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)

            MATCH (seg:Segment)-[:SPOKEN_BY]->(sp)

            RETURN
                seg.timestamp AS timestamp,
                seg.speaker AS speaker,
                seg.text AS text
            ORDER BY seg.index
            """,
            meeting_id=meeting_id,
            speaker=speaker,
            topic=topic,
        )

        rows = list(result)

    if not rows:
        return f"{speaker} did not mention {topic}"

    return "\n".join(
        f"[{r['timestamp']}] {r['speaker']}: {r['text']}"
        for r in rows
    )
    
def query_speaker_interaction_on_topic(
    meeting_id: str,
    speaker_a: str,
    speaker_b: str,
    topic: str
) -> str:
    """Query specific speakers about specific topic"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (sp1:Speaker {name: $speaker_a, meeting_id: $meeting_id})
            MATCH (sp2:Speaker {name: $speaker_b, meeting_id: $meeting_id})
            MATCH (sp1)-[:MENTIONED]->(t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
            MATCH (sp2)-[:MENTIONED]->(t)
            OPTIONAL MATCH (seg1:Segment)-[:SPOKEN_BY]->(sp1)
            OPTIONAL MATCH (seg2:Segment)-[:SPOKEN_BY]->(sp2)
            RETURN
                seg1.timestamp AS timestamp_a,
                seg1.speaker AS speaker_a_name,
                seg1.text AS text_a,
                seg2.timestamp AS timestamp_b,
                seg2.speaker AS speaker_b_name,
                seg2.text AS text_b
            ORDER BY seg1.index, seg2.index
            """,
            meeting_id=meeting_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            topic=topic,
        )
        rows = list(result)

    if not rows:
        return f"No interaction found between {speaker_a} and {speaker_b} on {topic}"

    formatted = []
    seen = set()
    for row in rows:
        for timestamp_key, speaker_key, text_key in (
            ("timestamp_a", "speaker_a_name", "text_a"),
            ("timestamp_b", "speaker_b_name", "text_b"),
        ):
            timestamp = row[timestamp_key]
            speaker = row[speaker_key]
            text = row[text_key]
            key = (timestamp, speaker, text)
            if timestamp and key not in seen:
                seen.add(key)
                formatted.append(f"[{timestamp}] {speaker}: {text}")

    return "\n".join(formatted)

def query_segments_by_topic(meeting_id: str, topic: str) -> str:
    """Get all segments mentioning a topic"""
    driver = get_neo4j_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (sp:Speaker {meeting_id: $meeting_id})-[:MENTIONED]->(t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
            OPTIONAL MATCH (seg:Segment)-[:SPOKEN_BY]->(sp)
            RETURN
                seg.timestamp AS timestamp,
                seg.speaker AS speaker,
                seg.text AS text
            ORDER BY seg.index
            """,
            meeting_id=meeting_id,
            topic=topic,
        )
        rows = list(result)

    if not rows:
        return f"No one mentioned '{topic}'"

    return "\n".join(
        f"[{row['timestamp']}] {row['speaker']}: {row['text']}"
        for row in rows
    )

def query_recent_segments_text(
    meeting_id: str,
    limit: int = 10,
) -> str:
    driver = get_neo4j_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:Meeting {id: $meeting_id})-[:HAS_SEGMENT]->(s:Segment)
            RETURN
                s.timestamp AS timestamp,
                s.speaker AS speaker,
                s.text AS text
            ORDER BY s.index DESC
            LIMIT $limit
            """,
            meeting_id=meeting_id,
            limit=limit,
        )

        records = list(result)

    if not records:
        return ""

    lines = [
        f"[{r['timestamp']}] {r['speaker']}: {r['text']}"
        for r in reversed(records)
    ]

    return "\n".join(lines)