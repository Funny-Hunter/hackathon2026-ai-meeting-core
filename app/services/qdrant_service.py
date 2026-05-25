from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.llm import get_embeddings
from app.models.transcript import MeetingTranscript
from app.utils.transcript_formatter import make_chunk_id
from app.utils.meeting_metadata import (
    get_meeting_date,
    get_meeting_datetime,
    get_topic_key,
)

import logging

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1536,
):
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {collection_name}")
    else:
        logger.info(f"Collection already exists: {collection_name}")


def merge_consecutive_turns(
    transcript: MeetingTranscript,
) -> list[dict]:
    """
    Merge ALL consecutive segments from the same speaker.
    Splitting is handled later by RecursiveCharacterTextSplitter.
    """
    chunks = []
    segments = transcript.segments

    if not segments:
        return chunks

    i = 0

    while i < len(segments):
        current_speaker = segments[i].speaker
        turn_group = []

        j = i
        while (
            j < len(segments)
            and segments[j].speaker == current_speaker
        ):
            turn_group.append(segments[j])
            j += 1

        merged_text = " ".join(
            s.text.strip()
            for s in turn_group
            if s.text.strip()
        )

        if merged_text:
            chunks.append(
                {
                    "speaker": current_speaker,
                    "timestamp": turn_group[0].timestamp,
                    "timestamp_end": turn_group[-1].timestamp,
                    "text": merged_text,
                    "meeting_id": transcript.meeting_id,
                }
            )

        i = j

    return chunks


def build_documents(
    transcript: MeetingTranscript,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Merge full speaker turns, then recursively split with overlap.
    """
    merged_chunks = merge_consecutive_turns(transcript)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs = []

    meeting_date = get_meeting_date(transcript)
    meeting_datetime = get_meeting_datetime(transcript)
    topic_key = get_topic_key(transcript)

    for chunk in merged_chunks:
        split_texts = splitter.split_text(chunk["text"])

        for idx, text in enumerate(split_texts):
            chunk_id = make_chunk_id(
                chunk["meeting_id"],
                chunk["speaker"],
                f"{chunk['timestamp']}_{idx}",
            )

            doc = Document(
                page_content=(
                    f"{chunk['speaker']} "
                    f"[{chunk['timestamp']}]: {text}"
                ),
                metadata={
                    "chunk_id": chunk_id,
                    "meeting_id": chunk["meeting_id"],
                    "meeting_date": meeting_date,
                    "meeting_datetime": meeting_datetime,
                    "title": transcript.title
                    or transcript.meeting_id,
                    "topic": transcript.topic
                    or transcript.title
                    or "",
                    "topic_key": topic_key,
                    "speaker": chunk["speaker"],
                    "timestamp": chunk["timestamp"],
                    "timestamp_end": chunk["timestamp_end"],
                    "split_index": idx,
                    "text": text,
                },
            )

            docs.append(doc)

    return docs


async def ingest_transcript(
    transcript: MeetingTranscript,
) -> int:
    settings = get_settings()
    client = get_qdrant_client()

    ensure_collection(
        client,
        settings.qdrant_collection,
    )

    docs = build_documents(transcript)
    embeddings = get_embeddings()

    try:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.meeting_id",
                            match=models.MatchValue(
                                value=transcript.meeting_id
                            ),
                        )
                    ]
                )
            ),
        )

        logger.info(
            f"Cleared old chunks for meeting: "
            f"{transcript.meeting_id}"
        )

    except Exception as e:
        logger.warning(
            f"Could not clear old chunks: {e}"
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    vector_store.add_documents(docs)

    logger.info(
        f"Indexed {len(docs)} chunks for "
        f"meeting: {transcript.meeting_id}"
    )

    return len(docs)


def get_retriever(
    meeting_id: str,
    k: int = 5,
    score_threshold: float = 0.5,
):
    settings = get_settings()
    client = get_qdrant_client()
    embeddings = get_embeddings()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
            "filter": models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.meeting_id",
                        match=models.MatchValue(
                            value=meeting_id
                        ),
                    )
                ]
            ),
        },
    )


async def search_documents_with_scores(
    meeting_id: str,
    query: str,
    k: int = 5,
    score_threshold: float = 0.5,
):
    settings = get_settings()
    client = get_qdrant_client()
    embeddings = get_embeddings()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    return await vector_store.asimilarity_search_with_relevance_scores(
        query=query,
        k=k,
        score_threshold=score_threshold,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.meeting_id",
                    match=models.MatchValue(
                        value=meeting_id
                    ),
                )
            ]
        ),
    )
