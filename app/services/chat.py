import logging
import json
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from app.core.llm import get_llm
from app.models.transcript import ChatResponse, SourceChunk
from app.services import neo4j_service, qdrant_service

# Graphiti path is paused for Neo4j-only mode.
# from app.services import graphiti_service

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    meeting_id: str
    question: str
    chat_history: list[BaseMessage]

    vector_context: str
    graph_context: str
    final_context: str
    sources: list[dict]
    answer: str


async def extract_entities_from_question(question: str) -> dict:
    """Extract topic and speakers from question"""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    Extract entities from question. Return JSON only.

    Question: {question}

    Return:
    {{
        "topic": "topic or null",
        "speakers": ["speaker names"],
        "question_type": "comparison|relationship|opinion|fact"
    }}
    """)

    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({"question": question})

    try:
        # Parse JSON
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        entities = json.loads(raw.strip())
        return entities
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
        return {"topic": None, "speakers": [], "question_type": "fact"}


async def vector_rag_node(state: ChatState) -> dict:
    """Retrieve semantic context from Qdrant for every user question."""
    logger.info(
        "CHAT QDRANT query_start meeting_id=%s question=%r",
        state["meeting_id"],
        state["question"],
    )
    docs_with_scores = await qdrant_service.search_documents_with_scores(
        meeting_id=state["meeting_id"],
        query=state["question"],
        k=5,
    )
    logger.info(
        "CHAT QDRANT query_result meeting_id=%s chunks=%s",
        state["meeting_id"],
        len(docs_with_scores),
    )

    if not docs_with_scores:
        logger.info(
            "CHAT QDRANT no_chunks meeting_id=%s question=%r",
            state["meeting_id"],
            state["question"],
        )

    sources = []
    context_lines = []
    for index, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata
        text = meta.get("text") or doc.page_content
        logger.info(
            "CHAT QDRANT chunk meeting_id=%s index=%s speaker=%s timestamp=%s score=%s text=%r",
            state["meeting_id"],
            index,
            meta.get("speaker", ""),
            meta.get("timestamp", ""),
            score,
            text,
        )
        sources.append({
            "speaker": meta.get("speaker", ""),
            "timestamp": meta.get("timestamp", ""),
            "text": text,
            "score": score,
        })
        context_lines.append(
            f"[{meta.get('timestamp', '')}] {meta.get('speaker', '')}: {text}"
        )

    context = "\n".join(context_lines) if context_lines else "No relevant Qdrant content found."
    return {
        "vector_context": context,
        "sources": sources,
    }


async def graph_rag_node(state: ChatState) -> dict:
    """Extract entities from the question, then retrieve matching Neo4j context."""
    logger.info(
        "CHAT NEO4J entity_extract_start meeting_id=%s question=%r",
        state["meeting_id"],
        state["question"],
    )
    entities = await extract_entities_from_question(state["question"])
    topic = entities.get("topic")
    speakers = entities.get("speakers", [])

    logger.info(
        "CHAT NEO4J entities meeting_id=%s topic=%s speakers=%s question_type=%s",
        state["meeting_id"],
        topic,
        speakers,
        entities.get("question_type"),
    )

    if topic and len(speakers) >= 2:
        query_name = "query_speaker_interaction_on_topic"
        graph_context = neo4j_service.query_speaker_interaction_on_topic(
            meeting_id=state["meeting_id"],
            speaker_a=speakers[0],
            speaker_b=speakers[1],
            topic=topic,
        )
    elif topic:
        query_name = "query_segments_by_topic"
        graph_context = neo4j_service.query_segments_by_topic(
            meeting_id=state["meeting_id"],
            topic=topic,
        )
    else:
        query_name = "query_all_segments_text"
        graph_context = neo4j_service.query_all_segments_text(
            meeting_id=state["meeting_id"]
        )

    lines = graph_context.splitlines()
    logger.info(
        "CHAT NEO4J query_result meeting_id=%s query=%s lines=%s chars=%s",
        state["meeting_id"],
        query_name,
        len(lines),
        len(graph_context),
    )
    for index, line in enumerate(lines, start=1):
        logger.info(
            "CHAT NEO4J context_line meeting_id=%s index=%s text=%r",
            state["meeting_id"],
            index,
            line,
        )

    return {"graph_context": graph_context}



async def synthesize_node(state: ChatState) -> dict:
    """Merge Qdrant semantic context and Neo4j entity context."""
    parts = []
    if state.get("vector_context"):
        parts.append("=== Qdrant Semantic Context ===\n" + state["vector_context"])
    if state.get("graph_context"):
        parts.append("=== Neo4j Entity Context ===\n" + state["graph_context"])

    final_context = "\n\n".join(parts) if parts else "No context available."
    return {"final_context": final_context}


_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant for meeting analysis.
    Answer the question accurately and completely using the provided context and chat history.
    If the context is not sufficient to answer, say so clearly.
    Respond in Vietnamese with a clear, structured answer."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """Meeting context:
    {context}

    Question: {question}"""),
    ])


async def answer_node(state: ChatState) -> dict:
    llm = get_llm()
    chain = _ANSWER_PROMPT | llm | StrOutputParser()

    answer = await chain.ainvoke({
        "context": state.get("final_context", ""),
        "question": state["question"],
        "chat_history": state.get("chat_history", []),
    })

    return {"answer": answer}


def build_chat_graph():
    builder = StateGraph(ChatState)

    builder.add_node("vector_rag", vector_rag_node)
    builder.add_node("graph_rag", graph_rag_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("generate_answer", answer_node)

    builder.set_entry_point("vector_rag")
    builder.add_edge("vector_rag", "graph_rag")
    builder.add_edge("graph_rag", "synthesize")
    builder.add_edge("synthesize", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


_chat_graph = None


def get_chat_graph():
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()
    return _chat_graph



async def chat_with_meeting(
    meeting_id: str,
    question: str,
    chat_history: list[dict],
) -> ChatResponse:
    """
    Main entry point cho chat API.
    chat_history: list of {role: "user"|"assistant", content: str}
    """
    graph = get_chat_graph()

    # Convert chat history sang LangChain messages
    lc_history: list[BaseMessage] = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    initial_state: ChatState = {
        "meeting_id": meeting_id,
        "question": question,
        "chat_history": lc_history,
        "vector_context": "",
        "graph_context": "",
        "final_context": "",
        "sources": [],
        "answer": "",
    }

    final_state = await graph.ainvoke(initial_state)

    sources = [
        SourceChunk(
            speaker=s.get("speaker", ""),
            timestamp=s.get("timestamp", ""),
            text=s.get("text", ""),
            score=float(s.get("score", 0.0)),
        )
        for s in final_state.get("sources", [])
    ]

    return ChatResponse(
        answer=final_state.get("answer", ""),
        sources=sources,
        route_used="hybrid",
    )
