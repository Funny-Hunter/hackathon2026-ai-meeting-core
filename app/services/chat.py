import logging
import json
from datetime import datetime
from typing import TypedDict, Literal
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from app.core.llm import get_llm
from app.models.transcript import ChatResponse, SourceChunk
from app.services.prompt import (
    ANSWER_PROMPT,
    DATABASE_ANSWER_PROMPT,
    DATABASE_LOOKUP_PLANNER_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    SPEAKER_ANALYTICS_PROMPT,
)
from app.services import neo4j_service, qdrant_service
from langgraph.types import Send

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """Shared state throughout the entire graph"""
    meeting_id: str
    question: str
    chat_history: list[BaseMessage]
    
    # Planning step
    intent: Literal["meeting_database_lookup", "meeting_content", "speaker_analytics"]
    operation: Literal["list", "count", "answer"] | None
    start_date: str | None
    end_date: str | None
    
    # RAG context
    vector_context: str
    graph_context: str
    final_context: str
    sources: list[dict]
    
    # Final answer
    answer: str
    route_used: str


def get_current_date() -> str:
    """Get current date in Ho Chi Minh timezone"""
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat()


# STEP 1: PLANNING NODE - Decide intent

async def invoke_database_lookup_planner(question: str, current_date: str) -> dict:
    """Use LLM to classify question intent"""
    llm = get_llm()
    parser = JsonOutputParser()

    chain = DATABASE_LOOKUP_PLANNER_PROMPT | llm | parser

    result = await chain.ainvoke({
        "question": question,
        "current_date": current_date,
        "format_instructions": parser.get_format_instructions(),
    })
    if not isinstance(result, dict):
        raise ValueError("Planner output must be dict")

    return result

async def plan_intent_node(state: ChatState) -> dict:
    logger.info(
        "CHAT PLAN question_start meeting_id=%s question=%r",
        state["meeting_id"],
        state["question"],
    )
    
    current_date = get_current_date()
    try:
        plan = await invoke_database_lookup_planner(
            state["question"],
            current_date
        )
    except Exception as exc:
        logger.warning("Database lookup planning failed: %s", exc)
        plan = {
            "intent": "meeting_content",
            "operation": "answer",
            "start_date": None,
            "end_date": None,
        }

    intent = plan.get("intent", "meeting_content")
    if intent not in (
        "meeting_database_lookup",
        "meeting_content",
        "speaker_analytics",
    ):
        intent = "meeting_content"

    operation = plan.get("operation", "answer")
    if operation not in ("list", "count", "answer"):
        operation = "answer"

    logger.info(
        "CHAT PLAN intent=%s operation=%s start_date=%s end_date=%s",
        intent,
        operation,
        plan.get("start_date"),
        plan.get("end_date"),
    )

    return {
        "intent": intent,
        "operation": operation,
        "start_date": plan.get("start_date"),
        "end_date": plan.get("end_date"),
    }


# BRANCH A: DATABASE LOOKUP PATH


def _format_date_range(start_date: str | None, end_date: str | None) -> str:
    """Format date range for Vietnamese message"""
    if start_date and end_date:
        return f" từ {start_date} đến {end_date}"
    if start_date:
        return f" từ {start_date}"
    if end_date:
        return f" đến {end_date}"
    return ""


async def database_answer_node(state: ChatState) -> dict:
    """
    Node A: Answer database metadata questions
    Returns meeting list/count from Neo4j
    """
    logger.info(
        "CHAT DATABASE query_start meeting_id=%s operation=%s start_date=%s end_date=%s",
        state["meeting_id"],
        state.get("operation"),
        state.get("start_date"),
        state.get("end_date"),
    )

    meetings = neo4j_service.list_meetings(
        start_date=state.get("start_date"),
        end_date=state.get("end_date"),
    )
    
    date_range = _format_date_range(
        state.get("start_date"), 
        state.get("end_date")
    )

    llm = get_llm()
    chain = DATABASE_ANSWER_PROMPT | llm | StrOutputParser()
    answer = await chain.ainvoke({
        "question": state["question"],
        "operation": state.get("operation") or "answer",
        "date_range": date_range or "không giới hạn",
        "meeting_count": len(meetings),
        "meetings_json": json.dumps(meetings, ensure_ascii=False, indent=2),
        "chat_history": state.get("chat_history", []),
    })

    logger.info(
        "CHAT DATABASE query_result count=%s operation=%s",
        len(meetings),
        state.get("operation"),
    )

    return {
        "answer": answer,
        "sources": [],
        "route_used": "database_lookup",
        "final_context": "",
        "vector_context": "",
        "graph_context": "",
    }


# BRANCH B: RAG CONTENT PIPELINE

async def extract_entities_from_question(question: str) -> dict:
    """Extract topic and speakers from question using LLM"""
    llm = get_llm()
    parser = JsonOutputParser()

    chain = ENTITY_EXTRACTION_PROMPT | llm | parser

    try:
        result = await chain.ainvoke({
            "question": question,
            "format_instructions": parser.get_format_instructions(),
        })
        if not isinstance(result, dict):
            raise ValueError("Entity output must be dict")

        return result
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
        return {
            "topic": None,
            "speakers": [],
            "question_type": "recent_context",
        }

async def vector_rag_node(state: ChatState) -> dict:
    """
    Node B.1: Retrieve semantic context from Qdrant
    Part of the hybrid RAG pipeline for content questions
    """
    logger.info("VECTOR_RAG START - %s", datetime.now().isoformat())

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
            "CHAT QDRANT chunk meeting_id=%s index=%s speaker=%s timestamp=%s score=%s",
            state["meeting_id"],
            index,
            meta.get("speaker", ""),
            meta.get("timestamp", ""),
            score,
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
    logger.info("VECTOR_RAG END - %s", datetime.now().isoformat())

    return {
        "vector_context": context,
        "sources": sources,
    }


async def graph_rag_node(state: ChatState) -> dict:
    """
    Node B.2: Extract entities and retrieve from Neo4j graph
    """
    logger.info("GRAPH_RAG START - %s", datetime.now().isoformat())

    logger.info(
        "CHAT NEO4J entity_extract_start meeting_id=%s question=%r",
        state["meeting_id"],
        state["question"],
    )

    entities = await extract_entities_from_question(state["question"])
    topic = entities.get("topic")
    speakers = entities.get("speakers", [])
    qtype = entities.get("question_type")

    logger.info(
        "CHAT NEO4J entities meeting_id=%s topic=%s speakers=%s question_type=%s",
        state["meeting_id"],
        topic,
        speakers,
        qtype,
    )

    if qtype == "speaker_statement" and speakers:
        query_name = "query_speaker_context"
        graph_context = neo4j_service.query_speaker_context(
            state["meeting_id"],
            speakers[0],
        )

    elif qtype == "topic_discussion" and topic:
        query_name = "query_segments_by_topic"
        graph_context = neo4j_service.query_segments_by_topic(
            state["meeting_id"],
            topic,
        )

    elif qtype == "speaker_topic" and speakers and topic:
        query_name = "query_speaker_context_on_topic"
        graph_context = neo4j_service.query_speaker_context_on_topic(
            state["meeting_id"],
            speakers[0],
            topic,
        )

    elif qtype == "speaker_pair_topic" and len(speakers) >= 2 and topic:
        query_name = "query_speaker_interaction_on_topic"
        graph_context = neo4j_service.query_speaker_interaction_on_topic(
            state["meeting_id"],
            speakers[0],
            speakers[1],
            topic,
        )

    elif qtype == "speaker_relationship" and len(speakers) >= 2:
        query_name = "query_speaker_relationship"
        graph_context = neo4j_service.query_speaker_relationship(
            state["meeting_id"],
            speakers[0],
            speakers[1],
        )

    else:
        query_name = "query_meeting_summary_context"
        graph_context = neo4j_service.query_recent_segments_text(
            state["meeting_id"],
            limit=20,
        )

    lines = graph_context.splitlines()

    logger.info(
        "CHAT NEO4J query_result meeting_id=%s query=%s lines=%s chars=%s",
        state["meeting_id"],
        query_name,
        len(lines),
        len(graph_context),
    )
    logger.info("GRAPH_RAG END - %s", datetime.now().isoformat())

    return {
        "graph_context": graph_context
    }


async def synthesize_node(state: ChatState) -> dict:
    """
    Node B.3: Merge vector and graph RAG contexts
    Combines semantic and entity-based contexts for answer generation
    """
    parts = []
    if state.get("vector_context"):
        parts.append("=== Qdrant Semantic Context ===\n" + state["vector_context"])
    if state.get("graph_context"):
        parts.append("=== Neo4j Entity Context ===\n" + state["graph_context"])

    final_context = "\n\n".join(parts) if parts else "No context available."
    
    logger.info(
        "CHAT SYNTHESIZE context_prepared meeting_id=%s length=%s",
        state["meeting_id"],
        len(final_context),
    )
    
    return {"final_context": final_context}


async def generate_answer_node(state: ChatState) -> dict:
    """
    Node B.4: Generate final answer using LLM
    Uses synthesized context and chat history to generate response
    """
    logger.info(
        "CHAT ANSWER generation_start meeting_id=%s",
        state["meeting_id"],
    )
    
    llm = get_llm()
    chain = ANSWER_PROMPT | llm | StrOutputParser()

    answer = await chain.ainvoke({
        "context": state.get("final_context", ""),
        "question": state["question"],
        "chat_history": state.get("chat_history", []),
    })

    logger.info(
        "CHAT ANSWER generation_complete meeting_id=%s answer_length=%s",
        state["meeting_id"],
        len(answer),
    )

    return {
        "answer": answer,
        "route_used": "hybrid",
    }

# BRANCH C: SPEAKER ANALYSIS


async def speaker_analytics_node(state: ChatState) -> dict:
    """
    Answer speaker statistics / participant questions
    """
    driver = neo4j_service.get_neo4j_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (sp:Speaker {meeting_id: $meeting_id})
            RETURN collect(DISTINCT sp.name) AS speakers
            """,
            meeting_id=state["meeting_id"],
        )

        row = result.single()

    speakers = row["speakers"] if row else []

    llm = get_llm()
    chain = SPEAKER_ANALYTICS_PROMPT | llm | StrOutputParser()
    answer = await chain.ainvoke({
        "question": state["question"],
        "meeting_id": state["meeting_id"],
        "speaker_count": len(speakers),
        "speakers_json": json.dumps(speakers, ensure_ascii=False, indent=2),
        "chat_history": state.get("chat_history", []),
    })

    return {
        "answer": answer,
        "sources": [],
        "route_used": "speaker_analytics",
        "final_context": "",
        "vector_context": "",
        "graph_context": "",
    }

# GRAPH BUILDER - LangGraph with conditional routing

def build_chat_graph():
    """
    Build the complete LangGraph with conditional routing
    
    Structure:
    - Entry: plan_intent_node (decides which branch to take)
    - Branch A: database_answer_node (for metadata questions)
    - Branch B: vector_rag_node → graph_rag_node → synthesize_node → generate_answer_node
    - Branch C: speak analysis
    - Exit: END
    """
    builder = StateGraph(ChatState)

    # Add all nodes
    builder.add_node("plan_intent", plan_intent_node)
    builder.add_node("database_answer", database_answer_node)
    builder.add_node("speaker_analytics", speaker_analytics_node)
    builder.add_node("vector_rag", vector_rag_node)
    builder.add_node("graph_rag", graph_rag_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Set entry point
    builder.set_entry_point("plan_intent")

    # CONDITIONAL ROUTING 
    # Routes based on intent determined in plan_intent_node
    def route_based_on_intent(state: ChatState):
        intent = state.get("intent")
        if intent == "meeting_database_lookup":
            return "database_answer"
        elif intent == "speaker_analytics":
            return "speaker_analytics"
        return [Send("vector_rag", state), Send("graph_rag", state)]

    builder.add_conditional_edges(
        "plan_intent",
        route_based_on_intent,
        ["database_answer", "speaker_analytics", "vector_rag", "graph_rag"]
    )

    builder.add_edge("database_answer", END)
    builder.add_edge("speaker_analytics", END)
    builder.add_edge("vector_rag", "synthesize")
    builder.add_edge("graph_rag", "synthesize")
    builder.add_edge("synthesize", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# Singleton instance
_chat_graph = None


def get_chat_graph():
    """Get or create the compiled graph"""
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()
    return _chat_graph



# MAIN ENTRY POINT - Now just delegates to the graph

async def chat_with_meeting(
    meeting_id: str,
    question: str,
    chat_history: list[dict],
) -> ChatResponse:
    logger.info(
        "CHAT START meeting_id=%s question=%r history_length=%s",
        meeting_id,
        question,
        len(chat_history),
    )

    # Convert chat history to LangChain messages
    lc_history: list[BaseMessage] = []
    for msg in chat_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    # Initialize state
    initial_state: ChatState = {
        "meeting_id": meeting_id,
        "question": question,
        "chat_history": lc_history,
        
        # Planning fields
        "intent": "meeting_content",
        "operation": None,
        "start_date": None,
        "end_date": None,
        
        # RAG fields
        "vector_context": "",
        "graph_context": "",
        "final_context": "",
        "sources": [],
        
        # Output fields
        "answer": "",
        "route_used": "",
    }

    # Invoke the graph - all routing happens inside
    graph = get_chat_graph()
    final_state = await graph.ainvoke(initial_state)

    logger.info(
        "CHAT END meeting_id=%s route=%s answer_length=%s sources=%s",
        meeting_id,
        final_state.get("route_used", "unknown"),
        len(final_state.get("answer", "")),
        len(final_state.get("sources", [])),
    )

    # Convert sources to response objects
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
        route_used=final_state.get("route_used", "unknown"),
    )
