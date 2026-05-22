import logging
from typing import TypedDict, Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from app.core.llm import get_llm
from app.models.transcript import ChatResponse, SourceChunk
from app.services import neo4j_service

# Embedding/Graphiti paths are paused for Neo4j-only mode.
# Uncomment these imports and the original node bodies when API keys are available.
# from app.services import graphiti_service, qdrant_service

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    meeting_id: str
    question: str
    chat_history: list[BaseMessage]

    # Routing
    route: Literal["vector_rag", "graph_rag", "hybrid"]

    # Retrieved contexts
    vector_context: str
    graph_context: str
    final_context: str

    # Sources returned to the client
    sources: list[dict]

    # Final answer
    answer: str


# ── Router node───────────────────────────────────────────────────────────────

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the router for a meeting RAG system.
Classify the question into exactly one of these three categories:

- "vector_rag": questions about specific content, quoted statements, or discussion details
  Examples: "What did they say about the deadline?", "Who proposed solution X?"

- "graph_rag": questions about relationships, who interacted with whom, or each person's topics
  Examples: "Did Speaker A and B agree?", "Who mentioned the budget?", "What is the relationship between A and C?"

- "hybrid": complex questions that require both retrieval methods
  Examples: "Summarize each person's view on issue X", "Compare A and B's opinions about the deadline"

Return only one of these three values: vector_rag, graph_rag, hybrid"""),
    ("human", "Question: {question}"),
])


async def router_node(state: ChatState) -> dict:
    llm = get_llm()
    chain = _ROUTER_PROMPT | llm | StrOutputParser()
    raw = await chain.ainvoke({"question": state["question"]})
    route = raw.strip().lower()
    if route not in ("vector_rag", "graph_rag", "hybrid"):
        route = "vector_rag"  # fallback
    logger.info(f"Routed to: {route}")
    return {"route": route}



async def vector_rag_node(state: ChatState) -> dict:
    # Qdrant embedding path, paused while running Neo4j-only mode:
    # retriever = qdrant_service.get_retriever(meeting_id=state["meeting_id"], k=5)
    # docs = await retriever.ainvoke(state["question"])
    #
    # sources = []
    # context_lines = []
    # for doc in docs:
    #     meta = doc.metadata
    #     sources.append({
    #         "speaker": meta.get("speaker", ""),
    #         "timestamp": meta.get("timestamp", ""),
    #         "text": meta.get("text", ""),
    #         "score": meta.get("score", 0.0),
    #     })
    #     context_lines.append(
    #         f"[{meta.get('timestamp', '')}] {meta.get('speaker', '')}: {meta.get('text', '')}"
    #     )

    segments = neo4j_service.query_all_segments(state["meeting_id"])
    sources = segments[:5]
    context_lines = [
        f"[{segment['timestamp']}] {segment['speaker']}: {segment['text']}"
        for segment in segments
    ]
    context = "\n".join(context_lines) if context_lines else "No relevant content found."
    return {
        "vector_context": context,
        "sources": sources,
    }


async def graph_rag_node(state: ChatState) -> dict:
    # Graphiti path, paused for Neo4j-only mode:
    # try:
    #     graph_context = await graphiti_service.search_graphiti(
    #         state["meeting_id"],
    #         state["question"],
    #     )
    # except Exception as exc:
    #     logger.warning("Graphiti search failed: %s", exc)
    #     graph_context = "Graph context is unavailable."

    graph_context = neo4j_service.query_all_segments_text(state["meeting_id"])
    return {"graph_context": graph_context}



async def synthesize_node(state: ChatState) -> dict:
    """Merge vector + graph context for the hybrid route."""
    parts = []
    if state.get("vector_context"):
        parts.append("=== Retrieved Content (Vector RAG) ===\n" + state["vector_context"])
    if state.get("graph_context"):
        parts.append("=== Relationship Information (Graph RAG) ===\n" + state["graph_context"])

    final_context = "\n\n".join(parts) if parts else "No context available."
    return {"final_context": final_context}



async def finalize_vector_node(state: ChatState) -> dict:
    return {"final_context": state.get("vector_context", "")}


async def finalize_graph_node(state: ChatState) -> dict:
    return {"final_context": state.get("graph_context", "")}



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


def route_after_router(state: ChatState) -> str:
    route = state.get("route", "vector_rag")
    if route == "hybrid":
        return "both"
    return route


def route_after_retrieval(state: ChatState) -> str:
    """After retrieval, route to synthesize or finalize."""
    return state.get("route", "vector_rag")



def build_chat_graph():
    builder = StateGraph(ChatState)

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("vector_rag", vector_rag_node)
    builder.add_node("graph_rag", graph_rag_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("finalize_vector", finalize_vector_node)
    builder.add_node("finalize_graph", finalize_graph_node)
    builder.add_node("generate_answer", answer_node)

    # Entry
    builder.set_entry_point("router")

    # Router → retrieval nodes
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "vector_rag": "vector_rag",
            "graph_rag": "graph_rag",
            "both": "vector_rag",  # hybrid: run vector first
        },
    )

    # Hybrid: sau vector_rag → tiếp tục chạy graph_rag
    # Non-hybrid: sau retrieval → finalize
    builder.add_conditional_edges(
        "vector_rag",
        lambda s: "synthesize" if s.get("route") == "hybrid" else "finalize_vector",
        {
            "synthesize": "graph_rag",   
            "finalize_vector": "finalize_vector",
        },
    )

    builder.add_conditional_edges(
        "graph_rag",
        lambda s: "synthesize" if s.get("route") in ("hybrid", "graph_rag") else "finalize_graph",
        {
            "synthesize": "synthesize",
            "finalize_graph": "finalize_graph",
        },
    )

    builder.add_edge("synthesize", "generate_answer")
    builder.add_edge("finalize_vector", "generate_answer")
    builder.add_edge("finalize_graph", "generate_answer")
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
        "route": "vector_rag",
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
        route_used=final_state.get("route", "vector_rag"),
    )
