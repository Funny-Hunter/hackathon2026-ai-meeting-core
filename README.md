# Hackathon 2026 AI Meeting Core

**The memory layer for AI-native organizations**

AI Meeting transforms conversations into **persistent organizational intelligence**, enabling teams to query, revisit, and act on meeting knowledge long after a meeting ends.

---

# Vision & Product Opportunity

## The Problem

Organizations spend hours in meetings every week, but most of that knowledge disappears once the meeting ends.

Important information is often lost because:

- Action items become fragmented across chats, documents, and project management tools
- Key decisions are buried inside long transcripts
- Teams lose context across weeks of discussions
- The same conversations happen repeatedly because past reasoning is hard to find
- New team members struggle to understand historical decisions


Existing meeting assistants mostly generate static summaries, but they cannot preserve knowledge as a searchable memory system that grows over time.

This creates execution drag, duplicated work, and reduced decision velocity.

---

## Our Solution: AI Meeting

AI Meeting transforms raw meeting conversations into **persistent, queryable organizational intelligence**.

The platform processes meetings end-to-end — from audio understanding to actionable execution workflows.

It provides:

### **Offline Meeting Intelligence**

Upload raw meeting audio and automatically:

- Convert speech to text with speaker-aware transcription
- Detect speakers and align timestamps
- Replay audio directly from transcript segments
- Correct transcription errors using LLM refinement

This turns raw meeting recordings into accurate, structured transcripts.

---

### **Structured Meeting Understanding**

AI automatically extracts:

- Participants
- Key discussion topics
- Decisions and outcomes
- Action items
- Open concerns and unresolved risks
- Recommended next steps

This converts unstructured conversations into structured organizational knowledge.

---

### **Hybrid Conversational Retrieval**

AI Meeting combines:

- **Semantic Memory** using Qdrant for contextual recall
- **Relational Memory** using Neo4j for graph reasoning
- **Speaker-aware chunking** for precise transcript retrieval
- **GraphRAG orchestration** using LangGraph for grounded multi-hop reasoning

Users can ask natural-language questions such as:

- What did Trang say about deployment risks?
- Who owns this action item?
- What concerns remain unresolved?
- How has this topic evolved across meetings?

The result is explainable conversational intelligence grounded in historical meeting context.

---

### **Visual Knowledge Intelligence**

Generate intelligent visualizations such as:

- Meeting mindmaps
- Speaker-topic relationship graphs
- Decision flow diagrams

This makes complex discussions easier to understand at a glance.

---

### **Workflow Automation**

AI converts decisions into execution-ready tasks.

Generated outputs include:

- Assignee
- Task name
- Task description
- Due date

Tasks can be automatically integrated into Jira workflows.

---

AI Meeting turns conversations into memory, memory into knowledge, and knowledge into action.

---

## Why This Matters

AI Meeting enables:

- Preserved decision rationale
- Accountability tracking
- Faster execution cycles
- Better onboarding
- Reduced repeated discussions
- Searchable institutional memory

For startups: faster iteration.

For enterprises: scalable knowledge continuity.

---

## Innovation

Unlike traditional summarizers, AI Meeting provides:

- Hybrid Graph + Vector Memory
- Entity-aware conversational reasoning
- Persistent meeting intelligence
- Explainable retrieval traces
- Long-term organizational context accumulation
- Replay audio from transcript timestamps
- Correct transcript errors with LLM refinement
- Visual Knowledge Intelligence by Intelligent meeting mindmaps
- Integration with Jira for automatic task creation
- Automatic slide generation

---

## Business Potential

Commercialization opportunities:

### SaaS Collaboration Intelligence
For remote teams and startups

- Searchable memory workspace
- Decision traceability
- AI-powered accountability management

### Enterprise Knowledge Infrastructure

- Meeting compliance traceability
- Cross-team memory graphs
- Historical decision intelligence

### Developer API Platform

Expose APIs for:

- Transcript intelligence
- Organizational memory graphs
- Conversational historical reasoning

---
## Chat Pipeline Architecture

![Architecture Overview](data/images/graph_rag.png)
![Meeting Intelligence Insights](data/images/insights_analyze.png)
![alt text](data/images/chat_langgraph_flow.svg)

## Key Features

- Import meeting transcripts from local JSON files.
- Index transcript chunks into Qdrant with OpenAI embeddings for semantic retrieval.
- Store meetings, speakers, segments, topics, and interaction relationships in Neo4j.
- Ask questions about imported meetings.
- Always answer with a hybrid RAG flow: Qdrant semantic context plus Neo4j entity/graph context.
- Log retrieved Qdrant chunks and Neo4j context lines for backend debugging.
- Serve a single-page web UI at `/`.
- Include sample data in `data/transcript_1.json`, `data/transcript_2.json`, and `data/transcript_3.json`.

## Tech Stack

- Python 3.11+
- FastAPI
- LangChain, LangGraph
- OpenAI API
- Neo4j
- Qdrant
- Docker Compose
- uv

## Project Structure

```text
.
├── app/
│   ├── core/                 # Configuration, logging, and OpenAI client
│   ├── models/               # Pydantic models
│   ├── services/             # Import, chat, Neo4j, Qdrant, Graphiti
│   ├── utils/                # Transcript formatting and metadata helpers
│   └── main.py               # FastAPI app + UI HTML
├── data/                     # Sample transcript JSON files
├── API.md                    # API endpoint documentation and examples
├── docker-compose.yml        # Local Neo4j + Qdrant services
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
└── .env.example              # Example environment variables
```

## Requirements

- Docker and Docker Compose
- Python 3.11 or later
- uv
- OpenAI API key

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create the local environment file:

```bash
cp .env.example .env
```

3. Update `OPENAI_API_KEY` in `.env`:

```env
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=meeting_transcripts

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
```

## Run Locally

1. Start Neo4j and Qdrant:

```bash
docker compose up -d
```

2. Run FastAPI:

```bash
uv run uvicorn app.main:app --reload
```

3. Open the application:

```text
http://localhost:8000
```

Neo4j Browser is available at:

```text
http://localhost:7474
```

Default credentials:

```text
username: neo4j
password: password123
```

## Usage

1. Open `http://localhost:8000`.
2. Select a transcript from the list on the left.
3. Click `Import to DB`.
4. After import completes, select the meeting from the dropdown on the right.
5. Type a question or use suggestion chips such as conclusions, assignments, risks, or deadlines.

## API Documentation

API docs are maintained separately in [`API.md`](API.md). When the app is running, FastAPI also serves interactive docs at `http://localhost:8000/docs` and `http://localhost:8000/redoc`.

## RAG Flow

The chat pipeline in `app/services/chat.py` is intentionally fixed to hybrid mode:

1. Convert chat history into LangChain messages.
2. Query Qdrant with the user question using OpenAI embeddings and `similarity_search_with_relevance_scores`.
3. Extract entities from the user question, including topic, speakers, and question type.
4. Query Neo4j with the extracted entities:
   - `topic + two speakers`: speaker interaction on that topic.
   - `topic`: segments connected to that topic.
   - no topic: all meeting segments as fallback.
5. Merge both sources into one prompt context:
   - `=== Qdrant Semantic Context ===`
   - `=== Neo4j Entity Context ===`
6. Ask the answer LLM to respond in Vietnamese.

The response `sources` field contains the Qdrant chunks returned by semantic retrieval. Neo4j context is logged on the backend and included in the final prompt context.

## Debug Logs

Application logs are configured in `app/core/logging.py` and enabled from `app/main.py`. When running the app with the default `LOG_LEVEL=INFO`, chat requests print backend retrieval details:

```text
CHAT QDRANT query_start meeting_id=... question='...'
CHAT QDRANT query_result meeting_id=... chunks=...
CHAT QDRANT chunk meeting_id=... index=... speaker=... timestamp=... score=... text='...'

CHAT NEO4J entity_extract_start meeting_id=... question='...'
CHAT NEO4J entities meeting_id=... topic=... speakers=[...] question_type=...
CHAT NEO4J query_result meeting_id=... query=... lines=... chars=...
CHAT NEO4J context_line meeting_id=... index=... text='...'
```

These logs are not shown in the UI; they are intended for debugging retrieval quality from the terminal running Uvicorn.

## Transcript Format

Transcript files are placed in the `data/` directory with the following format:

```json
{
  "transcript": {
    "meeting_id": "demo-product-sync",
    "title": "Product Sync - AI Meeting Assistant",
    "meeting_date": "2026-05-21",
    "topic": "AI Meeting Assistant",
    "segments": [
      {
        "speaker": "Linh",
        "timestamp": "00:00",
        "text": "Spoken content..."
      }
    ]
  }
}
```

Main fields:

- `meeting_id`: unique meeting identifier.
- `title`: display title.
- `meeting_date`: meeting date.
- `meeting_datetime`: meeting time in ISO-8601 format, if available.
- `topic`: stable topic used to link related meetings.
- `segments`: list of transcript turns containing `speaker`, `timestamp`, and `text`.

## Technical Notes

- `app/services/import_service.py` imports every transcript into both Qdrant and Neo4j. `qdrant_points` is the number of chunks indexed in Qdrant.
- `app/services/qdrant_service.py` builds overlapping speaker-turn chunks, embeds them with `OPENAI_EMBEDDING_MODEL`, and stores vectors plus metadata in Qdrant.
- `app/services/chat.py` always uses hybrid retrieval instead of asking a router model to choose a single retrieval path.
- Graphiti has a dedicated service in `app/services/graphiti_service.py`, but the current default path uses the direct Neo4j service.
- Imported meetings are stored in memory through the `_IMPORTED` variable, so they are lost when the app restarts.
- The import process calls OpenAI to extract topics per speaker, so `OPENAI_API_KEY` is required.

## Useful Commands

```bash
# Start local infrastructure
docker compose up -d

# Stop local infrastructure
docker compose down

# Run the development app
uv run uvicorn app.main:app --reload

# Run tests
uv run pytest
```