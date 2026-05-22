# Hackathon 2026 AI Meeting Core

Backend demo for an AI meeting analysis assistant. The application imports JSON transcripts from the `data/` directory, stores the meeting structure in Neo4j, and lets users ask questions about imported meetings through a simple web UI.

![alt text](data/images/graph_rag.png)

![alt text](data/images/insights_analyze.png)

## Key Features

- Import meeting transcripts from local JSON files.
- Store meetings, speakers, segments, topics, and interaction relationships in Neo4j.
- Ask questions about imported meetings.
- Route RAG requests with LangGraph: `vector_rag`, `graph_rag`, or `hybrid`.
- Serve a single-page web UI at `/`.
- Include sample data in `data/transcript_1.json` and `data/transcript_2.json`.
- Include Qdrant service and embedding code, although Qdrant ingestion is currently paused for Neo4j-only mode.

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
│   ├── core/                 # Configuration and OpenAI client
│   ├── models/               # Pydantic models
│   ├── services/             # Import, chat, Neo4j, Qdrant, Graphiti
│   ├── utils/                # Transcript formatting and metadata helpers
│   └── main.py               # FastAPI app + UI HTML
├── data/                     # Sample transcript JSON files
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

## API

### `GET /`

Returns the demo web UI.

### `GET /api/transcripts`

Lists transcripts in the `data/` directory and meetings imported during the current runtime.

Example response:

```json
{
  "files": [
    {
      "filename": "transcript_1.json",
      "meeting_id": "demo-product-sync",
      "title": "Product Sync - AI Meeting Assistant",
      "meeting_date": "2026-05-21",
      "segments": 7
    }
  ],
  "imported": []
}
```

### `POST /api/import`

Imports a transcript into Neo4j.

Request:

```json
{
  "filename": "transcript_1.json"
}
```

Example response:

```json
{
  "meeting_id": "demo-product-sync",
  "title": "Product Sync - AI Meeting Assistant",
  "meeting_date": "2026-05-21",
  "meeting_datetime": "2026-05-21",
  "topic_key": "product-sync-ai-meeting-assistant",
  "qdrant_points": 0,
  "graph": {
    "backend": "neo4j",
    "meeting_id": "demo-product-sync",
    "nodes": 16,
    "relationships": 32,
    "nodes_created": 16
  }
}
```

### `POST /api/chat`

Asks a question about an imported meeting.

Request:

```json
{
  "meeting_id": "demo-product-sync",
  "question": "Who is responsible for what?",
  "chat_history": []
}
```

Example response:

```json
{
  "answer": "Trang is responsible for checking the ingest pipeline and graph components...",
  "sources": [
    {
      "speaker": "Linh",
      "timestamp": "02:05",
      "text": "This week's MVP needs a local demo...",
      "score": 1.0
    }
  ],
  "route_used": "graph_rag"
}
```

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

- `app/services/import_service.py` currently runs in Neo4j-only mode. Because of that, `qdrant_points` returns `0`.
- Qdrant embedding code already exists in `app/services/qdrant_service.py`. To enable it again, uncomment the import and the `ingest_transcript(transcript)` call in `import_transcript_file()`.
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

# Run tests if test cases are added
uv run pytest
```
