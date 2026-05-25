# API Documentation

Base URL for local development:

```text
http://localhost:8000
```

The API is implemented in `app/main.py`. It is a FastAPI app, so OpenAPI documentation is also available when the server is running:

- `GET /docs`: Swagger UI.
- `GET /redoc`: ReDoc UI.
- `GET /openapi.json`: raw OpenAPI schema.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the demo single-page web UI. |
| `GET` | `/api/transcripts` | Lists transcript JSON files and meetings imported during the current runtime. |
| `POST` | `/api/import` | Imports a transcript file from `data/` into Qdrant and Neo4j. |
| `POST` | `/api/chat` | Asks a question about an imported meeting. |

## `GET /`

Returns the demo web UI as `text/html`.

Example:

```bash
curl http://localhost:8000/
```

## `GET /api/transcripts`

Lists transcript files in the local `data/` directory and meetings imported in the current app process.

Example:

```bash
curl http://localhost:8000/api/transcripts
```

Response:

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
  "imported": [
    {
      "meeting_id": "demo-product-sync",
      "title": "Product Sync - AI Meeting Assistant",
      "meeting_date": "2026-05-21",
      "segments": 7
    }
  ]
}
```

Fields:

- `files`: transcript files discovered from `data/*.json`.
- `imported`: meetings imported since the FastAPI process started. This list is stored in memory and resets when the app restarts.

## `POST /api/import`

Imports one local transcript file into both storage backends:

- Qdrant: transcript chunks with OpenAI embeddings.
- Neo4j: meeting, speaker, segment, topic, and interaction graph.

Request body:

```json
{
  "filename": "transcript_1.json"
}
```

Curl example:

```bash
curl -X POST http://localhost:8000/api/import \
  -H "Content-Type: application/json" \
  -d '{"filename":"transcript_1.json"}'
```

Response:

```json
{
  "meeting_id": "demo-product-sync",
  "title": "Product Sync - AI Meeting Assistant",
  "meeting_date": "2026-05-21",
  "meeting_datetime": "2026-05-21",
  "topic_key": "product-sync-ai-meeting-assistant",
  "qdrant_points": 7,
  "graph": {
    "backend": "neo4j",
    "meeting_id": "demo-product-sync",
    "nodes": 16,
    "relationships": 32,
    "nodes_created": 16
  }
}
```

Notes:

- `filename` is normalized with `Path(filename).name`, so only files directly under `data/` are loaded.
- Import requires Qdrant, Neo4j, and `OPENAI_API_KEY` to be configured.
- Re-importing the same `meeting_id` clears old Qdrant chunks and replaces the meeting graph data.

Error response:

```json
{
  "detail": "No such file or directory: 'data/missing.json'"
}
```

HTTP status: `500`.

## `POST /api/chat`

Asks a question about a meeting. The chat pipeline first classifies the question, then routes it to one of these paths:

- `database_lookup`: questions about listing or counting meetings.
- `speaker_analytics`: questions about participants or speaker-level information.
- `hybrid`: content questions using Qdrant semantic retrieval plus Neo4j graph/entity context.

Request body:

```json
{
  "meeting_id": "demo-product-sync",
  "question": "Ai phụ trách việc gì?",
  "chat_history": []
}
```

`meeting_id` is optional. If omitted, the app uses the most recently imported meeting. If no meeting has been imported yet, it loads the first transcript file from `data/` into memory for chat selection, but it does not run the full Qdrant/Neo4j import.

`chat_history` is optional and uses this shape:

```json
[
  {
    "role": "user",
    "content": "Cuộc họp này nói về gì?"
  },
  {
    "role": "assistant",
    "content": "Cuộc họp bàn về phạm vi MVP cho trợ lý họp AI."
  }
]
```

Curl example:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_id": "demo-product-sync",
    "question": "Ai phụ trách việc gì?",
    "chat_history": []
  }'
```

Hybrid response example:

```json
{
  "answer": "Nam phụ trách UI chat. Trang phụ trách kiểm tra pipeline ingest và phần graph.",
  "sources": [
    {
      "speaker": "Linh",
      "timestamp": "02:05",
      "text": "MVP tuần này cần có demo local. Nam phụ trách UI chat, Trang kiểm tra pipeline ingest và phần graph.",
      "score": 0.91
    }
  ],
  "route_used": "hybrid"
}
```

Database lookup example:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Có bao nhiêu cuộc họp trong tháng này?",
    "chat_history": []
  }'
```

Response:

```json
{
  "answer": "Trong tháng này có 3 cuộc họp đã được import.",
  "sources": [],
  "route_used": "database_lookup"
}
```

Speaker analytics example:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_id": "demo-product-sync",
    "question": "Cuộc họp có những ai tham gia?",
    "chat_history": []
  }'
```

Response:

```json
{
  "answer": "Cuộc họp có 3 người tham gia: Linh, Nam và Trang.",
  "sources": [],
  "route_used": "speaker_analytics"
}
```

Error response:

```json
{
  "detail": "No transcript files found in data/"
}
```

HTTP status: `400`.

## Data Models

### Transcript File Format

Transcript files must be placed in `data/` and use this shape:

```json
{
  "transcript": {
    "meeting_id": "demo-product-sync",
    "title": "Product Sync - AI Meeting Assistant",
    "meeting_date": "2026-05-21",
    "meeting_datetime": "2026-05-21T09:30:00+07:00",
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

Required fields:

- `meeting_id`
- `segments[].speaker`
- `segments[].timestamp`
- `segments[].text`

Optional fields:

- `title`
- `meeting_date`
- `meeting_datetime`
- `topic`

### Chat Source Chunk

`sources` contains Qdrant semantic retrieval chunks only. Neo4j graph context is included in the LLM prompt and backend logs, but it is not returned as a source item.

```json
{
  "speaker": "Linh",
  "timestamp": "02:05",
  "text": "Transcript chunk text",
  "score": 0.91
}
```

## Operational Requirements

Before calling import or chat endpoints, start local infrastructure and the API server:

```bash
docker compose up -d
uv run uvicorn app.main:app --reload
```

Required environment variables:

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
