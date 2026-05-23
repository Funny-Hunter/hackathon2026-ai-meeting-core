from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.logging import configure_app_logging
from app.services.chat import chat_with_meeting
from app.services.import_service import (
    get_imported_meetings,
    get_transcript_for_chat,
    import_transcript_file,
    list_data_transcripts,
)


configure_app_logging()
app = FastAPI(title="Meeting AI Import Demo")


class ImportRequest(BaseModel):
    filename: str


class ChatRequest(BaseModel):
    question: str
    meeting_id: str | None = None
    chat_history: list[dict] = []


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


@app.get("/api/transcripts")
async def transcripts() -> dict:
    return {"files": list_data_transcripts(), "imported": get_imported_meetings()}


@app.post("/api/import")
async def import_file(request: ImportRequest) -> dict:
    try:
        return await import_transcript_file(request.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    try:
        transcript = get_transcript_for_chat(request.meeting_id)
        response = await chat_with_meeting(
            transcript.meeting_id,
            request.question,
            request.chat_history,
        )
        return response.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


HTML = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Meeting AI Import</title>
  <style>
    :root {
      --ink: #17211d;
      --muted: #65726d;
      --line: #d8dedb;
      --paper: #f4f1ea;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-2: #b45309;
      --dark: #12302c;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(120deg, rgba(15,118,110,.13), transparent 42%),
        repeating-linear-gradient(90deg, rgba(18,48,44,.035) 0 1px, transparent 1px 52px),
        var(--paper);
    }

    main {
      width: min(1240px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 20px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1;
      letter-spacing: 0;
    }

    .kicker {
      margin: 0 0 7px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .08em;
    }

    .status {
      border: 1px solid var(--line);
      background: rgba(255,255,255,.76);
      padding: 9px 12px;
      font-size: 13px;
      font-weight: 800;
      color: var(--dark);
      white-space: nowrap;
    }

    .layout {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 18px;
      align-items: stretch;
    }

    section {
      border: 1px solid var(--line);
      background: rgba(255,255,255,.88);
      min-height: 690px;
      box-shadow: 0 18px 42px rgba(23,33,29,.11);
    }

    .left {
      padding: 18px;
      overflow: auto;
    }

    .right {
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    h2 {
      margin: 0 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }

    .file {
      display: grid;
      gap: 8px;
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      margin-bottom: 10px;
    }

    .file strong {
      line-height: 1.25;
    }

    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    button {
      border: 0;
      min-height: 40px;
      padding: 0 14px;
      color: white;
      background: var(--accent);
      font: inherit;
      font-weight: 900;
      cursor: pointer;
    }

    button.secondary {
      background: var(--dark);
    }

    button:disabled {
      opacity: .55;
      cursor: wait;
    }

    .import-result {
      margin-top: 16px;
      padding: 12px;
      border-left: 4px solid var(--accent-2);
      background: #fff7ed;
      color: #3f2a13;
      font-size: 13px;
      line-height: 1.55;
      white-space: pre-wrap;
    }

    .chat-head {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }

    select, input {
      min-height: 42px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      padding: 0 12px;
      font: inherit;
      outline: none;
    }

    select:focus, input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15,118,110,.14);
    }

    .messages {
      padding: 20px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,.46), rgba(244,241,234,.62));
    }

    .msg {
      width: min(88%, 720px);
      border: 1px solid var(--line);
      background: white;
      padding: 13px 14px;
      line-height: 1.5;
    }

    .msg.user {
      align-self: flex-end;
      background: var(--dark);
      border-color: var(--dark);
      color: white;
    }

    .msg.assistant {
      align-self: flex-start;
    }

    .sources {
      margin-top: 10px;
      display: grid;
      gap: 8px;
    }

    .source {
      border-left: 3px solid var(--accent);
      background: #edf7f5;
      padding: 8px 10px;
      font-size: 13px;
      color: #263530;
    }

    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 16px;
      border-top: 1px solid var(--line);
      background: rgba(255,255,255,.82);
    }

    .chips {
      grid-column: 1 / -1;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .chip {
      border: 1px solid var(--line);
      background: white;
      color: var(--dark);
      padding: 7px 9px;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
    }

    @media (max-width: 900px) {
      header { align-items: start; flex-direction: column; }
      .layout { grid-template-columns: 1fr; }
      section { min-height: 560px; }
      form { grid-template-columns: 1fr; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p class="kicker">Local import</p>
        <h1>Meeting AI</h1>
      </div>
      <div class="status" id="status">Loading data folder</div>
    </header>

    <div class="layout">
      <section class="left">
        <p class="kicker">data/*.json</p>
        <h2>Transcript files</h2>
        <div id="files"></div>
        <div class="import-result" id="result">Import một transcript để đổ vào Qdrant mock vector và Neo4j graph.</div>
      </section>

      <section class="right">
        <div class="chat-head">
          <div>
            <p class="kicker">Chat after import</p>
            <h2 style="margin:0">Ask imported meeting</h2>
          </div>
          <select id="meeting"></select>
        </div>
        <div class="messages" id="messages"></div>
        <form id="form">
          <input id="question" autocomplete="off" placeholder="VD: Ai phụ trách việc gì?" />
          <button id="send" type="submit">Ask</button>
          <div class="chips">
            <span class="chip" data-q="Cuộc họp này chốt những việc gì?">Kết luận</span>
            <span class="chip" data-q="Ai phụ trách việc gì?">Phân công</span>
            <span class="chip" data-q="Rủi ro hoặc vấn đề chính là gì?">Rủi ro</span>
            <span class="chip" data-q="Có nhắc tới deadline không?">Deadline</span>
          </div>
        </form>
      </section>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const files = $("files");
    const result = $("result");
    const status = $("status");
    const meeting = $("meeting");
    const messages = $("messages");
    const form = $("form");
    const input = $("question");
    const send = $("send");

    function addMessage(role, text, sources = []) {
      const node = document.createElement("div");
      node.className = `msg ${role}`;
      node.textContent = text;
      if (sources.length) {
        const sourceWrap = document.createElement("div");
        sourceWrap.className = "sources";
        sources.forEach((source) => {
          const item = document.createElement("div");
          item.className = "source";
          item.textContent = `[${source.timestamp}] ${source.speaker}: ${source.text}`;
          sourceWrap.appendChild(item);
        });
        node.appendChild(sourceWrap);
      }
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    function renderMeetings(imported) {
      meeting.innerHTML = "";
      imported.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.meeting_id;
        option.textContent = `${item.title} (${item.meeting_date || "no date"})`;
        meeting.appendChild(option);
      });
    }

    async function loadFiles() {
      const response = await fetch("/api/transcripts");
      const payload = await response.json();
      files.innerHTML = "";
      payload.files.forEach((item) => {
        const box = document.createElement("div");
        box.className = "file";
        box.innerHTML = `
          <strong>${item.title}</strong>
          <div class="meta">${item.filename}<br>${item.meeting_id} · ${item.meeting_date || "no date"} · ${item.segments} segments</div>
          <button data-file="${item.filename}">Import to DB</button>
        `;
        files.appendChild(box);
      });
      renderMeetings(payload.imported);
      status.textContent = `${payload.files.length} files in data/`;
      document.querySelectorAll("button[data-file]").forEach((button) => {
        button.addEventListener("click", () => importFile(button.dataset.file, button));
      });
    }

    async function importFile(filename, button) {
      button.disabled = true;
      status.textContent = "Importing to DB";
      result.textContent = `Importing ${filename}...`;
      try {
        const response = await fetch("/api/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Import failed");
        result.textContent =
          `Imported: ${payload.title}\\n` +
          `Meeting: ${payload.meeting_id}\\n` +
          `Datetime: ${payload.meeting_datetime || payload.meeting_date}\\n` +
          `Topic key: ${payload.topic_key}\\n` +
          `Qdrant points: ${payload.qdrant_points}\\n` +
          `Neo4j nodes: ${JSON.stringify(payload.graph.nodes)}\\n` +
          `Neo4j relationships: ${JSON.stringify(payload.graph.relationships)}`;
        await loadFiles();
        meeting.value = payload.meeting_id;
        addMessage("assistant", `Đã import ${payload.title}. Bạn có thể hỏi meeting này.`);
      } catch (error) {
        result.textContent = error.message;
      } finally {
        button.disabled = false;
        status.textContent = "Ready";
      }
    }

    async function ask(question) {
      addMessage("user", question);
      send.disabled = true;
      status.textContent = "Answering";
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ meeting_id: meeting.value || null, question }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Chat failed");
        addMessage("assistant", payload.answer, payload.sources || []);
      } catch (error) {
        addMessage("assistant", error.message);
      } finally {
        send.disabled = false;
        status.textContent = "Ready";
        input.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      input.value = "";
      ask(question);
    });

    document.querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => ask(chip.dataset.q));
    });

    loadFiles();
  </script>
</body>
</html>
"""
